"""Orchestration and CLI.

    python -m src.pipeline check                 # validate config + template
    python -m src.pipeline run [--inbox DIR]     # process, store, export
    python -m src.pipeline export [--all]        # re-render the register
    python -m src.pipeline review                # list low-confidence rows

Preprocessing is CPU-bound and the model call is IO-bound, so the two are
split: pages are prepared in a thread pool while extraction requests are kept
in flight against the server. The server does its own batching, so the useful
concurrency here is "enough requests queued to keep the GPU fed" rather than
one worker per document.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from . import excel_writer
from .extract import ExtractionError, extract
from .extract_cpu import extract_cpu
from .preprocess import SUPPORTED_SUFFIXES, prepare
from .schema import load_fields
from .store import Store, file_hash
from .validate import validate

log = logging.getLogger("pipeline")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    with path.open() as handle:
        return yaml.safe_load(handle)


def discover(inbox: Path) -> list[Path]:
    # Created rather than demanded: the inbox holds live invoices and so is
    # gitignored, meaning a fresh clone never has one.
    inbox.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in inbox.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


async def _process_one(
    path: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    config: dict[str, Any],
    fields: list,
    store: Store,
    loop: asyncio.AbstractEventLoop,
) -> tuple[Path, bool, str]:
    """Returns (path, success, message)."""
    try:
        digest = file_hash(path)
        if store.already_processed(digest):
            return path, True, "skipped (already processed)"

        # OpenCV releases the GIL, so a thread here genuinely overlaps with
        # in-flight requests rather than just deferring the work.
        pages = await loop.run_in_executor(
            None, prepare, path, config["preprocess"]
        )

        async with semaphore:
            if config["model"].get("backend", "vlm") == "ocr_llm":
                raw = await extract_cpu(client, pages, fields, config["model"], loop)
            else:
                raw = await extract(client, pages, fields, config["model"])

        result = validate(raw, fields, config["validation"])
        store.upsert(
            digest=digest,
            source_file=path.name,
            values=result.values,
            flags=result.flags,
            confidence=result.confidence,
        )

        if result.needs_review:
            return path, True, f"flagged ({len(result.flags)}): {result.flags[0]}"
        return path, True, "ok"

    except ExtractionError as exc:
        return path, False, f"extraction failed: {exc}"
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        log.debug("unhandled error on %s", path, exc_info=True)
        return path, False, f"{type(exc).__name__}: {exc}"


async def run(config: dict[str, Any], inbox: Path | None, no_export: bool) -> int:
    fields = load_fields(config)
    paths_cfg = config.get("paths", {})
    inbox = inbox or Path(paths_cfg.get("inbox", "inbox"))

    files = discover(inbox)
    if not files:
        print(f"No supported documents found in {inbox}")
        return 0

    model_cfg = config["model"]
    concurrency = int(model_cfg.get("concurrency", 12))

    # On the CPU backend the work happens locally, so oversubscribing cores
    # makes every document slower without raising throughput. Tesseract is
    # itself multi-threaded, hence cores/2 rather than cores.
    if model_cfg.get("backend") == "ocr_llm":
        safe = max(1, (os.cpu_count() or 4) // 2)
        if concurrency > safe:
            log.warning(
                "concurrency %d exceeds what %d cores can serve on the CPU "
                "backend; reducing to %d (set model.concurrency to override)",
                concurrency, os.cpu_count() or 4, safe,
            )
            concurrency = safe

    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()

    print(f"Processing {len(files)} document(s) at concurrency {concurrency}...")

    processed_dir = Path(paths_cfg.get("processed", "processed"))
    failed_dir = Path(paths_cfg.get("failed", "failed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    succeeded = failed = 0

    with Store(config["storage"]["db_path"]) as store:
        limits = httpx.Limits(max_connections=concurrency + 4)
        async with httpx.AsyncClient(
            base_url=model_cfg["base_url"],
            timeout=float(model_cfg.get("timeout_seconds", 180)),
            limits=limits,
        ) as client:
            tasks = [
                _process_one(p, client, semaphore, config, fields, store, loop)
                for p in files
            ]
            for coro in asyncio.as_completed(tasks):
                path, ok, message = await coro
                marker = "  ok " if ok else "FAIL "
                print(f"  {marker} {path.name}: {message}")

                destination = processed_dir if ok else failed_dir
                try:
                    shutil.move(str(path), str(destination / path.name))
                except OSError as exc:
                    log.warning("could not move %s: %s", path.name, exc)

                if ok:
                    succeeded += 1
                else:
                    failed += 1

        stats = store.stats()
        print(
            f"\n{succeeded} succeeded, {failed} failed. "
            f"Store holds {stats['total']} invoice(s), "
            f"{stats['flagged']} flagged for review."
        )

        if not no_export and stats["unexported"]:
            rows = store.pending_export()
            out = excel_writer.export(rows, config["excel"], fields)
            store.mark_exported([r["file_hash"] for r in rows])
            print(f"Register written to {out}")

    return 1 if failed else 0


def cmd_check(config: dict[str, Any]) -> int:
    fields = load_fields(config)
    print(f"Config OK — {len(fields)} field(s) defined.")

    try:
        warnings = excel_writer.verify_template(config["excel"], fields)
    except excel_writer.TemplateError as exc:
        print(f"\nTemplate problem:\n  {exc}")
        return 1

    if warnings:
        print("\nTemplate warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Template OK — column mapping matches headers.")

    base = config["model"]["base_url"]
    try:
        response = httpx.get(f"{base}/models", timeout=5)
        response.raise_for_status()
        served = [m.get("id") for m in response.json().get("data", [])]
        print(f"\nInference server reachable at {base}")
        print(f"  serving: {', '.join(served) or '(none reported)'}")
        if config["model"]["name"] not in served:
            print(
                f"  warning: configured model {config['model']['name']!r} "
                f"is not in that list"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"\nInference server NOT reachable at {base}: {exc}")
        print("  Start it with the vllm serve command in README.md.")
        return 1

    return 0


def cmd_export(config: dict[str, Any], export_all: bool) -> int:
    fields = load_fields(config)
    with Store(config["storage"]["db_path"]) as store:
        rows = store.all_rows() if export_all else store.pending_export()
        if not rows:
            print("Nothing to export.")
            return 0
        out = excel_writer.export(rows, config["excel"], fields)
        if not export_all:
            store.mark_exported([r["file_hash"] for r in rows])
        print(f"Wrote {len(rows)} row(s) to {out}")
    return 0


def cmd_review(config: dict[str, Any]) -> int:
    threshold = float(config["validation"].get("review_threshold", 0.8))
    with Store(config["storage"]["db_path"]) as store:
        rows = store.review_queue(threshold)
        if not rows:
            print(f"No invoices below confidence {threshold}.")
            return 0
        print(f"{len(rows)} invoice(s) below confidence {threshold}:\n")
        for row in rows:
            payload = json.loads(row["payload"])
            print(f"  {row['source_file']}  (confidence {row['confidence']})")
            print(
                f"    vendor: {payload.get('vendor_name')}   "
                f"total: {payload.get('total_amount')} {payload.get('currency') or ''}"
            )
            for flag in json.loads(row["flags"]):
                print(f"    - {flag}")
            print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="invoice-pipeline")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate config, template and server")

    run_parser = sub.add_parser("run", help="process the inbox")
    run_parser.add_argument("--inbox", type=Path, default=None)
    run_parser.add_argument(
        "--no-export", action="store_true", help="extract only, skip the .xlsx"
    )

    export_parser = sub.add_parser("export", help="render the register")
    export_parser.add_argument(
        "--all", action="store_true",
        help="include already-exported rows (rebuilds the full register)",
    )

    sub.add_parser("review", help="list low-confidence invoices")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)

    if args.command == "check":
        return cmd_check(config)
    if args.command == "run":
        return asyncio.run(run(config, args.inbox, args.no_export))
    if args.command == "export":
        return cmd_export(config, args.all)
    if args.command == "review":
        return cmd_review(config)
    return 1


if __name__ == "__main__":
    sys.exit(main())
