# On-premise invoice extraction → Excel register

Scans invoices (scanned PDFs and phone photos) and appends one row per invoice
to a copy of your Excel template. Nothing leaves your network.

Built for: **2,000–50,000 invoices/month**, **scanned/photo input**,
**fully on-premise**, **one row per invoice**.

---

## No GPU? Use the CPU backend

Set `model.backend: ocr_llm` in `config.yaml`. It OCRs each page with
Tesseract and runs a **text** model over the result, instead of a vision model
over the image. That removes the vision encoder and the 1,000–2,000 image
tokens of prefill per page — which is what makes a VLM impractical on CPU. A
reconstructed invoice page measures around 110 tokens, so only the ~200 tokens
of JSON output actually get generated.

Measured on 4 (weak) cores at 300 DPI, single-threaded:

| Stage | Time |
|---|---|
| Rasterise | 81 ms |
| Preprocess (autocrop + deskew + CLAHE) | 722 ms |
| OCR + line reconstruction | 793 ms |
| **Total, excluding generation** | **~1.5 s/invoice** |

So OCR is not the bottleneck — **generation is**. Budget roughly 10–20 s per
invoice per stream for a 7B model on server CPU, which puts even 50,000/month
(~1,700/day) inside an overnight batch window on a decent multi-core box.

Two tuning levers that matter a lot here:

- **Don't oversubscribe cores.** Running concurrency 12 on 4 cores took 67 s
  for 3 documents; capping it at cores/2 took 14 s for the same work. The
  pipeline now caps this automatically and warns.
- **Try `preprocess.dpi: 150`.** On clean scans it halved preprocess+OCR time
  (1,514 ms → 814 ms) with no field loss. Verify against *your* photos before
  adopting it — this was measured on a clean synthetic invoice.

**The honest trade:** OCR discards the visual layout that makes a VLM good at
this, so expect lower accuracy than the GPU path, worst on poor-quality
photos where OCR itself is the limiting factor. The validation layer is what
keeps it usable — an arithmetic cross-check catches a misread total no matter
which backend produced it.

**Why not pure rules instead of a model?** I tried. Label-anchored regex over
OCR boxes got **5/8 fields** on a *clean synthetic* invoice: `TOTAL` matched
the `Subtotal` line, and `VAT @ 20%` yielded `20`. Rules need per-layout
tuning and degrade badly on photos. A small model over OCR text costs seconds
and generalises across vendors.

### Serving on CPU

```bash
pip install "llama-cpp-python[server]"

python -m llama_cpp.server \
  --model qwen2.5-7b-instruct-q4_k_m.gguf \
  --n_ctx 8192 --n_threads $(nproc) --port 8000
```

Use the largest model your latency budget tolerates — 7B Q4 is a good start;
drop to 3B if too slow. llama.cpp compiles the JSON schema to a GBNF grammar,
so you keep the same structural guarantee as the GPU path.

---

## Why this stack

The on-premise constraint rules out the accuracy leaders — Azure Document
Intelligence, Claude, Gemini — so the question becomes which self-hostable
model is best at *structured extraction* specifically, not at OCR in general.

**Model: [NuExtract3](https://github.com/numindai/nuextract)** (4B, MIT licence).
Purpose-built for schema-driven extraction rather than page transcription, and
reported to beat Qwen3.5-9B by 17 points on NuMind's ~600-document benchmark
(which includes invoices) at under half the size. It takes a typed JSON
template directly, which is exactly this workload's shape.

Swapping models is a one-line config change, so these are drop-in alternatives:

| Model | Size | When to prefer it |
|---|---|---|
| **NuExtract3** *(default)* | 4B | Structured field extraction — this use case |
| Qwen3-VL-8B | 8B | Strongest general open-weight VLM; better on unusual layouts |
| PaddleOCR-VL | 0.9B | Highest throughput; best if you become GPU-bound |
| Granite-Docling + Docling | 258M | If you later need full document → structured Markdown |

**Serving: vLLM or SGLang.** Prefer **SGLang** here — its RadixAttention prefix
caching reuses the shared system prompt across requests, which is reported to
improve throughput 20–40% on repeated-template workloads. Invoices are exactly
that.

**Guided decoding (`guided_json` + xgrammar)** constrains the decoder to your
schema, so malformed JSON stops being a failure mode instead of just a rarer
one. NuMind's own benchmark reports 27 non-deserialisable outputs in 600
documents unconstrained; guided decoding takes that class of error to zero.

**Preprocessing matters more than model choice on phone photos.** Skew,
perspective, shadow and low effective DPI arrive together and each one
degrades extraction. The deskewer here is verified in the test suite to pull a
7° skew down to 0.68°.

**SQLite is the system of record; Excel is a rendering of it.** Excel cannot be
written concurrently, has no unique constraint, and a crash mid-write can take
the workbook with it. Extracting to SQLite first buys idempotent reruns
(content-hash dedup) and lets you regenerate the register after a template or
mapping change without re-running the model.

### Hardware options

A single 24GB GPU (RTX 4090 / L4 / A10) is comfortable for a 4B model. At
50,000/month — about 1,700/day — you have a very large margin; this workload
does not need multi-GPU.

If you don't have one, in rough order of value:

| Option | Cost | Notes |
|---|---|---|
| **Buy a GPU** | ~$800 used 3090 / ~$1,600 4090, one-time | Best long-run value and strictly on-premise. Pays back against rental inside a year. |
| **CPU backend** | £0 | Works today, no new spend. Lower accuracy — see above. |
| **Dedicated GPU server** | from ~€184/mo (Hetzner RTX 4000 SFF Ada) | Single-tenant hardware you control, not a shared AI API — but the data does leave your building, so it's a compliance question, not a technical one. |

Note the CPU and GPU backends share everything except `extract*.py` —
preprocessing, validation, storage and Excel output are identical. Switching
later is a one-line config change, so starting on CPU costs you nothing if you
add a GPU afterwards.

---

## Setup

### 1. Serve the model on your GPU host

```bash
pip install vllm

vllm serve numind/NuExtract3 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 10, "video": 0}' \
  --max-model-len 32768 \
  --guided-decoding-backend xgrammar \
  --port 8000
```

Tight on VRAM? Lower `--max-model-len` to `16384` and `image` to `4`.

### 2. Install the pipeline

```bash
pip install -r requirements.txt
```

### 3. Point it at your template

Put your register at `templates/invoice_register.xlsx`, then edit the
`excel.columns` block in `config.yaml` so each field maps to the right column
letter. Then:

```bash
python -m src.pipeline check
```

This validates the config, cross-checks every mapped column against your
template's actual headers, and confirms the model server is reachable. Run it
after any template change — a column letter that has drifted because someone
inserted a column is the failure that otherwise silently misaligns the whole
register.

### 4. Run

```bash
cp /path/to/invoices/*.pdf inbox/
python -m src.pipeline run
```

Output lands in `output/invoice_register_<timestamp>.xlsx`. Source files move
to `processed/` or `failed/`.

---

## Commands

| Command | Purpose |
|---|---|
| `check` | Validate config, template mapping and server before a run |
| `run` | Process the inbox, store results, export the register |
| `run --no-export` | Extract only; export separately later |
| `export` | Render rows not yet exported |
| `export --all` | Rebuild the full register from scratch |
| `review` | List low-confidence invoices, worst first |

---

## How reliability is achieved

Extraction accuracy alone is not the point — a 95%-accurate extractor with no
validation gives you a register where 1 row in 20 is quietly wrong and you
cannot tell which. Invoices are unusually well suited to verification because
they carry their own consistency constraint.

Every invoice is checked for:

- **`subtotal + tax = total`** within a rounding tolerance. This uses two
  fields to verify a third, so a misread in any of them surfaces.
- Required fields present
- Dates parseable and within a sane year range (catches OCR turning 2025 into 2925)
- Due date not preceding invoice date (catches swapped fields)
- Currency resolvable to an ISO code

Each independent problem halves the row's confidence score. Every row is
written to the register regardless — with its confidence and the specific
problems in audit columns — so nothing is silently dropped and `review` gives
you a queue sorted worst-first.

Amounts are written as real numbers and dates as real dates, not text. Text
-formatted numbers are the usual reason a finance team's totals row reads zero.

---

## Tuning

| Symptom | Change |
|---|---|
| CPU backend too slow | Lower `preprocess.dpi` to 150; use a smaller GGUF |
| CPU box thrashing | Lower `model.concurrency` (auto-capped at cores/2) |
| OCR misses non-English text | Set `model.ocr_lang`, e.g. `eng+deu`, install the language pack |
| Amounts misread on photos | Raise `preprocess.dpi`, `max_edge_px` |
| Autocrop removing content | Set `preprocess.autocrop: false` |
| GPU underutilised | Raise `model.concurrency` |
| Requests timing out | Raise `model.timeout_seconds`, lower `concurrency` |
| Too many rows flagged | Widen `validation.arithmetic_tolerance` |
| Need a new field | Add it to `fields:` **and** `excel.columns:` — nothing else |

---

## Known limits

- **openpyxl does not round-trip charts, images, pivot tables, or some
  conditional formatting.** Fine for a plain register; `check` warns you if
  your template contains anything at risk. If you need them preserved, swap
  `excel_writer.py` for ClosedXML (.NET, best fidelity, no Excel install) or
  xlwings (requires Excel).
- **No line-item extraction.** The schema is one row per invoice, per the
  register format. Line items need a child table and a second sheet.
- `.xlsm` macros are not preserved (would need `keep_vba=True`).

---

## Tests

```bash
python -m tests.test_pipeline
```

53 checks covering amount parsing across US/European/negative notations,
arithmetic validation, deskew effectiveness, PDF rasterisation, dedup
idempotency, and Excel template alignment. No pytest needed.

The full pipeline has additionally been verified end-to-end against a mock
inference server: concurrent extraction, arithmetic flagging, content-hash
dedup on rerun, file routing to `processed`/`failed`, graceful handling of
corrupt files and an unreachable server.

---

## Sources

- [NuExtract3 release](https://about.nuextract.ai/blog/nuextract-3-release) · [repo](https://github.com/numindai/nuextract)
- [Best open-source OCR & document VLMs to self-host, 2026](https://www.spheron.network/blog/best-open-source-ocr-vlm-self-host-gpu-cloud-2026/)
- [vLLM structured outputs](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html) · [XGrammar paper](https://arxiv.org/pdf/2411.15100)
- [Qwen3-VL technical report](https://arxiv.org/pdf/2511.21631)
- [Multi-stage field extraction of financial documents with OCR and compact VLMs](https://arxiv.org/pdf/2510.23066)
- [Open-source OCR for invoice extraction](https://invoicedataextraction.com/blog/open-source-ocr-invoice-extraction)
