# On-premise invoice extraction → Excel register

Scans invoices (scanned PDFs and phone photos) and appends one row per invoice
to a copy of your Excel template. Nothing leaves your network.

Built for: **2,000–50,000 invoices/month**, **scanned/photo input**,
**fully on-premise**, **one row per invoice**.

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

### Hardware

A single 24GB GPU (RTX 4090 / L4 / A10) is comfortable for a 4B model. At
50,000/month — about 1,700/day — you have a very large margin; this workload
does not need multi-GPU.

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
