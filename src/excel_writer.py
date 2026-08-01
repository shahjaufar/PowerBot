"""Renders the register into a copy of your .xlsx template.

Three rules this module enforces, each of which is a way people lose data:

1. The template is never opened for writing. Every run saves to a new file in
   `output/`, so a bad run costs you a file you can delete, not your template.

2. Values are written; styling is not. Setting `cell.value` leaves the
   template's fonts, borders and column widths intact — openpyxl only rebuilds
   what you touch.

3. Appends start below the last non-empty row, so repeat runs accumulate.

One caveat worth knowing: openpyxl does not round-trip charts, images, pivot
tables or some conditional formatting. For a plain register that is fine, and
`verify_template()` will warn you if your template contains anything at risk.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string
from openpyxl.worksheet.worksheet import Worksheet

from .schema import Field

log = logging.getLogger(__name__)

# Columns the pipeline adds beyond the extracted fields.
AUDIT_COLUMNS = {"source_file", "confidence", "review_flags"}


class TemplateError(RuntimeError):
    pass


def verify_template(cfg: dict[str, Any], fields: list[Field]) -> list[str]:
    """Check the template against the config before a run writes anything.

    Cheap to run and catches the two mistakes that otherwise surface as a
    silently misaligned register: a column letter that has drifted after
    someone inserted a column, and template features openpyxl will drop.
    """
    template = Path(cfg["template_path"])
    if not template.exists():
        raise TemplateError(
            f"template not found: {template}\n"
            f"Put your register template there, or point excel.template_path "
            f"at it in config.yaml."
        )

    warnings: list[str] = []
    workbook = load_workbook(template)
    sheet = _select_sheet(workbook, cfg.get("sheet_name"))

    if getattr(sheet, "_charts", None):
        warnings.append(
            f"template contains {len(sheet._charts)} chart(s); openpyxl will "
            f"drop them on save — use ClosedXML or xlwings if you need them"
        )
    if getattr(sheet, "_images", None):
        warnings.append(
            f"template contains {len(sheet._images)} image(s); openpyxl will "
            f"drop them on save"
        )
    if template.suffix.lower() == ".xlsm":
        warnings.append(
            "template is .xlsm; macros are only preserved if you load with "
            "keep_vba=True — this pipeline does not"
        )

    # Cross-check declared column letters against the template's own headers.
    header_row = int(cfg.get("header_row", 1))
    known = {f.name for f in fields} | AUDIT_COLUMNS
    for field_name, column in cfg.get("columns", {}).items():
        if field_name not in known:
            warnings.append(
                f"config maps unknown field {field_name!r} to column {column} "
                f"— it will never be written"
            )
            continue
        header = sheet.cell(row=header_row, column=column_index_from_string(column)).value
        if header is None:
            warnings.append(
                f"{field_name} -> column {column}: no header text in row "
                f"{header_row}; check the column letter is still correct"
            )

    workbook.close()
    return warnings


def _select_sheet(workbook: Any, sheet_name: str | None) -> Worksheet:
    if sheet_name:
        if sheet_name not in workbook.sheetnames:
            raise TemplateError(
                f"sheet {sheet_name!r} not in template "
                f"(available: {workbook.sheetnames})"
            )
        return workbook[sheet_name]
    return workbook.active


def _last_used_row(sheet: Worksheet, columns: list[int], header_row: int) -> int:
    """Last row with content in any mapped column.

    `sheet.max_row` counts rows that merely carry formatting, which on a
    styled template can be hundreds of empty rows — appending there would
    leave a large gap.
    """
    last = header_row
    for row_idx in range(header_row + 1, sheet.max_row + 1):
        if any(sheet.cell(row=row_idx, column=c).value not in (None, "") for c in columns):
            last = row_idx
    return last


def _cell_value(field_name: str, raw: Any, fields_by_name: dict[str, Field]) -> Any:
    """Convert a stored value into something Excel will treat as native.

    Amounts go in as float and dates as datetime, so the register can be
    summed and sorted. Text-formatted numbers are the usual reason a finance
    team's totals row reads zero.
    """
    if raw is None or raw == "":
        return None

    field = fields_by_name.get(field_name)
    if field is None:
        return raw

    if field.type in ("currency", "number"):
        try:
            return float(Decimal(str(raw)))
        except (InvalidOperation, ValueError):
            return raw

    if field.type == "date":
        if isinstance(raw, (date, datetime)):
            return raw
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            return raw

    return raw


def export(
    rows: list[Any],
    cfg: dict[str, Any],
    fields: list[Field],
    output_name: str | None = None,
) -> Path:
    """Append rows to a fresh copy of the template. Returns the output path."""
    template = Path(cfg["template_path"])
    if not template.exists():
        raise TemplateError(f"template not found: {template}")

    output_dir = Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / (output_name or f"invoice_register_{stamp}.xlsx")

    workbook = load_workbook(template)
    sheet = _select_sheet(workbook, cfg.get("sheet_name"))

    columns: dict[str, int] = {
        name: column_index_from_string(letter)
        for name, letter in cfg.get("columns", {}).items()
    }
    if not columns:
        raise TemplateError("excel.columns is empty in config.yaml")

    fields_by_name = {f.name: f for f in fields}
    header_row = int(cfg.get("header_row", 1))
    write_row = _last_used_row(sheet, list(columns.values()), header_row) + 1

    date_format = cfg.get("date_format", "yyyy-mm-dd")
    currency_format = cfg.get("currency_format", "#,##0.00")

    for row in rows:
        payload = json.loads(row["payload"])
        flags = json.loads(row["flags"])

        record: dict[str, Any] = dict(payload)
        record["source_file"] = row["source_file"]
        record["confidence"] = float(row["confidence"])
        record["review_flags"] = "; ".join(flags) if flags else ""

        for name, col_idx in columns.items():
            if name not in record:
                continue
            cell = sheet.cell(row=write_row, column=col_idx)
            cell.value = _cell_value(name, record[name], fields_by_name)

            field = fields_by_name.get(name)
            if field and field.type == "date" and cell.value is not None:
                cell.number_format = date_format
            elif field and field.type in ("currency", "number") and cell.value is not None:
                cell.number_format = currency_format

        write_row += 1

    workbook.save(out_path)
    workbook.close()
    log.info("wrote %d row(s) to %s", len(rows), out_path)
    return out_path
