"""Turns the `fields:` block of config.yaml into the artefacts the rest of the
pipeline needs: a JSON Schema for guided decoding, and a prompt fragment.

Keeping this derived rather than hand-written means the config file stays the
single place a new field has to be declared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Schema types are deliberately permissive: the model emits strings, and the
# validator coerces them. Asking a model for a JSON number invites it to invent
# one when the field is absent, whereas an empty string is an honest "not found".
_JSON_TYPE = {
    "string": "string",
    "date": "string",
    "currency": "string",
    "number": "string",
    "integer": "string",
}


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    required: bool
    description: str

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "Field":
        if "name" not in raw:
            raise ValueError(f"field entry is missing 'name': {raw!r}")
        ftype = raw.get("type", "string")
        if ftype not in _JSON_TYPE:
            raise ValueError(
                f"field {raw['name']!r} has unknown type {ftype!r}; "
                f"expected one of {sorted(_JSON_TYPE)}"
            )
        return cls(
            name=raw["name"],
            type=ftype,
            required=bool(raw.get("required", False)),
            description=raw.get("description", ""),
        )


def load_fields(config: dict[str, Any]) -> list[Field]:
    fields = [Field.from_config(f) for f in config.get("fields", [])]
    if not fields:
        raise ValueError("config.yaml defines no fields")
    names = [f.name for f in fields]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate field names in config: {sorted(duplicates)}")
    return fields


def json_schema(fields: list[Field]) -> dict[str, Any]:
    """JSON Schema handed to the inference server for guided decoding.

    Every field is marked required and `additionalProperties` is disabled, so
    the decoder is constrained to emit exactly our keys in exactly our order.
    A field the model cannot find comes back as "" rather than going missing,
    which keeps the downstream code free of key-presence checks.
    """
    return {
        "type": "object",
        "properties": {
            f.name: {"type": _JSON_TYPE[f.type], "description": f.description}
            for f in fields
        },
        "required": [f.name for f in fields],
        "additionalProperties": False,
    }


def prompt_fields_block(fields: list[Field]) -> str:
    """Human-readable field list embedded in the prompt.

    The schema constrains *shape*; this explains *meaning*. Both are needed —
    guided decoding will happily produce a well-formed wrong answer.
    """
    lines = []
    for f in fields:
        hint = {
            "date": " (ISO 8601: YYYY-MM-DD)",
            "currency": " (digits and at most one decimal point, no symbols or thousands separators)",
            "number": " (digits and at most one decimal point)",
            "integer": " (digits only)",
        }.get(f.type, "")
        need = "required" if f.required else "optional"
        lines.append(f"- {f.name} [{need}]{hint}: {f.description}")
    return "\n".join(lines)
