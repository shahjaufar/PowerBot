# Templates

Put your Excel register template here as `invoice_register.xlsx`, then map its
columns in the `excel.columns` block of `config.yaml`.

Actual workbooks are gitignored — the template is yours, not part of the repo.
Run `python -m src.pipeline check` after any template change to confirm the
column mapping still lines up with your headers.
