"""Convert R2RML string templates (``rr:template``) to Spark ``format_string`` / ``lit`` columns."""

import re

from pyspark.sql import Column
from pyspark.sql.functions import col, format_string, lit

# Column name is any run of characters other than ``}`` (typical SQL identifier in R2RML).
_BRACED_COLUMN = re.compile(r"\{([^}]+)\}")


def normalise_r2rml_sql_identifier(s: str) -> str:
    """
    R2RML / SQL-quoted identifiers in Turtle are often written as e.g. '\\\"Student\\\"';
    the rdflib lexical for the table name is then '\\\"Student\\\"' (one string with quotes),
    and braced {\\\"Name\\\"} in templates resolves to a column label including quotes.
    Strip a single layer of double-quotes for Spark identifiers.
    """
    t = s.strip()
    if len(t) >= 2 and t.startswith('"') and t.endswith('"'):
        return t[1:-1]
    return t


def _r2rml_template_to_printf_template(r2rml: str) -> tuple[str, list[str]]:
    """
    printf pattern for ``format_string`` (``%s`` per column) and column names in order.
    Literal ``%`` in the R2RML string is doubled for printf. If there are no braced
    column refs, the pattern is unchanged and the column list is empty.
    """
    if not _BRACED_COLUMN.search(r2rml):
        return r2rml, []

    columns: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        columns.append(normalise_r2rml_sql_identifier(m.group(1)))
        return "%s"

    pat = _BRACED_COLUMN.sub(_sub, r2rml.replace("%", "%%"))
    return pat, columns


def r2rml_template_to_format_string(r2rml: str) -> Column:
    """``lit`` if there are no column refs, else ``format_string`` (printf ``%s``)."""
    pat, names = _r2rml_template_to_printf_template(r2rml)
    if not names:
        return lit(pat)
    return format_string(pat, *(col(n) for n in names))
