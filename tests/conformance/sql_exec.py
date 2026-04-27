"""Run W3C SQL setup scripts in Spark (best-effort SQL-2008 → Spark SQL)."""

from __future__ import annotations

import re
from pathlib import Path

from pyspark.sql import SparkSession


def _split_statements(sql_text: str) -> list[str]:
    out: list[str] = []
    for raw in re.split(r";", sql_text):
        part = raw.strip()
        if part and not part.startswith("--"):
            out.append(part)
    return out


def run_sql_file(spark: SparkSession, sql_path: Path, database: str) -> None:
    """``CREATE DATABASE``, ``USE``, then each statement in ``sql_path``."""
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
    spark.sql(f"USE {database}")
    text = sql_path.read_text(encoding="utf-8", errors="replace")
    for stmt in _split_statements(text):
        if stmt:
            spark.sql(stmt)


def drop_test_database(spark: SparkSession, database: str) -> None:
    try:
        spark.sql(f"DROP DATABASE IF EXISTS {database} CASCADE")
    except Exception:  # noqa: BLE001
        try:
            spark.sql(f"DROP DATABASE IF EXISTS {database}")
        except Exception:
            pass


def sanitize_spark_db_name(w3c_id: str) -> str:
    """Build a valid Spark / Hive database name for this test id."""
    t = re.sub(r"[^0-9a-zA-Z_]+", "_", w3c_id).strip("_")
    t = t.lower()
    if not t[0].isalpha():
        t = "d_" + t
    return f"w3c_tc_{t[:200]}"
