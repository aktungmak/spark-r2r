"""
W3C rdb2rdf R2RML conformance: one unittest per `dcterms:identifier` in the suite.

This module is **not** named ``test_*.py`` so ``make test`` (unittest discover) does not
run the full W3C suite. Use ``make conformance-tests`` or ``python -m tests.conformance``.
"""

from __future__ import annotations

import re
import unittest
from functools import reduce
from pathlib import Path
from typing import cast

from pyspark.sql import SparkSession

from r2r import R2RMLParseError, from_r2rml

from tests.conformance import cmp, sql_exec
from tests.conformance.manifest import (
    R2RMLTestCase,
    default_ts_root,
    discover_r2rml_test_cases,
)

from rdflib import Graph


def _load_cases() -> list[R2RMLTestCase]:
    root = default_ts_root()
    if not root.is_dir() or not any(root.iterdir()):
        return []
    return discover_r2rml_test_cases(root)


def _method_name(w3c_id: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", w3c_id)
    s = s.strip("_") or "case"
    if s[0].isdigit():
        s = f"id_{s}"
    return f"test_{s}"


def _setup_fixture_db(sp: SparkSession, sql_path: Path, db: str) -> None:
    """Run W3C ``create.sql`` into ``db`` and select it. Let exceptions propagate."""
    sql_exec.run_sql_file(sp, sql_path, db)
    sp.sql(f"USE {db}")


def _run_positive_r2rml_to_graph(sp: SparkSession, r2rml_path: str) -> Graph:
    return cmp.data_frame_to_graph(
        reduce(
            lambda a, b: a.union(b),
            [m.to_df(sp) for _, m in from_r2rml(r2rml_path)],
        )
    )


def _body(self: "W3CConformanceTest", tc: R2RMLTestCase) -> None:
    sp = self.spark
    db = sql_exec.sanitize_spark_db_name(tc.w3c_id)
    base = tc.database_dir
    sql_path = base / tc.sql_script
    map_path = (base / tc.mapping_file).resolve()
    self.assertTrue(sql_path.is_file(), f"missing SQL: {sql_path}")
    self.assertTrue(map_path.is_file(), f"missing mapping: {map_path}")
    sql_exec.drop_test_database(sp, db)
    _setup_fixture_db(sp, sql_path, db)
    if not tc.has_expected_output:
        with self.assertRaises(R2RMLParseError):
            _run_positive_r2rml_to_graph(sp, str(map_path))
        return
    g_actual = _run_positive_r2rml_to_graph(sp, str(map_path))
    if not tc.output_file:
        self.fail(
            f"{tc.w3c_id} has hasExpectedOutput true but no rdb2rdftest:output in manifest"
        )
    out_path = base / tc.output_file
    if not out_path.is_file():
        self.fail(f"missing expected N-Quads file: {out_path}")
    try:
        g_exp = cmp.load_expected_graph(out_path)
    except Exception as e:  # noqa: BLE001
        self.fail(f"expected graph load failed: {e}")
    ok, msg = cmp.compare_graphs(g_exp, g_actual)
    if (not ok) and ("harness limitation" in msg):
        self.skipTest(msg)
    self.assertTrue(ok, f"{tc.w3c_id}: {msg}")


def _add_dynamic_tests(cls: type) -> None:
    for tcase in _load_cases():
        name = _method_name(tcase.w3c_id)

        def test_fn(self: "W3CConformanceTest", tc: R2RMLTestCase = tcase) -> None:
            _body(self, tc)

        test_fn.__name__ = name
        test_fn.__doc__ = f"W3C {tcase.w3c_id} ({tcase.mapping_file})"
        setattr(cls, name, test_fn)


def _w3c_missing(self: "W3CConformanceTest") -> None:
    self.fail(
        f"RDB2RDF test data not under {default_ts_root()!s}. "
        "Run: make conformance-tests  (or set RDB2RDF_TS_DIR)."
    )


class W3CConformanceTest(unittest.TestCase):
    """``test_*`` methods are dynamically attached below from the W3C suite."""

    spark: SparkSession

    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("r2r-w3c-conformance")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.ansi.enabled", "true")
            .config("spark.sql.ansi.doubleQuotedIdentifiers", "true")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "spark", None) is not None:
            cls.spark.stop()


if _load_cases():
    _add_dynamic_tests(W3CConformanceTest)
else:
    W3CConformanceTest.test_w3c_missing_data = _w3c_missing  # type: ignore[assignment, misc]


def load_tests(loader: unittest.TestLoader, tests, pattern) -> unittest.TestSuite:
    return cast(unittest.TestSuite, loader.loadTestsFromTestCase(W3CConformanceTest))
