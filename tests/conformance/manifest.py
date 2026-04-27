"""Load R2RML test cases from W3C rdb2rdf `manifest.ttl` files (rdb2rdftest vocabulary)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rdflib import Graph, Namespace, RDF
from rdflib.namespace import DC

RDB2RDFTEST = Namespace("http://purl.org/NET/rdb2rdf-test#")


@dataclass(frozen=True)
class R2RMLTestCase:
    """One W3C R2RML test (one row in a database folder manifest)."""

    w3c_id: str
    """dcterms:identifier, e.g. R2RMLTC0000"""
    database_dir: Path
    """Directory containing create.sql, mapping, and expected output."""
    sql_script: str
    """Filename of the SQL script (usually create.sql)."""
    mapping_file: str
    """Mapping document filename, e.g. r2rml.ttl or r2rmla.ttl"""
    output_file: str
    """Expected output, e.g. mapped.nq; may be empty when ``has_expected_output`` is false."""
    has_expected_output: bool
    """If False, a conforming engine should reject the mapping (expect an exception)."""


def _as_str(node) -> str:
    if node is None:
        return ""
    return str(node)


def _as_bool_obj(node) -> bool:
    """RDF boolean / XSD for rdb2rdftest:hasExpectedOutput."""
    if node is None:
        return True
    s = str(node).strip().lower()
    if s in ("false", "0", "http://www.w3.org/2001/xmlschema#false"):
        return False
    return True


def _iter_r2rml_cases_from_graph(
    g: Graph, database_dir: Path
) -> Iterator[R2RMLTestCase]:
    for tc in g.subjects(RDF.type, RDB2RDFTEST.R2RML):
        # W3C manifests use @prefix dcterms: <http://purl.org/dc/elements/1.1/> (``DC`` in rdflib).
        w3c_id = _as_str(g.value(tc, DC.identifier))
        if not w3c_id:
            warnings.warn(
                f"Skipping rdb2rdftest:R2RML node {tc!s} in {database_dir / 'manifest.ttl'}: "
                "no dcterms:identifier; test case not loaded.",
                UserWarning,
                stacklevel=2,
            )
            continue
        database_node = g.value(tc, RDB2RDFTEST.database)
        sql_file = "create.sql"
        if database_node:
            sfn = g.value(database_node, RDB2RDFTEST.sqlScriptFile)
            if sfn:
                sql_file = _as_str(sfn)
        mapping_f = g.value(tc, RDB2RDFTEST.mappingDocument)
        out_f = g.value(tc, RDB2RDFTEST.output)
        has_out = g.value(tc, RDB2RDFTEST.hasExpectedOutput)
        has_exp = _as_bool_obj(has_out)
        if not mapping_f:
            warnings.warn(
                f"Skipping R2RML test {w3c_id!r} in {database_dir / 'manifest.ttl'}: "
                "no mappingDocument; test case not loaded.",
                UserWarning,
                stacklevel=2,
            )
            continue
        out_s = _as_str(out_f) if out_f else ""
        if has_exp and not out_s:
            warnings.warn(
                f"Skipping R2RML test {w3c_id!r} in {database_dir / 'manifest.ttl'}: "
                "hasExpectedOutput is true but output is missing; test case not loaded.",
                UserWarning,
                stacklevel=2,
            )
            continue
        yield R2RMLTestCase(
            w3c_id=w3c_id,
            database_dir=database_dir,
            sql_script=sql_file,
            mapping_file=_as_str(mapping_f),
            output_file=out_s,
            has_expected_output=has_exp,
        )


def discover_r2rml_test_cases(ts_root: Path) -> list[R2RMLTestCase]:
    """
    Walk ``ts_root`` (unzipped W3C rdb2rdf test suite) for ``D*/manifest.ttl`` and return
    all ``rdb2rdftest:R2RML`` test case entries, sorted by W3C id.
    """
    ts_root = ts_root.resolve()
    if not ts_root.is_dir():
        return []
    out: list[R2RMLTestCase] = []
    for child in sorted(ts_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("D"):
            continue
        mf = child / "manifest.ttl"
        if not mf.is_file():
            continue
        g = Graph()
        g.parse(mf, format="turtle")
        for tc in _iter_r2rml_cases_from_graph(g, child):
            out.append(tc)
    out.sort(key=lambda t: t.w3c_id)
    return out


def default_ts_root() -> Path:
    """``RDB2RDF_TS_DIR`` env, else ``tests/rdb2rdf-test-cases`` under this repo (cwd-relative)."""
    env = os.environ.get("RDB2RDF_TS_DIR", "").strip()
    if env:
        return Path(env).resolve()
    return (Path(__file__).resolve().parent.parent / "rdb2rdf-test-cases").resolve()
