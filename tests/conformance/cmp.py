"""Compare W3C expected graphs (N-Quads / N-Triples) to Spark ``to_df()`` triple tables."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple, Union

from pyspark.sql import DataFrame
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import XSD

from r2r.mapping import (
    OBJECT_COLUMN,
    OBJECT_TYPE_COLUMN,
    PREDICATE_COLUMN,
    SUBJECT_COLUMN,
    RDF_TYPE_IRI,
)


def _align_xsd_string_literals(g: Graph) -> Graph:
    """
    W3C gold files often use plain string literals; `TripleMap` emits `xsd:string`.
    Strip ``xsd:string`` so ``rdflib.compare.isomorphic`` is meaningful.
    """
    h = Graph()
    for s, p, o in g:
        if (
            isinstance(o, Literal)
            and o.datatype is not None
            and o.datatype == XSD.string
        ):
            o = Literal(str(o))
        h.add((s, p, o))
    return h


def load_expected_graph(path: Path) -> Graph:
    """Load N-Quads, N-Triples, or Turtle from a W3C expected-output file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    non_comment = "\n".join(
        ln for ln in text.splitlines() if not re.match(r"^\s*#", ln) and ln.strip()
    )
    if not non_comment.strip():
        return Graph()
    g = Graph()
    last_err: Optional[Exception] = None
    for fmt in ("nquads", "nt", "turtle", "n3"):
        try:
            g = Graph()
            g.parse(data=non_comment, format=fmt)
            return g
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"Could not parse expected file {path}: {last_err}")


def _object_term(oval, otype: Optional[str]) -> Union[URIRef, Literal]:
    if oval is None:
        return Literal("")
    s = str(oval)
    tstr = (str(otype) if otype is not None else "").strip()
    if not tstr or tstr == "None" or tstr.lower() in ("null", "none"):
        if s.startswith("http://") or s.startswith("https://") or s.startswith("urn:"):
            return URIRef(s)
        return Literal(s)

    if tstr.startswith("@"):
        return Literal(s, lang=tstr[1:].strip() or "und")

    if tstr.startswith("http://") or tstr.startswith("https://"):
        if tstr == RDF_TYPE_IRI and s.startswith("http"):
            return URIRef(s)
        return Literal(s, datatype=URIRef(tstr))
    return Literal(s)


def data_frame_to_graph(df: DataFrame) -> Graph:
    """Build an rdflib graph from a `TripleMap.to_df` result (s, p, o, ot)."""
    g = Graph()
    for row in df.collect():
        d = row.asDict()
        s = d[SUBJECT_COLUMN]
        p = d[PREDICATE_COLUMN]
        o = d[OBJECT_COLUMN]
        ot = d.get(OBJECT_TYPE_COLUMN)
        st = _subject_term(s)
        pt = URIRef(str(p))
        ot_t = _object_term(o, str(ot) if ot is not None else None)
        g.add((st, pt, ot_t))
    return g


def _subject_term(s) -> Union[URIRef, BNode]:
    if s is None:
        return BNode("null")
    ss = str(s)
    if ss.startswith("_:"):
        return BNode(ss[2:] if ss.startswith("_:") else ss)
    if ss.startswith("http://") or ss.startswith("https://") or ss.startswith("urn:"):
        return URIRef(ss)
    return BNode(ss)


def _triple_n3_set(g: Graph) -> set[tuple[str, str, str]]:
    return {(a.n3(), b.n3(), c.n3()) for a, b, c in g}


def compare_graphs(expected: Graph, actual: Graph) -> Tuple[bool, str]:
    """``rdflib`` isomorphism after aligning string literals; fallback n3-set diff for errors."""
    e = _align_xsd_string_literals(expected)
    a = _align_xsd_string_literals(actual)
    if isomorphic(e, a):
        return True, ""
    ex = _triple_n3_set(e)
    ac = _triple_n3_set(a)
    if ex == ac:
        return True, ""
    missing = ex - ac
    extra = ac - ex
    return (
        False,
        f"graphs not isomorphic; n3 triple diff: {len(missing)} missing, {len(extra)} extra; "
        f"e.g. missing {next(iter(missing)) if missing else '—'}; "
        f"e.g. extra {next(iter(extra)) if extra else '—'}",
    )
