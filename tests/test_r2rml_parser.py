import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from pyspark.sql import SparkSession

from r2r import Mapping
from r2r.r2rml_parser import from_r2rml

# Minimal R2RML in Turtle: one triples map with logical table, subject map, and one predicate–object map.
# Expand this fixture as the parser implementation grows.
EXAMPLE_R2RML = """
@prefix rr: <http://www.w3.org/ns/r2rml#> .
@prefix ex: <http://example.com/ns#> .

<#TripleMap>
  a rr:TriplesMap ;
  rr:logicalTable [ rr:tableName "example_catalog.example_schema.products" ] ;
  rr:subjectMap [ rr:template "http://example.com/product/{ID}" ] ;
  rr:predicateObjectMap [
    rr:predicate ex:product_name ;
    rr:objectMap [ rr:column "product_name" ]
  ] .
"""


class TestR2RmlParser(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("r2rml-parser-test")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_from_r2rml_returns_list_of_mappings(self) -> None:
        """Parse example R2RML from a file; result materialized as a list of ``Mapping`` instances."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.ttl"
            path.write_text(EXAMPLE_R2RML, encoding="utf-8")
            mappings = list(from_r2rml(str(path), self.spark))
            self.assertIsInstance(mappings, list)
            for m in mappings:
                self.assertIsInstance(m, Mapping)


if __name__ == "__main__":
    unittest.main()
