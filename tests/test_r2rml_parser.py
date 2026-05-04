import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from pyspark.sql import SparkSession

from r2r import Mapping

# Minimal R2RML in Turtle: one triples map with logical table, subject map, and one predicate–object map.
# Expand this fixture as the parser implementation grows.
EXAMPLE_R2RML = """
@prefix rr: <http://www.w3.org/ns/r2rml#> .
@prefix ex: <http://example.com/ns#> .

ex:TripleMap
  a rr:TriplesMap ;
  rr:logicalTable [ rr:tableName "test_products" ] ;
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
        cls.spark.createDataFrame(
            [(1, "Laptop")], "ID int, product_name string"
        ).createOrReplaceTempView("test_products")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_simple_usage(self) -> None:
        """Parse example R2RML from a file; result materialised as a list of `TripleMap` instances."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.ttl"
            path.write_text(EXAMPLE_R2RML, encoding="utf-8")
            mapping = Mapping.from_r2rml(str(path))
            self.assertEqual(len(mapping), 1)
            self.assertEqual(
                len(
                    mapping.triple_maps[
                        "http://example.com/ns#TripleMap"
                    ].predicate_object_maps
                ),
                1,
            )
            self.assertEqual(
                mapping.to_df(self.spark).collect(),
                [
                    (
                        "http://example.com/product/1",
                        "http://example.com/ns#product_name",
                        "Laptop",
                        "http://www.w3.org/2001/XMLSchema#string",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
