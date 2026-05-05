import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from pyspark.sql import SparkSession

from r2r import Mapping
from r2r.mapping import (
    OBJECT_COLUMN,
    PREDICATE_COLUMN,
    RefObjectMap,
    SUBJECT_COLUMN,
)

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

# Parent triples map + rr:joinCondition on rr:objectMap (RefObjectMap).
EXAMPLE_R2RML_REF_OBJECT_MAP = """
@prefix rr: <http://www.w3.org/ns/r2rml#> .
@prefix ex: <http://example.com/ns#> .

ex:CustomerMap
  a rr:TriplesMap ;
  rr:logicalTable [ rr:tableName "parser_test_customers" ] ;
  rr:subjectMap [ rr:template "http://example.com/customer/{CustomerID}" ] .

ex:OrderMap
  a rr:TriplesMap ;
  rr:logicalTable [ rr:tableName "parser_test_orders" ] ;
  rr:subjectMap [ rr:template "http://example.com/order/{OrderID}" ] ;
  rr:predicateObjectMap [
    rr:predicate ex:forCustomer ;
    rr:objectMap [
      rr:parentTriplesMap ex:CustomerMap ;
      rr:joinCondition [
        rr:child "BuyerID" ;
        rr:parent "CustomerID"
      ]
    ]
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

    def test_parent_triples_map_and_join_conditions(self) -> None:
        """Parse rr:parentTriplesMap and rr:joinCondition into RefObjectMap."""
        order_iri = "http://example.com/ns#OrderMap"
        customer_iri = "http://example.com/ns#CustomerMap"

        self.spark.createDataFrame([(1,)], ["CustomerID"]).createOrReplaceTempView(
            "parser_test_customers"
        )
        self.spark.createDataFrame(
            [(100, 1)], ["OrderID", "BuyerID"]
        ).createOrReplaceTempView("parser_test_orders")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.ttl"
            path.write_text(EXAMPLE_R2RML_REF_OBJECT_MAP, encoding="utf-8")
            mapping = Mapping.from_r2rml(str(path))

        self.assertEqual(len(mapping), 2)

        _, obj_map = mapping.triple_maps[order_iri].predicate_object_maps[0]
        self.assertIsInstance(obj_map, RefObjectMap)
        self.assertEqual(obj_map.parent_triple_map, customer_iri)
        self.assertIsNotNone(obj_map.join_conditions)
        self.assertEqual(len(obj_map.join_conditions), 1)

        rows = mapping.triple_map_to_df(order_iri, self.spark).collect()
        print(rows)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row[SUBJECT_COLUMN], "http://example.com/order/100")
        self.assertEqual(row[PREDICATE_COLUMN], "http://example.com/ns#forCustomer")
        self.assertEqual(row[OBJECT_COLUMN], "http://example.com/customer/1")


if __name__ == "__main__":
    unittest.main()
