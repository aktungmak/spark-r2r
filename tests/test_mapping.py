import unittest
from typing import Optional
from unittest import TestCase

from pyspark import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import array_agg, col, concat, lit, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from r2r import Mapping, TripleMap
from r2r.mapping import (
    SUBJECT_COLUMN,
    PREDICATE_COLUMN,
    OBJECT_COLUMN,
    OBJECT_TYPE_COLUMN,
    RDF_TYPE_IRI,
    RefObjectMap,
)

XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"


class TestMapping(TestCase):
    """Unit tests for the r2r.Mapping and r2r.TripleMap classes."""

    @classmethod
    def setUpClass(cls):
        """Set up a local PySpark session for all tests."""
        cls.spark = SparkSession.getActiveSession() or SparkSession(SparkContext())

    @classmethod
    def tearDownClass(cls):
        """Clean up the Spark session after all tests."""
        cls.spark.stop()

    def setUp(self):
        """Set up test data for each test."""
        # Create sample data for users
        self.users_data = [
            (1, "john.doe@example.com", "John Doe", "admin"),
            (2, "jane.smith@example.com", "Jane Smith", "user"),
            (3, "bob.wilson@example.com", "Bob Wilson", "user"),
            (4, None, "Anonymous User", "guest"),  # Test null email
        ]
        self.users_schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("email", StringType(), True),
                StructField("name", StringType(), True),
                StructField("role", StringType(), True),
            ]
        )
        self.users_df = self.spark.createDataFrame(self.users_data, self.users_schema)

        # Create sample data for products
        self.products_data = [
            (101, "Laptop", "Electronics", 999.99),
            (102, "Mouse", "Electronics", 29.99),
            (103, "Book", "Literature", 19.99),
        ]
        self.products_schema = StructType(
            [
                StructField("product_id", IntegerType(), True),
                StructField("name", StringType(), True),
                StructField("category", StringType(), True),
                StructField(
                    "price", StringType(), True
                ),  # Using string for price to test different data types
            ]
        )
        self.products_df = self.spark.createDataFrame(
            self.products_data, self.products_schema
        )

        # Register tables in catalog for testing string source
        self.users_df.createOrReplaceTempView("test_users")
        self.products_df.createOrReplaceTempView("test_products")

    def test_basic_triple_map_init(self):
        """Test basic initialisation of a TripleMap."""
        triple_map = TripleMap(
            source_df=self.users_df,
            subject_map=col("id"),
            predicate_object_maps=[("name", col("name"))],
        )

        self.assertIsInstance(triple_map.source_df, DataFrame)
        self.assertEqual(triple_map.rdf_type, None)
        self.assertEqual(triple_map.filter, None)
        self.assertTrue(triple_map.filter_null_obj)
        self.assertEqual(triple_map.metadata_columns, {})

    def test_triple_map_init_with_all_parameters(self):
        """Test initialisation with all optional parameters."""
        triple_map = TripleMap(
            source_df=self.users_df,
            subject_map=col("id"),
            predicate_object_maps=[
                ("name", col("name")),
                ("email", col("email")),
            ],
            rdf_type="http://example.org/User",
            filter=col("role") != "guest",
            filter_null_obj=False,
            metadata_columns={"timestamp": lit("2024-01-01")},
        )

        self.assertEqual(triple_map.rdf_type, "http://example.org/User")
        self.assertIsNotNone(triple_map.filter)
        self.assertFalse(triple_map.filter_null_obj)
        self.assertEqual(len(triple_map.metadata_columns), 1)

    def test_triple_map_init_with_string_source(self):
        """Test initialisation with string table name as source."""
        triple_map = TripleMap(
            source_table="test_users",
            subject_map=col("id"),
            predicate_object_maps=[("name", col("name"))],
        )

        self.assertEqual(triple_map.source_table, "test_users")

    def test_triple_map_raises_when_no_source_provided(self):
        """__post_init__ rejects mappings with no table, query, or DataFrame source."""
        with self.assertRaisesRegex(
            ValueError,
            "Exactly one of source_table, source_query, or source_df must be provided",
        ):
            TripleMap(
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
            )

    def test_to_df_basic_functionality(self):
        """Test basic to_df functionality with DataFrame source."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[
                    ("name", col("name")),
                    ("email", col("email")),
                ],
            )
        )

        result_df = mapping.to_df(self.spark)

        # Check schema
        expected_columns = [
            SUBJECT_COLUMN,
            PREDICATE_COLUMN,
            OBJECT_COLUMN,
            OBJECT_TYPE_COLUMN,
        ]
        self.assertEqual(result_df.columns, expected_columns)

        # Check row count (2 predicates * 4 users = 8 rows, minus null email = 7 rows due to filter_null_obj)
        result_count = result_df.count()
        self.assertEqual(result_count, 7)  # One null email is filtered out

        # Check data content
        result_data = result_df.collect()
        subjects = [row[SUBJECT_COLUMN] for row in result_data]
        predicates = [row[PREDICATE_COLUMN] for row in result_data]
        objects = [row[OBJECT_COLUMN] for row in result_data]

        # Should have subject ids 1, 2, 3, 4 for name and 1, 2, 3 for email (4 is null)
        self.assertIn("1", subjects)
        self.assertIn("2", subjects)
        self.assertIn("3", subjects)
        self.assertIn("4", subjects)

        self.assertIn("name", predicates)
        self.assertIn("email", predicates)

        self.assertIn("John Doe", objects)
        self.assertIn("jane.smith@example.com", objects)

    def test_to_df_with_string_source(self):
        """Test to_df functionality with string table name as source."""
        mapping = Mapping(
            test_map=TripleMap(
                source_table="test_users",
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
            )
        )

        result_df = mapping.to_df(self.spark)

        # Should have 4 rows (one for each user)
        self.assertEqual(result_df.count(), 4)

        # Check that data is correctly loaded from the table
        result_data = result_df.collect()
        objects = [row[OBJECT_COLUMN] for row in result_data]
        self.assertIn("John Doe", objects)
        self.assertIn("Jane Smith", objects)

    def test_to_df_with_source_query(self):
        """Test to_df loads rows via spark.sql when source_query is provided."""
        mapping = Mapping(
            test_map=TripleMap(
                source_query="SELECT * FROM test_users WHERE role = 'admin'",
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
            )
        )

        result_df = mapping.to_df(self.spark)
        self.assertEqual(result_df.count(), 1)

        result_data = result_df.collect()
        objects = [row[OBJECT_COLUMN] for row in result_data]
        self.assertIn("John Doe", objects)
        self.assertNotIn("Jane Smith", objects)
        self.assertNotIn("Bob Wilson", objects)
        self.assertNotIn("Anonymous User", objects)

    def test_to_df_with_rdf_type(self):
        """Test to_df functionality with rdf_type specified."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
                rdf_type="http://example.org/User",
            )
        )

        result_data = mapping.to_df(self.spark).collect()
        # Should have 8 rows: 4 for name + 4 for rdf:type
        # self.assertEqual(len(result_data), 8)
        for row in result_data:
            if row[PREDICATE_COLUMN] == str(RDF_TYPE_IRI):
                self.assertEqual(row[OBJECT_COLUMN], "http://example.org/User")
            else:
                self.assertEqual(row[PREDICATE_COLUMN], "name")

    def test_to_df_with_rdf_type_list(self):
        """Test to_df functionality with rdf_type list."""
        test_types = ["http://example.org/User", "http://example.org/Person"]
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
                rdf_type=test_types,
            )
        )

        result_data = (
            mapping.to_df(self.spark)
            .filter(col(PREDICATE_COLUMN) == str(RDF_TYPE_IRI))
            .groupBy(SUBJECT_COLUMN, PREDICATE_COLUMN)
            .agg(array_agg(OBJECT_COLUMN).alias("rdf_type"))
            .collect()
        )
        self.assertEqual(len(result_data), 4)
        for row in result_data:
            self.assertSetEqual(
                set(row["rdf_type"]),
                set(test_types),
            )

    def test_to_df_with_filter(self):
        """Test to_df functionality with filter applied."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
                filter=col("role") != "guest",
            )
        )

        result_df = mapping.to_df(self.spark)

        # Should have 3 rows (filtering out the guest user)
        self.assertEqual(result_df.count(), 3)

        # Check that guest user is not in results
        result_data = result_df.collect()
        subjects = [row[SUBJECT_COLUMN] for row in result_data]
        self.assertNotIn(4, subjects)  # User ID 4 is the guest

    def test_to_df_with_filter_null_obj_false(self):
        """Test to_df functionality with filter_null_obj=False."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("email", col("email"))],
                filter_null_obj=False,
            )
        )

        result_df = mapping.to_df(self.spark)

        # Should have 4 rows (including the null email)
        self.assertEqual(result_df.count(), 4)

        # Check that null values are present
        result_data = result_df.collect()
        objects = [row[OBJECT_COLUMN] for row in result_data]
        null_count = sum(1 for obj in objects if obj is None)
        self.assertEqual(null_count, 1)

    def test_to_df_with_metadata_columns(self):
        """Test to_df functionality with metadata columns."""
        self.skipTest("Metadata columns not implemented yet")
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
                metadata_columns={
                    "timestamp": lit("2024-01-01"),
                    "source_system": lit("test_system"),
                },
            )
        )

        result_df = mapping.to_df(self.spark)

        # Check that metadata columns are included
        expected_columns = [
            SUBJECT_COLUMN,
            PREDICATE_COLUMN,
            OBJECT_COLUMN,
            OBJECT_TYPE_COLUMN,
            "timestamp",
            "source_system",
        ]
        self.assertEqual(result_df.columns, expected_columns)

        # Check metadata column values
        result_data = result_df.collect()
        timestamps = [row["timestamp"] for row in result_data]
        source_systems = [row["source_system"] for row in result_data]

        self.assertTrue(all(ts == "2024-01-01" for ts in timestamps))
        self.assertTrue(all(ss == "test_system" for ss in source_systems))

    def test_to_df_complex_expressions(self):
        """Test to_df with complex column expressions."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.products_df,
                subject_map=col("product_id"),
                predicate_object_maps=[
                    ("name", col("name")),
                    ("category", col("category")),
                    (
                        "expensive",
                        when(col("price").cast("double") > 100, lit("yes")).otherwise(
                            lit("no")
                        ),
                    ),
                ],
            )
        )

        result_df = mapping.to_df(self.spark)

        # Should have 9 rows (3 products * 3 predicates)
        self.assertEqual(result_df.count(), 9)

        # Check that complex expression works
        result_data = result_df.collect()
        expensive_objects = [
            row[OBJECT_COLUMN]
            for row in result_data
            if row[PREDICATE_COLUMN] == "expensive"
        ]

        self.assertIn("yes", expensive_objects)  # Laptop should be expensive
        self.assertIn("no", expensive_objects)  # Mouse and Book should not be expensive

    def test_to_df_empty_dataframe(self):
        """Test to_df with empty DataFrame."""
        empty_df = self.spark.createDataFrame([], self.users_schema)

        mapping = Mapping(
            test_map=TripleMap(
                source_df=empty_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
            )
        )

        result_df = mapping.to_df(self.spark)

        # Should have 0 rows
        self.assertEqual(result_df.count(), 0)

        # But should have correct schema
        expected_columns = [
            SUBJECT_COLUMN,
            PREDICATE_COLUMN,
            OBJECT_COLUMN,
            OBJECT_TYPE_COLUMN,
        ]
        self.assertEqual(result_df.columns, expected_columns)

    def test_multiple_triple_maps(self):
        """Test Mapping with multiple TripleMaps."""
        mapping = Mapping(
            triple_map1=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
            ),
            triple_map2=TripleMap(
                source_df=self.products_df,
                subject_map=col("product_id"),
                predicate_object_maps=[("name", col("name"))],
            ),
        )

        df = mapping.to_df(self.spark)

        # Should have combined count of 7 (4 users + 3 products)
        self.assertEqual(df.count(), 7)

        # Should have correct schema
        self.assertEqual(df.columns, ["s", "p", "o", "ot"])

    def test_object_type_with_typed_literals(self):
        """Test that object types are correctly set for typed literals."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[
                    ("name", (col("name"), XSD_STRING)),
                    # Cast id to string, but specify integer type
                    ("user_id_str", (col("id").cast("string"), XSD_INTEGER)),
                    # Leave id as integer to check that inferred type is correct
                    ("user_id_int", col("id")),
                ],
            )
        )

        result_data = mapping.to_df(self.spark).collect()

        for row in result_data:
            if row[PREDICATE_COLUMN] == "name":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], XSD_STRING)
            elif row[PREDICATE_COLUMN] == "user_id_str":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], XSD_INTEGER)
            elif row[PREDICATE_COLUMN] == "user_id_int":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], XSD_INTEGER)

    def test_object_type_with_language_tags(self):
        """Test that language-tagged literals use @lang format."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", (col("name"), "@en"))],
            )
        )

        result_data = mapping.to_df(self.spark).collect()
        for row in result_data:
            self.assertEqual(row[OBJECT_TYPE_COLUMN], "@en")

    def test_object_type_none_for_iris(self):
        """Test that IRIs (including rdf:type objects) have None as object type."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[
                    ("related", (col("email"), None))
                ],  # Treat as IRI
                rdf_type=lit("http://example.org/User"),
            )
        )

        result_data = mapping.to_df(self.spark).collect()
        for row in result_data:
            if row[PREDICATE_COLUMN] in [RDF_TYPE_IRI, "related"]:
                self.assertIsNone(row[OBJECT_TYPE_COLUMN])

    def test_object_type_inferred_from_schema(self):
        """Test that plain Column values infer object type from source schema."""
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[
                    ("name", col("name")),  # StringType -> xsd:string
                    ("id", col("id")),  # IntegerType -> xsd:integer
                ],
            )
        )

        result_data = mapping.to_df(self.spark).collect()
        for row in result_data:
            if row[PREDICATE_COLUMN] == "name":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], XSD_STRING)
            elif row[PREDICATE_COLUMN] == "id":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], XSD_INTEGER)

    def test_object_type_mixed_types(self):
        """Test mixing explicit types, language tags, IRIs, and inferred types."""
        XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[
                    ("name", (col("name"), XSD_STRING)),  # explicit typed literal
                    ("label", (col("name"), "@en")),  # language tag
                    ("sameAs", (col("email"), None)),  # explicit IRI (no type)
                    (
                        "role",
                        col("role"),
                    ),  # inferred from schema (StringType -> xsd:string)
                ],
            )
        )

        result_data = mapping.to_df(self.spark).collect()
        for row in result_data:
            if row[PREDICATE_COLUMN] == "name":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], XSD_STRING)
            elif row[PREDICATE_COLUMN] == "label":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], "@en")
            elif row[PREDICATE_COLUMN] == "sameAs":
                self.assertIsNone(row[OBJECT_TYPE_COLUMN])
            elif row[PREDICATE_COLUMN] == "role":
                self.assertEqual(row[OBJECT_TYPE_COLUMN], XSD_STRING)
            else:
                self.fail(f"Unexpected predicate: {row[PREDICATE_COLUMN]}")

    def test_ref_object_map_same_logical_source_no_join(self):
        """RefObjectMap without join: parent subject is taken from the same logical row."""
        rows = [(10, 99, "alpha"), (11, 99, "beta")]
        df = self.spark.createDataFrame(rows, ["assignee_id", "team_id", "label"])
        teams = TripleMap(
            source_df=df,
            subject_map=concat(lit("http://ex.example/team/"), col("team_id")),
            predicate_object_maps=[
                ("http://ex.example/teamLabel", col("label")),
            ],
        )
        assigns = TripleMap(
            source_df=df,
            subject_map=concat(lit("http://ex.example/person/"), col("assignee_id")),
            predicate_object_maps=[
                (
                    "http://ex.example/inTeam",
                    RefObjectMap(
                        parent_triple_map="teams",
                        join_conditions=None,
                    ),
                ),
            ],
        )
        mapping = Mapping(teams=teams, assigns=assigns)
        member_triples = (
            mapping.triple_map_to_df("assigns", self.spark)
            .filter(col(PREDICATE_COLUMN) == "http://ex.example/inTeam")
            .collect()
        )
        self.assertEqual(len(member_triples), 2)
        for row in member_triples:
            self.assertEqual(row[OBJECT_COLUMN], "http://ex.example/team/99")

    def test_ref_object_map_join_across_logical_sources(self):
        """RefObjectMap with join: child rows are joined to the parent logical table on rr:child / rr:parent."""
        users = self.spark.createDataFrame(
            [(1, "Ann"), (2, "Bob")],
            ["user_id", "name"],
        )
        orders = self.spark.createDataFrame(
            [(100, 1, 10.0), (101, 2, 20.0), (102, 1, 5.0)],
            ["order_id", "buyer_id", "total"],
        )
        users_tm = TripleMap(
            source_df=users,
            subject_map=concat(lit("http://ex.example/user/"), col("user_id")),
            predicate_object_maps=[("http://ex.example/name", col("name"))],
        )
        orders_tm = TripleMap(
            source_df=orders,
            subject_map=concat(lit("http://ex.example/order/"), col("order_id")),
            predicate_object_maps=[
                (
                    "http://ex.example/buyer",
                    RefObjectMap(
                        parent_triple_map="users",
                        join_conditions=[(col("buyer_id"), col("user_id"))],
                    ),
                ),
            ],
        )
        mapping = Mapping(users=users_tm, orders=orders_tm)
        buyer_triples = (
            mapping.triple_map_to_df("orders", self.spark)
            .filter(col(PREDICATE_COLUMN) == "http://ex.example/buyer")
            .collect()
        )
        self.assertEqual(len(buyer_triples), 3)
        by_order_subject = {
            row[SUBJECT_COLUMN]: row[OBJECT_COLUMN] for row in buyer_triples
        }
        self.assertEqual(
            by_order_subject["http://ex.example/order/100"],
            "http://ex.example/user/1",
        )
        self.assertEqual(
            by_order_subject["http://ex.example/order/101"],
            "http://ex.example/user/2",
        )
        self.assertEqual(
            by_order_subject["http://ex.example/order/102"],
            "http://ex.example/user/1",
        )


if __name__ == "__main__":
    unittest.main()
