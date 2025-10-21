import unittest
from typing import Optional
from unittest import TestCase

from pyspark import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from r2r import Mapping, SUBJECT_COLUMN, PREDICATE_COLUMN, OBJECT_COLUMN, RDF_TYPE_IRI, RDFS_DOMAIN_IRI


class TestMapping(TestCase):
    """Comprehensive unit tests for the r2r.Mapping class."""

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
        self.users_schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("email", StringType(), True),
            StructField("name", StringType(), True),
            StructField("role", StringType(), True)
        ])
        self.users_df = self.spark.createDataFrame(self.users_data, self.users_schema)

        # Create sample data for products
        self.products_data = [
            (101, "Laptop", "Electronics", 999.99),
            (102, "Mouse", "Electronics", 29.99),
            (103, "Book", "Literature", 19.99),
        ]
        self.products_schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("price", StringType(), True)  # Using string for price to test different data types
        ])
        self.products_df = self.spark.createDataFrame(self.products_data, self.products_schema)

        # Register tables in catalog for testing string source
        self.users_df.createOrReplaceTempView("test_users")
        self.products_df.createOrReplaceTempView("test_products")

    def test_basic_initialization(self):
        """Test basic initialization of Mapping class."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")}
        )

        self.assertIsInstance(mapping.source, DataFrame)
        self.assertEqual(mapping.rdf_type, None)
        self.assertEqual(mapping.filter, None)
        self.assertTrue(mapping.filter_null_obj)
        self.assertEqual(mapping.metadata_columns, {})

    def test_initialization_with_all_parameters(self):
        """Test initialization with all optional parameters."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name"), "email": col("email")},
            rdf_type="http://example.org/User",
            filter=col("role") != "guest",
            filter_null_obj=False,
            metadata_columns={"timestamp": lit("2024-01-01")}
        )

        self.assertEqual(mapping.rdf_type, "http://example.org/User")
        self.assertIsNotNone(mapping.filter)
        self.assertFalse(mapping.filter_null_obj)
        self.assertEqual(len(mapping.metadata_columns), 1)

    def test_string_source_initialization(self):
        """Test initialization with string table name as source."""
        mapping = Mapping(
            source="test_users",
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")}
        )

        self.assertEqual(mapping.source, "test_users")

    def test_rdfs_domain_without_rdf_type(self):
        """Test rdfs_domain method when rdf_type is None."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name"), "email": col("email")}
        )

        domain = mapping.rdfs_domain()
        self.assertEqual(domain, [])

    def test_rdfs_domain_with_rdf_type(self):
        """Test rdfs_domain method when rdf_type is provided."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name"), "email": col("email")},
            rdf_type="http://example.org/User"
        )

        domain = mapping.rdfs_domain()
        expected = [
            ("name", RDFS_DOMAIN_IRI, "http://example.org/User"),
            ("email", RDFS_DOMAIN_IRI, "http://example.org/User")
        ]
        self.assertEqual(domain, expected)

    def test_to_df_basic_functionality(self):
        """Test basic to_df functionality with DataFrame source."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name"), "email": col("email")}
        )

        result_df = mapping.to_df(self.spark)

        # Check schema
        expected_columns = [SUBJECT_COLUMN, PREDICATE_COLUMN, OBJECT_COLUMN]
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
        self.assertIn(1, subjects)
        self.assertIn(2, subjects)
        self.assertIn(3, subjects)
        self.assertIn(4, subjects)

        self.assertIn("name", predicates)
        self.assertIn("email", predicates)

        self.assertIn("John Doe", objects)
        self.assertIn("jane.smith@example.com", objects)

    def test_to_df_with_string_source(self):
        """Test to_df functionality with string table name as source."""
        mapping = Mapping(
            source="test_users",
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")}
        )

        result_df = mapping.to_df(self.spark)

        # Should have 4 rows (one for each user)
        self.assertEqual(result_df.count(), 4)

        # Check that data is correctly loaded from the table
        result_data = result_df.collect()
        objects = [row[OBJECT_COLUMN] for row in result_data]
        self.assertIn("John Doe", objects)
        self.assertIn("Jane Smith", objects)

    def test_to_df_with_rdf_type(self):
        """Test to_df functionality with rdf_type specified."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")},
            rdf_type="http://example.org/User"
        )

        result_df = mapping.to_df(self.spark)

        # Should have 8 rows: 4 for name + 4 for rdf:type
        self.assertEqual(result_df.count(), 8)

        # Check that rdf:type triples are included
        result_data = result_df.collect()
        predicates = [row[PREDICATE_COLUMN] for row in result_data]
        objects = [row[OBJECT_COLUMN] for row in result_data]

        self.assertIn(RDF_TYPE_IRI, predicates)
        self.assertIn("http://example.org/User", objects)

    def test_to_df_with_filter(self):
        """Test to_df functionality with filter applied."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")},
            filter=col("role") != "guest"
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
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"email": col("email")},
            filter_null_obj=False
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
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")},
            metadata_columns={
                "timestamp": lit("2024-01-01"),
                "source_system": lit("test_system")
            }
        )

        result_df = mapping.to_df(self.spark)

        # Check that metadata columns are included
        expected_columns = [SUBJECT_COLUMN, PREDICATE_COLUMN, OBJECT_COLUMN, "timestamp", "source_system"]
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
            source=self.products_df,
            subject_map=col("product_id"),
            predicate_object_maps={
                "name": col("name"),
                "category": col("category"),
                "expensive": when(col("price").cast("double") > 100, lit("yes")).otherwise(lit("no"))
            }
        )

        result_df = mapping.to_df(self.spark)

        # Should have 9 rows (3 products * 3 predicates)
        self.assertEqual(result_df.count(), 9)

        # Check that complex expression works
        result_data = result_df.collect()
        expensive_objects = [row[OBJECT_COLUMN] for row in result_data
                             if row[PREDICATE_COLUMN] == "expensive"]

        self.assertIn("yes", expensive_objects)  # Laptop should be expensive
        self.assertIn("no", expensive_objects)  # Mouse and Book should not be expensive

    def test_to_df_empty_dataframe(self):
        """Test to_df with empty DataFrame."""
        empty_df = self.spark.createDataFrame([], self.users_schema)

        mapping = Mapping(
            source=empty_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")}
        )

        result_df = mapping.to_df(self.spark)

        # Should have 0 rows
        self.assertEqual(result_df.count(), 0)

        # But should have correct schema
        expected_columns = [SUBJECT_COLUMN, PREDICATE_COLUMN, OBJECT_COLUMN]
        self.assertEqual(result_df.columns, expected_columns)

    def test_po_maps_private_method(self):
        """Test the _po_maps private method."""
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name"), "email": col("email")},
            rdf_type="http://example.org/User"
        )

        po_maps = list(mapping._po_maps())

        # Should have 3 items: rdf:type + 2 predicate_object_maps
        self.assertEqual(len(po_maps), 3)

        # Check that rdf:type comes first
        self.assertEqual(po_maps[0][0], RDF_TYPE_IRI)

        # Check that predicate_object_maps are included
        predicate_names = [item[0] for item in po_maps[1:]]
        self.assertIn("name", predicate_names)
        self.assertIn("email", predicate_names)

    def test_to_dp_method(self):
        """Test the to_dp method (basic functionality test)."""
        self.skipTest("Declarative Pipelines not available in test environment")
        mapping = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")}
        )

        result = mapping.to_dp(self.spark, "test_table")
        self.assertEqual(result, "test_table")

    def test_multiple_mappings_union(self):
        """Test creating multiple mappings and ensuring they can be unioned."""
        mapping1 = Mapping(
            source=self.users_df,
            subject_map=col("id"),
            predicate_object_maps={"name": col("name")}
        )

        mapping2 = Mapping(
            source=self.products_df,
            subject_map=col("product_id"),
            predicate_object_maps={"name": col("name")}
        )

        df1 = mapping1.to_df(self.spark)
        df2 = mapping2.to_df(self.spark)

        # Union the results (common pattern in the examples)
        union_df = df1.union(df2)

        # Should have combined count
        expected_count = df1.count() + df2.count()
        self.assertEqual(union_df.count(), expected_count)

        # Should have same schema
        self.assertEqual(df1.columns, df2.columns)


if __name__ == "__main__":
    unittest.main()
