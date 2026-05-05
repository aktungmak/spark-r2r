"""
Integration tests for `r2r.mapping.Mapping.to_dp` against a real Spark Connect
declarative pipeline (graph registration + ``start_run``).

These tests are skipped unless the optional Connect stack is installed; use:

    pip install '.[test-pipelines]'
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from pyspark import SparkConf
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession as PySparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from pyspark.testing.utils import connect_requirement_message, should_test_connect

from r2r import Mapping, TripleMap

_REPO_ROOT = Path(__file__).resolve().parents[1]
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def _run_mapping_declarative_pipeline(
    spark: PySparkSession,
    mapping: Mapping,
    materialized_view_name: str,
    *,
    materialize_intermediates: bool,
) -> None:
    """Register ``mapping`` on a new dataflow graph and run it to completion."""
    from pyspark.pipelines.graph_element_registry import (
        graph_element_registration_context,
    )
    from pyspark.pipelines.spark_connect_graph_element_registry import (
        SparkConnectGraphElementRegistry,
    )
    from pyspark.pipelines.spark_connect_pipeline import (
        create_dataflow_graph,
        handle_pipeline_events,
        start_run,
    )

    dataflow_graph_id = create_dataflow_graph(
        spark, default_catalog=None, default_database=None, sql_conf={}
    )
    registry = SparkConnectGraphElementRegistry(spark, dataflow_graph_id)
    with graph_element_registration_context(registry):
        mapping.to_dp(
            spark,
            materialized_view_name,
            materialize_intermediates=materialize_intermediates,
        )

    storage_dir = tempfile.mkdtemp(prefix="spark-r2r-dp-")
    storage_uri = Path(storage_dir).resolve().as_uri()
    try:
        result_iter = start_run(
            spark,
            dataflow_graph_id,
            full_refresh=[],
            full_refresh_all=False,
            refresh=[],
            dry=False,
            storage=storage_uri,
        )
        handle_pipeline_events(result_iter)
    finally:
        shutil.rmtree(storage_dir, ignore_errors=True)


@unittest.skipUnless(
    should_test_connect,
    connect_requirement_message or "Spark Connect test dependencies missing",
)
class TestMappingDeclarativePipeline(TestCase):
    """``Mapping.to_dp`` under Spark Connect + declarative pipeline graph APIs."""

    @classmethod
    def setUpClass(cls):
        conf = SparkConf(loadDefaults=True)
        remote = os.environ.get("SPARK_CONNECT_TESTING_REMOTE", "local[2]")
        cls.spark = (
            PySparkSession.builder.config(conf=conf)
            .appName(cls.__name__)
            .remote(remote)
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.spark.stop()
        finally:
            # Unlike pipeline ``storage=`` scratch dirs, managed tables use ``spark-warehouse/``
            # under the repo; remove it so fixed names like ``intermediate_0`` stay reusable.
            shutil.rmtree(_REPO_ROOT / "spark-warehouse", ignore_errors=True)

    def setUp(self):
        self.users_data = [
            (1, "john.doe@example.com", "John Doe", "admin"),
            (2, "jane.smith@example.com", "Jane Smith", "user"),
            (3, "bob.wilson@example.com", "Bob Wilson", "user"),
            (4, None, "Anonymous User", "guest"),
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

    def test_to_dp_materializes_output_view(self):
        mapping = Mapping(
            test_map=TripleMap(
                source_df=self.users_df,
                subject_map=col("id"),
                predicate_object_maps=[("name", col("name"))],
            )
        )
        _run_mapping_declarative_pipeline(
            self.spark,
            mapping,
            "test_table",
            materialize_intermediates=True,
        )
        result = self.spark.sql("SELECT * FROM test_table ORDER BY s")
        self.assertEqual(result.count(), 4)
        self.assertEqual(result.columns, ["s", "p", "o", "ot"])
        self.assertEqual(
            result.collect(),
            [
                ("1", "name", "John Doe", XSD_STRING),
                ("2", "name", "Jane Smith", XSD_STRING),
                ("3", "name", "Bob Wilson", XSD_STRING),
                ("4", "name", "Anonymous User", XSD_STRING),
            ],
        )
