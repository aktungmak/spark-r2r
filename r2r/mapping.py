import warnings
from dataclasses import dataclass, field
from functools import reduce
from typing import Iterator, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import lit
import pyspark.sql.types as spark_types

SUBJECT_COLUMN = "s"
PREDICATE_COLUMN = "p"
OBJECT_COLUMN = "o"
OBJECT_TYPE_COLUMN = "ot"
RDF_TYPE_IRI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_DOMAIN_IRI = "http://www.w3.org/2000/01/rdf-schema#domain"

XSD_PREFIX = "http://www.w3.org/2001/XMLSchema#"

# Sentinel to indicate object type should be inferred from schema
_INFER = object()

# Type alias: value is Column (infer type) or (Column, object_type)
# Use (Column, None) to explicitly mark as IRI (no type)
ObjectMapValue = Union[Column, tuple[Column, Optional[str]]]

_SPARK_TYPE_TO_XSD = {
    spark_types.StringType: f"{XSD_PREFIX}string",
    spark_types.IntegerType: f"{XSD_PREFIX}integer",
    spark_types.LongType: f"{XSD_PREFIX}long",
    spark_types.ShortType: f"{XSD_PREFIX}short",
    spark_types.ByteType: f"{XSD_PREFIX}byte",
    spark_types.FloatType: f"{XSD_PREFIX}float",
    spark_types.DoubleType: f"{XSD_PREFIX}double",
    spark_types.BooleanType: f"{XSD_PREFIX}boolean",
    spark_types.DateType: f"{XSD_PREFIX}date",
    spark_types.TimestampType: f"{XSD_PREFIX}dateTime",
    spark_types.DecimalType: f"{XSD_PREFIX}decimal",
}


def _spark_type_to_xsd(spark_type) -> str:
    """Map a Spark DataType to its corresponding XSD URI, defaulting to string."""
    return _SPARK_TYPE_TO_XSD.get(type(spark_type), f"{XSD_PREFIX}string")


@dataclass
class Mapping:
    """
    Represents a mapping from a relational table to an RDF representation
    using subject-predicate-object triples.
    """

    # The source, either as a DataFrame object or the name of a table
    # in the spark catalog.
    source: Union[str, DataFrame]
    # An expression used to form the subject of the output triples.
    # It is recommended that this be an IRI string.
    subject_map: Column
    # A list of (predicate, object map) pairs. Predicate is a string IRI or a Column
    # (R2RML). Object values can be:
    # - Column: object expression, type inferred from source schema
    # - (Column, str): object expression with explicit type (XSD IRI or "@lang")
    # - (Column, None): object expression treated as IRI (no type)
    predicate_object_maps: list[tuple[Union[str, Column], ObjectMapValue]]
    # An optional IRI defining the rdf:type of the table being mapped.
    rdf_type: Optional[str] = None
    # Optionally filter the data in the table before the transformation.
    filter: Optional[Column] = None
    # eliminate rows in the output with nulls in the object column
    filter_null_obj: Optional[bool] = True
    # Additional columns to be added alongside the subject, predicate and
    # object columns. A common example is to add a timestamp to the triple.
    metadata_columns: dict[str | Column, Column] = field(default_factory=dict)

    def rdfs_domain(self) -> list[tuple]:
        if self.rdf_type:
            return [
                (pred, RDFS_DOMAIN_IRI, self.rdf_type)
                for pred, _ in self.predicate_object_maps
            ]
        else:
            return []

    def _po_maps(self) -> Iterator[tuple[Column, Column, Optional[str]]]:
        """Yields (predicate, object_expr, object_type) tuples.

        object_type is either:
        - A string (explicit type IRI or language tag)
        - None (explicit IRI, no type)
        - _INFER sentinel (infer type from schema)
        """
        if self.rdf_type is not None:
            yield lit(RDF_TYPE_IRI), lit(self.rdf_type), None
        for predicate, value in self.predicate_object_maps:
            if isinstance(predicate, Column):
                predicate_expr = predicate
            else:
                predicate_expr = lit(predicate)
            if isinstance(value, tuple):
                object_expr, object_type = value
            else:
                object_expr, object_type = value, _INFER  # Infer from schema
            yield predicate_expr, object_expr, object_type

    def to_df(self, spark: SparkSession) -> DataFrame:
        """
        Build a DataFrame of triples based on the mapping.
        Output columns: s, p, o, ot (plus any metadata_columns).
        """
        if isinstance(self.source, str):
            source = spark.table(self.source)
        else:
            source = self.source
        metadata_columns = [
            mc.alias(name) for name, mc in self.metadata_columns.items()
        ]
        filter_expr = self.filter if self.filter is not None else lit(True)

        def resolve_object_type(object_expr: Column, object_type) -> Optional[str]:
            """Resolve _INFER sentinel to actual XSD type from expression."""
            if object_type is not _INFER:
                return object_type
            # Infer type from the column expression's result type
            result_type = source.select(object_expr).schema[0].dataType
            return _spark_type_to_xsd(result_type)

        map_queries = (
            source.select(
                self.subject_map.alias(SUBJECT_COLUMN),
                predicate_expr.alias(PREDICATE_COLUMN),
                object_expr.alias(OBJECT_COLUMN),
                lit(resolve_object_type(object_expr, object_type)).alias(
                    OBJECT_TYPE_COLUMN
                ),
                *metadata_columns,
            ).filter(filter_expr)
            for predicate_expr, object_expr, object_type in self._po_maps()
        )
        union_df = reduce(lambda df1, df2: df1.union(df2), map_queries)
        if self.filter_null_obj:
            union_df = union_df.dropna(subset=OBJECT_COLUMN)
        return union_df

    def to_dp(self, spark: SparkSession, name: str) -> str:
        """Create a Spark Declarative Pipelines flow for the mapping"""
        warnings.warn(
            "Please use to_materialized_view or to_temporary_view instead",
            DeprecationWarning,
        )
        return self.to_materialized_view(spark, name)

    def to_materialized_view(self, spark: SparkSession, name: str) -> str:
        """Create a Spark Declarative Pipelines materialised view for the mapping"""
        from pyspark.pipelines import materialized_view

        return self._to_dp(materialized_view, spark, name)

    def to_temporary_view(self, spark: SparkSession, name: str) -> str:
        """Create a Spark Declarative Pipelines temporary view for the mapping"""
        from pyspark.pipelines import temporary_view

        return self._to_dp(temporary_view, spark, name)

    def _to_dp(self, decorator, spark: SparkSession, name: str) -> str:
        @decorator(name=name)
        def t():
            return self.to_df(spark)

        return name
