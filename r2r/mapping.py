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


@dataclass
class RefObjectMap:
    """Represents an object map that refers to the subject of another mapping."""

    # The IRI of the parent mapping that this object map refers to.
    # The subject_map of the parent mapping will be used as the object of the output triples.
    parent_mapping: str
    # The join conditions to join the parent mapping to the current mapping.
    # If no join conditions are provided, the object is the subject of the parent mapping.
    # Otherwise, the first element of each tuple is the column from the current mapping
    # and the second element of each tuple is the column from the parent mapping.
    join_conditions: Optional[list[tuple[Column, Column]]] = None


PredicateMap = Union[str, Column]

# ObjectMap can be a Column (infer type) or (Column, object_type)
# Use (Column, None) to explicitly mark as IRI (no type)
# Alternatively, it can be a RefObjectMap
ObjectMap = Union[Column, tuple[Column, Optional[str]], RefObjectMap]


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
class TripleMap:
    """
    Represents a mapping from a relational table to an RDF representation
    using subject-predicate-object triples.
    """

    # An expression used to form the subject of the output triples.
    # It is recommended that this be an IRI.
    subject_map: Column
    # A list of (predicate map, object map) pairs.
    # Predicate maps can be:
    # - string IRI
    # - Column: predicate expression
    # Object maps can be:
    # - Column: object expression, type inferred from source schema
    # - (Column, str): object expression with explicit type (XSD IRI or "@lang")
    # - (Column, None): object expression treated as IRI (no type)
    # - RefObjectMap: object map that refers to the subject of another TripleMap
    predicate_object_maps: list[tuple[PredicateMap, ObjectMap]]
    # An optional IRI defining the rdf:type of the subject of the TripleMap.
    rdf_type: Optional[str] = None
    # Optionally filter the data in the table before the transformation.
    filter: Optional[Column] = None
    # eliminate rows in the output with nulls in the object column
    filter_null_obj: Optional[bool] = True
    # Additional columns to be added alongside the subject, predicate and
    # object columns. A common example is to add a timestamp to the triple.
    metadata_columns: dict[str | Column, Column] = field(default_factory=dict)
    # The source of the TripleMap, either a table name, a query, or a DataFrame.
    # Exactly one of source_table, source_query, or source_df must be provided.
    source_table: Optional[str] = None
    source_query: Optional[str] = None
    source_df: Optional[DataFrame] = None

    def __post_init__(self):
        if all(
            s is None for s in [self.source_table, self.source_query, self.source_df]
        ):
            raise ValueError(
                "Exactly one of source_table, source_query, or source_df must be provided"
            )

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
        Build a DataFrame of triples based on the TripleMap.
        Output columns: s, p, o, ot (plus any metadata_columns).
        """
        if self.source_table:
            source = spark.table(self.source_table)
        elif self.source_query:
            source = spark.sql(self.source_query)
        elif self.source_df:
            source = self.source_df
        else:
            assert False, "Unreachable"
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
        """Create a Spark Declarative Pipelines flow for the TripleMap"""
        warnings.warn(
            "Please use to_materialized_view or to_temporary_view instead",
            DeprecationWarning,
        )
        return self.to_materialized_view(spark, name)

    def to_materialized_view(self, spark: SparkSession, name: str) -> str:
        """Create a Spark Declarative Pipelines materialised view for the TripleMap"""
        from pyspark.pipelines import materialized_view

        return self._to_dp(materialized_view, spark, name)

    def to_temporary_view(self, spark: SparkSession, name: str) -> str:
        """Create a Spark Declarative Pipelines temporary view for the TripleMap"""
        from pyspark.pipelines import temporary_view

        return self._to_dp(temporary_view, spark, name)

    def _to_dp(self, decorator, spark: SparkSession, name: str) -> str:
        @decorator(name=name)
        def t():
            return self.to_df(spark)

        return name
