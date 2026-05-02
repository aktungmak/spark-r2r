from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import reduce
from typing import Iterator, Optional, Union

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, XSD

from .r2rml_template import (
    normalise_r2rml_sql_identifier,
    r2rml_template_to_format_string,
)

SUBJECT_COLUMN = "s"
PREDICATE_COLUMN = "p"
OBJECT_COLUMN = "o"
OBJECT_TYPE_COLUMN = "ot"
RDF_TYPE_IRI = str(RDF.type)
RDFS_DOMAIN_IRI = str(RDFS.domain)
TRIPLE_SCHEMA = StructType(
    [
        StructField(SUBJECT_COLUMN, StringType(), nullable=False),
        StructField(PREDICATE_COLUMN, StringType(), nullable=False),
        StructField(OBJECT_COLUMN, StringType(), nullable=False),
        StructField(OBJECT_TYPE_COLUMN, StringType(), nullable=True),
    ]
)

R2RML = Namespace("http://www.w3.org/ns/r2rml#")


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

PredicateObjectMap = tuple[PredicateMap, ObjectMap]

_SPARK_TYPE_TO_XSD = {
    spark_types.StringType: str(XSD.string),
    spark_types.IntegerType: str(XSD.integer),
    spark_types.LongType: str(XSD.long),
    spark_types.ShortType: str(XSD.short),
    spark_types.ByteType: str(XSD.byte),
    spark_types.FloatType: str(XSD.float),
    spark_types.DoubleType: str(XSD.double),
    spark_types.BooleanType: str(XSD.boolean),
    spark_types.DateType: str(XSD.date),
    spark_types.TimestampType: str(XSD.dateTime),
    spark_types.DecimalType: str(XSD.decimal),
}


def _spark_type_to_xsd(spark_type) -> str:
    """Map a Spark DataType to its corresponding XSD URI, defaulting to string."""
    return _SPARK_TYPE_TO_XSD.get(type(spark_type), str(XSD.string))


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
    predicate_object_maps: list[PredicateObjectMap]
    # An optional IRI or list of IRIs defining the rdf:type of the subject of the TripleMap.
    rdf_type: Optional[Union[str, Column, list[str]]] = None
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


class Mapping:
    """A collection of TripleMaps. TripleMaps are identified by their IRI and may depend each other."""

    @classmethod
    def from_r2rml(cls, r2rml_file: str) -> "Mapping":
        """Parse an R2RML file to extract TripleMaps."""
        triple_maps = {}
        g = Graph()
        g.parse(r2rml_file, format="turtle")

        for triple_map in g.subjects(RDF.type, R2RML.TriplesMap):

            ### Step 1: Extract source
            source_table = source_query = None
            logical_table = g.value(triple_map, R2RML.logicalTable)
            if table_name := g.value(logical_table, R2RML.tableName):
                source_table = normalise_r2rml_sql_identifier(str(table_name))
            elif sql_query := g.value(logical_table, R2RML.sqlQuery):
                source_query = str(sql_query)
            else:
                raise R2RMLParseError(f"Logical table not provided in {triple_map}")

            ### Step 2: Extract subject

            subject_map_expr = term_map_to_column(
                g, triple_map, R2RML.subject, R2RML.subjectMap
            )

            ### Step 3: Extract subject class
            # TODO: handle multiple subject classes
            subject_class = None
            if subject_map := g.value(triple_map, R2RML.subjectMap):
                subject_class = g.value(subject_map, R2RML["class"])

            ### Step 4: Extract predicate object maps

            predicate_object_maps = []
            for predicate_object_map in g.objects(triple_map, R2RML.predicateObjectMap):
                predicate = term_map_to_column(
                    g, predicate_object_map, R2RML.predicate, R2RML.predicateMap
                )

                # First try to parse as a RefObjectMap
                # We know its a RefObjectMap if there is an objectMap with an
                # rr:parentTriplesMap reference
                if (object_map := g.value(predicate_object_map, R2RML.objectMap)) and (
                    parent_triples_map := g.value(object_map, R2RML.parentTriplesMap)
                ):
                    parent_mapping = str(parent_triples_map)
                    join_conditions = [
                        (g.value(jc, R2RML.child), g.value(jc, R2RML.parent))
                        for jc in g.objects(predicate_object_map, R2RML.joinCondition)
                    ]
                    object_expr = RefObjectMap(
                        parent_mapping=parent_mapping,
                        join_conditions=join_conditions or None,
                    )
                # If not a RefObjectMap, it's a regular TermMap
                else:
                    object_expr = term_map_to_column(
                        g, predicate_object_map, R2RML.object, R2RML.objectMap
                    )

                predicate_object_maps.append((predicate, object_expr))

            triple_maps[triple_map] = TripleMap(
                source_table=source_table,
                source_query=source_query,
                subject_map=subject_map_expr,
                predicate_object_maps=predicate_object_maps,
                rdf_type=str(subject_class) if subject_class is not None else None,
            )

        return cls(triple_maps)

    def __init__(self, triple_maps: dict[str, TripleMap]):
        self.triple_maps = triple_maps

    def triple_map_to_df(self, triple_map_iri: str, spark: SparkSession) -> DataFrame:
        """
        Build a DataFrame of triples based on the TripleMap.
        Output columns: s, p, o, ot (plus any metadata_columns).
        """
        triple_map = self.triple_maps[triple_map_iri]

        if triple_map.source_table:
            source = spark.table(triple_map.source_table)
        elif triple_map.source_query:
            source = spark.sql(triple_map.source_query)
        elif triple_map.source_df:
            source = triple_map.source_df
        else:
            assert False, "Unreachable"

        metadata_columns = [
            mc.alias(name) for name, mc in triple_map.metadata_columns.items()
        ]
        filter_expr = triple_map.filter if triple_map.filter is not None else lit(True)

        dfs = (
            source.select(
                triple_map.subject_map.alias(SUBJECT_COLUMN),
                predicate_expr.alias(PREDICATE_COLUMN),
                object_expr.alias(OBJECT_COLUMN),
                object_type.alias(OBJECT_TYPE_COLUMN),
                *metadata_columns,
            ).filter(filter_expr)
            for predicate_expr, object_expr, object_type in self._po_maps(
                triple_map, spark
            )
        )
        dfs.append(self._rdf_types(triple_map, spark))
        union_df = reduce(lambda df1, df2: df1.union(df2), dfs)
        if triple_map.filter_null_obj:
            union_df = union_df.dropna(subset=OBJECT_COLUMN)
        return union_df

    def _rdf_types(self, triple_map: str, spark: SparkSession) -> DataFrame:
        if triple_map.rdf_type is None:
            return spark.createDataFrame([], schema=TRIPLE_SCHEMA)
        elif isinstance(triple_map.rdf_type, Column):
            return spark.createDataFrame(
                [(triple_map.subject_map, RDF_TYPE_IRI, triple_map.rdf_type)],
                schema=TRIPLE_SCHEMA,
            )
        elif isinstance(triple_map.rdf_type, str):
            return spark.createDataFrame(
                [(triple_map.subject_map, RDF_TYPE_IRI, lit(triple_map.rdf_type))],
                schema=TRIPLE_SCHEMA,
            )
        elif isinstance(triple_map.rdf_type, list):
            return spark.createDataFrame(
                [
                    (triple_map.subject_map, RDF_TYPE_IRI, lit(rdf_type))
                    for rdf_type in triple_map.rdf_type
                ],
                schema=TRIPLE_SCHEMA,
            )
        else:
            raise ValueError(f"Invalid rdf_type: {triple_map.rdf_type}")

    def _po_maps(
        self,
        source: DataFrame,
        predicate_object_maps: list[PredicateObjectMap],
        spark: SparkSession,
    ) -> Iterator[PredicateObjectMap]:
        """Yields (predicate, object_expr, object_type) tuples."""
        for predicate_map, object_map in predicate_object_maps:
            # Predicate
            if isinstance(predicate_map, Column):
                predicate_expr = predicate_map
            elif isinstance(predicate_map, str):
                predicate_expr = lit(predicate_map)
            else:
                raise ValueError(f"Invalid predicate map: {predicate_map}")

            # Object
            if isinstance(object_map, tuple):
                object_expr, object_type = object_map
            elif isinstance(object_map, RefObjectMap):
                other_df = self.triple_map_to_df(object_map.parent_mapping, spark)
                if object_map.join_conditions:
                    joined = source.join(
                        other_df,
                        [col(jc[0]) == col(jc[1]) for jc in object_map.join_conditions],
                        how="inner",
                    )
                object_expr = other_df.select(object_map.subject_map).alias(
                    OBJECT_COLUMN
                )
            elif isinstance(object_map, Column):
                # Infer type from the column expression's result type
                result_type = source.select(object_map).schema[0].dataType
                object_expr, object_type = object_map, _spark_type_to_xsd(result_type)
            else:
                raise ValueError(f"Invalid object map: {object_map}")

            yield predicate_expr, object_expr, object_type

    def to_df(self, spark: SparkSession) -> DataFrame:
        return reduce(
            lambda df1, df2: df1.union(df2),
            [tm.to_df(spark) for tm in self.triple_maps.values()],
        )


def term_map_to_column(
    g: Graph, triple_map: URIRef, shortcut_iri: URIRef, term_map_iri: URIRef
) -> Column:
    term_map = g.value(triple_map, term_map_iri)
    if shortcut := g.value(triple_map, shortcut_iri):
        return lit(str(shortcut))
    elif template := g.value(term_map, R2RML.template):
        return r2rml_template_to_format_string(str(template))
    elif column := g.value(term_map, R2RML.column):
        return col(normalise_r2rml_sql_identifier(str(column)))
    elif constant := g.value(term_map, R2RML.constant):
        return lit(str(constant))
    else:
        raise R2RMLParseError(f"Term map not supplied in {triple_map}")


class R2RMLParseError(Exception):
    pass
