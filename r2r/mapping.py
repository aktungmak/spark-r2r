from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import reduce
from typing import Optional, Union

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.functions import array, col, explode, lit
import pyspark.sql.types as st
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, XSD

from .r2rml_template import (
    normalise_r2rml_sql_identifier,
    r2rml_template_to_format_string,
)

SUBJECT_COLUMN = "s"
PREDICATE_COLUMN = "p"
OBJECT_COLUMN = "o"
OBJECT_TYPE_COLUMN = "ot"
RDF_TYPE_IRI = RDF.type
TRIPLE_SCHEMA = st.StructType(
    [
        st.StructField(SUBJECT_COLUMN, st.StringType(), nullable=False),
        st.StructField(PREDICATE_COLUMN, st.StringType(), nullable=False),
        st.StructField(OBJECT_COLUMN, st.StringType(), nullable=False),
        st.StructField(OBJECT_TYPE_COLUMN, st.StringType(), nullable=True),
    ]
)

R2RML = Namespace("http://www.w3.org/ns/r2rml#")


@dataclass
class RefObjectMap:
    """Represents an object map that refers to the subject of another mapping."""

    # The IRI of the parent TripleMap that this object map refers to.
    # The subject_map of the parent TripleMap will be used as the object of the output triples.
    parent_triple_map: str
    # The join conditions to join the parent TripleMap to the current TripleMap.
    # If no join conditions are provided, then the source is the current TripleMap.
    # Otherwise, the first element of each tuple is the column from the current TripleMap
    # and the second element of each tuple is the column from the parent TripleMap
    # which is used to join the parent TripleMap to the current TripleMap.
    join_conditions: Optional[list[tuple[Column, Column]]] = None

    def _to_join_expr(self) -> Column:
        if self.join_conditions:
            return [child == parent for child, parent in self.join_conditions]
        else:
            raise ValueError("No join conditions provided")


PredicateMap = Union[str, Column]

# ObjectMap can be a Column (infer type) or (Column, object_type)
# Use (Column, None) to explicitly mark as IRI (no type)
# Alternatively, it can be a RefObjectMap which refers to the subject of another TripleMap
ObjectMap = Union[Column, tuple[Column, Optional[str]], RefObjectMap]

PredicateObjectMap = tuple[PredicateMap, ObjectMap]

_SPARK_TYPE_TO_XSD = {
    st.StringType: XSD.string,
    st.IntegerType: XSD.integer,
    st.LongType: XSD.long,
    st.ShortType: XSD.short,
    st.ByteType: XSD.byte,
    st.FloatType: XSD.float,
    st.DoubleType: XSD.double,
    st.BooleanType: XSD.boolean,
    st.DateType: XSD.date,
    st.TimestampType: XSD.dateTime,
    st.DecimalType: XSD.decimal,
}


def _spark_type_to_xsd(spark_type) -> Column:
    """Map a Spark DataType to its corresponding XSD URI, defaulting to string."""
    return lit(_SPARK_TYPE_TO_XSD.get(type(spark_type), XSD.string))


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

            ### Step 3: Extract subject classes
            subject_classes = []
            if subject_map := g.value(triple_map, R2RML.subjectMap):
                for subject_class in g.objects(subject_map, R2RML["class"]):
                    subject_classes.append(str(subject_class))

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
                    parent_triple_map = str(parent_triples_map)
                    join_columns: list[tuple[Column, Column]] = []
                    for jc in g.objects(predicate_object_map, R2RML.joinCondition):
                        child = g.value(jc, R2RML.child)
                        parent = g.value(jc, R2RML.parent)
                        if child is None or parent is None:
                            raise R2RMLParseError(
                                f"joinCondition must supply rr:child and rr:parent in {jc}"
                            )
                        join_columns.append(
                            (
                                col(normalise_r2rml_sql_identifier(str(child))),
                                col(normalise_r2rml_sql_identifier(str(parent))),
                            )
                        )
                    object_expr = RefObjectMap(
                        parent_triple_map=parent_triple_map,
                        join_conditions=join_columns or None,
                    )
                # If not a RefObjectMap, it's a regular TermMap
                else:
                    object_expr = term_map_to_column(
                        g, predicate_object_map, R2RML.object, R2RML.objectMap
                    )

                predicate_object_maps.append((predicate, object_expr))

            triple_maps[str(triple_map)] = TripleMap(
                source_table=source_table,
                source_query=source_query,
                subject_map=subject_map_expr,
                predicate_object_maps=predicate_object_maps,
                rdf_type=subject_classes or None,
            )

        return cls(**triple_maps)

    def __init__(self, **triple_maps: TripleMap):
        self.triple_maps = triple_maps

    def _triple_map_source(self, triple_map_iri: str, spark: SparkSession) -> DataFrame:
        """Logical table rows for a TripleMap (same resolution as triple_map_to_df)."""
        triple_map = self.triple_maps[triple_map_iri]
        if triple_map.source_table:
            source = spark.table(triple_map.source_table)
        elif triple_map.source_query:
            source = spark.sql(triple_map.source_query)
        elif triple_map.source_df:
            source = triple_map.source_df
        else:
            assert False, "Unreachable"

        if triple_map.filter is not None:
            source = source.filter(triple_map.filter)
        return source

    def triple_map_to_df(self, triple_map_iri: str, spark: SparkSession) -> DataFrame:
        """
        Build a DataFrame of triples based on the TripleMap.
        Output columns: s, p, o, ot (plus any metadata_columns).
        """
        triple_map = self.triple_maps[triple_map_iri]

        subject_map = triple_map.subject_map.cast("string")

        source = self._triple_map_source(triple_map_iri, spark)

        # TODO: add metadata columns to the source DataFrame and ensure test coverage
        metadata_columns = [
            mc.alias(name) for name, mc in triple_map.metadata_columns.items()
        ]

        dfs = (
            self._po_map_to_df(
                source,
                subject_map,
                po_map,
                metadata_columns,
                spark,
            )
            for po_map in triple_map.predicate_object_maps
        )

        rdf_types = self._rdf_types(source, subject_map, triple_map.rdf_type, spark)

        union_df = reduce(lambda df1, df2: df1.union(df2), dfs, rdf_types)
        if triple_map.filter_null_obj:
            union_df = union_df.dropna(subset=OBJECT_COLUMN)
        return union_df

    def _rdf_types(
        self,
        source: DataFrame,
        subject_map: Column,
        rdf_type: Optional[Union[str, Column, list[str]]],
        spark: SparkSession,
    ) -> DataFrame:
        """Expand an rdf_type configuration into a DataFrame."""
        if rdf_type is None:
            return spark.createDataFrame([], schema=TRIPLE_SCHEMA)

        if isinstance(rdf_type, Column):
            rdf_type_expr = rdf_type
        elif isinstance(rdf_type, str):
            rdf_type_expr = lit(rdf_type)
        elif isinstance(rdf_type, list):
            rdf_type_expr = explode(array([lit(rt) for rt in rdf_type]))
        else:
            raise ValueError(f"Invalid rdf_type: {rdf_type}")

        return source.select(
            subject_map.alias(SUBJECT_COLUMN),
            lit(RDF_TYPE_IRI).alias(PREDICATE_COLUMN),
            rdf_type_expr.alias(OBJECT_COLUMN),
            lit(None).alias(OBJECT_TYPE_COLUMN),
        ).distinct()

    def _po_map_to_df(
        self,
        source: DataFrame,
        subject_map: Column,
        predicate_object_map: PredicateObjectMap,
        metadata_columns: list[Column],
        spark: SparkSession,
    ) -> DataFrame:
        """Expand a predicate_object_map into a DataFrame."""
        predicate_map, object_map = predicate_object_map
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
            object_type = lit(object_type)
        elif isinstance(object_map, RefObjectMap):
            if object_map.join_conditions:
                parent_source = self._triple_map_source(
                    object_map.parent_triple_map, spark
                )
                source = source.join(
                    parent_source,
                    object_map._to_join_expr(),
                    how="inner",
                )
            object_expr = self.triple_maps[object_map.parent_triple_map].subject_map
            object_type = lit(None)
        elif isinstance(object_map, Column):
            # Infer type from the column expression's result type
            result_type = source.select(object_map).schema[0].dataType
            object_expr, object_type = object_map, _spark_type_to_xsd(result_type)
        else:
            raise ValueError(f"Invalid object map: {object_map}")

        return source.select(
            subject_map.alias(SUBJECT_COLUMN),
            predicate_expr.alias(PREDICATE_COLUMN),
            object_expr.alias(OBJECT_COLUMN).cast("string"),
            object_type.alias(OBJECT_TYPE_COLUMN),
            *metadata_columns,
        )

    def to_df(self, spark: SparkSession) -> DataFrame:
        return reduce(
            lambda df1, df2: df1.union(df2),
            [self.triple_map_to_df(iri, spark) for iri in self.triple_maps.keys()],
        )

    def __len__(self) -> int:
        return len(self.triple_maps)


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
