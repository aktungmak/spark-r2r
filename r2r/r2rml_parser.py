from collections.abc import Iterator

from pyspark.sql import Column, SparkSession
from pyspark.sql.functions import col, lit
from rdflib import Graph, Namespace, RDF, URIRef

R2RML = Namespace("http://www.w3.org/ns/r2rml#")

from .mapping import Mapping
from .r2rml_template import r2rml_template_to_format_string


def from_r2rml(r2rml_file: str, spark: SparkSession) -> Iterator[Mapping]:
    """Parse an R2RML file and yield mappings."""
    g = Graph()
    g.parse(r2rml_file, format="turtle")
    for triple_map in g.subjects(RDF.type, R2RML.TriplesMap):

        ### Step 1: Extract source

        logical_table = g.value(triple_map, R2RML.logicalTable)
        if table_name := g.value(logical_table, R2RML.tableName):
            source = spark.table(table_name)
        elif sql_query := g.value(logical_table, R2RML.sqlQuery):
            source = spark.sql(sql_query)
        else:
            raise ValueError("Logical table not provided in {triple_map}")

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

        # TODO this should be a list of tuples to support multiple maps with the same predicate
        predicate_object_maps = {}
        for predicate_object_map in g.subjects(RDF.type, R2RML.PredicateObjectMap):
            predicate_expr = term_map_to_column(
                g, predicate_object_map, R2RML.predicate, R2RML.predicateMap
            )
            object_expr = term_map_to_column(
                g, predicate_object_map, R2RML.object, R2RML.objectMap
            )

            predicate_object_maps[predicate_expr] = object_expr

        yield Mapping(
            source=source,
            subject_map=subject_map_expr,
            predicate_object_maps=predicate_object_maps,
            rdf_type=str(subject_class) if subject_class is not None else None,
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
        return col(str(column))
    elif constant := g.value(term_map, R2RML.constant):
        return lit(str(constant))
    else:
        raise ValueError(f"Term map not supplied in {triple_map}")
