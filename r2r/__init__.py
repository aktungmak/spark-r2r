from dataclasses import dataclass, field
from functools import reduce
from typing import Iterator, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import lit

SUBJECT_COLUMN = "s"
PREDICATE_COLUMN = "p"
OBJECT_COLUMN = "o"
RDF_TYPE_IRI = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
RDFS_DOMAIN_IRI = "<http://www.w3.org/2000/01/rdf-schema#domain>"

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
    # A dictionary of predicate strings to column expressions that
    # will form the predicate and object columns of the result.
    predicate_object_maps: dict[str, Column]
    # An optional IRI defining the rdf:type of the table being mapped.
    rdf_type: Optional[str] = None
    # Optionally filter the data in the table before the transformation.
    filter: Optional[Column] = None
    # eliminate rows in the output with nulls in the object column
    filter_null_obj: Optional[bool] = True
    # Additional columns to be added alongside the subject, predicate and
    # object columns. A common example is to add a timestamp to the triple.
    metadata_columns: dict[str, Column] = field(default_factory=dict)

    def rdfs_domain(self) -> list[tuple]:
        if self.rdf_type:
            return [
                (pred, RDFS_DOMAIN_IRI, self.rdf_type) for pred in self.predicate_object_maps
            ]
        else:
            return []

    def _po_maps(self) -> Iterator[tuple[str, Column]]:
        if self.rdf_type is not None:
            yield RDF_TYPE_IRI, lit(self.rdf_type)
        yield from self.predicate_object_maps.items()

    def to_df(self, spark: SparkSession) -> DataFrame:
        """
        Build a DataFrame of triples based on the mapping.
        """
        if isinstance(self.source, str):
            source = spark.table(self.source)
        else:
            source = self.source
        metadata_columns = [
            mc.alias(name) for name, mc in self.metadata_columns.items()
        ]
        filter_expr = self.filter if self.filter is not None else lit(True)
        map_queries = (
            source.select(
                self.subject_map.alias(SUBJECT_COLUMN),
                lit(predicate).alias(PREDICATE_COLUMN),
                object_expr.alias(OBJECT_COLUMN),
                *metadata_columns,
            )
            .filter(filter_expr)
            for predicate, object_expr in self._po_maps()
        )
        union_df = reduce(lambda df1, df2: df1.union(df2), map_queries)
        if self.filter_null_obj:
            union_df = union_df.dropna(subset=OBJECT_COLUMN)
        return union_df

    def to_dlt(self, spark: SparkSession, name: str) -> str:
        """
        If you are using DLT, this can be used in a pipline to define
        a flow based on the mapping.
        """
        import dlt

        @dlt.table(name=name)
        def t():
            return self.to_df(spark)

        return name
