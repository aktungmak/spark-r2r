from dataclasses import dataclass, field
from functools import reduce
from typing import Iterator, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import lit

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
    # Additional columns to be added alongside the subject, predicate and
    # object columns. A common example is to add a timestamp to the triple.
    metadata_columns: dict[str, Column] = field(default_factory=dict)

    def rdfs_domain(self) -> list[tuple]:
        return [
            (pred, "rdfs:domain", self.rdf_type) for pred in self.predicate_object_maps
        ]

    def _po_maps(self) -> Iterator[tuple[str, Column]]:
        if self.rdf_type is not None:
            yield "rdf:type", lit(self.rdf_type)
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
        map_queries = (
            source.select(
                self.subject_map.alias("s"),
                lit(predicate).alias("p"),
                object_expr.alias("o"),
                *metadata_columns,
            )
            .filter(self.filter or lit(True))
            for predicate, object_expr in self._po_maps()
        )
        return reduce(lambda df1, df2: df1.union(df2), map_queries)

    def to_dlt(self, name: str) -> str:
        """
        If you are using DLT, this can be used in a pipline to define
        a flow based on the mapping.
        """
        try:
            import dlt
        except ImportError:
            raise ImportError("not running in DLT enabled cluster")

        @dlt.table(name=name)
        def t():
            return self.to_df()

        return name