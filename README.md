# Relational to RDF mapper

This is a simple package for performing [R2RML](https://www.w3.org/TR/r2rml/)
style tasks in Spark.

You can install it in your project as follows:
```commandline
pip install git+https://github.com/aktungmak/spark-r2r.git
```

The main entry point is the `Mapping` class:

```python
from r2r import Mapping

m = TripleMap(
        source="system.information_schema.catalogs",
        subject_map=iri.catalog("catalog_name"),
        rdf_type=iri.type("catalog"),
        predicate_object_maps=[
            (iri.pred("catalog_name"), col("catalog_name")),
            (iri.pred("catalog_owner_email"), col("catalog_owner")),
        ],
    )
```

In this example, a separate module called `iri` is used to encapsulate
the IRI formatting logic, but it can just be included as a string.

With this definition, we can then create a `DataFrame` based on this and perform
further transformations or save:

```python
df = m.to_df(yourSparkContext)
df.write.saveAsTable("example_catalog.schema.triples")
```

There are other options available, see the `examples` directory for more variations.

## Processing R2RML definitions

Basic support for R2RML definitions exists. If you would like to try it out, here is
a simple example (assuming that your mappings are in a file called `mapping.ttl`):

```python
from r2r import from_r2rml

for mapping in from_r2rml("mapping.ttl", spark):
    mapping.to_df(spark).write.saveAsTable(...)
```

Further support is gradually being added, to validate how much of the spec is currently
implemented you can run `make conformance-tests`.