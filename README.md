# Relational to RDF mapper

This is a simple package for performing [R2RML](https://www.w3.org/TR/r2rml/)
transformations in Spark. The triple-structured output dataframe is ready to
be queried with libraries like [sparql2sql](https://github.com/aktungmak/sparql2sql).

## Installation

You can install it in your project as follows:

```commandline
pip install git+https://github.com/aktungmak/spark-r2r.git
```

## Usage

If you already have an R2RML file that defines your mappings,
you can use it directly:

```python
from r2r import Mapping

mapping = Mapping.from_r2rml("your_r2rml_file.ttl")

# Access the triple-structured dataframe directly
df = mapping.to_df(spark)
df.write.saveAsTable("your_table_name")

# Alternatively you can create a Spark Declarative Pipeline
# that can incrementalise your mappings.
# Note: the below must be run within a pipeline!
mapping.to_dp(spark, "output_table")
```

Alternatively, you can instantiate the Mapping and TripleMap classes directly
in your Python code. Here is the Python version of [Example 2.3](https://www.w3.org/TR/r2rml/#example-simple)
in the R2RML spec:

```python
from pyspark.sql.functions import col, lit, format_string
from r2r import Mapping, TripleMap

EX = lambda x: format_string("http://example.com/ns#%s", lit(x))

mapping = Mapping(
    triples_map_1=TripleMap(
        source_table="catalog.schema.EMP",
        subject_map=format_string("http://data.example.com/employee/%s", col("EMPNO")),
        rdf_type=EX("Employee"),
        predicate_object_maps=[
            (EX("name"), col("ENAME")),
        ],
    )
)

# Now use the Mapping as above:
df = mapping.to_df(spark)
```

There are other options available, see the tests and `examples` directory for more variations.

## R2RML Conformance

Most R2RML constructs are supported, to validate how much of the spec is currently
implemented you can run `make conformance-tests`.