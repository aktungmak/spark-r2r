import unittest
from pyspark.sql import SparkSession
from r2r.r2rml_template import (
    r2rml_template_to_format_string,
    _r2rml_template_to_printf_template,
)


class TestR2RmlTemplateToPrintf(unittest.TestCase):
    def test_single_column(self) -> None:
        p, c = _r2rml_template_to_printf_template("http://example.com/product/{ID}")
        self.assertEqual("http://example.com/product/%s", p)
        self.assertEqual(["ID"], c)

    def test_doubles_percent_in_literal(self) -> None:
        p, c = _r2rml_template_to_printf_template("http://x/{a}?q=100%25")
        self.assertEqual("http://x/%s?q=100%%25", p)
        self.assertEqual(["a"], c)

    def test_two_columns(self) -> None:
        p, c = _r2rml_template_to_printf_template("http://ex.com/employee={E}/dept={D}")
        self.assertEqual("http://ex.com/employee=%s/dept=%s", p)
        self.assertEqual(["E", "D"], c)

    def test_no_columns_just_literal(self) -> None:
        p, c = _r2rml_template_to_printf_template(r"http://ex.org/path")
        self.assertEqual("http://ex.org/path", p)
        self.assertEqual([], c)


class TestR2RmlTemplateToFormatString(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("r2rml-template")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_e2e_replaces_id(self) -> None:
        df = self.spark.createDataFrame([("7", "hat")], ["ID", "product_name"])
        col_expr = r2rml_template_to_format_string("http://example.com/product/{ID}")
        out = df.select(col_expr.alias("s")).collect()[0].s
        self.assertEqual("http://example.com/product/7", out)


if __name__ == "__main__":
    unittest.main()
