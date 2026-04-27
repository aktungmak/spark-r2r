"""``python -m tests.conformance`` — run W3C R2RML conformance with a final summary line."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_ts = Path(__file__).resolve().parent.parent / "rdb2rdf-test-cases"
os.environ.setdefault("RDB2RDF_TS_DIR", str(_ts))


def main() -> int:
    from tests.conformance.manifest import default_ts_root
    from tests.conformance.w3c_r2rml import load_tests

    loader = unittest.TestLoader()
    suite = load_tests(loader, unittest.TestSuite(), None)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    n = result.testsRun
    skipped = len(getattr(result, "skipped", []) or [])
    passed = n - len(result.failures) - len(result.errors) - skipped
    print(
        f"\nW3C R2RML conformance: {n} test(s) run, "
        f"{passed} passed, {len(result.failures)} failed, {len(result.errors)} error(s), "
        f"{skipped} skipped. (RDB2RDF_TS_DIR={default_ts_root()!s})"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
