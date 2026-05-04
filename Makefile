venv: 
	python3 -m venv venv

install: venv
	venv/bin/pip install -r requirements.txt

# W3C rdb2rdf test suite (https://www.w3.org/TR/rdb2rdf-test-cases/)
RDB2RDF_ZIP_URL := https://dvcs.w3.org/hg/rdb2rdf-tests/raw-file/default/rdb2rdf-ts.zip
RDB2RDF_CACHE := tests/rdb2rdf-test-cases
RDB2RDF_ZIP := $(RDB2RDF_CACHE)/rdb2rdf-ts.zip
RDB2RDF_MANIFEST := $(RDB2RDF_CACHE)/D000-1table1column0rows/manifest.ttl

$(RDB2RDF_ZIP):
	mkdir -p $(RDB2RDF_CACHE)
	curl -L -f -o $@ $(RDB2RDF_ZIP_URL)

$(RDB2RDF_MANIFEST): $(RDB2RDF_ZIP)
	cd $(RDB2RDF_CACHE) && unzip -o -q rdb2rdf-ts.zip

# Local Spark warehouse dir is removed after each conformance test run
SPARK_WAREHOUSE := spark-warehouse

conformance-tests: install $(RDB2RDF_MANIFEST)
	RDB2RDF_TS_DIR=$(abspath $(RDB2RDF_CACHE)) venv/bin/python -m tests.conformance; \
	ec=$$?; \
	rm -rf $(SPARK_WAREHOUSE); \
	exit $$ec

conformance-tests-clean:
	rm -rf $(RDB2RDF_CACHE)

test: install
	venv/bin/python -m unittest discover -s tests
