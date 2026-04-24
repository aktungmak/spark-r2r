venv: 
	python3 -m venv venv

install: venv
	venv/bin/pip install -r requirements.txt

test: install
	venv/bin/python -m unittest discover -s tests
