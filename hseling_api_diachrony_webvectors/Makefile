all: run

clean:
	rm -rf venv && rm -rf *.egg-info && rm -rf dist && rm -rf *.log*

venv:
	virtualenv --python=python3 venv && venv/bin/pip install ../hseling_lib_diachrony_webvectors && venv/bin/pip install .

run: venv
	FLASK_APP=hseling_api_diachrony_webvectors venv/bin/flask run

test: venv
	HSELING_API_DIACHRONY_WEBVECTORS_SETTINGS=../settings.cfg venv/bin/python -m unittest discover -s tests

sdist: venv test
	venv/bin/python setup.py sdist
