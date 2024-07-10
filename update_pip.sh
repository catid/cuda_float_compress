#!/bin/bash
rm -rf dist/
pip install setuptools wheel twine
python setup.py sdist
twine upload dist/*
