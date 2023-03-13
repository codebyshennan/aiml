.PHONY: run clean

SHELL := /bin/bash

run: env/bin/activate
	./env/bin/python3 app.py

setup: requirements.txt
	pip install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf env

