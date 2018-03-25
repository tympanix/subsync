#!/usr/bin/env make

setup:
	python subsync/model/train_data.py

train:
	python subsync/model/train_ann.py

eval:
	python subsync/model/eval_ann.py

logloss:
	python subsync/model/eval_logloss.py

test:
	python subsync/model/test.py
.PHONY: test

freeze:
	pip freeze > requirements.txt
