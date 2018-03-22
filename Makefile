#!/usr/bin/env make

TRAIN_DIR = training

setup:
	python train_data.py

train:
	python train_ann.py

eval:
	python eval_ann.py

test:
	python test.py
.PHONY: test

clean:
	rm $(TRAIN_DIR)/*.wav

freeze:
	pip freeze > requirements.txt