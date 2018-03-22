#!/usr/bin/env make

TRAIN_DIR = training

setup:
	python train_data.py

train:
	python train_ann.py

inspect:
	python inspect_ann.py

clean:
	rm $(TRAIN_DIR)/*.wav

freeze:
	pip freeze > requirements.txt