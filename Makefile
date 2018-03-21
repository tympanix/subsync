#!/usr/bin/env make

TRAIN_DIR = training

clean:
	rm $(TRAIN_DIR)/*.wav

freeze:
	pip freeze > requirements.txt