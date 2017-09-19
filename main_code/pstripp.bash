#!/bin/bash

python preprocessing.py $1
python main.py $1 0.1 0.5 0
python postprocessing.py $1
