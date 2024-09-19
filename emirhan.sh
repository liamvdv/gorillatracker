#!/bin/bash

set -e

python3 src/gorillatracker/classification/emirhan_generate.py 
python3 src/gorillatracker/classification/emirhan_clustering.py