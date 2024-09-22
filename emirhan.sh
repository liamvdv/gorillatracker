#!/bin/bash

set -e

echo "Running embedding generation"
python3 src/gorillatracker/classification/emirhan_generate.py
echo "Running clustering"
python3 src/gorillatracker/classification/emirhan_clustering.py
