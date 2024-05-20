#!/bin/env python

# This script clears the output of a Jupyter notebook, use if GUI is unresponsive
# because the file is too large.

# pip install nbformat nbconvert

import sys
import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor


def clear_notebook_output(notebook_path, output_path=None):
    # Load the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Create a ClearOutputPreprocessor instance
    clear_output = ClearOutputPreprocessor()

    # Clear the notebook's output
    nb, _ = clear_output.preprocess(nb, {})

    # Define the output path
    if output_path is None:
        output_path = notebook_path

    # Write the cleared notebook back to a file
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clear_ipynb_outputs.py <notebook_path>")
        sys.exit(1)

    clear_notebook_output(sys.argv[1])
