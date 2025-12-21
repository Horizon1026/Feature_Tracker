#!/bin/bash

# Get the directory where the script is located
SEARCH_DIR=$(dirname "$(readlink -f "$0")")

echo "Cleaning __pycache__ directories in: $SEARCH_DIR"

# Recursively find and remove all __pycache__ directories
find "$SEARCH_DIR" -type d -name "__pycache__" -exec rm -rf {} +

echo "Cleanup completed."
