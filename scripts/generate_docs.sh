#!/usr/bin/env bash
# Automated documentation generation script

echo "[DOCS] Generating API documentation with pdoc..."
pip install pdoc3
pdoc --html src/ --output-dir docs/api --force
echo "[DOCS] API Documentation updated in docs/api/"
