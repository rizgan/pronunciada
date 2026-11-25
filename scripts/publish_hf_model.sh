#!/bin/sh
# Publish the model card and the small models to the Hugging Face model page.
#   sh scripts/publish_hf_model.sh            (after each release; needs `hf auth login`)
# Uploads huggingface/model/README.md and the model folders of pronunciada/data. The pronunciation
# dictionaries (pronunciada/data/lex) are data, not models, and are not uploaded.
set -e
cd "$(git rev-parse --show-toplevel)"
REPO="${1:-kimitonatsu/pronunciada}"
OUT="$(mktemp -d)"
cp huggingface/model/README.md "$OUT/"
for d in adapters calib coach prosody length native; do cp -R "pronunciada/data/$d" "$OUT/$d"; done
cp pronunciada/data/zipa_tokens.txt "$OUT/"
find "$OUT" -name ".DS_Store" -delete
hf upload "$REPO" "$OUT" . --repo-type model --commit-message "Pronunciada $(python3 -c 'import re;print(re.search(r"version = \"(.*?)\"", open("pyproject.toml").read()).group(1))')"
rm -rf "$OUT"
