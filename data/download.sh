#!/bin/bash
# Downloads the SECOM dataset from Kaggle
# Requires: pip install kaggle + API token in ~/.kaggle/kaggle.json
# Or just download manually from https://www.kaggle.com/datasets/paresh2047/uci-semcom

set -e

echo "Downloading SECOM dataset..."
kaggle datasets download -d paresh2047/uci-semcom -p .
unzip -o uci-semcom.zip
rm uci-semcom.zip
echo "Done! Files: secom.data, secom_labels.data"
