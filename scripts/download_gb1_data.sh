#!/bin/bash
# GB1 full measured-fitness table (149,361 variants, gb1_clade repo).
# The public package ships a built-in demo subset, so everything runs without this;
# install the full table only to reproduce full-pool numbers from the paper.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST="$HERE/data/gb1/Input"
mkdir -p "$DEST"
git clone --depth 1 https://github.com/chandar-lab/CLADE-gb1 "$DEST/clade" || \
  git clone --depth 1 https://github.com/samsledje/CLADE "$DEST/clade"
cp "$DEST/clade/Input/GB1.xlsx" "$DEST/GB1.xlsx" 2>/dev/null || \
  find "$DEST/clade" -name 'GB1.xlsx' -exec cp {} "$DEST/GB1.xlsx" \;
echo "GB1.xlsx installed at $DEST/GB1.xlsx"
