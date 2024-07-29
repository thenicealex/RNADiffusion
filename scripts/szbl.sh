#!/bin/bash

R_HOME="~/Projects/RNADiffusion"

rm $R_HOME/main.ipynb

# Check if rg exists
if command -v rg &> /dev/null; then
    rg "/home/fkli" ./ -g "!$R_HOME/scripts/*" | xargs sed -i "s/\/home\/fkli/\/lustre\/home\/fkli/g"
else
  # Check if grep exists
  if command -v grep &> /dev/null; then
    grep -irl --exclude-dir="$R_HOME/scripts" "/home/fkli" ./ | xargs sed -i "s/\/home\/fkli/\/lustre\/home\/fkli/g"
  else
    echo "Neither rg nor grep is installed. Exiting."
    exit 1
  fi
fi