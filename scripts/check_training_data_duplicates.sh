#!/bin/bash

# Usage: ./check_common_files.sh dir1 dir2

dir1="training_data"
dir2="training_data/validation"

# Check if directories exist
if [[ ! -d "$dir1" || ! -d "$dir2" ]]; then
  echo "Both arguments must be valid directories."
  exit 1
fi

# Get list of filenames in each directory (non-recursive)
files1=$(find "$dir1" -maxdepth 1 -type f -exec basename {} \;)
files2=$(find "$dir2" -maxdepth 1 -type f -exec basename {} \;)

# Compare and find common filenames
echo "Common files in $dir1 and $dir2:"
comm -12 <(echo "$files1" | sort) <(echo "$files2" | sort)
