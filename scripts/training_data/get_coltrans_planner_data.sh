#!/bin/bash

# Check if parameter n is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <n>"
    exit 1
fi

n=$1

data_dir="training_data"

# Ensure training_data directory exists
mkdir -p "$data_dir"

# Loop through matching directories
for dir in results/*_${n}robots; do
    if [ -d "$dir" ]; then
        # Extract folder name
        folder_name=$(basename "$dir")

        # Locate the required files
        traj_file=$(find "$dir" -type f -name "trajectory_opt.yaml" | head -n 1)
        env_file=$(find "$dir" -type f -name "env_inflated.yaml" | head -n 1)

        # Check if both required files exist
        if [ -f "$traj_file" ] && [ -f "$env_file" ]; then
            cp "$traj_file" "$data_dir/trajectory_${folder_name}.yaml"
            cp "$env_file" "$data_dir/env_${folder_name}.yaml"
            echo "Copied files for $folder_name"
        else
            echo "Skipping $folder_name: required files not found"
        fi
    fi
done

echo "Process completed."