#!/bin/bash

# Source directory where the files are located
src_dir="./uncompressed/train"

# Destination directory where the files will be moved
dest_dir="${src_dir}/../test"

# Check if the destination directory exists, create it if it doesn't
[ ! -d "$dest_dir" ] && mkdir -p "$dest_dir"

# Move 160,000 random files from source to destination
find "$src_dir" -type f -name '*.txt' | shuf -n 160000 | xargs -I{} mv {} "$dest_dir"

