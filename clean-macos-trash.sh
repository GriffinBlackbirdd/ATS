#!/bin/bash

# Target volume path
TARGET="/Volumes/T7 Shield"

# Check if volume exists
if [ ! -d "$TARGET" ]; then
  echo "❌ Error: Volume not found at $TARGET"
  exit 1
fi

echo "🧹 Cleaning metadata files from: $TARGET"

# Clean ._ files and .DS_Store files
find "$TARGET" \( -name '._*' -o -name '.DS_Store' \) -type f -print -delete

echo "✅ Cleanup complete. All ._ and .DS_Store files have been removed from $TARGET."
