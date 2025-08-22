#!/bin/bash
# Script to manage private files with skip-worktree

echo "Setting skip-worktree for private files..."

# CLAUDE.md in root
git update-index --skip-worktree CLAUDE.md

# All files in docs/ directory
find docs -type f -exec git update-index --skip-worktree {} \;

# All files in modification_logs/ directory
find modification_logs -type f -exec git update-index --skip-worktree {} \;

echo "Done! Files marked as skip-worktree:"
echo "===================================="

# List all skip-worktree files
git ls-files -v | grep '^S' | cut -c3-

echo "===================================="
echo "These files will be tracked locally but not pushed to remote."
echo ""
echo "To undo skip-worktree for a file, use:"
echo "git update-index --no-skip-worktree <filename>"