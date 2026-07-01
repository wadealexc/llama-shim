#!/bin/bash

# Cleanup log files older than 7 days
# Usage: ./cleanup-logs.sh [--dry-run]

LOG_DIR="./backend/logs"
DAYS_OLD=7
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory '$LOG_DIR' does not exist."
    exit 1
fi

# Find files older than 7 days (excluding .gitkeep)
mapfile -t files_to_delete < <(find "$LOG_DIR" -maxdepth 1 -type f -name "*.log" -mtime +$DAYS_OLD ! -name ".gitkeep" 2>/dev/null)

# Find files that will remain (newer than 7 days + .gitkeep)
mapfile -t files_to_remain < <(find "$LOG_DIR" -maxdepth 1 -type f -mtime -$DAYS_OLD ! -name ".gitkeep" 2>/dev/null)
if [ -f "$LOG_DIR/.gitkeep" ]; then
    files_to_remain+=("$LOG_DIR/.gitkeep")
fi

if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE ==="
    echo ""
    echo "Files that WILL BE DELETED (older than $DAYS_OLD days):"
    echo "--------------------------------------------------------"
    if [ ${#files_to_delete[@]} -eq 0 ]; then
        echo "  (none)"
    else
        for file in "${files_to_delete[@]}"; do
            echo "  - $file"
        done
    fi
    echo ""
    echo "Total files to delete: ${#files_to_delete[@]}"
    echo ""
    echo "Files that WILL REMAIN:"
    echo "-----------------------"
    if [ ${#files_to_remain[@]} -eq 0 ]; then
        echo "  (none)"
    else
        for file in "${files_to_remain[@]}"; do
            echo "  - $file"
        done
    fi
    echo ""
    echo "Total files remaining: ${#files_to_remain[@]}"
    echo ""
    echo "To execute deletion, run: $0"
else
    echo "Cleaning up log files older than $DAYS_OLD days in '$LOG_DIR'..."
    
    if [ ${#files_to_delete[@]} -eq 0 ]; then
        echo "No files to delete."
    else
        for file in "${files_to_delete[@]}"; do
            rm -f "$file"
            echo "Deleted: $file"
        done
        echo ""
        echo "Total files deleted: ${#files_to_delete[@]}"
    fi
fi
