#!/bin/bash

# Script to run the RSP project with standardized output formatting

# Display header
echo "==============================================="
echo "RADAR SIGNAL PROCESSING EXECUTION SCRIPT"
echo "==============================================="

if [ "$1" == "batch" ]; then
    echo "Running BATCH processing version..."
    ./build/rsp_batch
elif [ "$1" == "single" ]; then
    echo "Running SINGLE FRAME processing version..."
    ./build/rsp
else
    echo "Usage: $0 [batch|single]"
    echo "  batch  - Run batch processing version"
    echo "  single - Run single frame processing version"
    echo
    echo "Both versions provide similar output formatting for easy comparison."
    exit 1
fi

echo
echo "==============================================="
echo "Processing complete!"
echo "==============================================="
