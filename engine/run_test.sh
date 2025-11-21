#!/bin/bash
# Navigate to engine directory if not already there
if [[ "$(basename "$PWD")" != "engine" ]]; then
    if [ -d "engine" ]; then
        cd engine
    else
        echo "Error: Could not find engine directory"
        exit 1
    fi
fi

echo "Building C++ extension..."
python3 setup.py build_ext --inplace
if [ $? -eq 0 ]; then
    echo "Build successful."
    echo "Running test script..."
    python3 test_workflow.py
else
    echo "Build failed."
    exit 1
fi
