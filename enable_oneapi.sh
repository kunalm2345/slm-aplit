#!/bin/bash
# Source oneAPI environment variables

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
    echo "✓ oneAPI environment enabled (system)"
elif [ -f "$HOME/intel/oneapi/setvars.sh" ]; then
    source $HOME/intel/oneapi/setvars.sh
    echo "✓ oneAPI environment enabled (user)"
else
    echo "✗ oneAPI not found"
    echo "Install with: sh installer.sh --install-dir ~/intel/oneapi"
fi
