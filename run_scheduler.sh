#!/bin/bash
# Start the C++ scheduler

# Try to source oneAPI if available
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
elif [ -f "$HOME/intel/oneapi/setvars.sh" ]; then
    source $HOME/intel/oneapi/setvars.sh
fi

./split_inference/cpp/build/scheduler tcp://*:5555
