#!/bin/bash
# Quick fix for missing cppzmq (C++ bindings for ZeroMQ)
# Run this if you already have libzmq but setup fails with "zmq.hpp: No such file or directory"

set -e

echo "======================================================================"
echo "Installing cppzmq (C++ ZeroMQ bindings)"
echo "======================================================================"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Set local directory
LOCAL_DIR="$HOME/.local"
mkdir -p "$LOCAL_DIR/include"

echo "Downloading cppzmq headers..."
cd /tmp

# Clone cppzmq (header-only library)
if [ -d "cppzmq" ]; then
    rm -rf cppzmq
fi

git clone --depth 1 https://github.com/zeromq/cppzmq.git
cd cppzmq

# Copy header files to local include directory
echo "Installing headers to $LOCAL_DIR/include..."
cp *.hpp $LOCAL_DIR/include/

cd /tmp
rm -rf cppzmq

echo -e "${GREEN}✓ cppzmq installed to $LOCAL_DIR/include${NC}"

# Now rebuild the scheduler
echo ""
echo "======================================================================"
echo "Rebuilding C++ scheduler"
echo "======================================================================"

cd ~/slm-aplit/split_inference/cpp/build

echo "Reconfiguring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$LOCAL_DIR \
    -DCMAKE_PREFIX_PATH=$LOCAL_DIR \
    -DCMAKE_CXX_FLAGS="-I$LOCAL_DIR/include"

echo "Building..."
make -j$(nproc)

echo ""
echo -e "${GREEN}======================================================================"
echo "✓ Build complete!"
echo "======================================================================${NC}"
echo ""
echo "Scheduler binary: ~/slm-aplit/split_inference/cpp/build/scheduler"
echo ""
echo "Next steps:"
echo "  ./run_scheduler.sh"
