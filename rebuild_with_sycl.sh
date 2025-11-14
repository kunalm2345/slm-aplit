#!/bin/bash
# Rebuild scheduler with SYCL/iGPU support enabled

set -e

echo "========================================"
echo "REBUILDING WITH iGPU SUPPORT"
echo "========================================"

# 1. Source oneAPI environment
echo ""
echo "📦 Step 1: Activating oneAPI..."
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh --force
    echo "✓ oneAPI activated"
else
    echo "✗ ERROR: oneAPI not found at /opt/intel/oneapi/setvars.sh"
    exit 1
fi

# 2. Verify SYCL compiler
echo ""
echo "🔍 Step 2: Verifying SYCL compiler..."
if command -v icpx &> /dev/null; then
    echo "✓ icpx found: $(which icpx)"
    icpx --version | head -n 1
else
    echo "✗ ERROR: icpx compiler not found"
    exit 1
fi

# 3. Check for Intel GPU
echo ""
echo "🎮 Step 3: Checking for Intel GPU..."
if lspci | grep -i "intel.*graphics\|intel.*arc" > /dev/null; then
    echo "✓ Intel GPU detected:"
    lspci | grep -i "intel.*graphics\|intel.*arc"
else
    echo "⚠️  WARNING: No Intel GPU detected"
    echo "   Continuing anyway - SYCL will compile but may not find runtime device"
fi

# 4. Check GPU compute runtime
echo ""
echo "🔧 Step 4: Checking Level Zero runtime..."
if [ -f /usr/lib64/libze_loader.so.1 ] || [ -f /usr/lib/x86_64-linux-gnu/libze_loader.so.1 ]; then
    echo "✓ Level Zero loader found"
else
    echo "⚠️  WARNING: Level Zero runtime not found"
    echo "   You may need to install: intel-level-zero-gpu"
fi

# 5. Clean old build
echo ""
echo "🧹 Step 5: Cleaning old build..."
cd split_inference/cpp
rm -rf build
mkdir -p build
cd build

# 6. Configure with SYCL enabled
echo ""
echo "⚙️  Step 6: Configuring CMake with SYCL..."
cmake .. \
    -DENABLE_SYCL=ON \
    -DENABLE_ONEDNN=OFF \
    -DENABLE_VTUNE=OFF \
    -DCMAKE_BUILD_TYPE=Release

# 7. Build
echo ""
echo "🔨 Step 7: Building scheduler..."
make -j$(nproc)

if [ -f scheduler ]; then
    echo ""
    echo "✅ BUILD SUCCESSFUL!"
    echo ""
    echo "Scheduler binary: $(pwd)/scheduler"
    ls -lh scheduler
    
    # 8. Test SYCL device detection
    echo ""
    echo "🧪 Step 8: Testing SYCL device detection..."
    echo "   (This will start the scheduler briefly)"
    echo ""
    
    # Start scheduler in background and kill after 3 seconds
    timeout 3s ./scheduler 2>&1 | grep -A 5 "Initializing device executor" || true
    
    echo ""
    echo "========================================"
    echo "NEXT STEPS:"
    echo "========================================"
    echo "1. Update run_scheduler.sh to source oneAPI"
    echo "2. Run: ./run_scheduler.sh"
    echo "3. In another terminal: ./run_orchestrator.sh"
    echo ""
    
else
    echo ""
    echo "✗ BUILD FAILED - scheduler binary not found"
    exit 1
fi
