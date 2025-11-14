#!/bin/bash
# SAFE Setup script for shared server (no sudo required)
# Installs everything locally in user home directory

set -e  # Exit on error

echo "======================================================================"
echo "SPLIT CPU/iGPU INFERENCE - SAFE SETUP FOR SHARED SERVERS"
echo "======================================================================"
echo "This script installs everything locally (no sudo required)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set local installation directory
LOCAL_DIR="$HOME/.local"
mkdir -p "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR/bin"
mkdir -p "$LOCAL_DIR/lib"
mkdir -p "$LOCAL_DIR/include"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$LOCAL_DIR/bin:"* ]]; then
    export PATH="$LOCAL_DIR/bin:$PATH"
    echo "export PATH=\"$LOCAL_DIR/bin:\$PATH\"" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\"$LOCAL_DIR/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
fi

echo -e "${GREEN}Local installation directory: $LOCAL_DIR${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check system dependencies (no installation, just warning)
echo ""
echo "======================================================================"
echo "Step 1: Checking system dependencies"
echo "======================================================================"

MISSING_DEPS=()

for dep in cmake pkg-config python3 python3-venv gcc g++; do
    if ! command_exists $dep; then
        MISSING_DEPS+=($dep)
    else
        echo -e "${GREEN}✓ $dep found${NC}"
    fi
done

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo "Ask your admin to install: sudo apt-get install build-essential cmake pkg-config python3 python3-venv"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Install ZeroMQ and cppzmq locally (if not available)
echo ""
echo "======================================================================"
echo "Step 2: Setting up ZeroMQ and C++ bindings"
echo "======================================================================"

# Check for libzmq (C library)
if pkg-config --exists libzmq; then
    echo -e "${GREEN}✓ ZeroMQ (libzmq) found (system)${NC}"
    LIBZMQ_PREFIX=$(pkg-config --variable=prefix libzmq)
else
    echo "Installing ZeroMQ locally to $LOCAL_DIR..."
    cd /tmp
    
    if [ ! -f "zeromq-4.3.5.tar.gz" ]; then
        wget https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.tar.gz
    fi
    
    tar xzf zeromq-4.3.5.tar.gz
    cd zeromq-4.3.5
    
    ./configure --prefix=$LOCAL_DIR
    make -j$(nproc)
    make install
    
    cd /tmp
    rm -rf zeromq-4.3.5
    
    LIBZMQ_PREFIX=$LOCAL_DIR
    echo -e "${GREEN}✓ ZeroMQ installed locally${NC}"
fi

# Check for cppzmq (C++ header-only bindings)
if [ -f "$LIBZMQ_PREFIX/include/zmq.hpp" ] || [ -f "/usr/include/zmq.hpp" ] || [ -f "$LOCAL_DIR/include/zmq.hpp" ]; then
    echo -e "${GREEN}✓ cppzmq (C++ bindings) found${NC}"
else
    echo "Installing cppzmq (C++ bindings) to $LOCAL_DIR..."
    cd /tmp
    
    # cppzmq is header-only, just download and copy
    if [ ! -d "cppzmq" ]; then
        git clone --depth 1 https://github.com/zeromq/cppzmq.git
    fi
    
    cd cppzmq
    
    # Copy header files
    mkdir -p $LOCAL_DIR/include
    cp *.hpp $LOCAL_DIR/include/
    
    cd /tmp
    rm -rf cppzmq
    
    echo -e "${GREEN}✓ cppzmq installed locally${NC}"
fi

# Step 3: Skip oneAPI (not needed for basic functionality)
echo ""
echo "======================================================================"
echo "Step 3: Intel oneAPI"
echo "======================================================================"

if [ -d "/opt/intel/oneapi" ]; then
    echo -e "${GREEN}✓ oneAPI found at /opt/intel/oneapi (system)${NC}"
    echo "Run 'source /opt/intel/oneapi/setvars.sh' to use it"
elif [ -d "$HOME/intel/oneapi" ]; then
    echo -e "${GREEN}✓ oneAPI found at ~/intel/oneapi (user)${NC}"
    echo "Run 'source ~/intel/oneapi/setvars.sh' to use it"
else
    echo -e "${YELLOW}⚠️  oneAPI not found${NC}"
    echo "SYCL/iGPU features will be disabled (CPU-only mode)"
    echo ""
    echo "To install oneAPI locally (optional):"
    echo "  1. Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
    echo "  2. Run installer with: sh installer.sh --install-dir ~/intel/oneapi"
    echo ""
fi

# Step 4: Install Python dependencies in virtual environment
echo ""
echo "======================================================================"
echo "Step 4: Setting up Python environment"
echo "======================================================================"

# Create virtual environment in project directory (local to project)
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi
pwd
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python packages..."
pip install \
    torch --index-url https://download.pytorch.org/whl/cpu 
pip install transformers \
    numpy \
    psutil \
    einops \
    pyzmq \
    PyYAML \
    onnx \
    onnxruntime

echo -e "${GREEN}✓ Python dependencies installed in local venv${NC}"

# Step 5: Build C++ scheduler (without SYCL initially)
echo ""
echo "======================================================================"
echo "Step 5: Building C++ scheduler"
echo "======================================================================"

cd split_inference/cpp

# Clean previous build
rm -rf build
mkdir build
cd build

echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$LOCAL_DIR \
    -DCMAKE_PREFIX_PATH=$LOCAL_DIR \
    -DCMAKE_CXX_FLAGS="-I$LOCAL_DIR/include" \
    -DENABLE_SYCL=OFF \
    -DENABLE_ONEDNN=OFF \
    -DENABLE_VTUNE=OFF

echo "Building..."
make -j$(nproc)

echo -e "${GREEN}✓ Scheduler built successfully${NC}"

cd ../../..

# Step 6: Verify installation
echo ""
echo "======================================================================"
echo "Step 6: Verifying installation"
echo "======================================================================"

echo "Checking scheduler executable..."
if [ -f "split_inference/cpp/build/scheduler" ]; then
    echo -e "${GREEN}✓ Scheduler executable found${NC}"
else
    echo -e "${RED}✗ Scheduler executable not found${NC}"
fi

echo "Checking Python environment..."
pwd
source ./venv/bin/activate
python3 -c "import torch; import transformers; import zmq; print('✓ Python imports OK')"

# Step 7: Create helper scripts
echo ""
echo "======================================================================"
echo "Step 7: Creating helper scripts"
echo "======================================================================"

# Create run_scheduler.sh
cat > run_scheduler.sh << 'EOF'
#!/bin/bash
# Start the C++ scheduler

# Try to source oneAPI if available
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh
elif [ -f "$HOME/intel/oneapi/setvars.sh" ]; then
    source $HOME/intel/oneapi/setvars.sh
fi

./split_inference/cpp/build/scheduler tcp://*:5555
EOF

chmod +x run_scheduler.sh

# Create run_orchestrator.sh
cat > run_orchestrator.sh << 'EOF'
#!/bin/bash
# Start the Python orchestrator

source venv/bin/activate
python3 split_inference/python/orchestrator.py "$@"
EOF

chmod +x run_orchestrator.sh

# Create enable_oneapi.sh
cat > enable_oneapi.sh << 'EOF'
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
EOF

chmod +x enable_oneapi.sh

echo -e "${GREEN}✓ Helper scripts created${NC}"

# Summary
echo ""
echo "======================================================================"
echo "SETUP COMPLETE - SAFE FOR SHARED SERVER"
echo "======================================================================"
echo ""
echo "✓ All installations are local to this directory"
echo "✓ No system-wide changes made"
echo "✓ Safe to run on shared servers"
echo ""
echo "Installation locations:"
echo "  • Python venv: $(pwd)/venv"
echo "  • C++ scheduler: $(pwd)/split_inference/cpp/build/scheduler"
echo "  • Local libs: $LOCAL_DIR"
echo ""
echo "Next steps:"
echo "  1. Start the scheduler:"
echo "     ./run_scheduler.sh"
echo ""
echo "  2. In another terminal, run the orchestrator:"
echo "     ./run_orchestrator.sh"
echo ""
echo "  3. Or run CPU-only inference:"
echo "     source venv/bin/activate"
echo "     python3 cpu_inference.py --prompt 'Test' --max-tokens 50"
echo ""
echo "  4. Run tests:"
echo "     source venv/bin/activate"
echo "     python3 split_inference/tests/test_system.py"
echo ""
echo -e "${GREEN}Happy hacking! (on a shared server, safely)${NC}"
