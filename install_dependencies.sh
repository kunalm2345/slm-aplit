#!/bin/bash
# Installation script for CPU Inference Engine

echo "=========================================="
echo "Installing CPU Inference Dependencies"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Install dependencies one by one
echo "Installing dependencies..."
echo ""

echo "1/5 Installing numpy..."
python3 -m pip install numpy --no-cache-dir -q
echo "✓ numpy installed"

echo "2/5 Installing psutil..."
python3 -m pip install psutil --no-cache-dir -q
echo "✓ psutil installed"

echo "3/5 Installing einops..."
python3 -m pip install einops --no-cache-dir -q
echo "✓ einops installed"

echo "4/5 Installing PyTorch (CPU version - this may take a minute)..."
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir -q
echo "✓ PyTorch installed"

echo "5/5 Installing transformers..."
python3 -m pip install transformers --no-cache-dir -q
echo "✓ transformers installed"

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 -c "
import torch
import transformers
import numpy
import psutil
import einops
print('✅ torch:', torch.__version__)
print('✅ transformers:', transformers.__version__)
print('✅ numpy:', numpy.__version__)
print('✅ psutil:', psutil.__version__)
print('✅ einops:', einops.__version__)
"

echo ""
echo "=========================================="
echo "Testing inference engine..."
echo "=========================================="
python3 cpu_inference.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Inference engine loaded successfully!"
else
    echo "❌ Error loading inference engine"
    exit 1
fi

echo ""
echo "=========================================="
echo "✨ Installation complete!"
echo "=========================================="
echo ""
echo "Quick start commands:"
echo "  python3 cpu_inference.py --prompt 'Your prompt' --max-tokens 50"
echo "  python3 cpu_inference.py --interactive"
echo "  python3 cpu_inference.py --benchmark"
echo ""
