#!/bin/bash
# Quick start script for CPU inference

echo "=========================================="
echo "Phi-tiny-MoE CPU Inference Quick Start"
echo "=========================================="
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python3 -c "import torch, transformers" 2>/dev/null; then
    echo "❌ Dependencies not found. Installing..."
    pip install -r requirements_inference.txt
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "=========================================="
echo "Choose an option:"
echo "=========================================="
echo "1. Run a quick test"
echo "2. Generate text from a prompt"
echo "3. Interactive chat mode"
echo "4. Run comprehensive benchmark"
echo "5. Run test suite"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Running quick test..."
        python3 cpu_inference.py --prompt "What is AI?" --max-tokens 50
        ;;
    2)
        echo ""
        read -p "Enter your prompt: " prompt
        read -p "Max tokens to generate (default 100): " max_tokens
        max_tokens=${max_tokens:-100}
        python3 cpu_inference.py --prompt "$prompt" --max-tokens $max_tokens
        ;;
    3)
        echo ""
        echo "Starting interactive mode..."
        python3 cpu_inference.py --interactive
        ;;
    4)
        echo ""
        echo "Running comprehensive benchmark..."
        python3 cpu_inference.py --benchmark --benchmark-output benchmark_results.json
        ;;
    5)
        echo ""
        echo "Running test suite..."
        python3 test_inference.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
