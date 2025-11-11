# CPU Inference Engine for Phi-tiny-MoE

A simple, efficient inference engine for running the Phi-tiny-MoE model on CPU with comprehensive benchmarking capabilities.

## Features

✅ **CPU-optimized inference** - Runs on CPU with configurable precision (float32, float16, bfloat16)  
✅ **Comprehensive benchmarking** - Detailed metrics on throughput, latency, and memory  
✅ **Multiple modes** - Single generation, batch benchmarking, and interactive chat  
✅ **Streaming support** - Token-by-token generation for responsive UIs  
✅ **MoE insights** - Track expert selection patterns (planned feature)  

## Quick Start

### Installation

```bash
# Option 1: Use installation script (recommended)
./install_dependencies.sh

# Option 2: Manual installation (CPU-only PyTorch)
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install transformers numpy psutil einops

# Option 3: Use requirements file
python3 -m pip install -r requirements_inference.txt
```

**Note**: The installation uses CPU-only PyTorch which is much smaller and faster to install (~180MB vs ~2GB for GPU version).

### Basic Usage

```bash
# Run example generation
python cpu_inference.py --model-path .

# Generate from a custom prompt
python cpu_inference.py --prompt "Explain quantum computing" --max-tokens 200

# Interactive chat mode
python cpu_inference.py --interactive

# Run comprehensive benchmark
python cpu_inference.py --benchmark --benchmark-output results.json
```

## Usage Examples

### 1. Single Prompt Generation

```python
from cpu_inference import CPUInferenceEngine
import torch

# Initialize engine
engine = CPUInferenceEngine(
    model_path=".",
    device="cpu",
    dtype=torch.float32
)

# Generate text
text, metrics = engine.generate(
    prompt="What is the meaning of life?",
    max_new_tokens=100,
    temperature=0.7,
    benchmark=True
)

print(text)
metrics.print_summary()
```

### 2. Chat Interface

```python
# Chat with the model
messages = [
    {"role": "user", "content": "What is artificial intelligence?"},
]

response, metrics = engine.chat(
    messages=messages,
    max_new_tokens=150,
    temperature=0.7
)

print(response)
```

### 3. Streaming Generation

```python
# Stream tokens as they're generated
prompt = "Write a short story about a robot:"

print("Assistant: ", end='', flush=True)
for token in engine.generate_streaming(prompt, max_new_tokens=200):
    print(token, end='', flush=True)
print()
```

### 4. Comprehensive Benchmarking

```python
# Run benchmark suite
results = engine.run_benchmark_suite(
    prompts=[
        "What is machine learning?",
        "Explain neural networks.",
        "What are transformers?"
    ],
    max_new_tokens=100,
    num_runs=5
)

# Save results
engine.save_benchmark_results(results, "benchmark_results.json")
```

## Command Line Options

```
--model-path PATH       Path to model directory (default: current directory)
--prompt TEXT           Prompt for text generation
--max-tokens N          Maximum tokens to generate (default: 100)
--temperature FLOAT     Sampling temperature (default: 0.7)
--dtype TYPE            Model precision: float32/float16/bfloat16 (default: float32)
--benchmark             Run comprehensive benchmark suite
--benchmark-output PATH Output file for benchmark results (default: benchmark_results.json)
--interactive           Run in interactive chat mode
```

## Benchmark Metrics

The engine tracks comprehensive metrics:

### Timing Metrics
- **Total time** - End-to-end generation time
- **Prefill time** - Time to process input prompt
- **Decode time** - Time for autoregressive generation

### Throughput Metrics
- **Overall tokens/sec** - Total throughput
- **Prefill tokens/sec** - Input processing speed
- **Decode tokens/sec** - Generation speed

### Memory Metrics
- **Peak memory** - Maximum memory usage during generation

### Token Metrics
- **Prompt tokens** - Input length
- **Generated tokens** - Output length
- **Total tokens** - Sum of input + output

### MoE Metrics (Planned)
- **Expert selections** - Which experts were used and how often
- **Load balancing** - Distribution across experts

## Performance Tips

### 1. Choose the Right Precision

```python
# Fastest on most CPUs
engine = CPUInferenceEngine(model_path=".", dtype=torch.float32)

# Smaller memory footprint (if supported)
engine = CPUInferenceEngine(model_path=".", dtype=torch.bfloat16)
```

### 2. Adjust Generation Parameters

```python
# Faster but less creative (greedy)
text, _ = engine.generate(prompt, do_sample=False)

# More creative but slower
text, _ = engine.generate(
    prompt, 
    do_sample=True,
    temperature=0.9,
    top_p=0.95
)
```

### 3. Enable KV Caching

KV caching is enabled by default for faster generation:

```python
engine = CPUInferenceEngine(
    model_path=".",
    use_cache=True  # Default
)
```

## Model Architecture

This inference engine supports the **Phi-tiny-MoE** architecture:

- **Type**: Sparse Mixture-of-Experts (MoE) Transformer
- **Experts**: 16 total experts
- **Active experts**: 2 per token (top-2 routing)
- **Hidden size**: 4096
- **Intermediate size**: 448 (per expert)
- **Layers**: 32
- **Attention heads**: 16 (4 KV heads)
- **Vocabulary**: 32,064 tokens
- **Context length**: 4,096 tokens

## Next Steps: GPU and Split Execution

This CPU engine serves as a baseline. For GPU support and CPU/GPU splits:

1. **GPU inference** - Modify device parameter and add CUDA optimizations
2. **Split execution** - Place different experts on different devices
3. **Expert routing analysis** - Track which experts are most active
4. **Quantization** - Add INT8/INT4 quantization for faster inference

## Troubleshooting

### Out of Memory

```python
# Reduce precision
engine = CPUInferenceEngine(model_path=".", dtype=torch.float16)

# Reduce batch size (for multi-prompt benchmarks)
# Process prompts one at a time
```

### Slow Generation

```python
# Use greedy decoding
text, _ = engine.generate(prompt, do_sample=False)

# Reduce max_new_tokens
text, _ = engine.generate(prompt, max_new_tokens=50)
```

### Import Errors

Make sure flash_attn is NOT required for CPU inference by checking model loading:
```python
# The engine automatically uses CPU-compatible attention
# No flash_attn needed for CPU
```

## Example Output

```
🔧 Initializing CPU Inference Engine...
   Model: .
   Device: cpu
   Dtype: torch.float32
📖 Loading tokenizer...
🤖 Loading model...
✅ Model loaded in 12.34s
   MoE Configuration: 16 experts, 2 experts per token

🎯 Generating 100 tokens from 8 prompt tokens...

✨ Generated text:
What is artificial intelligence? Artificial intelligence (AI) refers to 
the simulation of human intelligence in machines that are programmed to 
think and learn like humans...

============================================================
BENCHMARK SUMMARY
============================================================

📊 Token Statistics:
  Prompt tokens:     8
  Generated tokens:  100
  Total tokens:      108

⏱️  Timing:
  Prefill time:      0.234s
  Decode time:       5.821s
  Total time:        6.055s

🚀 Throughput:
  Overall:           17.84 tokens/s
  Prefill:           34.19 tokens/s
  Decode:            17.18 tokens/s

💾 Memory:
  Peak memory:       8234.56 MB
============================================================
```

## License

Same as the base model - see LICENSE file in model directory.
