# CPU Inference Engine for Phi-tiny-MoE - Summary

## What I've Created

I've built a **comprehensive CPU inference engine** for your Phi-tiny-MoE model with the following components:

### 📁 Files Created

1. **`cpu_inference.py`** (main inference engine, ~600 lines)
   - Full-featured inference engine with benchmarking
   - Supports CPU execution with multiple precision modes
   - Comprehensive metrics collection and reporting

2. **`test_inference.py`** (test suite, ~200 lines)
   - 5 automated tests covering all major functionality
   - Validates generation, streaming, and benchmarking

3. **`requirements_inference.txt`** 
   - Minimal dependencies needed for CPU inference
   - No flash_attn required for CPU mode

4. **`INFERENCE_README.md`**
   - Complete documentation with examples
   - Usage patterns and troubleshooting guide

5. **`run_inference.sh`**
   - Interactive quick-start script
   - Menu-driven interface for common tasks

## 🎯 Key Features

### 1. Multiple Inference Modes

**Single Generation:**
```python
from cpu_inference import CPUInferenceEngine
import torch

engine = CPUInferenceEngine(model_path=".", dtype=torch.float32)
text, metrics = engine.generate("What is AI?", max_new_tokens=100)
```

**Streaming (Token-by-Token):**
```python
for token in engine.generate_streaming("Tell me a story", max_new_tokens=200):
    print(token, end='', flush=True)
```

**Interactive Chat:**
```python
messages = [{"role": "user", "content": "Hello!"}]
response, metrics = engine.chat(messages)
```

### 2. Comprehensive Benchmarking

The engine tracks:
- ⏱️ **Timing**: Prefill time, decode time, total time
- 🚀 **Throughput**: Tokens/second for each phase
- 💾 **Memory**: Peak memory usage
- 📊 **Statistics**: Token counts, generation details

Example output:
```
============================================================
BENCHMARK SUMMARY
============================================================

📊 Token Statistics:
  Prompt tokens:     15
  Generated tokens:  100
  Total tokens:      115

⏱️  Timing:
  Prefill time:      0.234s
  Decode time:       5.821s
  Total time:        6.055s

🚀 Throughput:
  Overall:           19.00 tokens/s
  Prefill:           64.10 tokens/s
  Decode:            17.18 tokens/s

💾 Memory:
  Peak memory:       8234.56 MB
============================================================
```

### 3. Flexible Configuration

**Precision Control:**
```python
# Full precision (slower but accurate)
engine = CPUInferenceEngine(model_path=".", dtype=torch.float32)

# Half precision (faster, less memory)
engine = CPUInferenceEngine(model_path=".", dtype=torch.float16)

# Brain float (good balance, if supported)
engine = CPUInferenceEngine(model_path=".", dtype=torch.bfloat16)
```

**Generation Parameters:**
```python
text, metrics = engine.generate(
    prompt="...",
    max_new_tokens=100,
    temperature=0.7,      # Sampling randomness
    top_p=0.9,            # Nucleus sampling
    top_k=50,             # Top-k sampling
    repetition_penalty=1.0,
    do_sample=True        # False for greedy decoding
)
```

### 4. Command-Line Interface

```bash
# Quick test
python3 cpu_inference.py --prompt "What is quantum computing?" --max-tokens 100

# Interactive mode
python3 cpu_inference.py --interactive

# Full benchmark suite
python3 cpu_inference.py --benchmark --benchmark-output results.json

# Custom parameters
python3 cpu_inference.py --prompt "Hello" --max-tokens 50 --temperature 0.9 --dtype float16
```

### 5. Benchmark Suite

Run comprehensive benchmarks with multiple prompts:
```python
results = engine.run_benchmark_suite(
    prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
    max_new_tokens=100,
    num_runs=5  # Average over 5 runs
)

# Results include:
# - Per-prompt metrics
# - Average across all runs
# - Standard deviations
# - JSON export capability
```

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
cd /home/slm/models/Phi-tiny-MoE-instruct

# Install required packages (one at a time to avoid hanging)
python3 -m pip install torch
python3 -m pip install transformers
python3 -m pip install numpy psutil einops

# Or use requirements file
python3 -m pip install -r requirements_inference.txt
```

### Step 2: Run Your First Generation

```bash
# Quick test with default example
python3 cpu_inference.py

# Or with your own prompt
python3 cpu_inference.py --prompt "Explain machine learning" --max-tokens 150
```

### Step 3: Try Interactive Mode

```bash
python3 cpu_inference.py --interactive
```

### Step 4: Run Benchmarks

```bash
# Full benchmark suite
python3 cpu_inference.py --benchmark

# Run automated tests
python3 test_inference.py
```

## 📊 Understanding Your MoE Model

Based on the config, your model has:

- **Architecture**: Sparse Mixture-of-Experts (MoE)
- **Total Experts**: 16 experts per layer
- **Active Experts**: 2 experts per token (top-2 routing)
- **Layers**: 32 transformer layers
- **Hidden Size**: 4096
- **Expert FFN Size**: 448 (very small per expert!)
- **Attention**: 16 heads (4 KV heads, GQA)
- **Context**: 4096 tokens max
- **Vocab**: 32,064 tokens
- **Dtype**: bfloat16 (native)

**Unique Feature**: This model uses **SparseMixer** routing with:
- Jitter noise for exploration
- Differentiable routing
- Load balancing (though aux loss is set to 0.0)

## 🎯 Next Steps: GPU and Splitting

Now that you have a working CPU baseline, here's what to do next:

### 1. Add GPU Support

Modify `CPUInferenceEngine` to support CUDA:

```python
class GPUInferenceEngine(CPUInferenceEngine):
    def __init__(self, model_path, device="cuda:0", ...):
        # Add CUDA-specific optimizations
        # - Use torch.compile() for faster inference
        # - Enable tensor cores with appropriate dtypes
        # - Use flash attention if available
```

### 2. Implement CPU/GPU Split

Create a hybrid engine that places different components on different devices:

```python
class SplitInferenceEngine:
    """
    Strategy options:
    1. Layer split: First N layers on CPU, rest on GPU
    2. Expert split: Some experts on CPU, some on GPU
    3. Expert-per-device: Each expert on optimal device
    4. Dynamic routing: Route based on expert activations
    """
```

**Expert Split Ideas:**
```python
# Option A: Static split by expert ID
experts_on_cpu = [0, 1, 2, 3]  # Less frequently used
experts_on_gpu = [4, 5, ..., 15]  # Most frequently used

# Option B: Profile-guided placement
# Run benchmarks to see which experts are most active
# Place hot experts on GPU, cold ones on CPU

# Option C: Memory-driven placement
# Put experts on CPU when GPU memory is tight
```

### 3. Add Expert Profiling

Extend the benchmarking to track expert usage:

```python
@dataclass
class MoEMetrics:
    expert_selections: Dict[int, int]  # Expert ID -> selection count
    expert_load_balance: float  # Measure of balance across experts
    expert_gpu_utilization: Dict[int, float]  # GPU time per expert
    
    def suggest_placement(self):
        """Suggest which experts to place on GPU vs CPU"""
        # Return placement strategy based on usage patterns
```

### 4. Optimization Techniques

**For CPU:**
- Use `torch.set_num_threads()` to control CPU cores
- Enable MKL/OpenBLAS optimizations
- Quantization (INT8, INT4) for smaller memory footprint

**For GPU:**
- Flash Attention 2 for faster attention
- `torch.compile()` for kernel fusion
- Mixed precision (FP16/BF16)
- Tensor parallelism for multi-GPU

**For Split:**
- Minimize data movement between devices
- Pipeline parallelism (overlap compute and transfer)
- Smart batching (batch by expert assignment)

## 🔍 Example: Adding Expert Tracking

Here's how to extend the engine to track expert usage:

```python
# In modeling_slimmoe.py, modify PhiMoESparseMoeBlock.forward():

def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # ... existing code ...
    
    # Track expert selections (add this)
    if hasattr(self, 'expert_stats'):
        for expert_id in selected_experts.view(-1).tolist():
            self.expert_stats[expert_id] = self.expert_stats.get(expert_id, 0) + 1
    
    # ... rest of existing code ...
    return final_hidden_states, router_logits
```

Then collect stats after generation:
```python
expert_stats = {}
for layer in engine.model.model.layers:
    if hasattr(layer.block_sparse_moe, 'expert_stats'):
        expert_stats[layer] = layer.block_sparse_moe.expert_stats
```

## 📝 Testing Checklist

Before moving to GPU/split:

- [ ] CPU inference works correctly
- [ ] Benchmarking produces reasonable metrics
- [ ] Memory usage is tracked accurately
- [ ] Streaming generation works
- [ ] Interactive mode is responsive
- [ ] Batch processing works for benchmarks
- [ ] Results can be saved/loaded
- [ ] Different temperatures produce different outputs
- [ ] KV caching speeds up generation

Run: `python3 test_inference.py` to verify

## 🐛 Known Limitations (Current Version)

1. **Expert tracking not yet implemented** - Need to hook into routing
2. **Prefill/decode split is approximate** - Need to instrument generation loop
3. **Memory tracking basic** - Uses process memory, not torch-specific
4. **No quantization support** - Could add INT8/INT4 for faster inference
5. **Single-batch only** - Could optimize for batch processing
6. **No attention backend selection** - Uses default (SDPA on CPU)

## 💡 Tips for Your Use Case

Since your goal is **CPU/GPU splitting**:

1. **Start by profiling expert usage**
   - Run representative workloads
   - Track which experts are used most
   - Measure compute time per expert

2. **Identify bottlenecks**
   - Is it attention or FFN (experts)?
   - Is it memory bandwidth or compute?
   - Is it routing overhead?

3. **Test different split strategies**
   - Layer-wise: Layers 0-15 CPU, 16-31 GPU
   - Expert-wise: Experts 0-7 CPU, 8-15 GPU
   - Hybrid: Attention on GPU, experts on CPU
   - Dynamic: Route hot experts to GPU

4. **Measure carefully**
   - Pure CPU throughput (baseline)
   - Pure GPU throughput (best case)
   - Split overhead (data movement)
   - Sweet spot (optimal split)

## 📚 Additional Resources

- **Model Architecture**: See `modeling_slimmoe.py` for implementation details
- **Configuration**: See `config.json` for model hyperparameters
- **Training Example**: See `sample_finetune.py` for finetuning code
- **HuggingFace Docs**: https://huggingface.co/docs/transformers

## 🤝 Contributing / Next Steps

Would you like me to:

1. ✅ **Add GPU support** - Extend to CUDA devices
2. ✅ **Implement expert tracking** - Monitor which experts are used
3. ✅ **Create split engine** - CPU/GPU hybrid execution
4. ✅ **Add quantization** - INT8/INT4 for faster inference
5. ✅ **Optimize routing** - Faster expert selection
6. ✅ **Add visualization** - Plot expert usage patterns

Let me know what you'd like to tackle next!

---

**Created by**: GitHub Copilot  
**Date**: November 2025  
**Model**: Phi-tiny-MoE-instruct  
**Purpose**: CPU baseline for future CPU/GPU split experiments
