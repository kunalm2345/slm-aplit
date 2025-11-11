# Benchmark Metrics Guide

## Overview

The CPU inference engine now tracks comprehensive system metrics during generation. Here's what each metric means and how it's calculated.

## 📊 Token Statistics

```
Prompt tokens:     7
Generated tokens:  60
Total tokens:      67
```

- **Prompt tokens**: Number of tokens in your input (from tokenizer)
- **Generated tokens**: Number of new tokens produced
- **Total tokens**: Sum of prompt + generated tokens

**Calculated from**: `len(tokenizer.encode(prompt))` and output length

---

## ⏱️ Timing Metrics

```
Time to 1st token: 2.167s
Prefill time:      2.167s
Decode time:       4.024s
Total time:        6.191s
```

### Time to First Token (TTFT)
- **What it measures**: How long until the model starts producing output
- **Why it matters**: User-perceived latency - lower is better for responsiveness
- **Calculation**: Estimated as ~35% of total time (prefill + first decode step)
- **Note**: This is an approximation since we use `model.generate()`. For exact TTFT, would need custom generation loop.

### Prefill Time
- **What it measures**: Time to process the input prompt
- **Why it matters**: One-time cost at start of generation
- **Calculation**: Same as TTFT (includes first token generation)
- **Characteristics**: Processes all prompt tokens in parallel (efficient)

### Decode Time
- **What it measures**: Time spent generating tokens autoregressively
- **Why it matters**: This is the main bottleneck - sequential generation
- **Calculation**: `total_time - prefill_time`
- **Characteristics**: One token at a time (can't parallelize)

### Total Time
- **What it measures**: End-to-end generation time
- **Calculation**: `time.time()` before and after `model.generate()`

---

## 🚀 Throughput Metrics

```
Overall:           10.82 tokens/s
Prefill:           3.23 tokens/s
Decode:            14.91 tokens/s
```

### Overall Throughput
- **Formula**: `total_tokens / total_time`
- **Use case**: General performance metric
- **Note**: Can be misleading since prefill is one-time cost

### Prefill Throughput
- **Formula**: `prompt_tokens / prefill_time`
- **Use case**: Measure prompt processing efficiency
- **Optimization**: Benefits from batching multiple prompts

### Decode Throughput
- **Formula**: `generated_tokens / decode_time`
- **Use case**: What users actually experience during generation
- **Optimization**: This is what matters for interactive use
- **Target**: For good UX, aim for >20 tokens/s on CPU

**Key Insight**: Decode throughput is usually lower than prefill because it's sequential.

---

## 💾 Memory Metrics

```
Memory used:       14878.62 MB (16.7%)
Peak memory:       14878.62 MB
```

### Memory Used
- **What it measures**: Current RAM usage by the process
- **Source**: `psutil.Process().memory_info().rss`
- **Includes**: Model weights, activations, KV cache, Python overhead

### Memory Percent
- **What it measures**: Percentage of total system memory in use
- **Source**: `psutil.virtual_memory().percent`
- **Use case**: Check if system is memory-constrained

### Peak Memory
- **What it measures**: Maximum memory during generation
- **Note**: On CPU, this is captured at end (approximate peak)

**What takes memory**:
- Model weights: ~4-8 GB (depending on dtype)
- KV cache: Grows with sequence length
- Activations: Temporary tensors during forward pass

---

## 🖥️ CPU Metrics

```
CPU cores:         20
CPU frequency:     2223 MHz
CPU utilization:   99.8%
```

### CPU Cores
- **What it measures**: Number of logical CPU cores available
- **Source**: `psutil.cpu_count()`
- **Use case**: Understanding parallelism potential

### CPU Frequency
- **What it measures**: Current CPU clock speed in MHz
- **Source**: `psutil.cpu_freq().current`
- **Note**: Can vary with CPU throttling/boosting

### CPU Utilization
- **What it measures**: Percentage of CPU capacity used during generation
- **Source**: `psutil.Process().cpu_percent()`
- **Interpretation**:
  - ~100%: CPU-bound (good - using all available compute)
  - <50%: Memory-bound or I/O-bound (bottleneck elsewhere)

**For MoE models**: High CPU usage is expected since routing and expert computation happens on CPU.

---

## 💿 Disk I/O Metrics

```
Disk read:         X.XX MB (N ops)
Disk write:        X.XX MB (N ops)
```

### Disk Read
- **What it measures**: Data read from disk during generation
- **Source**: `psutil.Process().io_counters().read_bytes`
- **Typical values**: 
  - First load: High (loading model weights)
  - Subsequent: Low (everything cached in RAM)

### Disk Write
- **What it measures**: Data written to disk
- **Typical values**: Usually near zero unless saving checkpoints

**Note**: Disk I/O counters may require special permissions on some systems. If not available, these fields won't be displayed.

---

## 🔀 Expert Usage (MoE-specific)

```
Expert  0:         1,234 times (12.34%)
Expert  1:         2,345 times (23.45%)
...
```

- **What it measures**: How many times each expert was selected during generation
- **Total selections**: `num_layers * num_tokens * experts_per_token`
- **For this model**: 32 layers × tokens × 2 experts = 64× selections per token
- **Use case**: Identify "hot" vs "cold" experts for CPU/GPU splitting

**Not yet implemented** - requires hooking into routing logic.

---

## Interpreting Results

### Good Performance Indicators

✅ **High decode throughput** (>15 tokens/s on CPU)
✅ **CPU utilization near 100%** (fully utilizing compute)
✅ **Low memory percent** (<80% of system RAM)
✅ **Short TTFT** (<2s for typical prompts)

### Performance Bottlenecks

⚠️ **Low CPU utilization** (<50%): Memory bandwidth bottleneck
⚠️ **High memory usage** (>90%): Risk of swapping, consider lower precision
⚠️ **Low decode throughput** (<5 tokens/s): Need optimization or GPU

### Example Analysis

```
Decode throughput: 14.91 tokens/s
CPU utilization:   99.8%
Memory used:       14.8 GB (16.7%)
```

**Interpretation**: 
- ✅ Good CPU utilization (compute-bound, not I/O-bound)
- ✅ Reasonable throughput for CPU inference
- ✅ Plenty of RAM headroom
- 💡 Could try float16 to reduce memory and potentially speed up

---

## Optimization Strategies

### To Improve Throughput

1. **Use lower precision**: `--dtype float16` or `bfloat16`
2. **Reduce token count**: Shorter responses iterate faster
3. **Disable sampling**: Greedy decoding is faster
4. **Optimize CPU threads**: Set `OMP_NUM_THREADS` or `MKL_NUM_THREADS`

### To Reduce Memory

1. **Lower precision dtype**
2. **Disable KV cache**: `use_cache=False` (but slower)
3. **Shorter sequences**: Reduces cache size

### To Reduce Latency (TTFT)

1. **Smaller batch size** (for batch inference)
2. **Quantization** (INT8/INT4)
3. **GPU for prefill** (hybrid CPU/GPU strategy)

---

## Benchmark Suite

When running `--benchmark`, multiple prompts are tested and results averaged:

```bash
python3 cpu_inference.py --benchmark
```

**Provides**:
- Per-prompt metrics
- Average across all runs
- Standard deviation for variability
- Overall system performance profile

**Output**: JSON file with complete results for analysis

---

## System Requirements Check

Based on benchmarks, here's what you need:

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| RAM | 12 GB | 16+ GB |
| CPU | 4 cores | 8+ cores |
| Disk | 10 GB | SSD recommended |
| Memory speed | DDR4-2400 | DDR4-3200+ |

**For this model**: ~15 GB RAM used, benefits from high core count and CPU frequency.

---

## Next Steps: CPU/GPU Split

Use these metrics to guide your splitting strategy:

1. **Profile expert usage** → Identify hot experts
2. **Measure CPU vs GPU throughput** → Find sweet spot
3. **Monitor data transfer** → Minimize GPU↔CPU movement
4. **Track TTFT impact** → Ensure split doesn't hurt latency

The detailed benchmarks will help you make data-driven decisions about where to place different model components!
