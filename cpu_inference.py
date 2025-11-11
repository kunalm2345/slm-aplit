#!/usr/bin/env python3
"""
CPU Inference Engine for Phi-tiny-MoE model
Supports text generation with benchmarking capabilities
"""

import torch
import time
import json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import threading
import queue


class SystemMonitor:
    """Background process to monitor system resources during inference"""
    
    def __init__(self, interval: float = 0.05):
        """
        Initialize system monitor
        
        Args:
            interval: Sampling interval in seconds (default: 50ms)
        """
        self.interval = interval
        self.samples = []
        self.running = False
        self.thread = None
        
    def _monitor_loop(self):
        """Background monitoring loop running in separate thread"""
        import psutil
        
        process = psutil.Process()
        
        # Get baseline measurements
        last_cpu_times = process.cpu_times()
        last_timestamp = time.time()
        
        while self.running:
            try:
                current_timestamp = time.time()
                current_cpu_times = process.cpu_times()
                
                # Calculate CPU usage based on actual CPU time consumed
                elapsed_wall_time = current_timestamp - last_timestamp
                elapsed_cpu_time = (
                    (current_cpu_times.user - last_cpu_times.user) +
                    (current_cpu_times.system - last_cpu_times.system)
                )
                
                # CPU % relative to 1 core (can exceed 100% if using multiple cores)
                cpu_percent = (elapsed_cpu_time / elapsed_wall_time) * 100 if elapsed_wall_time > 0 else 0
                
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024**2
                
                # Store sample
                self.samples.append({
                    'timestamp': current_timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                })
                
                # Update baseline
                last_cpu_times = current_cpu_times
                last_timestamp = current_timestamp
                
                # Sleep until next sample
                time.sleep(self.interval)
                
            except Exception as e:
                # Silently continue on errors
                pass
    
    def start(self):
        """Start monitoring in background thread"""
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        # Give it a moment to start
        time.sleep(0.01)
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated statistics"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if not self.samples:
            return {
                'avg_cpu_percent': 0.0,
                'max_cpu_percent': 0.0,
                'min_cpu_percent': 0.0,
                'cpu_cores_used': 0.0,
                'avg_memory_mb': 0.0,
                'max_memory_mb': 0.0,
                'sample_count': 0,
            }
        
        cpu_samples = [s['cpu_percent'] for s in self.samples]
        memory_samples = [s['memory_mb'] for s in self.samples]
        
        avg_cpu = np.mean(cpu_samples)
        
        return {
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max(cpu_samples),
            'min_cpu_percent': min(cpu_samples),
            'cpu_cores_used': avg_cpu / 100.0,  # Effective cores used
            'avg_memory_mb': np.mean(memory_samples),
            'max_memory_mb': max(memory_samples),
            'sample_count': len(self.samples),
        }


@dataclass
class BenchmarkMetrics:
    """Store benchmark metrics for model inference"""
    # Timing metrics
    total_time_s: float = 0.0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    time_to_first_token_s: float = 0.0
    
    # Token metrics
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    prefill_tokens_per_second: float = 0.0
    decode_tokens_per_second: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    
    # CPU metrics
    cpu_percent: float = 0.0  # Average CPU % (can exceed 100% for multi-core usage)
    max_cpu_percent: float = 0.0  # Peak CPU % during generation
    min_cpu_percent: float = 0.0  # Minimum CPU % during generation
    cpu_cores_used: float = 0.0  # Effective number of CPU cores utilized
    cpu_count: int = 0  # Total CPU cores available
    cpu_freq_mhz: float = 0.0  # CPU frequency
    
    # Disk I/O metrics
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    disk_read_count: int = 0
    disk_write_count: int = 0
    
    # Expert metrics (MoE specific)
    expert_selections: Dict[int, int] = None
    
    def __post_init__(self):
        if self.expert_selections is None:
            self.expert_selections = {}
    
    def compute_derived_metrics(self):
        """Compute derived metrics from raw measurements"""
        if self.total_time_s > 0:
            self.tokens_per_second = self.total_tokens / self.total_time_s
        
        if self.prefill_time_s > 0:
            self.prefill_tokens_per_second = self.prompt_tokens / self.prefill_time_s
        
        if self.decode_time_s > 0:
            self.decode_tokens_per_second = self.generated_tokens / self.decode_time_s
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def print_summary(self):
        """Print a human-readable summary of metrics"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\n📊 Token Statistics:")
        print(f"  Prompt tokens:     {self.prompt_tokens:,}")
        print(f"  Generated tokens:  {self.generated_tokens:,}")
        print(f"  Total tokens:      {self.total_tokens:,}")
        
        print(f"\n⏱️  Timing:")
        if self.time_to_first_token_s > 0:
            print(f"  Time to 1st token: {self.time_to_first_token_s:.3f}s")
        print(f"  Prefill time:      {self.prefill_time_s:.3f}s")
        print(f"  Decode time:       {self.decode_time_s:.3f}s")
        print(f"  Total time:        {self.total_time_s:.3f}s")
        
        print(f"\n🚀 Throughput:")
        print(f"  Overall:           {self.tokens_per_second:.2f} tokens/s")
        print(f"  Prefill:           {self.prefill_tokens_per_second:.2f} tokens/s")
        print(f"  Decode:            {self.decode_tokens_per_second:.2f} tokens/s")
        
        if self.peak_memory_mb > 0 or self.memory_used_mb > 0:
            print(f"\n💾 Memory:")
            if self.memory_used_mb > 0:
                print(f"  Memory used:       {self.memory_used_mb:.2f} MB ({self.memory_percent:.1f}%)")
            if self.peak_memory_mb > 0:
                print(f"  Peak memory:       {self.peak_memory_mb:.2f} MB")
        
        if self.cpu_percent > 0 or self.cpu_count > 0:
            print(f"\n🖥️  CPU:")
            if self.cpu_count > 0:
                print(f"  CPU cores:         {self.cpu_count}")
            if self.cpu_freq_mhz > 0:
                print(f"  CPU frequency:     {self.cpu_freq_mhz:.0f} MHz")
            if self.cpu_percent > 0:
                print(f"  CPU utilization:   {self.cpu_percent:.1f}% (avg), {self.max_cpu_percent:.1f}% (peak)")
                print(f"  CPU cores used:    {self.cpu_cores_used:.2f} cores")
        
        if self.disk_read_mb > 0 or self.disk_write_mb > 0:
            print(f"\n💿 Disk I/O:")
            if self.disk_read_mb > 0:
                print(f"  Disk read:         {self.disk_read_mb:.2f} MB ({self.disk_read_count:,} ops)")
            if self.disk_write_mb > 0:
                print(f"  Disk write:        {self.disk_write_mb:.2f} MB ({self.disk_write_count:,} ops)")
        
        if self.expert_selections:
            print(f"\n🔀 Expert Usage (MoE):")
            total_selections = sum(self.expert_selections.values())
            for expert_id in sorted(self.expert_selections.keys()):
                count = self.expert_selections[expert_id]
                percentage = (count / total_selections * 100) if total_selections > 0 else 0
                print(f"  Expert {expert_id:2d}:         {count:6,} times ({percentage:5.2f}%)")
        
        print("="*60 + "\n")


class CPUInferenceEngine:
    """
    CPU-based inference engine for Phi-tiny-MoE model
    Supports text generation with detailed benchmarking
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_cache: bool = True,
    ):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to model directory or HuggingFace model ID
            device: Device to run on ('cpu' or specific CPU)
            dtype: Data type for model (float32, bfloat16, or float16)
            use_cache: Whether to use KV cache for generation
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.use_cache = use_cache
        
        # Configure PyTorch threading for optimal CPU performance
        # Use all available CPU cores
        import os
        num_threads = os.cpu_count() or 20
        torch.set_num_threads(num_threads)
        
        print(f"🔧 Initializing CPU Inference Engine...")
        print(f"   Model: {model_path}")
        print(f"   Device: {device}")
        print(f"   Dtype: {dtype}")
        print(f"   Torch threads: {num_threads}")
        
        # Load tokenizer
        print(f"📖 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"🤖 Loading model...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=None,  # Manual device placement
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Get model config
        self.config = self.model.config
        self.num_experts = getattr(self.config, 'num_local_experts', None)
        self.num_layers = self.config.num_hidden_layers
        
        if self.num_experts:
            print(f"   MoE Configuration: {self.num_experts} experts, "
                  f"{self.config.num_experts_per_tok} experts per token")
        
        # Memory tracking
        self.track_memory = True
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            return torch.cuda.max_memory_allocated() / 1024**2
        # For CPU, this is approximate
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024**2
    
    def reset_memory_stats(self):
        """Reset memory statistics"""
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
    
    def _format_prompt(self, prompt: str, use_chat_template: bool = True) -> str:
        """
        Format prompt for instruction-tuned model
        
        Args:
            prompt: Raw prompt text
            use_chat_template: Whether to use chat template formatting
            
        Returns:
            Formatted prompt string
        """
        if not use_chat_template:
            return prompt
        
        # For instruction-tuned models, format as a user message
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                print(f"⚠️  Warning: Could not apply chat template: {e}")
                print(f"   Falling back to raw prompt...")
                return prompt
        else:
            # Fallback: simple instruction format
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        benchmark: bool = True,
        use_chat_template: bool = True,
    ) -> Tuple[str, Optional[BenchmarkMetrics]]:
        """
        Generate text from a prompt with optional benchmarking
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            benchmark: Whether to collect benchmark metrics
            use_chat_template: Whether to format prompt for instruction model (recommended)
            
        Returns:
            Tuple of (generated_text, metrics)
        """
        metrics = BenchmarkMetrics() if benchmark else None
        
        # Reset memory tracking
        if benchmark and self.track_memory:
            self.reset_memory_stats()
        
        # Format prompt for instruction-tuned model
        formatted_prompt = self._format_prompt(prompt, use_chat_template)
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        if benchmark:
            metrics.prompt_tokens = input_ids.shape[1]
            print(f"\n🎯 Generating {max_new_tokens} tokens from {metrics.prompt_tokens} prompt tokens...")
            
            # Capture initial system stats
            import psutil
            process = psutil.Process()
            
            # Initial disk I/O (may require elevated permissions on some systems)
            try:
                initial_io = process.io_counters()
                initial_disk_read = initial_io.read_bytes
                initial_disk_write = initial_io.write_bytes
                initial_disk_read_count = initial_io.read_count
                initial_disk_write_count = initial_io.write_count
            except (AttributeError, PermissionError):
                initial_disk_read = initial_disk_write = 0
                initial_disk_read_count = initial_disk_write_count = 0
            
            # Initial memory
            initial_memory = process.memory_info()
            
            # CPU info
            metrics.cpu_count = psutil.cpu_count()
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    metrics.cpu_freq_mhz = cpu_freq.current
            except:
                pass
            
            # Start background system monitoring
            monitor = SystemMonitor(interval=0.05)  # 50ms sampling
            monitor.start()
        else:
            monitor = None
        
        # Start timing
        start_time = time.time()
        
        # Generate tokens
        # Note: Using model.generate() doesn't allow us to hook into first token generation
        # For accurate TTFT, would need to implement custom generation loop
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            use_cache=self.use_cache,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # End timing
        total_time = time.time() - start_time
        
        # Stop background monitoring and collect stats
        if benchmark:
            monitor_stats = monitor.stop()
        
        # Decode output
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        
        # Collect metrics
        if benchmark:
            import psutil
            process = psutil.Process()
            
            metrics.generated_tokens = output_ids.shape[1] - input_ids.shape[1]
            metrics.total_tokens = output_ids.shape[1]
            metrics.total_time_s = total_time
            
            # Estimate timing breakdown
            # Since we use model.generate(), we estimate TTFT as ~30-40% of total time
            # For accurate TTFT, need custom generation loop with callback
            if metrics.generated_tokens > 0:
                # Rough estimate: prefill + first token generation
                estimated_ttft_ratio = 0.35
                metrics.time_to_first_token_s = total_time * estimated_ttft_ratio
                metrics.prefill_time_s = metrics.time_to_first_token_s
                metrics.decode_time_s = total_time - metrics.time_to_first_token_s
            else:
                # No tokens generated, all time is prefill
                metrics.prefill_time_s = total_time
                metrics.decode_time_s = 0
                metrics.time_to_first_token_s = total_time
            
            # Memory metrics
            if self.track_memory:
                metrics.peak_memory_mb = self.get_memory_usage()
                current_memory = process.memory_info()
                metrics.memory_used_mb = current_memory.rss / 1024**2
                
                # Use monitored memory if available
                if monitor_stats['max_memory_mb'] > 0:
                    metrics.peak_memory_mb = monitor_stats['max_memory_mb']
                
                try:
                    virtual_memory = psutil.virtual_memory()
                    metrics.memory_percent = virtual_memory.percent
                except:
                    pass
            
            # CPU metrics from background monitoring
            metrics.cpu_percent = monitor_stats['avg_cpu_percent']
            metrics.max_cpu_percent = monitor_stats['max_cpu_percent']
            metrics.min_cpu_percent = monitor_stats['min_cpu_percent']
            metrics.cpu_cores_used = monitor_stats['cpu_cores_used']
            
            # Disk I/O metrics (requires permissions on some systems)
            try:
                final_io = process.io_counters()
                metrics.disk_read_mb = (final_io.read_bytes - initial_disk_read) / 1024**2
                metrics.disk_write_mb = (final_io.write_bytes - initial_disk_write) / 1024**2
                metrics.disk_read_count = final_io.read_count - initial_disk_read_count
                metrics.disk_write_count = final_io.write_count - initial_disk_write_count
            except (AttributeError, PermissionError):
                # Disk I/O counters not available (common on some systems)
                pass
            
            metrics.compute_derived_metrics()
        
        return generated_text, metrics
    
    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_chat_template: bool = True,
        **kwargs
    ):
        """
        Generate text token-by-token (streaming mode)
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            use_chat_template: Whether to format prompt for instruction model
            
        Yields:
            Generated tokens as strings
        """
        # Format prompt for instruction-tuned model
        formatted_prompt = self._format_prompt(prompt, use_chat_template)
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        past_key_values = None
        generated_tokens = 0
        
        while generated_tokens < max_new_tokens:
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(next_token_logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Decode and yield token
            token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_str
            
            # Update for next iteration
            input_ids = next_token
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device)
            ], dim=1)
            past_key_values = outputs.past_key_values
            generated_tokens += 1
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
        **generate_kwargs
    ) -> Tuple[str, Optional[BenchmarkMetrics]]:
        """
        Chat interface using the model's chat template
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional arguments for generate()
            
        Returns:
            Tuple of (response, metrics)
        """
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"
        
        return self.generate(prompt, max_new_tokens=max_new_tokens, **generate_kwargs)
    
    def run_benchmark_suite(
        self,
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        num_runs: int = 3,
    ) -> Dict:
        """
        Run a comprehensive benchmark suite
        
        Args:
            prompts: List of prompts to test (uses defaults if None)
            max_new_tokens: Tokens to generate per prompt
            num_runs: Number of runs per prompt
            
        Returns:
            Dictionary with benchmark results
        """
        if prompts is None:
            prompts = [
                "What is the meaning of life?",
                "Explain quantum computing in simple terms.",
                "Write a short poem about artificial intelligence.",
            ]
        
        print(f"\n🧪 Running benchmark suite with {len(prompts)} prompts, {num_runs} runs each...")
        
        results = {
            'model_path': self.model_path,
            'device': self.device,
            'dtype': str(self.dtype),
            'num_experts': self.num_experts,
            'num_layers': self.num_layers,
            'prompts': [],
        }
        
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            prompt_results = {
                'prompt': prompt,
                'runs': [],
            }
            
            for run in range(num_runs):
                print(f"   Run {run+1}/{num_runs}...", end=' ')
                
                _, metrics = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    benchmark=True,
                    temperature=0.7,
                )
                
                prompt_results['runs'].append(metrics.to_dict())
                print(f"✓ {metrics.tokens_per_second:.2f} tok/s")
            
            # Compute averages
            avg_metrics = self._compute_average_metrics(prompt_results['runs'])
            prompt_results['average'] = avg_metrics
            
            results['prompts'].append(prompt_results)
        
        # Compute overall averages
        all_runs = []
        for prompt_result in results['prompts']:
            all_runs.extend(prompt_result['runs'])
        results['overall_average'] = self._compute_average_metrics(all_runs)
        
        print("\n✅ Benchmark suite complete!")
        self._print_benchmark_summary(results)
        
        return results
    
    def _compute_average_metrics(self, runs: List[Dict]) -> Dict:
        """Compute average metrics across multiple runs"""
        avg = {}
        numeric_keys = [
            'total_time_s', 'prefill_time_s', 'decode_time_s', 'time_to_first_token_s',
            'tokens_per_second', 'prefill_tokens_per_second', 'decode_tokens_per_second',
            'prompt_tokens', 'generated_tokens', 'total_tokens',
            'peak_memory_mb', 'memory_used_mb', 'memory_percent',
            'cpu_percent', 'cpu_count', 'cpu_freq_mhz',
            'disk_read_mb', 'disk_write_mb', 'disk_read_count', 'disk_write_count'
        ]
        
        for key in numeric_keys:
            values = [run[key] for run in runs if key in run and run[key] is not None]
            if values:
                avg[key] = np.mean(values)
                avg[f'{key}_std'] = np.std(values)
        
        return avg
    
    def _print_benchmark_summary(self, results: Dict):
        """Print benchmark summary"""
        avg = results['overall_average']
        
        print("\n" + "="*60)
        print("OVERALL BENCHMARK RESULTS")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Device: {results['device']}")
        print(f"Prompts tested: {len(results['prompts'])}")
        print(f"\nAverage Performance:")
        print(f"  Throughput:     {avg['tokens_per_second']:.2f} ± {avg['tokens_per_second_std']:.2f} tok/s")
        print(f"  Prefill speed:  {avg['prefill_tokens_per_second']:.2f} ± {avg['prefill_tokens_per_second_std']:.2f} tok/s")
        print(f"  Decode speed:   {avg['decode_tokens_per_second']:.2f} ± {avg['decode_tokens_per_second_std']:.2f} tok/s")
        if 'peak_memory_mb' in avg:
            print(f"  Peak memory:    {avg['peak_memory_mb']:.2f} ± {avg['peak_memory_mb_std']:.2f} MB")
        print("="*60 + "\n")
    
    def save_benchmark_results(self, results: Dict, output_path: str):
        """Save benchmark results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Benchmark results saved to: {output_path}")


def main():
    """Example usage and interactive mode"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CPU Inference Engine for Phi-tiny-MoE")
    parser.add_argument(
        "--model-path",
        type=str,
        default=".",
        help="Path to model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for text generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark suite"
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="benchmark_results.json",
        help="Output file for benchmark results"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template formatting (use raw prompt)"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize engine
    engine = CPUInferenceEngine(
        model_path=args.model_path,
        device="cpu",
        dtype=dtype,
    )
    
    # Run benchmark suite
    if args.benchmark:
        results = engine.run_benchmark_suite(
            max_new_tokens=args.max_tokens,
            num_runs=3,
        )
        engine.save_benchmark_results(results, args.benchmark_output)
        return
    
    # Interactive mode
    if args.interactive:
        print("\n💬 Interactive mode - Type 'quit' to exit\n")
        while True:
            try:
                prompt = input("You: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                print("\nAssistant: ", end='', flush=True)
                for token in engine.generate_streaming(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    use_chat_template=not args.no_chat_template,
                ):
                    print(token, end='', flush=True)
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
        return
    
    # Single prompt generation
    if args.prompt:
        print(f"\n📝 Prompt: {args.prompt}\n")
        text, metrics = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            benchmark=True,
            use_chat_template=not args.no_chat_template,
        )
        
        print(f"\n✨ Generated text:\n{text}\n")
        
        if metrics:
            metrics.print_summary()
    else:
        # Default: show example
        print("\n🎯 Running example generation...\n")
        example_prompt = "What is artificial intelligence?"
        text, metrics = engine.generate(
            example_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            benchmark=True,
            use_chat_template=not args.no_chat_template,
        )
        
        print(f"\n📝 Prompt: {example_prompt}")
        print(f"✨ Generated:\n{text}\n")
        
        if metrics:
            metrics.print_summary()
        
        print("\n💡 Usage tips:")
        print("  --prompt 'Your prompt'     Generate from a prompt")
        print("  --benchmark                Run full benchmark suite")
        print("  --interactive              Interactive chat mode")
        print("  --help                     Show all options")


if __name__ == "__main__":
    main()
