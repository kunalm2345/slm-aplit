#!/usr/bin/env python3
"""
Quick test script for the CPU inference engine
"""

import torch
from cpu_inference import CPUInferenceEngine

def test_basic_generation():
    """Test basic text generation"""
    print("\n" + "="*60)
    print("TEST 1: Basic Text Generation")
    print("="*60)
    
    engine = CPUInferenceEngine(
        model_path=".",
        device="cpu",
        dtype=torch.float32
    )
    
    prompt = "Hello, I am"
    print(f"\nPrompt: '{prompt}'")
    
    text, metrics = engine.generate(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.7,
        benchmark=True
    )
    
    print(f"\nGenerated: {text}")
    print(f"\n✅ Basic generation works!")
    print(f"   Speed: {metrics.tokens_per_second:.2f} tokens/s")
    
    return True


def test_longer_generation():
    """Test longer text generation with benchmarking"""
    print("\n" + "="*60)
    print("TEST 2: Longer Generation with Full Metrics")
    print("="*60)
    
    engine = CPUInferenceEngine(
        model_path=".",
        device="cpu",
        dtype=torch.float32
    )
    
    prompt = "Artificial intelligence is"
    print(f"\nPrompt: '{prompt}'")
    
    text, metrics = engine.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7,
        benchmark=True
    )
    
    print(f"\nGenerated: {text}")
    metrics.print_summary()
    
    return True


def test_streaming():
    """Test streaming generation"""
    print("\n" + "="*60)
    print("TEST 3: Streaming Generation")
    print("="*60)
    
    engine = CPUInferenceEngine(
        model_path=".",
        device="cpu",
        dtype=torch.float32
    )
    
    prompt = "The future of computing is"
    print(f"\nPrompt: '{prompt}'")
    print("\nStreaming output: ", end='', flush=True)
    
    token_count = 0
    for token in engine.generate_streaming(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.7
    ):
        print(token, end='', flush=True)
        token_count += 1
    
    print(f"\n\n✅ Streaming works! Generated {token_count} tokens")
    
    return True


def test_different_temperatures():
    """Test generation with different temperatures"""
    print("\n" + "="*60)
    print("TEST 4: Different Temperatures")
    print("="*60)
    
    engine = CPUInferenceEngine(
        model_path=".",
        device="cpu",
        dtype=torch.float32
    )
    
    prompt = "Once upon a time"
    temperatures = [0.1, 0.7, 1.0]
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        text, _ = engine.generate(
            prompt=prompt,
            max_new_tokens=20,
            temperature=temp,
            benchmark=False
        )
        print(f"{text}")
    
    print("\n✅ Temperature control works!")
    
    return True


def test_mini_benchmark():
    """Test mini benchmark suite"""
    print("\n" + "="*60)
    print("TEST 5: Mini Benchmark Suite")
    print("="*60)
    
    engine = CPUInferenceEngine(
        model_path=".",
        device="cpu",
        dtype=torch.float32
    )
    
    results = engine.run_benchmark_suite(
        prompts=[
            "What is AI?",
            "Explain computers.",
        ],
        max_new_tokens=30,
        num_runs=2
    )
    
    print("\n✅ Benchmark suite completed!")
    
    # Save results
    engine.save_benchmark_results(results, "test_benchmark.json")
    print("   Results saved to test_benchmark.json")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "🧪"*30)
    print("CPU INFERENCE ENGINE TEST SUITE")
    print("🧪"*30)
    
    tests = [
        ("Basic Generation", test_basic_generation),
        ("Longer Generation", test_longer_generation),
        ("Streaming", test_streaming),
        ("Temperature Control", test_different_temperatures),
        ("Mini Benchmark", test_mini_benchmark),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    else:
        print("🎉 All tests passed!")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
