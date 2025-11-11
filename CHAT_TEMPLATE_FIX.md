# Chat Template Fix for Instruction-Tuned Model

## Problem

The Phi-tiny-MoE-instruct model is **instruction-tuned**, meaning it expects prompts to be formatted in a specific chat/instruction format. When you provided raw prompts like "The first US president was...", the model tried to interpret it as an instruction and generated structured responses with citations and system tags like `[RESPONSE]`, `[SYS]`, etc.

## Solution

The inference engine now **automatically formats prompts** using the model's chat template. This wraps your prompt in the proper format that the instruction-tuned model expects.

### What Changed

**Before:**
```python
# Raw prompt sent directly to model
"The first US president was..."
```

**After:**
```python
# Automatically formatted as a chat message
<|user|>
The first US president was...
<|end|>
<|assistant|>
```

This tells the model: "A user is asking a question, please provide a direct answer."

## Usage

### Default Behavior (Recommended)

Chat template formatting is **enabled by default**:

```bash
# This now works correctly!
python3 cpu_inference.py --prompt "The first US president was" --max-tokens 50
```

**Output:**
```
The first President of the United States was George Washington, who served 
from April 30, 1789, to March 4, 1797. He played a pivotal role in the 
American Revolution and was unanimously elected...
```

### Disable Chat Template (If Needed)

If you ever need to send raw prompts without formatting:

```bash
python3 cpu_inference.py --prompt "Your raw prompt" --no-chat-template
```

## Examples

### ✅ Good - Chat Template Enabled (Default)

```bash
# Simple question
python3 cpu_inference.py --prompt "What is AI?" --max-tokens 50

# Completion task
python3 cpu_inference.py --prompt "Explain quantum computing" --max-tokens 100

# Interactive chat
python3 cpu_inference.py --interactive
```

### Python API

```python
from cpu_inference import CPUInferenceEngine

engine = CPUInferenceEngine(model_path=".")

# Chat template automatically applied
text, metrics = engine.generate(
    "The first US president was",
    max_new_tokens=50,
    use_chat_template=True  # Default
)

print(text)  # Proper answer about George Washington
```

### Disable Chat Template

```python
# For raw text completion (rare use case)
text, metrics = engine.generate(
    "Once upon a time",
    max_new_tokens=50,
    use_chat_template=False
)
```

## Why This Matters

Instruction-tuned models are trained to:
1. **Respond to instructions** - "Explain...", "Write...", "What is..."
2. **Follow a specific format** - User message → Assistant response
3. **Provide direct answers** - Not web search results or citations

Without proper formatting, the model may:
- ❌ Generate system messages and metadata
- ❌ Try to simulate a web search interface
- ❌ Produce structured but unhelpful output

With chat template:
- ✅ Generates direct, coherent responses
- ✅ Follows conversational patterns
- ✅ Provides informative answers

## Technical Details

The chat template is automatically loaded from the model's tokenizer configuration. The engine checks if the tokenizer has a chat template and applies it:

```python
if hasattr(self.tokenizer, 'apply_chat_template'):
    messages = [{"role": "user", "content": prompt}]
    formatted = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
```

If the chat template is not available, it falls back to a simple format:
```python
f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
```

## Updated Commands

All these now work correctly:

```bash
# Single generation
python3 cpu_inference.py --prompt "What is machine learning?" --max-tokens 100

# Interactive chat (best for conversations)
python3 cpu_inference.py --interactive

# Benchmark with proper formatting
python3 cpu_inference.py --benchmark

# Custom temperature
python3 cpu_inference.py --prompt "Write a poem" --temperature 0.9 --max-tokens 100
```

## Comparison

### Before Fix (Raw Prompt)
```
Input: "The first US president was..."
Output: 
3. Title: "A Brief History of the U.S. Presidency - History.com"
Snippet: The presidency of George Washington is often cited...
[/RESPONSE]
A: [SYS]expand: 3 [/SYS]
```
❌ Confusing, trying to simulate a search interface

### After Fix (Chat Template)
```
Input: "The first US president was..."
Output:
The first President of the United States was George Washington, who served 
from April 30, 1789, to March 4, 1797. He played a pivotal role in the 
American Revolution and was unanimously elected as the first president.
```
✅ Clear, direct answer

## Summary

- ✅ **Chat template now enabled by default** for instruction-tuned models
- ✅ **Automatic formatting** - prompts are wrapped in proper chat format
- ✅ **Better responses** - model generates coherent, direct answers
- ✅ **Optional override** - use `--no-chat-template` if needed
- ✅ **Works everywhere** - CLI, Python API, interactive mode, benchmarks

The model now works as expected for an instruction-tuned assistant!
