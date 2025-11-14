# Setup Scripts Comparison - Which One to Use?

## 🔴 `setup_split_inference.sh` - **DANGEROUS for Shared Servers**

### ⚠️ Issues:
- ❌ Requires `sudo` access
- ❌ Installs packages system-wide (`sudo apt-get install`)
- ❌ Installs oneAPI to `/opt/intel/oneapi` (system-wide)
- ❌ Could break dependencies for other users
- ❌ Could fail if you don't have admin rights

### ✅ Use this script ONLY if:
- You have root/sudo access
- You're on your own machine/VM
- You're the only user on the system

---

## ✅ `setup_split_inference_safe.sh` - **SAFE for Shared Servers**

### ✅ Safe because:
- ✅ **NO `sudo` required** - everything installed locally
- ✅ Python venv in project directory (`./venv`)
- ✅ ZeroMQ installed to `~/.local` (if needed)
- ✅ C++ build in project directory
- ✅ All files stay in your user space
- ✅ Won't affect other users

### 📦 Installation locations:
```
Your project dir/
├── venv/                          # Python virtual environment
├── split_inference/cpp/build/     # C++ scheduler binary
└── ...

~/.local/                          # Local user libraries
├── bin/                           # Local binaries
├── lib/                           # Local libraries
└── include/                       # Local headers
```

### ✅ Use this script if:
- You're on a shared server (like `anjuna3.dashlab.in`)
- You don't have sudo access
- You want to avoid breaking things for others
- **THIS IS THE RECOMMENDED VERSION FOR YOUR CASE**

---

## 📋 Quick Comparison Table

| Feature | `setup_split_inference.sh` | `setup_split_inference_safe.sh` |
|---------|---------------------------|--------------------------------|
| Requires sudo | ❌ YES | ✅ NO |
| System-wide changes | ❌ YES | ✅ NO |
| Safe for shared servers | ❌ NO | ✅ YES |
| oneAPI installation | `/opt/intel/oneapi` | Skipped (optional user install) |
| Python packages | venv (local) | venv (local) |
| ZeroMQ | System-wide | `~/.local` |
| Safe for other users | ❌ NO | ✅ YES |

---

## 🚀 Recommended: Use the SAFE version

```bash
# On the shared server (anjuna3)
cd ~/slm-aplit
./setup_split_inference_safe.sh
```

This will:
1. ✅ Check if dependencies exist (no installation)
2. ✅ Install ZeroMQ locally to `~/.local` (if needed)
3. ✅ Create Python venv in project directory
4. ✅ Build C++ scheduler locally
5. ✅ Create helper scripts

**Everything stays in your home directory - completely safe!**

---

## 🔧 If Dependencies Are Missing

If you're missing system dependencies (cmake, gcc, etc.), you'll need to:

### Option 1: Ask admin to install (recommended)
```bash
# Admin runs (one-time, helps everyone):
sudo apt-get install build-essential cmake pkg-config python3 python3-venv
```

### Option 2: Use existing module system
Many shared servers have module systems:
```bash
module load cmake
module load gcc
module load python/3.10
```

### Option 3: Install locally (advanced)
For cmake, gcc, etc. - possible but complicated. Usually better to ask admin.

---

## 🎯 What You Should Do

Since you ran `scp -r ../slm-aplit slm@anjuna3.dashlab.in:~/`, you should:

```bash
# 1. SSH to the server
ssh slm@anjuna3.dashlab.in

# 2. Go to project directory
cd ~/slm-aplit

# 3. Run the SAFE setup script
./setup_split_inference_safe.sh

# 4. If it says dependencies are missing, check for modules:
module avail  # See what's available
module load cmake gcc python/3.10  # Load what you need

# 5. Or ask admin to install:
# "Could you please install: build-essential cmake pkg-config python3 python3-venv"
```

---

## ⚡ Quick Test After Setup

```bash
# Activate Python environment
source venv/bin/activate

# Run tests (CPU-only mode)
python3 split_inference/tests/test_system.py

# Expected output:
# ✓ Config Loading: PASSED
# ✓ Cpu Fallback: PASSED
# (Scheduler tests will be skipped if not running)
```

---

## 🆘 If Something Goes Wrong

The safe script won't break anything because:
- All files are in your home directory
- Can be completely removed with: `rm -rf ~/slm-aplit`
- No system-wide changes
- Other users unaffected

**Bottom line**: Use `setup_split_inference_safe.sh` - it's designed specifically for shared servers like yours! 🎯
