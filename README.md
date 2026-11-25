# Agni — High-Performance PyTorch Engine Fork (`agni-torch`)

Agni is an independent, AGPL-3.0 high-performance fork of **PyTorch**, built specifically to provide optimized native GPU binary kernels (`cubins`) for modern GPU architectures (including Ada Lovelace `sm_89` and Blackwell `sm_121`) along with full forward-compatibility JIT support and immediate support for Python 3.14/3.15.

The name **Agni** names the Vedic deity of fire — matching the flame root of PyTorch. Distributed as the **`agni-torch`** package with full `import torch` API compatibility.

---

## 🔥 Key Features & Capabilities

### ⚡ Custom Architecture Cubins & PTX (`sm_89` & `sm_121`)
- **Current**: Native `sm_89` (RTX 40-series) and `sm_121` (RTX 50-series) cubin embedding across all fatbins, keeping forward JIT PTX (`compute_121`) intact.
- **Planned / Roadmap**: Single-arch per-GPU specialized wheels (`agni-torch-sm89`, `agni-torch-sm121`) for minimal memory footprint and zero unused kernel overhead.

### 🐍 Next-Gen Python Support (Python 3.14 & 3.15)
- **Current**: Full CPython 3.14 and 3.15 C-API extension support built from day one.
- **Planned / Roadmap**: Free-threaded (no-GIL) CPython build optimization and JIT engine integration.

### 🚀 Modern Toolchain & CUDA Floor
- **Current**: CUDA 13.0+ and ROCm 6.0+ toolchain baseline with dead code / legacy architecture cleanup.
- **Planned / Roadmap**: APEX quantization acceleration and low-overhead memory allocator patches.

---

## 🏗️ Architecture & Technology Stack

- **Core Engine**: C++20, CUDA 13.0, ATen tensor library, PyTorch C++ core.
- **Package Name**: `agni-torch` (`import torch`).
- **Licensing**: AGPL-3.0 (with original PyTorch BSD-3-Clause notices retained).

---

## 🛠️ Building & Running

```bat
:: Build Release binary / wheel
build.bat Release

:: Run verification suite
run.bat
```

---

## 📜 License & Acknowledgements

Agni is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0). PyTorch is a trademark of The PyTorch Foundation; Agni is an independent fork and is not affiliated with or endorsed by them. See [`LICENSE`](LICENSE) for details.
