---
name: install-gpu-stack
description: Install Intel GPU driver, OpenCL, and CM environment from scratch
trigger: Use when user wants to install GPU stack, set up OpenCL environment, or install CM compiler
---

# Install Intel GPU Stack Skill

Complete installation of Intel GPU driver, OpenCL runtime, Level Zero, and CM compiler environment for Ubuntu 22.04.

## What it installs

1. **Intel GPU Driver** (i915/xe kernel module)
2. **Intel Compute Runtime** (OpenCL + Level Zero)
3. **CM Compiler** (clangFEWrapper v11 + headers)
4. **Environment Configuration** (CM_FE_DIR, ldconfig)

## Quick Start

```bash
# Full installation
./install_intel_gpu_stack.sh

# Skip specific components
./install_intel_gpu_stack.sh --skip-driver  # Skip GPU driver
./install_intel_gpu_stack.sh --skip-compute # Skip OpenCL/Level Zero
./install_intel_gpu_stack.sh --skip-cm      # Skip CM compiler
```

## Prerequisites

### System Requirements

- **OS**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **GPU**: Intel Arc (DG2) or Intel Flex (ATS-M) series
- **Kernel**: 5.15+ (6.2+ recommended for Arc)
- **Privileges**: sudo access required
- **Internet**: For apt-get package installation

### Required Files - ComputeSDK

Download Intel ComputeSDK and place at:

```text
$HOME/CM/
├── ComputeSDK_Linux/
│   └── usr/
│       ├── include/cm/          # CM headers
│       └── lib/x86_64-linux-gnu/ # Libraries
└── ComputeSDK_Linux_internal_2025_WW41/
    └── drivers/IGC-22.04/        # .deb packages
```

**Required .deb packages** (in IGC-22.04/ directory):

- `libigdgmm12_22.8.13871.19835-main_amd64.deb`
- `libigc1_2.20.30698.19835-main_amd64.deb`
- `libigdfcl1_2.20.30698.19835-main_amd64.deb`
- `intel-igc-cm_1.0.1176-main_amd64.deb`
- `intel-opencl-icd_25.40.035542.19835-1main_amd64.deb`
- `intel-level-zero-gpu_1.13.035542.19835-1main_amd64.deb`
- `intel-igc-cm-devel_1.0.1176-main_amd64.deb`

### How to Obtain ComputeSDK

**Option 1: Intel Internal Access**

Download from Intel internal package repositories or contact your Intel representative.

**Option 2: Build from Source** (advanced)

- IGC: https://github.com/intel/intel-graphics-compiler
- CM Compiler: https://github.com/intel/cm-compiler
- Compute Runtime: https://github.com/intel/compute-runtime

## Installation Steps

The script performs these steps automatically:

### Step 1: GPU Driver Installation

- Detects Intel GPU
- Installs kernel headers and build tools
- Configures i915/xe kernel module
- Note: Ubuntu 22.04 includes i915 driver by default

**Manual verification:**
```bash
lsmod | grep i915
lspci | grep VGA
```

### Step 2: Compute Runtime Installation

Installs these packages:
- `libigdgmm12` - Graphics Memory Management Library
- `libigc1` - Intel Graphics Compiler
- `libigdfcl1` - Intel Graphics Common Library
- `intel-opencl-icd` - OpenCL ICD loader
- `intel-level-zero-gpu` - Level Zero runtime
- `intel-igc-cm` - CM compiler frontend

**Manual verification:**
```bash
clinfo -l
dpkg -l | grep intel-opencl-icd
```

### Step 3: CM Compiler Installation

- Installs `intel-igc-cm-devel` package
- Copies clangFEWrapper v11 to `/usr/local/computesdk/lib/`
- Installs CM headers to `/usr/include/cm/`
- Configures ldconfig

**Manual verification:**
```bash
ls -la /usr/local/computesdk/lib/libclangFEWrapper.so.11
ls -la /usr/include/cm/cm/cm_common.h
```

### Step 4: Environment Configuration

Updates `~/.bashrc` with:
```bash
export CM_FE_DIR=/usr/local/computesdk/lib
```

Configures system library paths via:
```text
/etc/ld.so.conf.d/computesdk.conf
```

### Step 5: Verification

Tests all components:
- GPU driver loaded
- OpenCL platform available
- Level Zero runtime present
- clangFEWrapper interface version
- CM headers installed

## Post-Installation

### 1. Reload Environment

```bash
source ~/.bashrc
```

### 2. Verify Installation

```bash
./check_cm_environment.sh
```

Expected output shows all green checkmarks (✓).

### 3. Test Compilation

```bash
cd opencl/tests
python test_cm.py
```

Expected output:
```text
[100%] Built target csrc
GPU device [0] : Intel(R) Arc(TM) A770 Graphics ...
t.shape=[2, 3], t.numpy()=
[[1. 2. 0.]
 [3. 4. 0.]]
[[1.23 1.23 0.  ]
 [1.23 1.23 0.  ]]
```

### 4. Reboot (if driver was installed)

If GPU driver was installed or updated:
```bash
sudo reboot
```

## Customization

Edit the script to change default paths:

```bash
# At top of install_intel_gpu_stack.sh
COMPUTESDK_PACKAGES_DIR="$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04"
COMPUTESDK_HEADERS="$HOME/CM/ComputeSDK_Linux/usr/include/cm"
COMPUTESDK_LIBS="$HOME/CM/ComputeSDK_Linux/usr/lib/x86_64-linux-gnu"
```

## Troubleshooting

### Issue: No Intel GPU detected

**Error:** `ERROR: No Intel GPU detected!`

**Solution:** 
- Verify GPU is installed: `lspci | grep VGA`
- Check if GPU is recognized in BIOS/UEFI
- Ensure PCI slot is properly seated

### Issue: ComputeSDK packages not found

**Error:** `ERROR: ComputeSDK packages not found at: ...`

**Solution:**
- Download ComputeSDK from Intel
- Extract to `$HOME/CM/`
- Verify directory structure matches prerequisites

### Issue: Package installation fails

**Error:** `dpkg: dependency problems`

**Solution:**
```bash
sudo apt-get update
sudo apt-get install -f -y
```

### Issue: OpenCL not available after installation

**Symptoms:** `clinfo -l` shows no platforms

**Solution:**
1. Check ICD loader: `ls -la /etc/OpenCL/vendors/intel.icd`
2. Verify runtime: `ls -la /usr/lib/x86_64-linux-gnu/intel-opencl/`
3. Check permissions: User must be in `video` and `render` groups
   ```bash
   sudo usermod -aG video,render $USER
   newgrp video
   ```
4. Reboot if driver was just installed

### Issue: CM compilation fails with header not found

**Error:** `fatal error: cm/cm.h: No such file or directory`

**Solution:**
- Verify headers: `ls -la /usr/include/cm/`
- Run: `sudo cp -r $HOME/CM/ComputeSDK_Linux/usr/include/cm /usr/include/`

### Issue: clangFEWrapper interface mismatch

**Error:** `incompatible clangFEWrapper interface: expected = 10, loaded = 11`

**Solution:**
- This means old v10 libraries are still present
- Remove old versions: `sudo apt-get remove intel-igc-opencl`
- Run: `sudo ldconfig`
- Rerun installation script

## Advanced Options

### Using Intel's Official Repositories (Alternative)

Instead of local .deb files, you can use Intel's APT repository:

```bash
# Add Intel graphics repository
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
    sudo tee /etc/apt/sources.list.d/intel-graphics.list

sudo apt-get update
sudo apt-get install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero
```

**Note:** Repository versions may differ from ComputeSDK versions.

### Building from Source

For the latest features or custom builds:

1. **IGC (Intel Graphics Compiler)**
   ```bash
   git clone https://github.com/intel/intel-graphics-compiler.git
   cd intel-graphics-compiler
   # Follow build instructions in repo
   ```

2. **CM Compiler**
   ```bash
   git clone https://github.com/intel/cm-compiler.git
   cd cm-compiler
   # Follow build instructions in repo
   ```

## Version Information

This installation configures:

- **clangFEWrapper**: interface version 11
- **IGC**: 2.20.30698.19835
- **OpenCL ICD**: 25.40.035542.19835
- **Level Zero GPU**: 1.13.035542.19835
- **libigdgmm12**: 22.8.13871.19835

## Related Scripts

- **install_intel_gpu_stack.sh** - This installation script
- **check_cm_environment.sh** - Environment verification script
- **README_CM_ENVIRONMENT.md** - User guide

## Additional Resources

- Intel GPU Documentation: https://dgpu-docs.intel.com/
- OpenCL Guide: https://www.khronos.org/opencl/
- Level Zero Specification: https://spec.oneapi.io/level-zero/
- CM Language Guide: Intel internal documentation
