# CM Environment Setup Guide

This directory contains tools to install, check, and upgrade your Intel GPU driver, OpenCL, and CM compiler environment.

## Quick Start

### For Fresh Installation

```bash
# Install complete GPU stack (driver + OpenCL + CM)
./install_intel_gpu_stack.sh

# After installation, reload environment
source ~/.bashrc
```

### For Existing Installation

```bash
# Check current environment
./check_cm_environment.sh

# Check and upgrade if needed
./check_cm_environment.sh --upgrade
```

## What's Included

- **install_intel_gpu_stack.sh** - Fresh installation of GPU driver, OpenCL, and CM
- **check_cm_environment.sh** - Check and upgrade existing CM environment
- **.claude/skills/install-gpu-stack.md** - Claude Code skill for installation
- **.claude/skills/check-cm-env.md** - Claude Code skill for environment management

## Prerequisites

### Intel ComputeSDK Required

Download Intel ComputeSDK and place it at:

```text
$HOME/CM/
├── ComputeSDK_Linux/
│   └── usr/
│       ├── include/cm/          # CM headers
│       └── lib/x86_64-linux-gnu/ # Libraries
└── ComputeSDK_Linux_internal_2025_WW41/
    └── drivers/IGC-22.04/        # .deb packages
```

### Required Packages

The following .deb packages are needed (in IGC-22.04/ directory):

- libigdgmm12_22.8.13871.19835-main_amd64.deb
- libigc1_2.20.30698.19835-main_amd64.deb
- libigdfcl1_2.20.30698.19835-main_amd64.deb
- intel-igc-cm_1.0.1176-main_amd64.deb
- intel-opencl-icd_25.40.035542.19835-1main_amd64.deb
- intel-level-zero-gpu_1.13.035542.19835-1main_amd64.deb
- intel-igc-cm-devel_1.0.1176-main_amd64.deb

## Customization

If your ComputeSDK is in a different location, edit `check_cm_environment.sh` and update:

```bash
COMPUTESDK_PACKAGES_DIR="$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04"
COMPUTESDK_HEADERS="$HOME/CM/ComputeSDK_Linux/usr/include/cm"
COMPUTESDK_LIBS="$HOME/CM/ComputeSDK_Linux/usr/lib/x86_64-linux-gnu"
```

## What Gets Configured

The upgrade configures:

1. **clangFEWrapper v11** at `/usr/local/computesdk/lib/`
2. **IGC** (Intel Graphics Compiler) version 2.20.30698.19835
3. **OpenCL Runtime** intel-opencl-icd 25.40.035542.19835
4. **CM Headers** at `/usr/include/cm/`
5. **CM_FE_DIR** environment variable in `~/.bashrc`
6. **ldconfig** at `/etc/ld.so.conf.d/computesdk.conf`

## After Upgrade

```bash
# Reload environment
source ~/.bashrc

# Verify
cd opencl/tests
python test_cm.py
```

## Using the Claude Code Skill

If you have Claude Code, use:

```text
/check-cm-env
```

This will run the environment check and provide guidance.

## Troubleshooting

### ComputeSDK Not Found

**Error:** `ERROR: ComputeSDK packages not found`

**Solution:** Download ComputeSDK from Intel and extract to `$HOME/CM/`

### Permission Denied

**Error:** `sudo: a password is required`

**Solution:** The upgrade requires sudo access. Run from a terminal where you can enter your password.

### Old clangFEWrapper Conflicts

**Error:** `incompatible clangFEWrapper interface`

**Solution:** Run `./check_cm_environment.sh --upgrade` to install v11

## System Requirements

- Ubuntu 22.04 or similar Linux distribution
- Intel GPU (Arc, Flex, or compatible)
- sudo privileges
- gcc compiler
- Internet connection (for apt-get install -f)

## Support

For issues or questions about:

- **Script functionality**: Check the inline comments in `check_cm_environment.sh`
- **CM programming**: Refer to Intel CM documentation
- **OpenCL issues**: Check `/var/log/` for OpenCL runtime logs

## Version Information

This setup is configured for:

- clangFEWrapper: interface version 11
- IGC: 2.20.30698.19835
- OpenCL ICD: 25.40.035542.19835
- libigdgmm12: 22.8.13871.19835

Updated: 2026-04-27
