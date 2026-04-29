# Intel GPU + CM Environment Quick Reference

## Installation Commands

```bash
# Fresh installation (full stack)
./install_intel_gpu_stack.sh

# Selective installation
./install_intel_gpu_stack.sh --skip-driver   # Skip GPU driver
./install_intel_gpu_stack.sh --skip-compute  # Skip OpenCL/Level Zero
./install_intel_gpu_stack.sh --skip-cm       # Skip CM compiler
```

## Environment Check

```bash
# Check environment status
./check_cm_environment.sh

# Check and auto-upgrade if needed
./check_cm_environment.sh --upgrade
```

## Claude Code Skills

```text
/install-gpu-stack   # Install GPU driver, OpenCL, CM compiler
/check-cm-env        # Check and upgrade CM environment
```

## Verification Commands

```bash
# GPU driver
lsmod | grep i915
lspci | grep VGA

# OpenCL
clinfo -l

# CM compiler
ls -la /usr/local/computesdk/lib/libclangFEWrapper.so.11
echo $CM_FE_DIR

# CM headers
ls -la /usr/include/cm/cm/
```

## Test Commands

```bash
# Simple CM test
cd opencl/tests
python test_cm.py

# Page attention test
cd opencl/tests/pageatten
python test_pa.py
```

## Common Issues

| Issue | Command |
|-------|---------|
| Reload environment | `source ~/.bashrc` |
| Check OpenCL devices | `clinfo -l` |
| Verify library paths | `ldconfig -p \| grep clangFE` |
| Check CM_FE_DIR | `echo $CM_FE_DIR` |
| Test GPU access | `ls -la /dev/dri/` |

## Expected Configuration

- **GPU**: Intel Arc/Flex (detected by `lspci`)
- **Driver**: i915 or xe kernel module
- **OpenCL**: intel-opencl-icd v25.40.035542.19835
- **Level Zero**: intel-level-zero-gpu v1.13.035542.19835
- **IGC**: libigc1 v2.20.30698.19835
- **CM Compiler**: clangFEWrapper interface v11
- **CM Headers**: /usr/include/cm/
- **CM_FE_DIR**: /usr/local/computesdk/lib

## File Locations

| Component | Location |
|-----------|----------|
| Scripts | `./install_intel_gpu_stack.sh`<br>`./check_cm_environment.sh` |
| Skills | `./.claude/skills/install-gpu-stack.md`<br>`./.claude/skills/check-cm-env.md` |
| Libraries | `/usr/local/computesdk/lib/` |
| Headers | `/usr/include/cm/` |
| Config | `~/.bashrc` (CM_FE_DIR)<br>`/etc/ld.so.conf.d/computesdk.conf` |
| Packages | `$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04/` |

## Troubleshooting Quick Fixes

```bash
# Permission issues
sudo usermod -aG video,render $USER
newgrp video

# Library cache issues
sudo ldconfig

# Missing headers
sudo cp -r $HOME/CM/ComputeSDK_Linux/usr/include/cm /usr/include/

# Wrong CM_FE_DIR
echo "export CM_FE_DIR=/usr/local/computesdk/lib" >> ~/.bashrc
source ~/.bashrc

# Verify all components
./check_cm_environment.sh
```

## Getting Help

1. Run environment check: `./check_cm_environment.sh`
2. Check installation guide: `cat README_CM_ENVIRONMENT.md`
3. Review skills: `cat .claude/skills/*.md`
4. Test with simple example: `cd opencl/tests && python test_cm.py`
