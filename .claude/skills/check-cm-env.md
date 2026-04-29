---
name: check-cm-env
description: Check and upgrade OCL/CM/clangFEWrapper v11 environment
trigger: Use when the user asks to check CM environment, verify OpenCL setup, or upgrade clangFEWrapper
---

# Check CM Environment Skill

This skill checks and optionally upgrades the OCL/CM/clangFEWrapper v11 environment.

## What it checks:

1. **clangFEWrapper** - CM compiler frontend (interface version 11)
2. **IGC** - Intel Graphics Compiler (version 2.20.30698.19835)
3. **OpenCL Runtime** - intel-opencl-icd (version 25.40.035542.19835)
4. **CM Headers** - Located at `/usr/include/cm/`
5. **CM_FE_DIR** - Environment variable pointing to `/usr/local/computesdk/lib`
6. **ldconfig** - System library configuration
7. **CM Compilation** - Tests if CM code can compile

## Usage:

### Check only (no changes):
```bash
./check_cm_environment.sh
```

### Check and upgrade if needed:
```bash
./check_cm_environment.sh --upgrade
```

## Expected Configuration:

- **clangFEWrapper**: `/usr/local/computesdk/lib/libclangFEWrapper.so.11` (interface v11)
- **IGC**: Version 2.20.30698.19835
- **OpenCL Runtime**: intel-opencl-icd 25.40.035542.19835
- **CM Headers**: `/usr/include/cm/cm/`
- **CM_FE_DIR**: `/usr/local/computesdk/lib` (in `~/.bashrc`)
- **ldconfig**: `/etc/ld.so.conf.d/computesdk.conf` → `/usr/local/computesdk/lib`

## Prerequisites:

### ComputeSDK Packages

The upgrade requires Intel ComputeSDK packages. By default, the script looks for them at:
```
$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04/
```

**Required packages:**
- `libigdgmm12_22.8.13871.19835-main_amd64.deb`
- `libigc1_2.20.30698.19835-main_amd64.deb`
- `libigdfcl1_2.20.30698.19835-main_amd64.deb`
- `intel-igc-cm_1.0.1176-main_amd64.deb`
- `intel-opencl-icd_25.40.035542.19835-1main_amd64.deb`
- `intel-level-zero-gpu_1.13.035542.19835-1main_amd64.deb`
- `intel-igc-cm-devel_1.0.1176-main_amd64.deb`

**CM Headers and Libraries:**
```
$HOME/CM/ComputeSDK_Linux/usr/include/cm/
$HOME/CM/ComputeSDK_Linux/usr/lib/x86_64-linux-gnu/
```

### How to obtain ComputeSDK:

1. **Download from Intel**: Contact Intel or download from internal repositories
2. **Extract to**: `$HOME/CM/`
3. **Verify structure**:
   ```
   $HOME/CM/
   ├── ComputeSDK_Linux/
   │   └── usr/
   │       ├── include/cm/
   │       └── lib/x86_64-linux-gnu/
   └── ComputeSDK_Linux_internal_2025_WW41/
       └── drivers/IGC-22.04/*.deb
   ```

### Custom Paths:

If your ComputeSDK is in a different location, edit the script and update these variables at the top:
```bash
COMPUTESDK_PACKAGES_DIR="$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04"
COMPUTESDK_HEADERS="$HOME/CM/ComputeSDK_Linux/usr/include/cm"
COMPUTESDK_LIBS="$HOME/CM/ComputeSDK_Linux/usr/lib/x86_64-linux-gnu"
```

## Common Issues:

### Issue: Old clangFEWrapper v8/v10 found
**Solution**: Run with `--upgrade` to install v11

### Issue: CM_FE_DIR not set or pointing to wrong location
**Solution**: The upgrade will add/update it in `~/.bashrc`
**Manual fix**: Add to `~/.bashrc`:
```bash
export CM_FE_DIR=/usr/local/computesdk/lib
```
Then run: `source ~/.bashrc`

### Issue: CM headers not found
**Solution**: Run upgrade to install intel-igc-cm-devel package and copy headers

### Issue: "incompatible clangFEWrapper interface" error
**Cause**: AdaptorCM expects v10 but loading v11, or vice versa
**Solution**: Run upgrade to ensure consistent v11 across all components

### Issue: ComputeSDK packages not found
**Error**: `ERROR: ComputeSDK packages not found at: ...`
**Solution**: Download ComputeSDK and place packages in the expected directory, or update the script paths

## Upgrade Process:

The upgrade will:
1. Install libigdgmm12 dependency
2. Install/upgrade IGC, OpenCL runtime, and intel-igc-cm packages
3. Copy clangFEWrapper v11 to `/usr/local/computesdk/lib/`
4. Install CM headers to `/usr/include/cm/`
5. Configure ldconfig at `/etc/ld.so.conf.d/computesdk.conf`
6. Add/update CM_FE_DIR in `~/.bashrc`

**After upgrade**, run:
```bash
source ~/.bashrc
```

## Verification:

Test that CM code compiles:
```bash
cd ~/OCL/aboutSHW/opencl/tests
python test_cm.py
```

Expected output:
```
[100%] Built target csrc
 GPU device [0] : Intel(R) Arc(TM) A770 Graphics ...
t.shape=[2, 3], t.numpy()=
[[1. 2. 0.]
 [3. 4. 0.]]
[[1.23 1.23 0.  ]
 [1.23 1.23 0.  ]]
```

## Environment Variables:

The script uses these environment variables:
- `$HOME` - User's home directory
- `$USER` - Current username
- `CM_FE_DIR` - Should be set to `/usr/local/computesdk/lib`

## Related Files:

- **Script**: `./check_cm_environment.sh` (in current directory)
- **Config**: `~/.bashrc` (CM_FE_DIR export)
- **Headers**: `/usr/include/cm/`
- **Libraries**: `/usr/local/computesdk/lib/`
- **Packages**: `$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04/`
- **ldconfig**: `/etc/ld.so.conf.d/computesdk.conf`

## System Requirements:

- Ubuntu 22.04 or similar
- Intel GPU (Arc, Flex, or compatible)
- sudo privileges (for installing packages and system configuration)
- gcc (for compiling verification code)
