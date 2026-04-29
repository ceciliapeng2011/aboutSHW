# OpenCL Environment Fix

## Problem
The OpenCL applications (clinfo, Python test scripts) were crashing with segmentation faults.

## Root Cause
**Library conflict:** Multiple Intel OpenCL runtime libraries (`libigdrcl.so`) exist on the system:

1. **System version** (correct): `/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so` (22.9 MB)
2. **CM SDK version** (incompatible): `/home/ceciliapeng/CM/ComputeSDK_Linux/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so` (40.2 MB)
3. **OpenVINO paths**: Also added by setupvars.sh

The `LD_LIBRARY_PATH` was set to search CM SDK and OpenVINO directories first, causing the wrong (older/incompatible) runtime to be loaded, which crashed when trying to initialize the GPU.

## Solution Applied

### 1. Updated `.bashrc`
- Removed `CM_FE_DIR` from `LD_LIBRARY_PATH` (kept only for `CM_FE_DIR` env var)
- Changed default venv from OpenVINO to OCL
- Added warning to `OV0` function about conflicts

### 2. Created helper scripts

**`run_ocl_tests.sh`** - Run tests with clean environment:
```bash
cd ~/OCL/aboutSHW
./run_ocl_tests.sh python opencl/tests/test_cl.py
```

**`setup_ocl_env.sh`** - Clean current shell environment:
```bash
source ~/OCL/aboutSHW/setup_ocl_env.sh
```

## Usage

### For OpenCL Development (default)
Open a new terminal - it will automatically activate the OCL venv with clean environment.

```bash
cd ~/OCL/aboutSHW/opencl/tests
python test_cl.py  # Should work now
```

### For OpenVINO Development
```bash
source /home/ceciliapeng/openvino.venv/bin/activate
OV0
# Note: This will break OpenCL due to library conflicts
```

### If Switching Between Them
Either:
- Open separate terminals for each
- Or use `run_ocl_tests.sh` to run OpenCL scripts regardless of environment

## Verification

Test OpenCL is working:
```bash
clinfo -l
```

Expected output:
```
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics
```

## Files Modified
- `/home/ceciliapeng/.bashrc` - Fixed LD_LIBRARY_PATH, changed default venv
- Created: `run_ocl_tests.sh` - Helper to run tests in clean environment
- Created: `setup_ocl_env.sh` - Helper to fix current shell

## Technical Details
The conflict happens because:
1. Both runtimes export the same symbol names
2. The OpenCL ICD loader (`/usr/lib/x86_64-linux-gnu/libOpenCL.so.1`) dynamically loads vendor implementations via `/etc/OpenCL/vendors/intel.icd`
3. When the wrong `libigdrcl.so` is loaded first via `LD_LIBRARY_PATH`, it crashes during GPU initialization
4. The system version (from `intel-opencl-icd` package) is the correct one for your GPU

## Recommendation
Keep OpenCL and OpenVINO work in separate terminal sessions to avoid library conflicts.
