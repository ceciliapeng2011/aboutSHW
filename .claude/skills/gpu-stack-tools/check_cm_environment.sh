#!/bin/bash
# Check and optionally upgrade OCL/CM/clangFEWrapper v11 environment
# Usage: ./check_cm_environment.sh [--upgrade]

set -e

UPGRADE=false
if [ "$1" = "--upgrade" ]; then
    UPGRADE=true
fi

# Configuration - adjust these paths if needed
COMPUTESDK_PACKAGES_DIR="$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04"
COMPUTESDK_HEADERS="$HOME/CM/ComputeSDK_Linux/usr/include/cm"
COMPUTESDK_LIBS="$HOME/CM/ComputeSDK_Linux/usr/lib/x86_64-linux-gnu"

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m' # No Color

echo -e "${COLOR_BLUE}=== OCL/CM/clangFEWrapper Environment Check ===${COLOR_NC}"
echo ""

# Function to check interface version
check_clangfe_version() {
    local lib_path=$1
    if [ ! -f "$lib_path" ]; then
        echo "N/A"
        return
    fi

    cat > /tmp/check_ver_$$.c << 'EOF'
#include <stdio.h>
#include <dlfcn.h>
int main(int argc, char **argv) {
    void *handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) return 1;
    typedef int (*get_version_func)();
    get_version_func get_version = (get_version_func) dlsym(handle, "IntelCMClangFEGetInterfaceVersion");
    if (!get_version) { dlclose(handle); return 1; }
    printf("%d", get_version());
    dlclose(handle);
    return 0;
}
EOF
    gcc -o /tmp/check_ver_$$ /tmp/check_ver_$$.c -ldl 2>/dev/null
    local version=$(/tmp/check_ver_$$ "$lib_path" 2>/dev/null || echo "N/A")
    rm -f /tmp/check_ver_$$ /tmp/check_ver_$$.c
    echo "$version"
}

# 1. Check clangFEWrapper
echo -e "${COLOR_YELLOW}[1/7] Checking clangFEWrapper...${COLOR_NC}"
CLANGFE_MAIN="/usr/local/computesdk/lib/libclangFEWrapper.so.11"
CLANGFE_SYSTEM="/lib/x86_64-linux-gnu/libclangFEWrapper.so.11"
CLANGFE_OLD="/usr/lib/x86_64-linux-gnu/libclangFEWrapper.so.8"

if [ -f "$CLANGFE_MAIN" ]; then
    VERSION=$(check_clangfe_version "$CLANGFE_MAIN")
    if [ "$VERSION" = "11" ]; then
        echo -e "  ${COLOR_GREEN}✓${COLOR_NC} $CLANGFE_MAIN (interface v$VERSION)"
    else
        echo -e "  ${COLOR_RED}✗${COLOR_NC} $CLANGFE_MAIN (interface v$VERSION, expected v11)"
    fi
else
    echo -e "  ${COLOR_RED}✗${COLOR_NC} $CLANGFE_MAIN not found"
fi

if [ -f "$CLANGFE_OLD" ]; then
    echo -e "  ${COLOR_RED}⚠${COLOR_NC} Old version found: $CLANGFE_OLD (should be removed)"
fi

# 2. Check IGC
echo ""
echo -e "${COLOR_YELLOW}[2/7] Checking IGC (Intel Graphics Compiler)...${COLOR_NC}"
IGC_VERSION=$(dpkg -l | grep "^ii.*libigc1" | awk '{print $3}' || echo "not installed")
if [[ "$IGC_VERSION" == *"2.20.30698.19835"* ]]; then
    echo -e "  ${COLOR_GREEN}✓${COLOR_NC} libigc1: $IGC_VERSION"
else
    echo -e "  ${COLOR_RED}✗${COLOR_NC} libigc1: $IGC_VERSION (expected 2.20.30698.19835)"
fi

# 3. Check OpenCL Runtime
echo ""
echo -e "${COLOR_YELLOW}[3/7] Checking OpenCL Runtime...${COLOR_NC}"
OCL_VERSION=$(dpkg -l | grep "^ii.*intel-opencl-icd" | awk '{print $3}' || echo "not installed")
if [[ "$OCL_VERSION" == *"25.40.035542.19835"* ]]; then
    echo -e "  ${COLOR_GREEN}✓${COLOR_NC} intel-opencl-icd: $OCL_VERSION"
else
    echo -e "  ${COLOR_RED}✗${COLOR_NC} intel-opencl-icd: $OCL_VERSION (expected 25.40.035542.19835)"
fi

OCL_RUNTIME="/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so"
if [ -f "$OCL_RUNTIME" ]; then
    echo -e "  ${COLOR_GREEN}✓${COLOR_NC} OpenCL ICD: $OCL_RUNTIME"
else
    echo -e "  ${COLOR_RED}✗${COLOR_NC} OpenCL ICD not found"
fi

# 4. Check CM Headers
echo ""
echo -e "${COLOR_YELLOW}[4/7] Checking CM Headers...${COLOR_NC}"
if [ -f "/usr/include/cm/cm/cm_common.h" ]; then
    echo -e "  ${COLOR_GREEN}✓${COLOR_NC} CM headers installed at /usr/include/cm/"
elif [ -d "$COMPUTESDK_HEADERS" ]; then
    echo -e "  ${COLOR_RED}✗${COLOR_NC} CM headers not in /usr/include/cm/ (available at $COMPUTESDK_HEADERS)"
else
    echo -e "  ${COLOR_RED}✗${COLOR_NC} CM headers not found"
    echo -e "  ${COLOR_YELLOW}ℹ${COLOR_NC} Download ComputeSDK and place headers at: \$HOME/CM/ComputeSDK_Linux/usr/include/cm"
fi

# 5. Check CM_FE_DIR
echo ""
echo -e "${COLOR_YELLOW}[5/7] Checking CM_FE_DIR environment...${COLOR_NC}"
if [ "$CM_FE_DIR" = "/usr/local/computesdk/lib" ]; then
    echo -e "  ${COLOR_GREEN}✓${COLOR_NC} CM_FE_DIR=$CM_FE_DIR"
else
    echo -e "  ${COLOR_RED}✗${COLOR_NC} CM_FE_DIR=$CM_FE_DIR (expected /usr/local/computesdk/lib)"
    echo -e "  ${COLOR_YELLOW}ℹ${COLOR_NC} Add to ~/.bashrc: export CM_FE_DIR=/usr/local/computesdk/lib"
fi

# 6. Check ldconfig
echo ""
echo -e "${COLOR_YELLOW}[6/7] Checking ldconfig...${COLOR_NC}"
if [ -f "/etc/ld.so.conf.d/computesdk.conf" ]; then
    LDCONF=$(cat /etc/ld.so.conf.d/computesdk.conf | grep -v "^#")
    if [ "$LDCONF" = "/usr/local/computesdk/lib" ]; then
        echo -e "  ${COLOR_GREEN}✓${COLOR_NC} /etc/ld.so.conf.d/computesdk.conf configured"
    else
        echo -e "  ${COLOR_RED}✗${COLOR_NC} /etc/ld.so.conf.d/computesdk.conf incorrect: $LDCONF"
    fi
else
    echo -e "  ${COLOR_RED}✗${COLOR_NC} /etc/ld.so.conf.d/computesdk.conf not found"
fi

# 7. Test compilation
echo ""
echo -e "${COLOR_YELLOW}[7/7] Testing CM compilation...${COLOR_NC}"
TEST_DIR="/tmp/cm_test_$$"
mkdir -p "$TEST_DIR"
cat > "$TEST_DIR/test.cm" << 'EOF'
#include <cm/cm.h>
extern "C" _GENX_MAIN_ void test_kernel(svmptr_t out) {
    vector<int, 1> v = 42;
    cm_svm_block_write(out, v);
}
EOF

if command -v ocloc &> /dev/null; then
    if ocloc compile -file "$TEST_DIR/test.cm" -device dg2 -output "$TEST_DIR/test.bin" &>/dev/null; then
        echo -e "  ${COLOR_GREEN}✓${COLOR_NC} CM compilation test passed"
    else
        echo -e "  ${COLOR_RED}✗${COLOR_NC} CM compilation test failed"
    fi
else
    echo -e "  ${COLOR_YELLOW}⚠${COLOR_NC} ocloc not found, skipping compilation test"
fi
rm -rf "$TEST_DIR"

echo ""
echo -e "${COLOR_BLUE}=== Summary ===${COLOR_NC}"

# Check if upgrade is needed
NEEDS_UPGRADE=false
if [ ! -f "$CLANGFE_MAIN" ]; then NEEDS_UPGRADE=true; fi
if [[ "$IGC_VERSION" != *"2.20.30698.19835"* ]]; then NEEDS_UPGRADE=true; fi
if [[ "$OCL_VERSION" != *"25.40.035542.19835"* ]]; then NEEDS_UPGRADE=true; fi
if [ ! -f "/usr/include/cm/cm/cm_common.h" ]; then NEEDS_UPGRADE=true; fi

if [ "$NEEDS_UPGRADE" = true ]; then
    echo -e "${COLOR_RED}Environment needs upgrade!${COLOR_NC}"

    if [ "$UPGRADE" = true ]; then
        # Check if packages directory exists
        if [ ! -d "$COMPUTESDK_PACKAGES_DIR" ]; then
            echo ""
            echo -e "${COLOR_RED}ERROR: ComputeSDK packages not found at:${COLOR_NC}"
            echo "  $COMPUTESDK_PACKAGES_DIR"
            echo ""
            echo "Please download Intel ComputeSDK and place packages at:"
            echo "  \$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04/"
            echo ""
            echo "Or update COMPUTESDK_PACKAGES_DIR at the top of this script."
            exit 1
        fi

        echo ""
        echo -e "${COLOR_BLUE}=== Starting Upgrade ===${COLOR_NC}"

        # Upgrade process
        cd "$COMPUTESDK_PACKAGES_DIR"

        echo "Installing dependencies and packages..."
        sudo dpkg -i libigdgmm12_22.8.13871.19835-main_amd64.deb
        sudo dpkg -i libigc1_2.20.30698.19835-main_amd64.deb \
            libigdfcl1_2.20.30698.19835-main_amd64.deb \
            intel-igc-cm_1.0.1176-main_amd64.deb \
            intel-opencl-icd_25.40.035542.19835-1main_amd64.deb \
            intel-level-zero-gpu_1.13.035542.19835-1main_amd64.deb \
            intel-igc-cm-devel_1.0.1176-main_amd64.deb

        sudo apt-get install -f -y

        echo "Copying clangFEWrapper v11..."
        sudo mkdir -p /usr/local/computesdk/lib
        sudo cp "$COMPUTESDK_LIBS"/libclangFEWrapper.so* /usr/local/computesdk/lib/
        sudo cp "$COMPUTESDK_LIBS"/libigc.so* /usr/local/computesdk/lib/
        sudo cp "$COMPUTESDK_LIBS"/libigdfcl.so* /usr/local/computesdk/lib/

        if [ -d "$COMPUTESDK_HEADERS" ]; then
            echo "Installing CM headers..."
            sudo cp -r "$COMPUTESDK_HEADERS" /usr/include/
        else
            echo -e "${COLOR_YELLOW}WARNING: CM headers not found at $COMPUTESDK_HEADERS${COLOR_NC}"
        fi

        echo "Updating ldconfig..."
        echo "/usr/local/computesdk/lib" | sudo tee /etc/ld.so.conf.d/computesdk.conf
        sudo ldconfig

        echo "Updating ~/.bashrc..."
        if ! grep -q "export CM_FE_DIR=/usr/local/computesdk/lib" ~/.bashrc; then
            echo "" >> ~/.bashrc
            echo "# CM Compiler Frontend Directory" >> ~/.bashrc
            echo "export CM_FE_DIR=/usr/local/computesdk/lib" >> ~/.bashrc
        else
            sed -i 's|^export CM_FE_DIR=.*|export CM_FE_DIR=/usr/local/computesdk/lib|' ~/.bashrc
        fi

        echo -e "${COLOR_GREEN}Upgrade completed!${COLOR_NC}"
        echo "Please run: source ~/.bashrc"
    else
        echo ""
        echo "To upgrade, run:"
        echo "  $0 --upgrade"
        echo ""
        echo -e "${COLOR_YELLOW}Note:${COLOR_NC} Upgrade requires ComputeSDK packages at:"
        echo "  $COMPUTESDK_PACKAGES_DIR"
        echo ""
        echo "If not available, download from Intel and adjust COMPUTESDK_PACKAGES_DIR"
        echo "at the top of this script."
    fi
else
    echo -e "${COLOR_GREEN}✓ Environment is correctly configured!${COLOR_NC}"
    echo ""
    echo "Configuration:"
    echo "  - clangFEWrapper: interface v11"
    echo "  - IGC: 2.20.30698.19835"
    echo "  - OpenCL runtime: 25.40.035542.19835"
    echo "  - CM headers: /usr/include/cm/"
fi
