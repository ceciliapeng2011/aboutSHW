#!/bin/bash
# Install Intel GPU Driver, OpenCL, and CM Environment
# Compatible with Ubuntu 22.04 LTS and Intel Arc/Flex GPUs
# Usage: ./install_intel_gpu_stack.sh [--skip-driver] [--skip-compute] [--skip-cm]

set -e

# Color codes
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m'

# Configuration - adjust if needed
COMPUTESDK_PACKAGES_DIR="$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04"
COMPUTESDK_HEADERS="$HOME/CM/ComputeSDK_Linux/usr/include/cm"
COMPUTESDK_LIBS="$HOME/CM/ComputeSDK_Linux/usr/lib/x86_64-linux-gnu"

# Parse command line options
INSTALL_DRIVER=true
INSTALL_COMPUTE=true
INSTALL_CM=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-driver) INSTALL_DRIVER=false; shift ;;
        --skip-compute) INSTALL_COMPUTE=false; shift ;;
        --skip-cm) INSTALL_CM=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${COLOR_BLUE}====================================================${COLOR_NC}"
echo -e "${COLOR_BLUE}   Intel GPU Stack Installation for Ubuntu 22.04   ${COLOR_NC}"
echo -e "${COLOR_BLUE}====================================================${COLOR_NC}"
echo ""

# Detect GPU
echo -e "${COLOR_YELLOW}[0/5] Detecting Intel GPU...${COLOR_NC}"
GPU_INFO=$(lspci | grep -i "VGA.*Intel" || true)
if [ -z "$GPU_INFO" ]; then
    echo -e "${COLOR_RED}ERROR: No Intel GPU detected!${COLOR_NC}"
    echo "This script is designed for Intel Arc/Flex GPUs"
    exit 1
fi
echo -e "${COLOR_GREEN}✓${COLOR_NC} Detected: $GPU_INFO"
echo ""

# Check OS
OS_VERSION=$(lsb_release -rs)
if [ "$OS_VERSION" != "22.04" ]; then
    echo -e "${COLOR_YELLOW}WARNING: This script is tested on Ubuntu 22.04${COLOR_NC}"
    echo "Current version: $OS_VERSION"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# STEP 1: Install/Update Intel GPU Driver
# ============================================================================
if [ "$INSTALL_DRIVER" = true ]; then
    echo -e "${COLOR_BLUE}[1/5] Installing Intel GPU Driver...${COLOR_NC}"

    # Check if driver is already installed
    if lsmod | grep -q "^i915\|^xe"; then
        echo -e "${COLOR_GREEN}✓${COLOR_NC} Intel GPU kernel driver already loaded"
    else
        echo "Installing GPU driver..."

        # Install required packages
        sudo apt-get update
        sudo apt-get install -y \
            linux-headers-$(uname -r) \
            dkms \
            build-essential

        # For Ubuntu 22.04, the i915 driver is included in the kernel
        # If you need a newer version, you can add Intel's graphics repository:
        # wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
        #     sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
        # echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
        #     sudo tee /etc/apt/sources.list.d/intel-graphics.list
        # sudo apt-get update
        # sudo apt-get install -y intel-i915-dkms intel-fw-gpu

        echo -e "${COLOR_GREEN}✓${COLOR_NC} Kernel GPU driver installed"
        echo -e "${COLOR_YELLOW}Note: Reboot may be required for driver changes${COLOR_NC}"
    fi

    # Verify driver
    if lsmod | grep -q "i915\|xe"; then
        echo -e "${COLOR_GREEN}✓${COLOR_NC} GPU driver verified"
    else
        echo -e "${COLOR_RED}✗${COLOR_NC} GPU driver not loaded"
    fi
    echo ""
else
    echo -e "${COLOR_YELLOW}[1/5] Skipping GPU driver installation${COLOR_NC}"
    echo ""
fi

# ============================================================================
# STEP 2: Install Intel Compute Runtime (Level Zero + OpenCL)
# ============================================================================
if [ "$INSTALL_COMPUTE" = true ]; then
    echo -e "${COLOR_BLUE}[2/5] Installing Intel Compute Runtime...${COLOR_NC}"

    # Check if packages directory exists
    if [ ! -d "$COMPUTESDK_PACKAGES_DIR" ]; then
        echo -e "${COLOR_RED}ERROR: ComputeSDK packages not found at:${COLOR_NC}"
        echo "  $COMPUTESDK_PACKAGES_DIR"
        echo ""
        echo "Download Intel ComputeSDK WW41'25 from:"
        echo "  Linux: https://gfx-assets.fm.intel.com/artifactory/gfx-compute-sdk-fm/releases/internal/Linux/2025_WW41/artifacts/ComputeSDK_Linux_internal_2025_WW41.tar.gz"
        echo ""
        echo "Extract to \$HOME/CM/ so the structure is:"
        echo "  \$HOME/CM/ComputeSDK_Linux_internal_2025_WW41/drivers/IGC-22.04/"
        echo ""
        echo "Required .deb packages (will be in the extracted archive):"
        echo "  - libigdgmm12_22.8.13871.19835-main_amd64.deb"
        echo "  - libigc1_2.20.30698.19835-main_amd64.deb"
        echo "  - libigdfcl1_2.20.30698.19835-main_amd64.deb"
        echo "  - intel-igc-cm_1.0.1176-main_amd64.deb"
        echo "  - intel-opencl-icd_25.40.035542.19835-1main_amd64.deb"
        echo "  - intel-level-zero-gpu_1.13.035542.19835-1main_amd64.deb"
        echo "  - intel-igc-cm-devel_1.0.1176-main_amd64.deb"
        echo ""
        echo "Installation commands:"
        echo "  cd \$HOME/CM"
        echo "  wget https://gfx-assets.fm.intel.com/artifactory/gfx-compute-sdk-fm/releases/internal/Linux/2025_WW41/artifacts/ComputeSDK_Linux_internal_2025_WW41.tar.gz"
        echo "  tar -xzf ComputeSDK_Linux_internal_2025_WW41.tar.gz"
        exit 1
    fi

    cd "$COMPUTESDK_PACKAGES_DIR"

    echo "Installing dependencies..."
    sudo apt-get install -y ocl-icd-libopencl1

    echo "Installing Intel compute packages..."
    sudo dpkg -i libigdgmm12_22.8.13871.19835-main_amd64.deb || true
    sudo dpkg -i libigc1_2.20.30698.19835-main_amd64.deb || true
    sudo dpkg -i libigdfcl1_2.20.30698.19835-main_amd64.deb || true
    sudo dpkg -i intel-igc-cm_1.0.1176-main_amd64.deb || true
    sudo dpkg -i intel-opencl-icd_25.40.035542.19835-1main_amd64.deb || true
    sudo dpkg -i intel-level-zero-gpu_1.13.035542.19835-1main_amd64.deb || true

    # Fix any dependency issues
    sudo apt-get install -f -y

    echo -e "${COLOR_GREEN}✓${COLOR_NC} Intel Compute Runtime installed"

    # Verify OpenCL
    if command -v clinfo &> /dev/null; then
        if clinfo -l | grep -q "Intel"; then
            echo -e "${COLOR_GREEN}✓${COLOR_NC} OpenCL runtime verified"
        else
            echo -e "${COLOR_YELLOW}⚠${COLOR_NC} OpenCL runtime installed but no Intel platform found"
        fi
    else
        echo "Installing clinfo for verification..."
        sudo apt-get install -y clinfo
    fi
    echo ""
else
    echo -e "${COLOR_YELLOW}[2/5] Skipping Intel Compute Runtime installation${COLOR_NC}"
    echo ""
fi

# ============================================================================
# STEP 3: Install CM Compiler (clangFEWrapper v11)
# ============================================================================
if [ "$INSTALL_CM" = true ]; then
    echo -e "${COLOR_BLUE}[3/5] Installing CM Compiler (clangFEWrapper v11)...${COLOR_NC}"

    # Install devel package for headers
    if [ -f "$COMPUTESDK_PACKAGES_DIR/intel-igc-cm-devel_1.0.1176-main_amd64.deb" ]; then
        cd "$COMPUTESDK_PACKAGES_DIR"
        sudo dpkg -i intel-igc-cm-devel_1.0.1176-main_amd64.deb || true
        sudo apt-get install -f -y
    fi

    # Copy clangFEWrapper and IGC libraries
    if [ -d "$COMPUTESDK_LIBS" ]; then
        echo "Installing clangFEWrapper v11..."
        sudo mkdir -p /usr/local/computesdk/lib
        sudo cp "$COMPUTESDK_LIBS"/libclangFEWrapper.so* /usr/local/computesdk/lib/
        sudo cp "$COMPUTESDK_LIBS"/libigc.so* /usr/local/computesdk/lib/
        sudo cp "$COMPUTESDK_LIBS"/libigdfcl.so* /usr/local/computesdk/lib/
        echo -e "${COLOR_GREEN}✓${COLOR_NC} clangFEWrapper v11 installed to /usr/local/computesdk/lib"
    else
        echo -e "${COLOR_RED}ERROR: ComputeSDK libraries not found at:${COLOR_NC}"
        echo "  $COMPUTESDK_LIBS"
        exit 1
    fi

    # Copy CM headers
    if [ -d "$COMPUTESDK_HEADERS" ]; then
        echo "Installing CM headers..."
        sudo cp -r "$COMPUTESDK_HEADERS" /usr/include/
        echo -e "${COLOR_GREEN}✓${COLOR_NC} CM headers installed to /usr/include/cm"
    else
        echo -e "${COLOR_YELLOW}WARNING: CM headers not found at $COMPUTESDK_HEADERS${COLOR_NC}"
    fi

    # Configure ldconfig
    echo "Configuring library paths..."
    echo "/usr/local/computesdk/lib" | sudo tee /etc/ld.so.conf.d/computesdk.conf
    sudo ldconfig
    echo -e "${COLOR_GREEN}✓${COLOR_NC} Library paths configured"
    echo ""
else
    echo -e "${COLOR_YELLOW}[3/5] Skipping CM Compiler installation${COLOR_NC}"
    echo ""
fi

# ============================================================================
# STEP 4: Configure Environment Variables
# ============================================================================
echo -e "${COLOR_BLUE}[4/5] Configuring environment variables...${COLOR_NC}"

# Check if CM_FE_DIR is already in bashrc
if grep -q "CM_FE_DIR" ~/.bashrc; then
    echo "Updating CM_FE_DIR in ~/.bashrc..."
    sed -i 's|^export CM_FE_DIR=.*|export CM_FE_DIR=/usr/local/computesdk/lib|' ~/.bashrc
else
    echo "Adding CM_FE_DIR to ~/.bashrc..."
    echo "" >> ~/.bashrc
    echo "# Intel CM Compiler Frontend Directory" >> ~/.bashrc
    echo "export CM_FE_DIR=/usr/local/computesdk/lib" >> ~/.bashrc
fi

echo -e "${COLOR_GREEN}✓${COLOR_NC} Environment variables configured in ~/.bashrc"
echo ""

# ============================================================================
# STEP 5: Verification
# ============================================================================
echo -e "${COLOR_BLUE}[5/5] Verifying installation...${COLOR_NC}"

# Check GPU driver
if lsmod | grep -q "i915\|xe"; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} GPU driver loaded: $(lsmod | grep -E '^i915|^xe' | awk '{print $1}')"
else
    echo -e "${COLOR_RED}✗${COLOR_NC} GPU driver not loaded"
fi

# Check OpenCL
if command -v clinfo &> /dev/null && clinfo -l 2>&1 | grep -q "Intel"; then
    OCL_PLATFORM=$(clinfo -l 2>&1 | grep "Platform #" | head -1)
    OCL_DEVICE=$(clinfo -l 2>&1 | grep "Device #" | head -1 | sed 's/^[[:space:]]*//')
    echo -e "${COLOR_GREEN}✓${COLOR_NC} OpenCL: $OCL_PLATFORM - $OCL_DEVICE"
else
    echo -e "${COLOR_RED}✗${COLOR_NC} OpenCL not available"
fi

# Check Level Zero
if [ -f "/usr/lib/x86_64-linux-gnu/libze_loader.so.1" ]; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} Level Zero runtime installed"
else
    echo -e "${COLOR_YELLOW}⚠${COLOR_NC} Level Zero runtime not found"
fi

# Check clangFEWrapper
if [ -f "/usr/local/computesdk/lib/libclangFEWrapper.so.11" ]; then
    # Check interface version
    cat > /tmp/check_ver_$$.c << 'EOF'
#include <stdio.h>
#include <dlfcn.h>
int main() {
    void *handle = dlopen("/usr/local/computesdk/lib/libclangFEWrapper.so.11", RTLD_LAZY);
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
    VERSION=$(/tmp/check_ver_$$ 2>/dev/null || echo "N/A")
    rm -f /tmp/check_ver_$$ /tmp/check_ver_$$.c
    echo -e "${COLOR_GREEN}✓${COLOR_NC} clangFEWrapper v11 (interface version $VERSION)"
else
    echo -e "${COLOR_RED}✗${COLOR_NC} clangFEWrapper not installed"
fi

# Check CM headers
if [ -f "/usr/include/cm/cm/cm_common.h" ]; then
    echo -e "${COLOR_GREEN}✓${COLOR_NC} CM headers installed"
else
    echo -e "${COLOR_RED}✗${COLOR_NC} CM headers not found"
fi

echo ""
echo -e "${COLOR_BLUE}====================================================${COLOR_NC}"
echo -e "${COLOR_GREEN}Installation Complete!${COLOR_NC}"
echo -e "${COLOR_BLUE}====================================================${COLOR_NC}"
echo ""
echo "Next steps:"
echo "1. Reload environment: source ~/.bashrc"
echo "2. Verify setup: ./check_cm_environment.sh"
echo "3. Test CM code: cd opencl/tests && python test_cm.py"
echo ""

# Check if reboot recommended
if [ "$INSTALL_DRIVER" = true ]; then
    echo -e "${COLOR_YELLOW}Note: If you installed/updated GPU drivers, a reboot is recommended.${COLOR_NC}"
    echo ""
fi

echo "Installed versions:"
dpkg -l | grep -E "intel-opencl-icd|intel-level-zero-gpu|libigc1|intel-igc-cm" | grep "^ii" | awk '{printf "  - %-30s %s\n", $2, $3}'
