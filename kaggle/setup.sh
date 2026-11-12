#!/usr/bin/env bash
# Installs build dependencies on a Kaggle (or any Ubuntu) machine, registers the NVIDIA OpenCL
# driver if it is present, and builds build/mass-prime-siever.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export DEBIAN_FRONTEND=noninteractive
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq cmake g++ libboost-dev libboost-json-dev libboost-date-time-dev \
    ocl-icd-opencl-dev opencl-headers clinfo >/dev/null

# The NVIDIA container runtime mounts libnvidia-opencl.so.1 but usually no ICD file for it.
if [ ! -e /etc/OpenCL/vendors/nvidia.icd ]; then
    NV_OCL="$(ldconfig -p | awk '/libnvidia-opencl.so.1/ {print $NF; exit}')"
    if [ -n "$NV_OCL" ]; then
        $SUDO mkdir -p /etc/OpenCL/vendors
        echo "$NV_OCL" | $SUDO tee /etc/OpenCL/vendors/nvidia.icd >/dev/null
    fi
fi
echo "== OpenCL devices =="
clinfo -l || true

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release -DSIEVER_NATIVE=ON >/dev/null
cmake --build "$ROOT/build" -j"$(nproc)"
echo "Built $ROOT/build/mass-prime-siever"
