#!/usr/bin/env bash
# GPU environment probe for xmrig-cuda RandomX (run on the mining host).
# Collects ROCm/GPU facts and computes the valid RandomX intensity range.
# Usage: bash gpu_probe.sh [device_index]   (default 0)
set -u
DEV="${1:-0}"

echo "================ XMRI-CUDA GPU PROBE ================"
date
echo "host: $(hostname) | kernel: $(uname -r)"
echo

echo "--- ROCm installation ---"
for v in /opt/rocm/.info/version /opt/rocm/.info/version-*; do
    [ -f "$v" ] && echo "$v: $(cat "$v")" && break
done
command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null
echo "rocm path candidates: $(ls -d /opt/rocm* 2>/dev/null | tr '\n' ' ')"
echo

echo "--- Discovered GPU agents (rocminfo) ---"
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | awk '
        /^  Name:/        { name=$2 }
        /Marketing Name/  { sub(/.*Marketing Name:[[:space:]]*/,""); mkt=$0 }
        /^  Compute Unit:/       { cu=$3 }
        /Max Clock Frequency/    { clk=$NF }
        /Wavesize|Wavefront Size/ { wf=$NF }
        /Max Size:/ && inlds { lds=$3; inlds=0 }
        /Group Segment Size/{ inlds=1 }
        /Max Waves Per CU|Maximum number of waves/ { mw=$NF }
        /^  Name:.*gfx/ { gfx=name }
        END { }
        /Marketing Name|Compute Unit|Wavefront|Max Clock|gfx/ { print }
    ' | sort -u
    echo
    echo "--- Full agent dump for gfx devices (grep details) ---"
    rocminfo 2>/dev/null | grep -E "Name:|Marketing|Compute Unit|Waves|Max Clock|Pool.*size|Segment" | grep -B1 -A4 -i "gfx" | head -60
else
    echo "rocminfo not found in PATH"
fi
echo

echo "--- rocm-smi ---"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram --showuse --showclocks 2>/dev/null
else
    echo "rocm-smi not found in PATH"
fi
echo

echo "--- VRAM arithmetic (RandomX per-hash cost = 2 MB scratchpad + ~2 GB dataset) ---"
if command -v rocm-smi >/dev/null 2>&1; then
    TOTAL=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Total Memory \(B\)|VRAM Total/ {print $NF; exit}')
    USED=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used Memory \(B\)|VRAM Used/ {print $NF; exit}')
    if [ -n "${TOTAL:-}" ]; then
        TB=$(echo "$TOTAL" | awk '{printf "%.1f", $1/1073741824}')
        UB=$(echo "${USED:-0}" | awk '{printf "%.1f", $1/1073741824}')
        FREE=$(echo "$TOTAL ${USED:-0}" | awk '{printf "%d", ($1-$2)}')
        echo "VRAM total=${TB} GB used=${UB} GB free_bytes=${FREE}"
        # Keep a 2 GB aperture safety margin below the VA wall on top of the dataset.
        SAFE=$(echo "$FREE" | awk '{v=int(($1 - 2147483648 - 2147483648)/2097152); v=v-(v%32); print v}')
        echo "max safe intensity (~2 MB/hash, dataset+margin reserved): ${SAFE}"
        echo
        echo "suggested configs (threads x blocks = intensity, must be multiple of 32):"
        for t in 16 32 64 128; do
            for b in 16 32 64 96 128; do
                i=$((t*b))
                if [ $((i % 32)) -eq 0 ] && [ "$i" -ge 1024 ] && [ "$i" -le "${SAFE:-16384}" ]; then
                    printf "  threads=%-4s blocks=%-4s intensity=%s\n" "$t" "$b" "$i"
                fi
            done
        done
    else
        echo "could not parse VRAM sizes from rocm-smi"
    fi
else
    echo "rocm-smi unavailable; skipping VRAM math"
fi
echo

echo "--- xmrig plugin check ---"
if [ -f ./xmrig-rocm.dll ] || [ -f ./xmrig-rocm.so ] || [ -f ./libxmrig-rocm.so ]; then
    ls -la ./*xmrig-rocm* 2>/dev/null
else
    echo "(no xmrig-rocm plugin in cwd - run from the xmrig binary directory)"
fi
echo "================ PROBE DONE - paste everything back ================"
