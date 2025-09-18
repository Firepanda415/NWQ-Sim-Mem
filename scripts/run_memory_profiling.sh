#!/bin/bash

# Script to run ADAPT-VQE with comprehensive GPU memory profiling
# Usage: ./run_memory_profiling.sh [input_file] [NumParticles] [output_file]

echo "========================================"
echo "  NWQSim ADAPT-VQE Memory Profiling"
echo "========================================"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. GPU information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits
    echo
else
    echo "Warning: CUDA not detected. GPU profiling may not work."
    echo
fi

# Set input parameters
INPUT_FILE=${1:-"$HOME/NSMem/H2O_1.75_Eq_11-Orbitals_DUCC3_H2O-1.75_Eq_DUCC3_10-electrons_11-Orbitals.out-xacc"}
P=${2:-"10"}
OUTPUT_FILE=${2:-"memory_profile_output.log"}

echo "Configuration:"
echo "  Input file: $INPUT_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Build type: GPU-enabled ADAPT-VQE"
echo

# Check if the executable exists
if [ ! -f "$HOME/NSMem/build/vqe/nwq_vqe" ]; then
    echo "Error: $HOME/NSMem/build/vqe/nwq_vqe not found."
    echo "Please build the project first with:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DCUDA_ENABLED=ON"
    echo "  make -j"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found."
    echo "Available example files:"
    find examples/molecules/ -name "*.yaml" 2>/dev/null | head -5
    exit 1
fi

echo "Starting ADAPT-VQE with GPU memory profiling..."
echo "Output will be saved to: $OUTPUT_FILE"
echo

# Run with memory profiling - redirect both stdout and stderr
{
    echo "=== ADAPT-VQE Memory Profiling Session ===" 
    echo "Date: $(date)"
    
    # System info
    echo "System Info:"
    echo "  CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
    echo "  Total RAM: $(free -h | grep 'Mem:' | awk '{print $2}')"
    echo "  Available RAM: $(free -h | grep 'Mem:' | awk '{print $7}')"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Info:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    fi
    echo
    
    # Initial memory state
    echo "=== Initial Memory State ==="
    echo "CPU Memory:"
    free -h
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory:"
        nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits
    fi
    echo
    
    # Run the actual simulation
    echo "Starting simulation..."
    # $HOME/NSMem/build/vqe/nwq_vqe -f "$INPUT_FILE" -p 6 -v --abstol 1e-8 --maxeval 1000 -o LN_BOBYQA --adapt -ag 1e-4
    $HOME/NSMem/build/vqe/nwq_vqe -b NVGPU -f "$INPUT_FILE" -p "$P" -v --abstol 1e-6 --maxeval 5000 -o LN_COBYLA --adapt -ag 1e-3 -am 120
    
    echo
    echo "=== Final Memory State ==="
    echo "CPU Memory:"
    free -h
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory:"
        nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits
    fi
    echo "Memory profiling complete."
    
} 2>&1 | tee "$OUTPUT_FILE"

echo
echo "========================================"
echo "Memory profiling completed!"
echo "Results saved to: $OUTPUT_FILE"
echo
echo "To analyze the results:"
echo "  grep -A 10 'ADAPT-VQE GPU Memory Profiling' $OUTPUT_FILE"
echo "  grep 'Memory used by ADAPT operators' $OUTPUT_FILE"
echo "  grep 'GPU Memory Breakdown' $OUTPUT_FILE"
echo "========================================"