#!/bin/bash

# This is a comment
echo "Starting automated tasks..."

# Your Linux commands go here



# Get custom name from command line argument, use default if not provided
FOLDER_NAME=${1:-"NSMem"}
echo "Install in folder: $FOLDER_NAME"


# git clone https://github.com/pnnl/NWQ-Sim.git $FOLDER_NAME


git clone https://github.com/Firepanda415/NWQ-Sim-Mem.git $FOLDER_NAME


source ~/"$FOLDER_NAME"/environment/setup_perlmutter.sh
cd ~/"$FOLDER_NAME"

git submodule init
git submodule update

cd ~/"$FOLDER_NAME"/vqe/nlopt
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC && make -j10

cd ~/"$FOLDER_NAME"
rm -rf build
mkdir build; cd build
cmake .. -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_CUDA_HOST_COMPILER=CC -DCUDA_ENABLED=ON -DCMAKE_BUILD_TYPE=Release && make -j10

echo "NWQSim completed!"