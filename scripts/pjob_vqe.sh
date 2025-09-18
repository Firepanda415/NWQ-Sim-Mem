#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J v2
#SBATCH --mail-user=muqing.zheng@pnnl.gov
#SBATCH -o printouts/out_%x_%j.txt
#SBATCH -e printouts/err_%x_%j.txt
#SBATCH --mail-type=ALL
#SBATCH -t 47:00:00
#SBATCH -A m4243

##### ---- For Perlmutter ---- #####

#OpenMP settings:
export OMP_NUM_THREADS=64
export OMP_PLACES=cores
export OMP_PROC_BIND=close




# VQE parameters
NWQSIM_FOLDER="NSNew"
BACKEND="NVGPU"

HAMILTONIAN_PATH=${1} #"$HOME/HamData/bztz66ducc3.hamil"                  #"$HOME/HamData/B_red_ducc_8-5-xacc"
NUM_PARTICLES=${2} #16 6

OPTIMIZER="LN_NEWUOA"                   #"LN_BOBYQA"              # NLopt optimizer
MAX_ITERATIONS=10000
TOLERANCE=1e-6
UB=1
LB=-1
MAX_SECS=144000
# OUTPUT_FILE="vqe_results_${SLURM_JOBID}.txt"



echo "Starting VQE simulation on Perlmutter"
echo "Job ID: ${SLURM_JOBID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Tasks: ${SLURM_NTASKS}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Timestamp: $(date)"
echo "================================================"

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1
source ~/${NWQSIM_FOLDER}/environment/setup_perlmutter.sh
srun -n 1 -c 128 --cpu_bind=cores -G 1 --gpu-bind=single:1 ./${NWQSIM_FOLDER}/build/vqe/nwq_vqe -b ${BACKEND} -f ${HAMILTONIAN_PATH} -p ${NUM_PARTICLES} -v --abstol ${TOLERANCE} -ub ${UB} -lb ${LB} --maxeval ${MAX_ITERATIONS} -o ${OPTIMIZER} --maxtime ${MAX_SECS} --sym 2
squeue --user zhen002


echo "================================================"
echo "VQE simulation completed"
echo "End timestamp: $(date)"
echo "Output file: ${OUTPUT_FILE}"

# Display GPU utilization summary
nvidia-smi --query-gpu=gpu_name,gpu_bus_id,memory.used,utilization.gpu \
    --format=csv >> gpu_utilization_${SLURM_JOBID}.log