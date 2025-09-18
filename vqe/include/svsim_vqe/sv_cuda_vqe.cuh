#ifndef VQE_CUDA_STATE
#define VQE_CUDA_STATE
#include "state.hpp"
#include "svsim/sv_cuda.cuh"
#include "vqe_state.hpp"
#include "utils.hpp"
#include "circuit/ansatz.hpp"
#include "observable/hamiltonian.hpp"
#include "nlopt.hpp"
#include "private/cuda_util.cuh"
#include <memory>
#include <cmath>

namespace NWQSim
{
  namespace VQE {
    class SV_CUDA_VQE: public VQEState, public SV_CUDA {
      public:
        SV_CUDA_VQE(std::shared_ptr<Ansatz> a, 
                   std::shared_ptr<Hamiltonian> h, 
                   nlopt::algorithm optimizer_algorithm,
                   Callback _callback,
                   IdxType seed = 0,
                   OptimizerSettings opt_settings = OptimizerSettings()): 
                                      SV_CUDA(a->num_qubits()),
                                      VQEState(a, h, optimizer_algorithm, _callback, seed, opt_settings) {
        // resize the expectation value data structure (to-do, just make a double)
        // allocate space for host, device memory for observable lists
      };

      virtual void allocate_observables(IdxType size) override {
        // GPU MEMORY PROFILING: Before observable allocation
        size_t gpu_free_before, gpu_total;
        cudaMemGetInfo(&gpu_free_before, &gpu_total);
        
        SAFE_ALOC_GPU(obsvals_dev, size * sizeof(ObservableList));
        obsvals.resize(size);
        obsvec.resize(size);
        
        // GPU MEMORY PROFILING: After observable allocation
        size_t gpu_free_after, gpu_total_after;
        cudaMemGetInfo(&gpu_free_after, &gpu_total_after);
        size_t allocated_memory = gpu_free_before - gpu_free_after;
        
        std::cout << "\nGPU Observable Allocation:" << std::endl;
        std::cout << "  Allocated " << size << " ObservableList structs" << std::endl;
        std::cout << "  Memory allocated: " << (allocated_memory / (1024.0*1024.0)) << " MB" << std::endl;
        std::cout << "  Struct size: " << sizeof(ObservableList) << " bytes" << std::endl;
        std::cout << "  Expected memory: " << (size * sizeof(ObservableList) / (1024.0*1024.0)) << " MB" << std::endl;
      };
      virtual void fill_obslist(IdxType index) override {
        /**
         * @brief Add an entry to the ObservableList data structure
         */
        ObservableList obs;
        
        obs.numterms = zmasks[index].size();
        IdxType isize = obs.numterms * sizeof(IdxType);
        IdxType vsize = obs.numterms * sizeof(ValType);
        SAFE_ALOC_GPU(obs.zmasks, isize);
        obs.exp_output = 0;
        SAFE_ALOC_GPU(obs.coeffs, vsize);
        // allocate memory for the zmasks and coefficients
        cudaSafeCall(cudaMemcpy(obs.zmasks, zmasks[index].data(), isize,
                                cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(obs.coeffs, coeffs[index].data(), vsize,
                                cudaMemcpyHostToDevice));
          // std::cout << "Filling list: " << index << " " <<(obsvals_dev + index) << std::endl;
        ObservableList* obs_device = (ObservableList*)((void*)obsvals_dev + index * sizeof(ObservableList));
        obsvals[index] = obs;
        obsvec[index] = obs_device;
        measurement->EXPECT(obs_device); 
      };
      ~SV_CUDA_VQE()
      {
          cudaSafeCall(cudaMemcpy(obsvals.data(), obsvals_dev, obsvals.size() * sizeof(ObservableList),
                                  cudaMemcpyDeviceToHost));
          for (auto o: obsvals) {
            SAFE_FREE_GPU(o.zmasks);
            SAFE_FREE_GPU(o.coeffs);
          }
          SAFE_FREE_GPU(obsvals_dev);
      }
      virtual void call_simulator() override {

        cudaSafeCall(cudaMemcpy(obsvals_dev, obsvals.data(), obsvals.size() * sizeof(ObservableList),
                                    cudaMemcpyHostToDevice));
        reset_state();
        sim(ansatz);
        sim(measurement);
        cudaDeviceSynchronize();
        cudaSafeCall(cudaMemcpy(obsvals.data(), obsvals_dev, obsvals.size() * sizeof(ObservableList),
                                    cudaMemcpyDeviceToHost));
        // for (size_t i = 0; i < dim; i++) {
        //   std::cout << "(" << sv_real_cpu[i] * sv_real_cpu[i] + sv_imag_cpu[i] * sv_imag_cpu[i] << "), ";
        // }
        // std::cout << std::endl;
        expectation_value = 0;
        size_t index = 0;
        for (auto o: obsvals) {
          expectation_value += o.exp_output;
          index++;
        }
      };
      virtual void call_simulator(std::shared_ptr<Ansatz> _measurement, bool reset) override { 
        if (reset) {
          reset_state();
          sim(ansatz);
        }    
        sim(_measurement);
        cudaDeviceSynchronize();
      };
      virtual void set_exp_gate(std::shared_ptr<Ansatz> circuit, ObservableList* o, std::vector<IdxType>& _zmasks, std::vector<ValType>& _coeffs) override {
        // GPU MEMORY PROFILING: Track individual observable memory
        static size_t total_observable_memory = 0;
        
        size_t gpu_free_before, gpu_total;
        cudaMemGetInfo(&gpu_free_before, &gpu_total);
        
        ObservableList obs;
        obs.numterms = _zmasks.size();
        IdxType isize = obs.numterms * sizeof(IdxType);
        IdxType vsize = obs.numterms * sizeof(ValType);
        SAFE_ALOC_GPU(obs.zmasks, isize);
        obs.exp_output = 0;
        SAFE_ALOC_GPU(obs.coeffs, vsize);
        cudaSafeCall(cudaMemcpy(obs.zmasks, _zmasks.data(), isize,
                                cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(obs.coeffs, _coeffs.data(), _coeffs.size() * sizeof(ValType),
                                cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(o, &obs, sizeof(ObservableList),
                                cudaMemcpyHostToDevice));
        circuit->EXPECT(o);
        
        // GPU MEMORY PROFILING: After individual observable setup
        size_t gpu_free_after, gpu_total_after;
        cudaMemGetInfo(&gpu_free_after, &gpu_total_after);
        size_t this_observable_memory = gpu_free_before - gpu_free_after;
        total_observable_memory += this_observable_memory;
      };
      virtual void delete_observables(ObservableList* observables, IdxType size) override {
        std::vector<ObservableList> obs_temp (size);
            cudaSafeCall(cudaMemcpy(obs_temp.data(), observables, size * sizeof(ObservableList),
                                    cudaMemcpyDeviceToHost));
        for (auto o: obs_temp) {
          SAFE_FREE_GPU(o.zmasks);
          SAFE_FREE_GPU(o.coeffs);
        }
        SAFE_FREE_GPU(observables);
      };

      virtual void get_exp_values(const std::vector<ObservableList*>& observables, std::vector<IdxType> sizes, std::vector<ValType>& output) override {
        std::fill(output.begin(), output.end(), 0);
        for (size_t i = 0; i < observables.size(); i++) {
          std::vector<ObservableList> host_temp(sizes[i]);
          cudaSafeCall(cudaMemcpy(host_temp.data(), observables[i], sizes[i] * sizeof(ObservableList),
                                    cudaMemcpyDeviceToHost));
          
          for (ObservableList obs: host_temp) {
            // std::cout << obs.exp_output << std::endl;
            output[i] += obs.exp_output;
          }
          // output.at(i) = observables.at(i)->exp_output;
        }
      };

      virtual void allocate_observables(ObservableList*& observables, IdxType size) override {
        SAFE_ALOC_GPU(observables, size * sizeof(ObservableList));
      };
      
      // PHASE 1 MEMORY FIX: Override to actually deallocate CUDA simulation state  
      virtual void deallocate_simulation_state() override {
        // Free GPU memory
        if (sv_real) SAFE_FREE_GPU(sv_real);
        if (sv_imag) SAFE_FREE_GPU(sv_imag);
        if (m_real) SAFE_FREE_GPU(m_real);
        if (m_imag) SAFE_FREE_GPU(m_imag);
        
        // Free CPU memory  
        if (sv_real_cpu) SAFE_FREE_HOST_CUDA(sv_real_cpu);
        if (sv_imag_cpu) SAFE_FREE_HOST_CUDA(sv_imag_cpu);
        
        // Reset pointers to null
        sv_real = sv_imag = m_real = m_imag = nullptr;
        sv_real_cpu = sv_imag_cpu = nullptr;
        std::cout << ">>>>>>     FLAG: Deallocate _simulation_state!      <<<<<<<" << std::endl;
      };
      
      virtual void reallocate_simulation_state() override {
        // Only reallocate if memory was actually freed
        if (sv_real == nullptr) {
          // CPU side initialization
          SAFE_ALOC_HOST_CUDA(sv_real_cpu, sv_size);
          SAFE_ALOC_HOST_CUDA(sv_imag_cpu, sv_size);
          memset(sv_real_cpu, 0, sv_size);
          memset(sv_imag_cpu, 0, sv_size);
          sv_real_cpu[0] = 1.;  // |0⟩ state
          
          // GPU side allocation
          SAFE_ALOC_GPU(sv_real, sv_size);
          SAFE_ALOC_GPU(sv_imag, sv_size);
          SAFE_ALOC_GPU(m_real, sv_size + sizeof(ValType));
          SAFE_ALOC_GPU(m_imag, sv_size + sizeof(ValType));
          
          // Copy initial state to GPU
          cudaSafeCall(cudaMemcpy(sv_real, sv_real_cpu, sv_size, cudaMemcpyHostToDevice));
          cudaSafeCall(cudaMemcpy(sv_imag, sv_imag_cpu, sv_size, cudaMemcpyHostToDevice));
          cudaSafeCall(cudaMemset(m_real, 0, sv_size + sizeof(ValType)));
          cudaSafeCall(cudaMemset(m_imag, 0, sv_size + sizeof(ValType)));
        }
      };
    protected:
        ObservableList* obsvals_dev;
        std::vector<ObservableList> obsvals;

};
  };
} // namespace NWQSim

#endif