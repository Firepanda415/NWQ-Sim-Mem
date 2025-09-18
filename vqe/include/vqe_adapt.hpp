#ifndef VQE_ADAPT_HH
#define VQE_ADAPT_HH
#include "state.hpp"
#include "vqe_state.hpp"
#include "environment.hpp"
#include "utils.hpp"
#include "circuit/dynamic_ansatz.hpp"
#include <cstddef>
namespace NWQSim {
  namespace VQE {
    const std::complex<ValType> imag = {0, 1.0}; // constant `i` value
    /**
    * @brief  Implementation of ADAPT-VQE optimizer
    * @note   Uses commutator-based gradient calculation with either Fermionic or Pauli operator pools
    */
    class AdaptVQE {
      protected:
        std::shared_ptr<DynamicAnsatz> ansatz; // Iterative Ansatz being constructed
        std::shared_ptr<VQEState> state; // VQE state for energy calculation and ansatz optimization
        std::vector<ValType> gradient_magnitudes; // vector of gradient magnitudes
        std::shared_ptr<Hamiltonian> hamil; // Hamiltonian observable
        std::vector<std::shared_ptr<Ansatz> > gradient_measurement; // Measurement circuit for gradient estimation
        std::vector<std::vector<std::vector<double> > > commutator_coeffs; // Coefficients for commutator operators
        std::vector<std::vector<std::vector<IdxType> > > commutator_zmasks; // Zmasks for commutator operators
        std::vector<ObservableList*> gradient_observables; // Vector of structure pointers for measurement circuit
        std::vector<IdxType> observable_sizes; // Stores the number of commuting cliques for each commutator  
        size_t num_pauli_terms_total; // MZ: Total number of Pauli terms in all commutators
        size_t num_commuting_groups; // MZ: Total number of commuting groups (measurement cliques)

    public: 
        size_t get_numpauli() const { return num_pauli_terms_total;}; // MZ: Get total Pauli terms
        void set_numpauli(size_t value) { num_pauli_terms_total = value; }; //MZ
        
        size_t get_numcommutinggroups() const { return num_commuting_groups;}; // MZ: Get number of commuting groups
        void set_numcommutinggroups(size_t value) { num_commuting_groups = value; }; //MZ

      //Ctor
      AdaptVQE(std::shared_ptr<DynamicAnsatz> _ans, std::shared_ptr<VQEState> backend, std::shared_ptr<Hamiltonian> _hamil): 
            ansatz(_ans), 
            state(backend), 
            hamil(_hamil) {
      }
      //Dtor
      ~AdaptVQE() {
        size_t index = 0;
        for (auto i: gradient_observables)
          state->delete_observables(i, observable_sizes[index++]);
      }

     /**
      * @brief  Calculate the commutator of two sums over Pauli strings
      * @note   Uses a std::unordered_map to keep track of coefficients to avoid duplicate/zero terms. Computes [oplist1, oplist2]
      * @param  oplist1: Sum over Pauli terms, first operator in commutator
      * @param  oplist2: Sum over Pauli terms, second operator in commutator
      * @param  summation: std::unordered_map to track coefficient sums
      * @retval None
      */
      void commutator(std::vector<PauliOperator>& oplist1, 
                      std::vector<PauliOperator>& oplist2, 
                      PauliMap& summation) {
          for (auto p1: oplist1) {
            for (auto p2: oplist2) {
              auto p12 = (p1 * p2);
              auto p21 = (p2 * p1);
              // multiply by -i to account for phase from Hermitian UCCSD flag in JW constructor
              p21 *= -1.0 * imag;
              p12 *= imag;
              // Track coefficients for each Pauli term
              if (summation.find(p12) == summation.end()) {
                summation[p12] = p12.getCoeff();
              } else {
                summation[p12] += p12.getCoeff();
              }
              if (summation.find(p21) == summation.end()) {
                summation[p21] = p21.getCoeff();
              } else {
                summation[p21] += p21.getCoeff();
              }
            }
          }
      }
      /**
       * @brief  Construct commutator for each operator in ansatz pool
       * @note   This is the *big* slow (yes that grammar was intentional), O(N^8) scaling 
       * * TODO: Get O(N^6) scaling with reduced density matrix approach
       * @retval None
       */
      void make_commutators() {
        std::shared_ptr<Hamiltonian> hamil = state->get_hamiltonian();
        const auto& pauli_strings = hamil->getPauliOperators();
        const auto& pauli_op_pool = ansatz->get_pauli_op_pool();
        IdxType poolsize = pauli_op_pool.size();
        
        // MEMORY PROFILING: Initial state (both GPU and CPU)
        #ifdef CUDA_ENABLED
        size_t gpu_free_initial = 0, gpu_total = 0;
        #endif
        
        // CPU memory profiling variables
        size_t vm_rss_kb = 0, vm_peak_kb = 0, vm_size_kb = 0;
        
        if (state->get_process_rank() == 0) {
          // CPU memory profiling
          std::ifstream status_file("/proc/self/status");
          std::string line;
          
          while (std::getline(status_file, line)) {
            if (line.find("VmRSS:") == 0) {
              sscanf(line.c_str(), "VmRSS: %zu kB", &vm_rss_kb);
            } else if (line.find("VmPeak:") == 0) {
              sscanf(line.c_str(), "VmPeak: %zu kB", &vm_peak_kb);
            } else if (line.find("VmSize:") == 0) {
              sscanf(line.c_str(), "VmSize: %zu kB", &vm_size_kb);
            }
          }
          
          std::cout << "\n=== ADAPT-VQE Memory Profiling (CPU + GPU) ===" << std::endl;
          std::cout << "Initial CPU memory:" << std::endl;
          std::cout << "  RSS (Resident Set Size): " << (vm_rss_kb / 1024.0) << " MB" << std::endl;
          std::cout << "  Peak memory usage: " << (vm_peak_kb / 1024.0) << " MB" << std::endl;
          std::cout << "  Virtual memory size: " << (vm_size_kb / 1024.0) << " MB" << std::endl;
          
          #ifdef CUDA_ENABLED
          cudaMemGetInfo(&gpu_free_initial, &gpu_total);
          std::cout << "\nInitial GPU memory:" << std::endl;
          std::cout << "  Total GPU memory: " << (gpu_total / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          std::cout << "  Free GPU memory: " << (gpu_free_initial / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          std::cout << "  Used GPU memory: " << ((gpu_total - gpu_free_initial) / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          #endif
          
          std::cout << "\nStarting UCCSD operator pool allocation..." << std::endl;
          std::cout << "  Pool size: " << poolsize << " operators" << std::endl;
          std::cout << "  Hamiltonian terms: " << pauli_strings.size() << std::endl;
        }
        
        commutator_coeffs.resize(poolsize);
        gradient_measurement.resize(poolsize);
        commutator_zmasks.resize(poolsize);
        gradient_magnitudes.resize(poolsize);
        gradient_observables.resize(poolsize);
        observable_sizes.resize(poolsize);
        // size_t num_pauli_terms_total = 0; // MZ: want to pass this value out
        num_pauli_terms_total = 0; // MZ: Initialize the total number of Pauli terms in all commutators
        num_commuting_groups = 0; // MZ: Initialize the total number of commuting groups
        for (size_t i = 0; i < poolsize; i++) {
          // MEMORY OPTIMIZATION: Progress reporting every 50 operators with memory usage
          if (state->get_process_rank() == 0 && i % 50 == 0) {
            // Get current CPU memory usage
            std::ifstream status_file("/proc/self/status");
            std::string line;
            size_t vm_rss_kb = 0;
            while (std::getline(status_file, line)) {
              if (line.find("VmRSS:") == 0) {
                sscanf(line.c_str(), "VmRSS: %zu kB", &vm_rss_kb);
                break;
              }
            }
            
            #ifdef CUDA_ENABLED
            size_t gpu_free, gpu_total;
            cudaMemGetInfo(&gpu_free, &gpu_total);
            std::cout << "Processing operator " << i << "/" << poolsize 
                      << " - CPU: " << (vm_rss_kb/1024.0) << " MB, GPU: " << (gpu_free/1e9) << " GB free" << std::endl;
            #else
            std::cout << "Processing operator " << i << "/" << poolsize 
                      << " - CPU: " << (vm_rss_kb/1024.0) << " MB" << std::endl;
            #endif
          }
          
          // Get all of the ungrouped Pauli strings for this commutator 
          std::unordered_map<PauliOperator,  std::complex<double>, PauliHash> pmap;

          std::vector<PauliOperator> oplist = pauli_op_pool[i];
          for (auto hamil_oplist: pauli_strings) {
            commutator(hamil_oplist, oplist, pmap);
          }
          
          // Filter out the Paulis with zero coeffients
          std::vector<PauliOperator> comm_ops;
          comm_ops.reserve(pmap.size());
          for (auto pair: pmap) {
            if (abs(pair.second.real()) > 1e-10 || abs(pair.second.imag()) > 1e-10) {
              PauliOperator op(pair.first);
              op.setCoeff(pair.second);
              comm_ops.push_back(op);
            }
          }
          
          // MEMORY OPTIMIZATION: Explicitly clear and shrink pmap to free memory immediately
          pmap.clear();
          std::unordered_map<PauliOperator, std::complex<double>, PauliHash>().swap(pmap);

          num_pauli_terms_total += comm_ops.size();
          // Create commuting groups using the (nonoverlapping) Sorted Insertion heuristic (see Crawford et. al 2021)
          std::list<std::vector<IdxType>> cliques;
          sorted_insertion(comm_ops, cliques, false);
          
          // COMMUTING CLIQUES DEBUGGING: Show clique grouping efficiency
          if (state->get_process_rank() == 0 && (i < 5 || i % 100 == 0)) {
            std::cout << "Operator " << i << ": " << comm_ops.size() << " Pauli terms -> " << cliques.size() << " cliques";
            if (comm_ops.size() > 0) {
              std::cout << " (efficiency: " << std::fixed << std::setprecision(1) 
                        << (100.0 * cliques.size() / comm_ops.size()) << "%)";
            }
            
            // Show details for first few operators
            if (i < 3) {
              auto clique_iter = cliques.begin();
              for (size_t c = 0; c < std::min((size_t)3, cliques.size()); c++) {
                std::cout << "\n    Clique " << c << " has " << clique_iter->size() << " terms";
                clique_iter++;
              }
            }
            std::cout << std::endl;
          }

          // For each clique, we want to make an ObservableList object to compute expectation values after diagonalization
          auto cliqueiter = cliques.begin();
          commutator_coeffs[i].resize(cliques.size()); // Pauli coefficient storage
          commutator_zmasks[i].resize(cliques.size()); // ZMask storage (in diagonal QWC basis this is pauli.xmask | pauli.zmask)
          state->allocate_observables(gradient_observables[i], cliques.size());
          std::vector<IdxType> qubit_mapping (ansatz->num_qubits());
          std::iota(qubit_mapping.begin(), qubit_mapping.end(), 0);
          observable_sizes[i] = cliques.size();
          num_commuting_groups += cliques.size(); // MZ: Accumulate total number of commuting groups
          gradient_measurement[i] = std::make_shared<Ansatz>(ansatz->num_qubits());
          // For each clique, construct a measurement circuit and append
          for (size_t j = 0; j < cliques.size(); j++) {
            std::vector<IdxType>& clique = *cliqueiter;
            std::vector<PauliOperator> commuting_group (clique.size());
            std::transform(clique.begin(), clique.end(),
              commuting_group.begin(), [&] (IdxType ind) {return comm_ops[ind];}); 
            // Get a Pauli string that's the logical `or` over each stabilizer state, append to zmasks and coeff data structures within loop  
            PauliOperator common = make_common_op(commuting_group, 
                                                  commutator_zmasks[i][j], 
                                                  commutator_coeffs[i][j]);
            
            Measurement circ1 (common, false); // QWC measurement circuit $U_M$
            gradient_measurement[i]->compose(circ1, qubit_mapping);         // add to gradient measurement
            // add a gate to compute the expectation values   
            state->set_exp_gate(gradient_measurement[i], gradient_observables[i] + j, commutator_zmasks[i][j], commutator_coeffs[i][j]);
            Measurement circ2 (common, true); // inverse of the measurement circuit $U_M^\dagger$
            gradient_measurement[i]->compose(circ2, qubit_mapping);  // add the inverse
            cliqueiter++;
          }
          
          // PHASE 1 STATE VECTOR MEMORY FIX: Deallocate accumulated state vectors after each operator
          state->deallocate_simulation_state();
          
        }
        
        // MEMORY PROFILING: Final state (both GPU and CPU)
        if (state->get_process_rank() == 0) {
          // CPU memory profiling - final state
          std::ifstream status_file_final("/proc/self/status");
          std::string line_final;
          size_t vm_rss_final_kb = 0, vm_peak_final_kb = 0, vm_size_final_kb = 0;
          
          while (std::getline(status_file_final, line_final)) {
            if (line_final.find("VmRSS:") == 0) {
              sscanf(line_final.c_str(), "VmRSS: %zu kB", &vm_rss_final_kb);
            } else if (line_final.find("VmPeak:") == 0) {
              sscanf(line_final.c_str(), "VmPeak: %zu kB", &vm_peak_final_kb);
            } else if (line_final.find("VmSize:") == 0) {
              sscanf(line_final.c_str(), "VmSize: %zu kB", &vm_size_final_kb);
            }
          }
          
          #ifdef CUDA_ENABLED
          size_t gpu_free_final, gpu_total_final;
          cudaMemGetInfo(&gpu_free_final, &gpu_total_final);
          size_t memory_used_by_operators = gpu_free_initial - gpu_free_final;
          #endif
          
          std::cout << "\n=== ADAPT-VQE Memory Usage Summary ===" << std::endl;
          std::cout << "Generated " << pauli_op_pool.size() << " commutators with " << num_pauli_terms_total << " Individual Pauli Strings" << std::endl;
          
          std::cout << "\nFinal CPU memory state:" << std::endl;
          std::cout << "  RSS (Resident Set Size): " << (vm_rss_final_kb / 1024.0) << " MB" << std::endl;
          std::cout << "  Peak memory usage: " << (vm_peak_final_kb / 1024.0) << " MB" << std::endl;
          std::cout << "  Virtual memory size: " << (vm_size_final_kb / 1024.0) << " MB" << std::endl;
          std::cout << "  CPU memory increase from operators: " << ((vm_rss_final_kb - vm_rss_kb) / 1024.0) << " MB" << std::endl;
          
          #ifdef CUDA_ENABLED
          std::cout << "\nFinal GPU memory state:" << std::endl;
          std::cout << "  Free GPU memory: " << (gpu_free_final / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          std::cout << "  GPU memory used by ADAPT operators: " << (memory_used_by_operators / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          #endif
          
          // Calculate detailed memory breakdown and validate clique computation
          size_t total_cliques_validation = 0;
          for (size_t i = 0; i < poolsize; i++) {
            total_cliques_validation += observable_sizes[i];
          }
          
          // Validation check: num_commuting_groups should equal total_cliques_validation
          if (num_commuting_groups != total_cliques_validation) {
            std::cout << "WARNING: Commuting groups count mismatch! num_commuting_groups=" << num_commuting_groups 
                      << " vs validation=" << total_cliques_validation << std::endl;
          }
          
          size_t total_cliques = total_cliques_validation;
          
          std::cout << "\nDetailed memory breakdown:" << std::endl;
          std::cout << "  Total operators in pool: " << poolsize << std::endl;
          std::cout << "  Total commuting groups: " << total_cliques << std::endl;
          std::cout << "  Total Pauli terms: " << num_pauli_terms_total << std::endl;
          std::cout << "  Average Pauli terms per operator: " << (num_pauli_terms_total / poolsize) << std::endl;
          std::cout << "  Average commuting groups per operator: " << (total_cliques / poolsize) << std::endl;
          
          // Estimate memory usage by component
          size_t coeff_memory = num_pauli_terms_total * sizeof(double);
          size_t zmask_memory = num_pauli_terms_total * sizeof(IdxType);
          size_t observable_structs = total_cliques * sizeof(ObservableList);
          
          std::cout << "\nEstimated memory by component:" << std::endl;
          std::cout << "  Coefficients: " << (coeff_memory / (1024.0*1024.0)) << " MB" << std::endl;
          std::cout << "  Z-masks: " << (zmask_memory / (1024.0*1024.0)) << " MB" << std::endl;
          std::cout << "  ObservableList structs: " << (observable_structs / (1024.0*1024.0)) << " MB" << std::endl;
          std::cout << "  Total estimated: " << ((coeff_memory + zmask_memory + observable_structs) / (1024.0*1024.0)) << " MB" << std::endl;
          std::cout << "==========================================" << std::endl;
          
          #ifndef CUDA_ENABLED
          std::cout << "Generated " << pauli_op_pool.size() << " commutators with " << num_pauli_terms_total << " (possibly degenerate) Individual Pauli Strings" << std::endl;
          #endif
        }
      }


      /**
       * @brief  Main ADAPT_VQE optimization loop
       * @note   Can terminate either due to gradient convergence, maxeval limit, or due to function value convergence
       * @param  parameters: Reference to ansatz parameter vector (output)
       * @param  ene: Reference to estimated energy (output)
       * @param  maxiter: Max iteration limit
       * @param  abstol: Tolerance for gradient norm
       * @param  fvaltol: Tolerance for function value convergence
       * @retval None
       */
      void optimize(std::vector<double>& parameters, ValType& ene, IdxType maxiter, ValType abstol = 1e-5, ValType fvaltol = 1e-7) {
        ene = hamil->getEnv().constant;
        state->initialize();
        state->set_adaptresult(9); // MZ: initialize convergence flag to "Other"
        ValType constant = ene;
        IdxType iter = 0;
        ValType prev_ene = 1 + ene;
        const auto& pauli_op_pool = ansatz->get_pauli_op_pool();
        while(iter < maxiter) {
          prev_ene = ene;
          IdxType max_ind = 0; 
          double max_ene = -MAXFLOAT;
          
          // PHASE 1 STATE VECTOR MEMORY FIX: Reallocate state vectors for new iteration
          state->reallocate_simulation_state();
          
          // Gradient estimation
          bool first = true;
          for (auto grad_circuit: gradient_measurement) {
            state->call_simulator(grad_circuit, first); // compute the commutator expvals
            first = false;
          }
          std::fill(gradient_magnitudes.begin(), gradient_magnitudes.end(), 0);
          state->get_exp_values(gradient_observables, observable_sizes, gradient_magnitudes); // Get the commutator expvals from ObservableList structures (possibly in device memory)
          // compute the norm
          double grad_norm = std::sqrt(std::accumulate(gradient_magnitudes.begin(), gradient_magnitudes.end(), 0.0, [] (ValType a, ValType b) {
            return a + b * b;
          }));
          // if the gradient converged, break
          if (grad_norm < abstol) {
            state->set_adaptresult(0); // MZ: converged flag
            break;
          }
          // else find the index of the gradient element with the largest magnitude
          max_ind = std::max_element(gradient_magnitudes.begin(),
                                     gradient_magnitudes.end(),
                                     [] (ValType a, ValType b) {return abs(a) < abs(b);}) - gradient_magnitudes.begin();
          // Add a new parameter for the new operator
          ValType paramval = 0.0;
          // Slap the operator on the back of the ansatz
          ansatz->add_operator(max_ind, paramval); 
          parameters.push_back(paramval);
          // VQE Optimzation step
          state->optimize(parameters, ene);
          IdxType num_func_evals = state->get_iteration();
          // Print update
          if (state->get_process_rank() == 0) {
            if (iter == 0) {
              std::cout << "\n----------- Iteration Summary -----------\n" << std::left
                        << std::setw(8) << " Iter"
                        << std::setw(27) << "Objective Value"
                        << std::setw(9) << "# Evals"
                        << std::setw(11) << "Grad Norm"
                        << std::setw(13) << "|  Depth"
                        << std::setw(11) << "#1q Gates"
                        << std::setw(11) << "#2q Gates"
                        << std::setw(12) << "GPU Mem(GB)"
                        << std::setw(46) << "|  Selected Operator"
                        << std::endl;
              std::cout << std::string(132, '-') << std::endl;
            }
            NWQSim::CircuitMetrics metrics = ansatz -> circuit_metrics();
            
            // MEMORY MONITORING: Get current GPU memory usage
            #ifdef CUDA_ENABLED
            size_t gpu_free, gpu_total;
            cudaMemGetInfo(&gpu_free, &gpu_total);
            double gpu_used_gb = (gpu_total - gpu_free) / 1e9;
            
            // DETAILED MEMORY PROFILING: Show memory breakdown during optimization
            if (iter % 5 == 0) {  // Every 5 iterations
              // CPU memory during optimization
              std::ifstream status_opt("/proc/self/status");
              std::string line_opt;
              size_t vm_rss_opt_kb = 0;
              
              while (std::getline(status_opt, line_opt)) {
                if (line_opt.find("VmRSS:") == 0) {
                  sscanf(line_opt.c_str(), "VmRSS: %zu kB", &vm_rss_opt_kb);
                  break;
                }
              }
              
              std::cout << "\n--- Memory Breakdown (Iteration " << iter << ") ---" << std::endl;
              std::cout << "  CPU RSS: " << (vm_rss_opt_kb / 1024.0) << " MB" << std::endl;
              std::cout << "  GPU Total: " << (gpu_total / 1e9) << " GB" << std::endl;
              std::cout << "  GPU Used: " << gpu_used_gb << " GB" << std::endl;
              std::cout << "  GPU Free: " << (gpu_free / 1e9) << " GB" << std::endl;
              std::cout << "  Ansatz depth: " << metrics.depth << std::endl;
              std::cout << "  Parameters: " << parameters.size() << std::endl;
              std::cout << "-----------------------------------------------" << std::endl;
            }
            #else
            double gpu_used_gb = 0.0;
            #endif
            
            std::cout << std::left << " "
                      << std::setw(7) << iter
                      << std::setw(27) << std::fixed << std::setprecision(14) << ene
                      << std::setw(9) << std::fixed << num_func_evals
                      << std::setw(11) << std::scientific << std::setprecision(3) << grad_norm;
            // Print circuit metrics in scientific notation if they are greater than 1e6
            const double print_threshold = 1e6;
            if (metrics.depth > print_threshold) {
              std::cout << "|  " << std::setw(10) << std::fixed << metrics.depth
                                << std::setw(11) << std::fixed << metrics.one_q_gates
                                << std::setw(11) << std::fixed << metrics.two_q_gates;
            } else {
              std::cout<< "|  " << std::setw(10) << std::scientific << std::setprecision(3) << metrics.depth
                              << std::setw(11) << std::scientific << std::setprecision(3) << metrics.one_q_gates
                              << std::setw(11) << std::scientific << std::setprecision(3) << metrics.two_q_gates;
            }
            // Print GPU memory usage and selected operator
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << gpu_used_gb
                      << "|  " << ansatz->get_operator_string(max_ind) << std::endl;
          }

          // If the function value converged, then break
          if (abs((ene - prev_ene)) < fvaltol) {
            state->set_adaptresult(1); // MZ: converged flag
            break;
          }
          iter++;
        }
        state->set_adaptrounds(iter); // MZ: record numebr of ADAPT rounds
        if (iter >= maxiter) {
          state->set_adaptresult(2); // MZ: converged flag
        }
      }

    };
  };
};


#endif