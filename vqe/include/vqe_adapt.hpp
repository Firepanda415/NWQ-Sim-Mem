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
        size_t num_pauli_terms_total; // MZ: Total number of Pauli terms in the commutator
        size_t num_comm_cliques; // MZ: Total number of commuting cliques
        size_t num_commuting_groups; // MZ: Total number of commuting groups (measurement cliques)

    public: 
        size_t get_numpauli() const { return num_pauli_terms_total;}; // MZ
        void set_numpauli(size_t value) { num_pauli_terms_total = value; }; //MZ
        
        size_t get_numcommutinggroups() const { return num_commuting_groups;}; // MZ: Get number of commuting groups
        void set_numcommutinggroups(size_t value) { num_commuting_groups = value; }; //MZ
        
        size_t  get_numcomm() const { return num_comm_cliques;}; // MZ
        void set_numcomm(size_t value) { num_comm_cliques = value; }; //MZ

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
        
        // Memory profiling variables (in bytes for precision)
        size_t cumulative_observable_memory_bytes = 0;
        size_t cumulative_coefficient_memory_bytes = 0;
        size_t cumulative_zmask_memory_bytes = 0;
        size_t cumulative_clique_memory_bytes = 0;
        size_t cumulative_pauli_map_memory_bytes = 0;
        size_t cumulative_comm_ops_memory_bytes = 0;
        size_t total_observables_created = 0;
        size_t total_pauli_maps_created = 0;
        size_t max_single_operator_memory_bytes = 0;
        
        // Initial memory state tracking
        size_t initial_cpu_rss = 0;
        #ifdef CUDA_ENABLED
        size_t initial_gpu_free = 0, gpu_total = 0;
        #endif
        
        if (state->get_process_rank() == 0) {
          // Get initial CPU memory
          std::ifstream status_file("/proc/self/status");
          std::string line;
          while (std::getline(status_file, line)) {
            if (line.find("VmRSS:") == 0) {
              sscanf(line.c_str(), "VmRSS: %zu kB", &initial_cpu_rss);
              break;
            }
          }
          
          #ifdef CUDA_ENABLED
          cudaMemGetInfo(&initial_gpu_free, &gpu_total);
          #endif
          
          std::cout << "\n=== ADAPT-VQE Memory Profiling (CPU + GPU) ===" << std::endl;
          std::cout << "Initial CPU memory:" << std::endl;
          std::cout << "  RSS (Resident Set Size): " << std::fixed << std::setprecision(2) << (initial_cpu_rss / 1024.0) << " MB" << std::endl;
          
          #ifdef CUDA_ENABLED
          std::cout << "Initial GPU memory:" << std::endl;
          std::cout << "  Total GPU memory: " << std::fixed << std::setprecision(4) << (gpu_total / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          std::cout << "  Free GPU memory: " << std::fixed << std::setprecision(4) << ((initial_gpu_free) / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          std::cout << "  Used GPU memory: " << std::fixed << std::setprecision(4) << ((gpu_total - initial_gpu_free) / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          #endif
          
          std::cout << "\nStarting UCCSD operator pool allocation..." << std::endl;
          std::cout << "  Pool size: " << poolsize << " operators" << std::endl;
          std::cout << "  Hamiltonian terms: " << pauli_strings.size() << std::endl;
        }
        
        // Summary tracking variables
        size_t summary_interval = 10;
        size_t last_summary_cpu_rss = initial_cpu_rss;
        size_t batch_pauli_terms = 0;
        size_t batch_cliques = 0;
        double batch_efficiency_sum = 0.0;
        
        // allocate memory for commutator structures
        commutator_coeffs.resize(poolsize);
        gradient_measurement.resize(poolsize);
        commutator_zmasks.resize(poolsize);
        gradient_magnitudes.resize(poolsize);
        gradient_observables.resize(poolsize);
        observable_sizes.resize(poolsize);
        num_pauli_terms_total = 0;
        num_commuting_groups = 0;
        for (size_t i = 0; i < poolsize; i++) {
          // Memory tracking before operator processing
          size_t cpu_rss_before = 0;
          #ifdef CUDA_ENABLED
          size_t gpu_free_before = 0, gpu_total_before = 0;
          #endif
          
          if (state->get_process_rank() == 0) {
            // Get CPU memory before this operator
            std::ifstream status_file("/proc/self/status");
            std::string line;
            while (std::getline(status_file, line)) {
              if (line.find("VmRSS:") == 0) {
                sscanf(line.c_str(), "VmRSS: %zu kB", &cpu_rss_before);
                break;
              }
            }
            
            #ifdef CUDA_ENABLED
            cudaMemGetInfo(&gpu_free_before, &gpu_total_before);
            #endif
            
            std::cout << "Processing operator " << i << "/" << poolsize << " - CPU: " 
                      << std::fixed << std::setprecision(2) << (cpu_rss_before / 1024.0) << " MB";
            #ifdef CUDA_ENABLED
            std::cout << ", GPU: " << std::fixed << std::setprecision(4) << (gpu_free_before / (1024.0*1024.0*1024.0)) << " GB free";
            #endif
            std::cout << std::endl;
          }
          
          // Get all of the ungrouped Pauli strings for this commutator 
          std::unordered_map<PauliOperator,  std::complex<double>, PauliHash> pmap;
          std::vector<PauliOperator> oplist = pauli_op_pool[i];
          
          // Track memory before commutator computation
          size_t operator_start_memory_bytes = 0;
          if (state->get_process_rank() == 0) {
            operator_start_memory_bytes = oplist.size() * sizeof(PauliOperator);
          }
          
          for (auto hamil_oplist: pauli_strings) {
            commutator(hamil_oplist, oplist, pmap);
          }
          
          // Track Pauli map memory
          size_t pauli_map_memory_bytes = 0;
          if (state->get_process_rank() == 0) {
            pauli_map_memory_bytes = pmap.size() * (sizeof(PauliOperator) + sizeof(std::complex<double>) + 64); // 64 bytes for hash table overhead
            cumulative_pauli_map_memory_bytes += pauli_map_memory_bytes;
            total_pauli_maps_created++;
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

          // Track comm_ops memory
          size_t comm_ops_memory_bytes = 0;
          if (state->get_process_rank() == 0) {
            comm_ops_memory_bytes = comm_ops.size() * sizeof(PauliOperator);
            cumulative_comm_ops_memory_bytes += comm_ops_memory_bytes;
          }

          num_pauli_terms_total += comm_ops.size();
          
          // Create commuting groups using the (nonoverlapping) Sorted Insertion heuristic (see Crawford et. al 2021)
          std::list<std::vector<IdxType>> cliques;
          sorted_insertion(comm_ops, cliques, false);
          
          // Calculate detailed memory usage for this operator's data structures
          size_t operator_coefficient_memory_bytes = 0;
          size_t operator_zmask_memory_bytes = 0;
          size_t operator_clique_memory_bytes = 0;
          size_t operator_total_memory_bytes = 0;
          
          // Detailed STL container size tracking
          size_t pmap_actual_bytes = 0;
          size_t comm_ops_actual_bytes = 0;
          size_t cliques_actual_bytes = 0;
          
          if (state->get_process_rank() == 0) {
            // Measure actual STL container sizes
            pmap_actual_bytes = pmap.size() * (sizeof(PauliOperator) + sizeof(std::complex<double>)) + 
                               pmap.bucket_count() * sizeof(void*) + 
                               pmap.size() * 32; // hash table overhead
            
            comm_ops_actual_bytes = comm_ops.capacity() * sizeof(PauliOperator);
            
            // Calculate cliques list memory (complex nested structure)
            cliques_actual_bytes = 0;
            for (const auto& clique : cliques) {
              cliques_actual_bytes += sizeof(std::vector<IdxType>) + clique.capacity() * sizeof(IdxType);
            }
            cliques_actual_bytes += cliques.size() * sizeof(std::list<std::vector<IdxType>>::value_type);
            
            // More accurate memory calculations
            operator_coefficient_memory_bytes = comm_ops.size() * sizeof(std::complex<double>); // coefficients are complex
            operator_zmask_memory_bytes = comm_ops.size() * sizeof(IdxType) * 2; // both xmask and zmask
            operator_clique_memory_bytes = cliques.size() * sizeof(ObservableList) + 
                                         cliques.size() * 128; // ObservableList + estimated overhead
            operator_total_memory_bytes = operator_coefficient_memory_bytes + operator_zmask_memory_bytes + 
                                        operator_clique_memory_bytes + comm_ops_memory_bytes + pauli_map_memory_bytes;
            
            cumulative_coefficient_memory_bytes += operator_coefficient_memory_bytes;
            cumulative_zmask_memory_bytes += operator_zmask_memory_bytes;
            cumulative_clique_memory_bytes += operator_clique_memory_bytes;
            
            if (operator_total_memory_bytes > max_single_operator_memory_bytes) {
              max_single_operator_memory_bytes = operator_total_memory_bytes;
            }
            
            double efficiency = (comm_ops.size() > 0) ? (100.0 * cliques.size() / comm_ops.size()) : 0.0;
            std::cout << "Operator " << i << ": " << comm_ops.size() << " Pauli terms -> " 
                      << cliques.size() << " cliques (efficiency: " << std::fixed << std::setprecision(1) 
                      << efficiency << "%) - Memory: " << std::fixed << std::setprecision(2) 
                      << (operator_total_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
            
            // Detailed STL container breakdown (before cleanup)
            std::cout << "  STL Containers (temp): PauliMap=" << std::fixed << std::setprecision(2) << (pmap_actual_bytes / (1024.0 * 1024.0)) 
                      << "MB, CommOps=" << (comm_ops_actual_bytes / (1024.0 * 1024.0))
                      << "MB, Cliques=" << (cliques_actual_bytes / (1024.0 * 1024.0)) << "MB" << std::endl;
            
            // Show container efficiency with detailed sizes
            double pmap_overhead = (pmap_actual_bytes > pauli_map_memory_bytes) ? 
                                  ((double)pmap_actual_bytes / pauli_map_memory_bytes) : 1.0;
            double comm_ops_overhead = (comm_ops_actual_bytes > comm_ops_memory_bytes) ? 
                                      ((double)comm_ops_actual_bytes / comm_ops_memory_bytes) : 1.0;
            std::cout << "  Container Overhead: PauliMap=" << std::fixed << std::setprecision(1) << pmap_overhead 
                      << "x, CommOps=" << comm_ops_overhead << "x" << std::endl;
            
            // Detailed container diagnostics
            std::cout << "  Container Details: CommOps size=" << comm_ops.size() 
                      << ", capacity=" << comm_ops.capacity() 
                      << ", PauliMap size=" << pmap.size() 
                      << ", buckets=" << pmap.bucket_count() << std::endl;
            
            // Show accumulated data structure sizes
            size_t accumulated_coeffs = i * comm_ops.size() * sizeof(std::complex<double>);
            size_t accumulated_zmasks = i * comm_ops.size() * sizeof(IdxType) * 2;
            size_t accumulated_observables = total_observables_created * sizeof(ObservableList);
            std::cout << "  Accumulated Data: Coeffs=" << std::fixed << std::setprecision(2) 
                      << (accumulated_coeffs / (1024.0 * 1024.0)) << "MB, ZMasks=" 
                      << (accumulated_zmasks / (1024.0 * 1024.0)) << "MB, Observables=" 
                      << (accumulated_observables / (1024.0 * 1024.0)) << "MB" << std::endl;
            
            // Update batch tracking
            batch_pauli_terms += comm_ops.size();
            batch_cliques += cliques.size();
            batch_efficiency_sum += efficiency;
          }

          
          // For each clique, we want to make an ObservableList object to compute expectation values after diagonalization
          auto cliqueiter = cliques.begin();
          commutator_coeffs[i].resize(cliques.size()); // Pauli coefficient storage
          commutator_zmasks[i].resize(cliques.size()); // ZMask storage (in diagonal QWC basis this is pauli.xmask | pauli.zmask)
          state->allocate_observables(gradient_observables[i], cliques.size());
          std::vector<IdxType> qubit_mapping (ansatz->num_qubits());
          std::iota(qubit_mapping.begin(), qubit_mapping.end(), 0);
          observable_sizes[i] = cliques.size();
          num_commuting_groups += cliques.size();
          gradient_measurement[i] = std::make_shared<Ansatz>(ansatz->num_qubits());
          
          // Track observables created for this operator
          total_observables_created += cliques.size();
          cumulative_observable_memory_bytes += (cliques.size() * sizeof(ObservableList));
          
          // Memory tracking before circuit construction (after STL container setup) - all processes
          size_t cpu_rss_before_circuits = 0;
          std::ifstream status_file_circuits("/proc/self/status");
          std::string line_circuits;
          while (std::getline(status_file_circuits, line_circuits)) {
            if (line_circuits.find("VmRSS:") == 0) {
              sscanf(line_circuits.c_str(), "VmRSS: %zu kB", &cpu_rss_before_circuits);
              break;
            }
          }
          
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
          
          // Shrink vectors to actual size to reclaim over-allocated memory
          if (gradient_measurement[i] != nullptr) {
            // Shrink main gates vector
            if (gradient_measurement[i]->gates != nullptr) {
              gradient_measurement[i]->gates->shrink_to_fit();
            }
            
            // Shrink nested gate_parameter_pointers structure
            auto gate_pointers = gradient_measurement[i]->getParamGatePointers();
            if (gate_pointers != nullptr) {
              gate_pointers->shrink_to_fit(); // Shrink outer vector
              for (auto& inner_vec : *gate_pointers) {
                inner_vec.shrink_to_fit(); // Shrink each inner vector
              }
            }
            
            // Shrink other Ansatz vectors
            if (gradient_measurement[i]->getParams() != nullptr) {
              gradient_measurement[i]->getParams()->shrink_to_fit();
            }
            if (gradient_measurement[i]->getParamGateIndices() != nullptr) {
              gradient_measurement[i]->getParamGateIndices()->shrink_to_fit();
            }
          }
          
          // Memory tracking immediately after circuit construction (before cleanup)
          size_t cpu_rss_after_circuits_before_cleanup = 0;
          if (state->get_process_rank() == 0) {
            std::ifstream status_file_circuits_after("/proc/self/status");
            std::string line_circuits_after;
            while (std::getline(status_file_circuits_after, line_circuits_after)) {
              if (line_circuits_after.find("VmRSS:") == 0) {
                sscanf(line_circuits_after.c_str(), "VmRSS: %zu kB", &cpu_rss_after_circuits_before_cleanup);
                break;
              }
            }
          }
          
          // Store cliques size before cleanup for static variable
          size_t cliques_count = cliques.size();
          
          // Memory cleanup to prevent accumulation during operator processing
          pmap.clear();
          std::unordered_map<PauliOperator, std::complex<double>, PauliHash>().swap(pmap);
          comm_ops.clear();
          comm_ops.shrink_to_fit();
          cliques.clear();
          
          // Memory tracking after cleanup to isolate persistent circuit memory (all processes)
          size_t cpu_rss_after_cleanup = 0;
          double persistent_circuit_memory = 0.0;
          double circuit_memory_mb = 0.0;
          double total_memory_with_temps = 0.0;
          double temporary_memory_cleaned = 0.0;
          
          // Get memory state after cleanup for all processes
          std::ifstream status_file_cleanup("/proc/self/status");
          std::string line_cleanup;
          while (std::getline(status_file_cleanup, line_cleanup)) {
            if (line_cleanup.find("VmRSS:") == 0) {
              sscanf(line_cleanup.c_str(), "VmRSS: %zu kB", &cpu_rss_after_cleanup);
              break;
            }
          }
          
          if (state->get_process_rank() == 0) {
            // Calculate memory components (only for output)
            total_memory_with_temps = (cpu_rss_after_circuits_before_cleanup - cpu_rss_before_circuits) / 1024.0;
            persistent_circuit_memory = (cpu_rss_after_cleanup - cpu_rss_before_circuits) / 1024.0;
            temporary_memory_cleaned = total_memory_with_temps - persistent_circuit_memory;
            
            std::cout << "  Circuit + Temp Memory: +" << std::fixed << std::setprecision(2) << total_memory_with_temps 
                      << " MB, Persistent Circuit: +" << std::fixed << std::setprecision(2) << persistent_circuit_memory 
                      << " MB, Cleaned Temp: " << std::fixed << std::setprecision(2) << temporary_memory_cleaned << " MB";
            
            // Verify cleanup efficiency
            double total_temp_stl = (pmap_actual_bytes + comm_ops_actual_bytes + cliques_actual_bytes) / (1024.0 * 1024.0);
            double cleanup_efficiency = (total_temp_stl > 0) ? (temporary_memory_cleaned / total_temp_stl) * 100.0 : 100.0;
            std::cout << " (cleanup: " << std::fixed << std::setprecision(1) << cleanup_efficiency << "%)" << std::endl;
            
          }
          
          // Calculate circuit memory for all processes with bounds checking
          if (cpu_rss_after_cleanup > 0 && cpu_rss_before_circuits > 0 && cpu_rss_after_cleanup >= cpu_rss_before_circuits) {
            circuit_memory_mb = (cpu_rss_after_cleanup - cpu_rss_before_circuits) / 1024.0;
          } else {
            circuit_memory_mb = 0.0; // Fallback for edge cases
          }
          
          if (state->get_process_rank() == 0) {
            // Show cumulative circuit memory growth
            static double total_circuit_memory = 0.0;
            static double prev_circuit_memory = 0.0;
            static size_t prev_cliques = 0;
            static size_t total_circuits_created = 0;
            total_circuit_memory += circuit_memory_mb;
            total_circuits_created += cliques_count; // Use stored count before cleanup
            std::cout << "  Cumulative Circuit Memory: " << std::fixed << std::setprecision(2) 
                      << total_circuit_memory << " MB (Total circuits: " << total_circuits_created << ")" << std::endl;
            
            // Comprehensive Ansatz memory analysis - all members and overhead
            if (gradient_measurement[i] != nullptr) {
              // Error checking for null pointers
              auto theta_ptr = gradient_measurement[i]->getParams();
              auto param_gates_ptr = gradient_measurement[i]->getParamGateIndices();
              auto gates_ptr = gradient_measurement[i]->gates;
              
              if (theta_ptr == nullptr || param_gates_ptr == nullptr || gates_ptr == nullptr) {
                std::cout << "  Ansatz Analysis: Null pointer detected, skipping analysis" << std::endl;
              } else {
                // 1. shared_ptr overhead (control blocks for reference counting)
                size_t shared_ptr_overhead = 4 * (sizeof(std::shared_ptr<void>) + 32); // 4 shared_ptrs + control blocks
                
                // 2. theta vector (parameters) with shared_ptr overhead
                size_t theta_data_bytes = theta_ptr->capacity() * sizeof(ValType);
                size_t theta_total_bytes = theta_data_bytes + sizeof(std::vector<ValType>);
                
                // 3. parameterized_gates vector with shared_ptr overhead  
                size_t param_gates_data_bytes = param_gates_ptr->capacity() * sizeof(IdxType);
                size_t param_gates_total_bytes = param_gates_data_bytes + sizeof(std::vector<IdxType>);
                
                // 4. gate_parameter_pointers - complex nested structure with shared_ptr
                size_t gate_param_pointers_bytes = 0;
                auto gate_pointers = gradient_measurement[i]->getParamGatePointers();
                if (gate_pointers != nullptr) {
                  // Outer vector overhead
                  gate_param_pointers_bytes += gate_pointers->capacity() * sizeof(std::vector<std::pair<IdxType, ValType>>);
                  gate_param_pointers_bytes += sizeof(std::vector<std::vector<std::pair<IdxType, ValType>>>);
                  
                  // Inner vectors overhead and data
                  for (const auto& inner_vec : *gate_pointers) {
                    gate_param_pointers_bytes += inner_vec.capacity() * sizeof(std::pair<IdxType, ValType>);
                    gate_param_pointers_bytes += sizeof(std::vector<std::pair<IdxType, ValType>>); // vector object overhead
                  }
                }
                
                // 5. gate_coefficients vector (missing from previous analysis!)
                size_t gate_coeffs_bytes = 0; // Cannot access directly, estimate based on param gates
                if (param_gates_ptr != nullptr) {
                  gate_coeffs_bytes = param_gates_ptr->size() * sizeof(ValType) + sizeof(std::vector<ValType>);
                }
                
                // 6. excitation_index_map unordered_map<string, IdxType> (potential memory hog!)
                size_t excitation_map_bytes = 0;
                // Estimate: assume average string length of 20 chars, map overhead of 32 bytes per entry
                // This could be HUGE with many operators
                size_t estimated_map_entries = std::min((size_t)100, param_gates_ptr->size()); // conservative estimate
                excitation_map_bytes = estimated_map_entries * (20 + sizeof(IdxType) + 32) + 64; // map overhead
                
                // 7. Circuit gates from inherited Circuit class (actual Gate objects) - use existing gates_ptr
                size_t circuit_gates_data_bytes = gates_ptr->capacity() * sizeof(Gate);
                size_t circuit_gates_total_bytes = circuit_gates_data_bytes + sizeof(std::vector<Gate>);
                
                // 8. Ansatz object overhead (vtable, padding, etc.)
                size_t ansatz_object_overhead = sizeof(Ansatz) + sizeof(Circuit); // inheritance overhead
                
                // 9. ansatz_name string
                size_t ansatz_name_bytes = 32; // estimated string overhead
                
                // Total calculation
                size_t total_ansatz_bytes = shared_ptr_overhead + theta_total_bytes + param_gates_total_bytes + 
                                           gate_param_pointers_bytes + gate_coeffs_bytes + excitation_map_bytes +
                                           circuit_gates_total_bytes + ansatz_object_overhead + ansatz_name_bytes;
                
                std::cout << "  Complete Ansatz Memory Breakdown:" << std::endl;
                std::cout << "    shared_ptr overhead: " << std::fixed << std::setprecision(2) 
                          << (shared_ptr_overhead / 1024.0) << " KB" << std::endl;
                std::cout << "    theta (params): " << std::fixed << std::setprecision(2) 
                          << (theta_total_bytes / 1024.0) << " KB (" << theta_ptr->size() << "/" << theta_ptr->capacity() << ")" << std::endl;
                std::cout << "    parameterized_gates: " << std::fixed << std::setprecision(2) 
                          << (param_gates_total_bytes / 1024.0) << " KB (" << param_gates_ptr->size() << "/" << param_gates_ptr->capacity() << ")" << std::endl;
                std::cout << "    gate_parameter_pointers: " << std::fixed << std::setprecision(2) 
                          << (gate_param_pointers_bytes / 1024.0) << " KB (nested structure)" << std::endl;
                std::cout << "    gate_coefficients: " << std::fixed << std::setprecision(2) 
                          << (gate_coeffs_bytes / 1024.0) << " KB (estimated)" << std::endl;
                std::cout << "    excitation_index_map: " << std::fixed << std::setprecision(2) 
                          << (excitation_map_bytes / 1024.0) << " KB (string map)" << std::endl;
                std::cout << "    circuit_gates: " << std::fixed << std::setprecision(2) 
                          << (circuit_gates_total_bytes / 1024.0) << " KB (" << gates_ptr->size() << "/" << gates_ptr->capacity() << ")" << std::endl;
                std::cout << "    object_overhead: " << std::fixed << std::setprecision(2) 
                          << ((ansatz_object_overhead + ansatz_name_bytes) / 1024.0) << " KB" << std::endl;
                std::cout << "    Total Ansatz Object: " << std::fixed << std::setprecision(3) 
                          << (total_ansatz_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
                
                // Memory efficiency analysis with division by zero protection
                size_t useful_data = theta_ptr->size() * sizeof(ValType) + 
                                    param_gates_ptr->size() * sizeof(IdxType) + 
                                    gates_ptr->size() * sizeof(Gate);
                double efficiency = (useful_data > 0 && total_ansatz_bytes > 0) ? (100.0 * useful_data / total_ansatz_bytes) : 0.0;
                std::cout << "    Memory Efficiency: " << std::fixed << std::setprecision(1) << efficiency 
                          << "% (useful data vs total)" << std::endl;
                
                // Identify the largest memory consumer with STL safety
                std::vector<std::pair<std::string, size_t>> components = {
                  {"circuit_gates", circuit_gates_total_bytes},
                  {"gate_parameter_pointers", gate_param_pointers_bytes},
                  {"theta", theta_total_bytes},
                  {"excitation_map", excitation_map_bytes},
                  {"shared_ptr_overhead", shared_ptr_overhead}
                };
                
                if (!components.empty() && total_ansatz_bytes > 0) {
                  auto max_component = std::max_element(components.begin(), components.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
                  
                  std::cout << "    Memory Hotspot: " << max_component->first << " (" 
                            << std::fixed << std::setprecision(1) << (100.0 * max_component->second / total_ansatz_bytes) 
                            << "% of object)" << std::endl;
                } else {
                  std::cout << "    Memory Hotspot: Unable to determine (no valid components)" << std::endl;
                }
              } // end of null pointer check
            }
            
            // Track circuit memory per clique for analysis with division protection
            if (i > 0 && prev_cliques > 0 && cliques_count > 0) {
              double memory_per_clique_current = circuit_memory_mb / cliques_count;
              double memory_per_clique_prev = prev_circuit_memory / prev_cliques;
              
              std::cout << "  Memory/Clique: " << std::fixed << std::setprecision(3) 
                        << memory_per_clique_current << " MB (prev: " 
                        << memory_per_clique_prev << " MB)" << std::endl;
            } else if (cliques_count > 0) {
              double memory_per_clique_current = circuit_memory_mb / cliques_count;
              std::cout << "  Memory/Clique: " << std::fixed << std::setprecision(3) 
                        << memory_per_clique_current << " MB" << std::endl;
            } else {
              std::cout << "  Memory/Clique: No cliques available for analysis" << std::endl;
            }
            
            // Track for next iteration
            prev_circuit_memory = circuit_memory_mb;
            prev_cliques = cliques_count;
          }
          
          // Memory tracking after operator processing (using cleanup memory)
          size_t cpu_rss_after = cpu_rss_after_cleanup;
          #ifdef CUDA_ENABLED
          size_t gpu_free_after = 0, gpu_total_after = 0;
          cudaMemGetInfo(&gpu_free_after, &gpu_total_after);
          #endif
          
          if (state->get_process_rank() == 0) {
            // Print intermediate memory checkpoint using persistent memory
            double cpu_growth_mb = (cpu_rss_after - cpu_rss_before) / 1024.0;
            std::cout << "  → Net Memory Delta: CPU +" << std::fixed << std::setprecision(2) << cpu_growth_mb << " MB";
            
            #ifdef CUDA_ENABLED
            if (gpu_free_before > gpu_free_after) {
              double gpu_consumed_mb = (gpu_free_before - gpu_free_after) / (1024.0 * 1024.0);
              std::cout << ", GPU +" << std::fixed << std::setprecision(2) << gpu_consumed_mb << " MB";
            }
            #endif
            
            // Detailed memory accounting analysis
            double total_data_structures_mb = (operator_total_memory_bytes / (1024.0 * 1024.0));
            double unaccounted_memory = cpu_growth_mb - circuit_memory_mb - total_data_structures_mb;
            std::cout << " [Circuit: " << std::fixed << std::setprecision(1) << circuit_memory_mb 
                      << "MB + Data: " << std::fixed << std::setprecision(1) << total_data_structures_mb 
                      << "MB + Overhead: " << std::fixed << std::setprecision(1) << unaccounted_memory << "MB]";
            std::cout << std::endl;
            
            // Periodic summary every 10 operators
            if ((i + 1) % summary_interval == 0 || i == poolsize - 1) {
              size_t current_cpu_rss = cpu_rss_after;
              double batch_memory_growth = (current_cpu_rss - last_summary_cpu_rss) / 1024.0;
              double avg_efficiency = (summary_interval > 0) ? (batch_efficiency_sum / summary_interval) : 0.0;
              
              std::cout << "\n--- SUMMARY: Operators " << (i + 1 - summary_interval) << "-" << i << " ---" << std::endl;
              std::cout << "  Batch Memory Growth: +" << std::fixed << std::setprecision(2) << batch_memory_growth << " MB" << std::endl;
              std::cout << "  Pauli Terms Generated: " << batch_pauli_terms << std::endl;
              std::cout << "  Commuting Groups: " << batch_cliques << std::endl;
              std::cout << "  Average Efficiency: " << std::fixed << std::setprecision(1) << avg_efficiency << "%" << std::endl;
              std::cout << "  Cumulative Memory: " << std::fixed << std::setprecision(2) << (current_cpu_rss / 1024.0) << " MB" << std::endl;
              
              // Memory per operator analysis
              double avg_memory_per_op = batch_memory_growth / summary_interval;
              std::cout << "  Average Memory/Operator: " << std::fixed << std::setprecision(2) << avg_memory_per_op << " MB" << std::endl;
              
              // Theoretical vs actual comparison
              double avg_pauli_per_op = (double)batch_pauli_terms / summary_interval;
              double theoretical_mb_per_op = (avg_pauli_per_op * 16 + avg_pauli_per_op * 8 + batch_cliques * 128 / summary_interval) / (1024.0 * 1024.0);
              double bloat_factor = avg_memory_per_op / theoretical_mb_per_op;
              std::cout << "  Theoretical Memory/Op: " << std::fixed << std::setprecision(2) << theoretical_mb_per_op 
                        << " MB (bloat factor: " << std::fixed << std::setprecision(1) << bloat_factor << "x)" << std::endl;
              
              // Memory analysis data
              std::cout << "  Memory Growth Rate: " << std::fixed << std::setprecision(1) << (batch_memory_growth / summary_interval) << " MB/op" << std::endl;
              std::cout << "  Memory Bloat Factor: " << std::fixed << std::setprecision(1) << bloat_factor << "x" << std::endl;
              
              // Memory fragmentation analysis
              size_t total_heap_size = 0;
              size_t total_heap_free = 0;
              std::ifstream meminfo("/proc/meminfo");
              std::string line_mem;
              while (std::getline(meminfo, line_mem)) {
                if (line_mem.find("MemTotal:") == 0) {
                  sscanf(line_mem.c_str(), "MemTotal: %zu kB", &total_heap_size);
                } else if (line_mem.find("MemAvailable:") == 0) {
                  sscanf(line_mem.c_str(), "MemAvailable: %zu kB", &total_heap_free);
                  break;
                }
              }
              
              double memory_pressure = 100.0 * (1.0 - (double)total_heap_free / total_heap_size);
              std::cout << "  System Memory Pressure: " << std::fixed << std::setprecision(1) << memory_pressure << "%" << std::endl;
              std::cout << "--- END SUMMARY ---\n" << std::endl;
              
              // Reset batch counters
              last_summary_cpu_rss = current_cpu_rss;
              batch_pauli_terms = 0;
              batch_cliques = 0;
              batch_efficiency_sum = 0.0;
            }
          }
        }
        
        // Final memory analysis and categorical breakdown
        if (state->get_process_rank() == 0) {
          // Get final memory state
          size_t final_cpu_rss = 0;
          std::ifstream status_file_final("/proc/self/status");
          std::string line_final;
          while (std::getline(status_file_final, line_final)) {
            if (line_final.find("VmRSS:") == 0) {
              sscanf(line_final.c_str(), "VmRSS: %zu kB", &final_cpu_rss);
              break;
            }
          }
          
          #ifdef CUDA_ENABLED
          size_t final_gpu_free = 0, final_gpu_total = 0;
          cudaMemGetInfo(&final_gpu_free, &final_gpu_total);
          #endif
          
          std::cout << "\n=== COMMUTATORS GENERATION COMPLETE ===" << std::endl;
          std::cout << "Final Memory State:" << std::endl;
          std::cout << "  Final CPU RSS: " << std::fixed << std::setprecision(2) << (final_cpu_rss / 1024.0) << " MB" << std::endl;
          std::cout << "  CPU Memory Growth: " << std::fixed << std::setprecision(2) << ((final_cpu_rss - initial_cpu_rss) / 1024.0) << " MB" << std::endl;
          
          #ifdef CUDA_ENABLED
          std::cout << "  Final GPU Free: " << std::fixed << std::setprecision(4) << (final_gpu_free / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          size_t gpu_memory_consumed = (initial_gpu_free > final_gpu_free) ? (initial_gpu_free - final_gpu_free) : 0;
          std::cout << "  GPU Memory Consumed: " << std::fixed << std::setprecision(4) << (gpu_memory_consumed / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          #endif
          
          std::cout << "\nCategorical Memory Breakdown:" << std::endl;
          std::cout << "  Total Operators Processed: " << poolsize << std::endl;
          std::cout << "  Total Pauli Terms Generated: " << num_pauli_terms_total << std::endl;
          std::cout << "  Total Commuting Groups: " << num_commuting_groups << std::endl;
          std::cout << "  Total Observables Created: " << total_observables_created << std::endl;
          std::cout << "  Total Pauli Maps Created: " << total_pauli_maps_created << std::endl;
          
          std::cout << "\nDetailed Memory Usage by Category:" << std::endl;
          std::cout << "  Coefficient Storage: " << std::fixed << std::setprecision(3) << (cumulative_coefficient_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  Z-mask Storage: " << std::fixed << std::setprecision(3) << (cumulative_zmask_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  Observable Structures: " << std::fixed << std::setprecision(3) << (cumulative_observable_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  Clique Metadata: " << std::fixed << std::setprecision(3) << (cumulative_clique_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  Pauli Map Storage: " << std::fixed << std::setprecision(3) << (cumulative_pauli_map_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  Comm Ops Storage: " << std::fixed << std::setprecision(3) << (cumulative_comm_ops_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          
          // Circuit memory breakdown estimate
          double non_circuit_memory_mb = (cumulative_coefficient_memory_bytes + cumulative_zmask_memory_bytes + 
                                         cumulative_observable_memory_bytes + cumulative_clique_memory_bytes +
                                         cumulative_pauli_map_memory_bytes + cumulative_comm_ops_memory_bytes) / (1024.0 * 1024.0);
          double total_growth_mb = (final_cpu_rss - initial_cpu_rss) / 1024.0;
          double estimated_circuit_memory_mb = total_growth_mb - non_circuit_memory_mb;
          
          std::cout << "\nCircuit Memory Analysis:" << std::endl;
          std::cout << "  Non-Circuit Memory: " << std::fixed << std::setprecision(3) << non_circuit_memory_mb << " MB" << std::endl;
          std::cout << "  Estimated Circuit Memory: " << std::fixed << std::setprecision(3) << estimated_circuit_memory_mb << " MB" << std::endl;
          std::cout << "  Circuit Memory Percentage: " << std::fixed << std::setprecision(1) 
                    << (100.0 * estimated_circuit_memory_mb / total_growth_mb) << "%" << std::endl;
          
          // Comprehensive cumulative Ansatz memory analysis
          size_t total_ansatz_objects = gradient_measurement.size();
          size_t gradient_measurement_vector_overhead = gradient_measurement.capacity() * sizeof(std::shared_ptr<Ansatz>);
          
          // Accumulate all Ansatz memory components across all operators
          size_t cumulative_shared_ptr_overhead = 0;
          size_t cumulative_theta_memory = 0;
          size_t cumulative_param_gates_memory = 0;
          size_t cumulative_gate_param_pointers_memory = 0;
          size_t cumulative_gate_coeffs_memory = 0;
          size_t cumulative_excitation_maps_memory = 0;
          size_t cumulative_circuit_gates_memory = 0;
          size_t cumulative_object_overhead = 0;
          
          for (size_t j = 0; j < gradient_measurement.size(); j++) {
            if (gradient_measurement[j] != nullptr) {
              // Error checking for null pointers
              auto theta_ptr = gradient_measurement[j]->getParams();
              auto param_gates_ptr = gradient_measurement[j]->getParamGateIndices();
              auto gates_ptr = gradient_measurement[j]->gates;
              
              if (theta_ptr == nullptr || param_gates_ptr == nullptr || gates_ptr == nullptr) {
                // Skip this object if any pointers are null
                continue;
              }
              
              // shared_ptr overhead per object
              cumulative_shared_ptr_overhead += 4 * (sizeof(std::shared_ptr<void>) + 32);
              
              // theta vectors
              cumulative_theta_memory += theta_ptr->capacity() * sizeof(ValType) + sizeof(std::vector<ValType>);
              
              // parameterized_gates vectors
              cumulative_param_gates_memory += param_gates_ptr->capacity() * sizeof(IdxType) + sizeof(std::vector<IdxType>);
              
              // gate_parameter_pointers nested structures
              auto gate_pointers = gradient_measurement[j]->getParamGatePointers();
              if (gate_pointers != nullptr) {
                cumulative_gate_param_pointers_memory += gate_pointers->capacity() * sizeof(std::vector<std::pair<IdxType, ValType>>) +
                                                         sizeof(std::vector<std::vector<std::pair<IdxType, ValType>>>);
                for (const auto& inner_vec : *gate_pointers) {
                  cumulative_gate_param_pointers_memory += inner_vec.capacity() * sizeof(std::pair<IdxType, ValType>) +
                                                           sizeof(std::vector<std::pair<IdxType, ValType>>);
                }
              }
              
              // gate_coefficients vectors (estimated)
              cumulative_gate_coeffs_memory += param_gates_ptr->size() * sizeof(ValType) + sizeof(std::vector<ValType>);
              
              // excitation_index_map string maps (estimated)
              size_t estimated_map_entries = std::min((size_t)100, param_gates_ptr->size());
              cumulative_excitation_maps_memory += estimated_map_entries * (20 + sizeof(IdxType) + 32) + 64;
              
              // circuit_gates vectors
              cumulative_circuit_gates_memory += gates_ptr->capacity() * sizeof(Gate) + sizeof(std::vector<Gate>);
              
              // object overhead
              cumulative_object_overhead += sizeof(Ansatz) + sizeof(Circuit) + 32; // ansatz_name
            }
          }
          
          size_t total_ansatz_system_memory = cumulative_shared_ptr_overhead + cumulative_theta_memory + 
                                             cumulative_param_gates_memory + cumulative_gate_param_pointers_memory +
                                             cumulative_gate_coeffs_memory + cumulative_excitation_maps_memory +
                                             cumulative_circuit_gates_memory + cumulative_object_overhead;
          
          std::cout << "\nComprehensive Cumulative Ansatz Memory:" << std::endl;
          std::cout << "  Total Ansatz Objects: " << total_ansatz_objects << std::endl;
          std::cout << "  gradient_measurement Vector Overhead: " << std::fixed << std::setprecision(3) 
                    << (gradient_measurement_vector_overhead / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All shared_ptr overhead: " << std::fixed << std::setprecision(3) 
                    << (cumulative_shared_ptr_overhead / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All theta vectors: " << std::fixed << std::setprecision(3) 
                    << (cumulative_theta_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All parameterized_gates: " << std::fixed << std::setprecision(3) 
                    << (cumulative_param_gates_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All gate_parameter_pointers: " << std::fixed << std::setprecision(3) 
                    << (cumulative_gate_param_pointers_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All gate_coefficients: " << std::fixed << std::setprecision(3) 
                    << (cumulative_gate_coeffs_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All excitation_maps: " << std::fixed << std::setprecision(3) 
                    << (cumulative_excitation_maps_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All circuit_gates: " << std::fixed << std::setprecision(3) 
                    << (cumulative_circuit_gates_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  All object overhead: " << std::fixed << std::setprecision(3) 
                    << (cumulative_object_overhead / (1024.0 * 1024.0)) << " MB" << std::endl;
          std::cout << "  Total Ansatz System Memory: " << std::fixed << std::setprecision(3) 
                    << (total_ansatz_system_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
          
          // Identify the largest cumulative memory consumer with safety checks
          std::vector<std::pair<std::string, size_t>> cumulative_components = {
            {"circuit_gates", cumulative_circuit_gates_memory},
            {"gate_parameter_pointers", cumulative_gate_param_pointers_memory},
            {"theta_vectors", cumulative_theta_memory},
            {"excitation_maps", cumulative_excitation_maps_memory},
            {"shared_ptr_overhead", cumulative_shared_ptr_overhead}
          };
          
          if (!cumulative_components.empty() && total_ansatz_system_memory > 0) {
            auto max_cumulative = std::max_element(cumulative_components.begin(), cumulative_components.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
            
            std::cout << "  System Memory Hotspot: " << max_cumulative->first << " (" 
                      << std::fixed << std::setprecision(1) << (100.0 * max_cumulative->second / total_ansatz_system_memory) 
                      << "% of all Ansatz memory)" << std::endl;
          } else {
            std::cout << "  System Memory Hotspot: Unable to determine (no valid data)" << std::endl;
          }
          
          // Circuit density analysis with division by zero protection
          double avg_circuit_memory_per_operator = (poolsize > 0) ? estimated_circuit_memory_mb / poolsize : 0.0;
          double avg_cliques_per_operator = (poolsize > 0) ? (double)num_commuting_groups / poolsize : 0.0;
          std::cout << "  Average Circuit Memory/Operator: " << std::fixed << std::setprecision(3) 
                    << avg_circuit_memory_per_operator << " MB (" << std::fixed << std::setprecision(1) 
                    << avg_cliques_per_operator << " cliques/op)" << std::endl;
          
          double total_estimated_mb = (cumulative_coefficient_memory_bytes + cumulative_zmask_memory_bytes + 
                                     cumulative_observable_memory_bytes + cumulative_clique_memory_bytes +
                                     cumulative_pauli_map_memory_bytes + cumulative_comm_ops_memory_bytes) / (1024.0 * 1024.0);
          std::cout << "  Total Estimated: " << std::fixed << std::setprecision(3) << total_estimated_mb << " MB" << std::endl;
          std::cout << "  Largest Single Operator: " << std::fixed << std::setprecision(3) << (max_single_operator_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          
          // Memory analysis with division protection
          if (total_growth_mb > 0) {
            double efficiency_ratio = total_estimated_mb / total_growth_mb;
            std::cout << "\nMemory Analysis:" << std::endl;
            std::cout << "  Estimated vs Actual Growth: " << std::fixed << std::setprecision(1) << (efficiency_ratio * 100.0) << "%" << std::endl;
            std::cout << "  Memory Efficiency Ratio: " << std::fixed << std::setprecision(2) << efficiency_ratio << std::endl;
          } else {
            std::cout << "\nMemory Analysis:" << std::endl;
            std::cout << "  No memory growth detected for analysis" << std::endl;
          }
          
          // Enhanced GPU memory analysis
          #ifdef CUDA_ENABLED
          std::cout << "\nGPU Memory Analysis:" << std::endl;
          std::cout << "  Initial Free: " << std::fixed << std::setprecision(4) << (initial_gpu_free / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          std::cout << "  Final Free: " << std::fixed << std::setprecision(4) << (final_gpu_free / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          std::cout << "  Net Consumed: " << std::fixed << std::setprecision(4) << (gpu_memory_consumed / (1024.0*1024.0*1024.0)) << " GB" << std::endl;
          
          // Calculate expected GPU memory (statevector size based on number of qubits)
          size_t num_qubits = ansatz->num_qubits();
          size_t statevector_size_bytes = (1ULL << num_qubits) * sizeof(std::complex<double>); // 2^n_qubits * 16 bytes
          double expected_gpu_mb = statevector_size_bytes / (1024.0 * 1024.0);
          std::cout << "  Expected Statevector (" << num_qubits << " qubits): " << std::fixed << std::setprecision(2) << expected_gpu_mb << " MB" << std::endl;
          
          double gpu_vs_expected_ratio = (expected_gpu_mb > 0) ? 
                                        (gpu_memory_consumed * 1024.0) / expected_gpu_mb : 0.0;
          std::cout << "  GPU vs Expected Ratio: " << std::fixed << std::setprecision(2) << gpu_vs_expected_ratio << std::endl;
          #endif
          
          std::cout << "\nGenerated " << poolsize << " commutators with " << num_pauli_terms_total << " Individual Pauli Strings" << std::endl;
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
                        << std::setw(58) << "|  Selected Operator"
                        << std::endl;
              std::cout << std::string(120, '-') << std::endl;
            }
            NWQSim::CircuitMetrics metrics = ansatz -> circuit_metrics();
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
            // Print the selected operator
            std::cout << "|  " << ansatz->get_operator_string(max_ind) << std::endl;
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