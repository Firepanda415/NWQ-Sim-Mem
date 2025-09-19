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
            
            // Detailed STL container breakdown
            std::cout << "  STL Containers: PauliMap=" << std::fixed << std::setprecision(2) << (pmap_actual_bytes / (1024.0 * 1024.0)) 
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
          
          // Memory tracking before circuit construction
          size_t cpu_rss_before_circuits = 0;
          if (state->get_process_rank() == 0) {
            std::ifstream status_file_circuits("/proc/self/status");
            std::string line_circuits;
            while (std::getline(status_file_circuits, line_circuits)) {
              if (line_circuits.find("VmRSS:") == 0) {
                sscanf(line_circuits.c_str(), "VmRSS: %zu kB", &cpu_rss_before_circuits);
                break;
              }
            }
          }
          
          // For each clique, construct a measurement circuit and append
          size_t circuit_gate_count_before = 0;
          size_t circuit_instruction_count_before = 0;
          if (state->get_process_rank() == 0) {
            circuit_gate_count_before = gradient_measurement[i]->get_gate_count();
            circuit_instruction_count_before = gradient_measurement[i]->get_instruction_count();
          }
          
          for (size_t j = 0; j < cliques.size(); j++) {
            std::vector<IdxType>& clique = *cliqueiter;
            std::vector<PauliOperator> commuting_group (clique.size());
            std::transform(clique.begin(), clique.end(),
              commuting_group.begin(), [&] (IdxType ind) {return comm_ops[ind];}); 
            // Get a Pauli string that's the logical `or` over each stabilizer state, append to zmasks and coeff data structures within loop  
            PauliOperator common = make_common_op(commuting_group, 
                                                  commutator_zmasks[i][j], 
                                                  commutator_coeffs[i][j]);
            
            // Track memory before individual circuit construction
            size_t clique_cpu_before = 0;
            if (state->get_process_rank() == 0 && j == 0) {
              std::ifstream status_clique("/proc/self/status");
              std::string line_clique;
              while (std::getline(status_clique, line_clique)) {
                if (line_clique.find("VmRSS:") == 0) {
                  sscanf(line_clique.c_str(), "VmRSS: %zu kB", &clique_cpu_before);
                  break;
                }
              }
            }
            
            Measurement circ1 (common, false); // QWC measurement circuit $U_M$
            gradient_measurement[i]->compose(circ1, qubit_mapping);         // add to gradient measurement
            // add a gate to compute the expectation values   
            state->set_exp_gate(gradient_measurement[i], gradient_observables[i] + j, commutator_zmasks[i][j], commutator_coeffs[i][j]);
            Measurement circ2 (common, true); // inverse of the measurement circuit $U_M^\dagger$
            gradient_measurement[i]->compose(circ2, qubit_mapping);  // add the inverse
            
            // Track memory after individual circuit construction (sample first few)
            if (state->get_process_rank() == 0 && j < 3) {
              size_t clique_cpu_after = 0;
              std::ifstream status_clique_after("/proc/self/status");
              std::string line_clique_after;
              while (std::getline(status_clique_after, line_clique_after)) {
                if (line_clique_after.find("VmRSS:") == 0) {
                  sscanf(line_clique_after.c_str(), "VmRSS: %zu kB", &clique_cpu_after);
                  break;
                }
              }
              
              double clique_memory_delta = (clique_cpu_after - clique_cpu_before) / 1024.0;
              std::cout << "    Clique " << j << " Memory: +" << std::fixed << std::setprecision(3) 
                        << clique_memory_delta << " MB (" << commuting_group.size() << " Pauli ops)" << std::endl;
            }
            
            cliqueiter++;
          }
          
          // Detailed circuit construction analysis
          if (state->get_process_rank() == 0) {
            size_t circuit_gate_count_after = gradient_measurement[i]->get_gate_count();
            size_t circuit_instruction_count_after = gradient_measurement[i]->get_instruction_count();
            size_t gates_added = circuit_gate_count_after - circuit_gate_count_before;
            size_t instructions_added = circuit_instruction_count_after - circuit_instruction_count_before;
            
            std::cout << "  Circuit Details: +" << gates_added << " gates, +" << instructions_added 
                      << " instructions (" << cliques.size() << " cliques * 2 circuits each)" << std::endl;
            
            // Memory per gate analysis
            double circuit_memory_mb = (cpu_rss_after_circuits - cpu_rss_before_circuits) / 1024.0;
            if (gates_added > 0) {
              double memory_per_gate_kb = (circuit_memory_mb * 1024.0) / gates_added;
              std::cout << "  Memory/Gate: " << std::fixed << std::setprecision(2) << memory_per_gate_kb 
                        << " KB (" << std::fixed << std::setprecision(1) << (gates_added / (double)cliques.size()) 
                        << " gates/clique)" << std::endl;
            }
            
            // Circuit object memory analysis
            size_t estimated_circuit_object_bytes = sizeof(std::shared_ptr<Ansatz>) + 
                                                   gates_added * 64 + // estimated gate storage
                                                   instructions_added * 32; // estimated instruction storage
            std::cout << "  Estimated Circuit Object: " << std::fixed << std::setprecision(3) 
                      << (estimated_circuit_object_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          }
          
          // Memory tracking after circuit construction
          size_t cpu_rss_after_circuits = 0;
          if (state->get_process_rank() == 0) {
            std::ifstream status_file_circuits_after("/proc/self/status");
            std::string line_circuits_after;
            while (std::getline(status_file_circuits_after, line_circuits_after)) {
              if (line_circuits_after.find("VmRSS:") == 0) {
                sscanf(line_circuits_after.c_str(), "VmRSS: %zu kB", &cpu_rss_after_circuits);
                break;
              }
            }
            
            double circuit_memory_mb = (cpu_rss_after_circuits - cpu_rss_before_circuits) / 1024.0;
            std::cout << "  Circuit Construction: +" << std::fixed << std::setprecision(2) << circuit_memory_mb 
                      << " MB (" << cliques.size() << " circuits)" << std::endl;
            
            // Show cumulative circuit memory growth
            static double total_circuit_memory = 0.0;
            static double prev_circuit_memory = 0.0;
            static size_t prev_cliques = 0;
            static size_t total_circuits_created = 0;
            total_circuit_memory += circuit_memory_mb;
            total_circuits_created += cliques.size();
            std::cout << "  Cumulative Circuit Memory: " << std::fixed << std::setprecision(2) 
                      << total_circuit_memory << " MB (Total circuits: " << total_circuits_created << ")" << std::endl;
            
            // gradient_measurement vector analysis
            size_t gradient_measurement_vector_bytes = gradient_measurement.capacity() * sizeof(std::shared_ptr<Ansatz>);
            size_t accumulated_ansatz_objects = (i + 1) * sizeof(Ansatz); // approximation
            std::cout << "  gradient_measurement Vector: " << std::fixed << std::setprecision(3) 
                      << (gradient_measurement_vector_bytes / (1024.0 * 1024.0)) << " MB capacity, " 
                      << std::fixed << std::setprecision(3) << (accumulated_ansatz_objects / (1024.0 * 1024.0)) 
                      << " MB ansatz objects" << std::endl;
            
            // Track circuit memory per clique for analysis
            if (i > 0 && prev_cliques > 0) {
              double memory_per_clique_current = circuit_memory_mb / cliques.size();
              double memory_per_clique_prev = prev_circuit_memory / prev_cliques;
              
              std::cout << "  Memory/Clique: " << std::fixed << std::setprecision(3) 
                        << memory_per_clique_current << " MB (prev: " 
                        << memory_per_clique_prev << " MB)" << std::endl;
            } else if (cliques.size() > 0) {
              double memory_per_clique_current = circuit_memory_mb / cliques.size();
              std::cout << "  Memory/Clique: " << std::fixed << std::setprecision(3) 
                        << memory_per_clique_current << " MB" << std::endl;
            }
            
            // Track for next iteration
            prev_circuit_memory = circuit_memory_mb;
            prev_cliques = cliques.size();
          }
          
          // Memory tracking after operator processing
          size_t cpu_rss_after = 0;
          #ifdef CUDA_ENABLED
          size_t gpu_free_after = 0, gpu_total_after = 0;
          #endif
          
          if (state->get_process_rank() == 0) {
            // Get CPU memory after this operator
            std::ifstream status_file_after("/proc/self/status");
            std::string line_after;
            while (std::getline(status_file_after, line_after)) {
              if (line_after.find("VmRSS:") == 0) {
                sscanf(line_after.c_str(), "VmRSS: %zu kB", &cpu_rss_after);
                break;
              }
            }
            
            #ifdef CUDA_ENABLED
            cudaMemGetInfo(&gpu_free_after, &gpu_total_after);
            #endif
            
            // Print intermediate memory checkpoint
            double cpu_growth_mb = (cpu_rss_after - cpu_rss_before) / 1024.0;
            std::cout << "  → Memory delta: CPU +" << std::fixed << std::setprecision(2) << cpu_growth_mb << " MB";
            
            #ifdef CUDA_ENABLED
            if (gpu_free_before > gpu_free_after) {
              double gpu_consumed_mb = (gpu_free_before - gpu_free_after) / (1024.0 * 1024.0);
              std::cout << ", GPU +" << std::fixed << std::setprecision(2) << gpu_consumed_mb << " MB";
            }
            #endif
            
            // Memory growth ratio
            double growth_ratio = (operator_total_memory_bytes > 0) ? 
                                (cpu_growth_mb * 1024.0 * 1024.0) / operator_total_memory_bytes : 0.0;
            std::cout << " [" << std::fixed << std::setprecision(1) << growth_ratio << "x]";
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
          
          // Memory cleanup to prevent accumulation during operator processing
          pmap.clear();
          std::unordered_map<PauliOperator, std::complex<double>, PauliHash>().swap(pmap);
          comm_ops.clear();
          comm_ops.shrink_to_fit();
          cliques.clear();
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
          double actual_growth_mb = (final_cpu_rss - initial_cpu_rss) / 1024.0;
          double estimated_circuit_memory_mb = actual_growth_mb - non_circuit_memory_mb;
          
          std::cout << "\nCircuit Memory Analysis:" << std::endl;
          std::cout << "  Non-Circuit Memory: " << std::fixed << std::setprecision(3) << non_circuit_memory_mb << " MB" << std::endl;
          std::cout << "  Estimated Circuit Memory: " << std::fixed << std::setprecision(3) << estimated_circuit_memory_mb << " MB" << std::endl;
          std::cout << "  Circuit Memory Percentage: " << std::fixed << std::setprecision(1) 
                    << (100.0 * estimated_circuit_memory_mb / actual_growth_mb) << "%" << std::endl;
          
          // gradient_measurement vector final analysis
          size_t final_gradient_measurement_bytes = gradient_measurement.size() * sizeof(std::shared_ptr<Ansatz>);
          std::cout << "  gradient_measurement Vector Size: " << gradient_measurement.size() << " entries, " 
                    << std::fixed << std::setprecision(3) << (final_gradient_measurement_bytes / (1024.0 * 1024.0)) 
                    << " MB vector overhead" << std::endl;
          
          // Circuit density analysis
          double avg_circuit_memory_per_operator = estimated_circuit_memory_mb / poolsize;
          double avg_cliques_per_operator = (double)num_commuting_groups / poolsize;
          std::cout << "  Average Circuit Memory/Operator: " << std::fixed << std::setprecision(3) 
                    << avg_circuit_memory_per_operator << " MB (" << std::fixed << std::setprecision(1) 
                    << avg_cliques_per_operator << " cliques/op)" << std::endl;
          
          double total_estimated_mb = (cumulative_coefficient_memory_bytes + cumulative_zmask_memory_bytes + 
                                     cumulative_observable_memory_bytes + cumulative_clique_memory_bytes +
                                     cumulative_pauli_map_memory_bytes + cumulative_comm_ops_memory_bytes) / (1024.0 * 1024.0);
          std::cout << "  Total Estimated: " << std::fixed << std::setprecision(3) << total_estimated_mb << " MB" << std::endl;
          std::cout << "  Largest Single Operator: " << std::fixed << std::setprecision(3) << (max_single_operator_memory_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
          
          // Memory analysis
          double actual_growth_mb = (final_cpu_rss - initial_cpu_rss) / 1024.0;
          if (actual_growth_mb > 0) {
            double efficiency_ratio = total_estimated_mb / actual_growth_mb;
            std::cout << "\nMemory Analysis:" << std::endl;
            std::cout << "  Estimated vs Actual Growth: " << std::fixed << std::setprecision(1) << (efficiency_ratio * 100.0) << "%" << std::endl;
            std::cout << "  Memory Efficiency Ratio: " << std::fixed << std::setprecision(2) << efficiency_ratio << std::endl;
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