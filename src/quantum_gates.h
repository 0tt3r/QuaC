#ifndef QUANTUM_GATES_H_
#define QUANTUM_GATES_H_

#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include "qsystem_p.h"
#include <petscksp.h>
#include <petscts.h>

typedef enum {
  NULL_GATE = -1000,
  CZ_ARP = -7,
  CUSTOM2QGATE = -6,
  CZX  = -5,
  CmZ  = -4,
  CZ   = -3,
  CXZ  = -2,
  CNOT = -1,
  HADAMARD = 1,
  SIGMAX = 2,
  SIGMAY = 3,
  SIGMAZ = 4,
  EYE    = 5,
  RX     = 6,
  RY     = 7,
  RZ     = 8,
  U1     = 9,
  U2     = 10,
  U3     = 11,
  CUSTOM1QGATE = 12
} gate_type;


typedef void(*custom_gate_func_type)(PetscScalar*,PetscInt,PetscInt,void*);

struct quantum_gate_struct{
  PetscReal time,run_time; //run_time is how long the gate takes
  gate_type my_gate_type;
  PetscInt *qubit_numbers,num_qubits;
  void (*_get_val_j_from_global_i)(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
  void (*_get_val_j_from_global_i_sys)(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
  void (*gate_func_wf)(qvec,Vec,PetscInt*,void*);
  custom_gate_func_type custom_func;
  PetscReal theta,phi,lambda; //Only used for rotation gates
  void *gate_ctx;
};

struct gate_layer_struct{
  PetscReal time; //Time struct should be applied
  PetscInt num_gates;
  struct quantum_gate_struct *gate_list; //List of gates to apply at that time
};


PetscScalar _get_val_in_subspace_gate(PetscInt,gate_type,PetscInt,PetscInt*,PetscInt*);
void add_gate(PetscReal,gate_type,...);
void _construct_gate_mat(gate_type,int*,Mat);
void _apply_gate(struct quantum_gate_struct,Vec);
void _change_basis_ij_pair(PetscInt*,PetscInt*,PetscInt,PetscInt);
PetscErrorCode _QG_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _QG_PostEventFunction(TS,PetscInt,PetscInt [],PetscReal,Vec,PetscBool,void*);

PetscErrorCode _QC_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _QC_PostEventFunction(TS,PetscInt,PetscInt [],PetscReal,Vec,PetscBool,void*);

void create_circuit(circuit*,PetscInt);
void destroy_circuit(circuit*);
void add_gate_to_circuit(circuit*,PetscReal,gate_type,...);
void add_circuit_to_circuit(circuit*,circuit,PetscReal);
void start_circuit_at_time(circuit*,PetscReal);

void _get_val_j_from_global_i_gates(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void combine_circuit_to_mat(Mat*,circuit);
void combine_circuit_to_mat2(Mat*,circuit);

void _get_val_j_from_global_i_super_gates(PetscInt,struct quantum_gate_struct,PetscInt*,
                                          PetscScalar*,PetscInt*,PetscScalar*,PetscInt);
void combine_circuit_to_super_mat(Mat*,circuit);
void _initialize_gate_function_array();
void _check_gate_type(gate_type,int*);
void _get_n_after_2qbit(PetscInt*,int[],PetscInt,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
void _get_n_after_1qbit(PetscInt,int,PetscInt,PetscInt*,PetscInt*);
void HADAMARD_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void CNOT_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void SIGMAX_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void SIGMAY_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void SIGMAZ_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void EYE_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void RX_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void RY_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void RZ_get_val_j_from_global_i(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);

struct quantum_gate_struct _quantum_gate_list[MAX_GATES];
extern int _num_quantum_gates;
extern int _min_gate_enum; // Minimum gate enumeration number
extern int _gate_array_initialized;
void (*_get_val_j_functions_gates[MAX_GATES])(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
circuit _circuit_list[MAX_GATES];
extern int _num_circuits;

#endif
