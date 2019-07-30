#ifndef QUANTUM_CIRCUITS_H_
#define QUANTUM_CIRCUITS_H_

#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include "qsystem_p.h"
#include "quantum_gates.h"
#include <petscksp.h>
#include <petscts.h>

void apply_circuit_to_sys(qsystem,circuit*,PetscReal);
void combine_circuit_to_mat_sys(qsystem,Mat*,circuit);
void schedule_circuit_layers(qsystem,circuit*);

void _get_n_after_2qbit_sys(qsystem,PetscInt*,int[],PetscInt,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
void _get_n_after_1qbit_sys(qsystem,PetscInt,int,PetscInt,PetscInt*,PetscInt*);
void HADAMARD_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void CNOT_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void SIGMAX_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void SIGMAY_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void SIGMAZ_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void EYE_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void RX_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void RY_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void RZ_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void U1_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void U2_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void U3_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);

void _change_basis_ij_pair_sys(qsystem,PetscInt*,PetscInt*,PetscInt,PetscInt);
void _initialize_gate_function_array_sys();
PetscErrorCode _sys_QC_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _sys_QC_PostEventFunction(TS,PetscInt,PetscInt [],PetscReal,Vec,PetscBool,void*);
void _apply_gate_sys(qsystem,struct quantum_gate_struct,Vec);


void (*_get_val_j_functions_gates_sys[MAX_GATES])(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);

#endif
