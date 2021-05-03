#ifndef QUANTUM_CIRCUITS_H_
#define QUANTUM_CIRCUITS_H_

#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include "qsystem_p.h"
#include "quantum_gates.h"
#include <petscksp.h>
#include <petscts.h>

void add_gate_to_circuit_sys(circuit*,PetscReal,gate_type,...);
void apply_circuit_to_sys(qsystem,circuit*,PetscReal);
void combine_circuit_to_mat_sys(qsystem,Mat*,circuit);
void schedule_circuit_layers(qsystem,circuit*);
void _get_n_after_2qbit_sys(qsystem,PetscInt*,PetscInt[],PetscInt,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
void _get_n_after_1qbit_sys(qsystem,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*);
void CUSTOM2Q_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void HADAMARD_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void CZX_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void CmZ_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void CZ_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void CXZ_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void CZ_ARP_get_val_j_from_global_i_sys(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
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
void (*_get_val_j_functions_gates_sys[MAX_GATES])(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);

void get_probs_pauli_1sys(qvec,operator,PetscScalar probs[2]);
void apply_gate2(qvec,gate_type,...);
void apply_projective_measurement_qubit(qvec,PetscScalar*,PetscInt,...);
void apply_projective_measurement_qubit_list(qvec,PetscScalar*,PetscInt,operator*);

void apply_single_qb_measurement_error_probs(PetscReal*,PetscInt,PetscReal,PetscReal,PetscInt);
void apply_single_qb_measurement_error_state(qvec,PetscReal,PetscReal,PetscInt);
void _apply_single_qb_measurement_error_wf(qvec,PetscReal,PetscReal,PetscInt);


void CUSTOM2Q_gate_func_wf(qvec,Vec,PetscInt*,void*,PetscBool);//need to be void?
void HADAMARD_gate_func_wf(qvec,Vec,PetscInt*,void*,PetscBool);
void SIGMAX_gate_func_wf(qvec,Vec,PetscInt*,void*,PetscBool);
void RZ_gate_func_wf(qvec,Vec,PetscInt*,void*,PetscBool);
void CNOT_gate_func_wf(qvec,Vec,PetscInt*,void*,PetscBool);
void U3_gate_func_wf(qvec,Vec,PetscInt*,void*,PetscBool);


void (*_get_gate_func_wf(gate_type))(qvec,Vec,PetscInt*,void*,PetscBool);
void _change_basis_ij_pair_sys(qsystem,PetscInt*,PetscInt*,PetscInt,PetscInt);
void _initialize_gate_function_array_sys();
PetscErrorCode _sys_QC_EventFunction(qsystem,PetscReal,PetscScalar*);
PetscErrorCode _sys_QC_PostEventFunction(TS,PetscInt,PetscInt [],PetscReal,Vec,PetscBool,void*);
void _apply_gate_sys(qsystem,struct quantum_gate_struct,Vec);


void (*_get_val_j_functions_gates_sys[MAX_GATES])(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);

void apply_circuit_to_qvec2(circuit,qvec);
void apply_circuit_to_qvec(qsystem,circuit,qvec);
void _apply_gate2(struct quantum_gate_struct,qvec);
PetscInt flipBit(PetscInt,PetscInt);
PetscInt insertTwoZeroBits(PetscInt,PetscInt,PetscInt);
PetscInt insertZeroBit(PetscInt,PetscInt);

#endif
