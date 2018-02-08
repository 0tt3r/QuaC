#ifndef QUANTUM_GATES_H_
#define QUANTUM_GATES_H_

#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include <petscksp.h>
#include <petscts.h>

typedef enum {
  CNOT = -1,
  HADAMARD = 1,
  SIGMAX = 2,
  SIGMAY = 3,
  SIGMAZ = 4
} gate_type;


struct quantum_gate_struct{
  PetscReal time;
  gate_type my_gate_type;
  int *qubit_numbers;
};

typedef struct circuit{
  PetscInt num_gates,gate_list_size,current_gate;
  PetscReal start_time;
  struct quantum_gate_struct *gate_list;
} circuit;

PetscScalar _get_val_in_subspace_gate(PetscInt,gate_type,PetscInt,PetscInt*,PetscInt*);
void add_gate(PetscReal,gate_type,...);
void _construct_gate_mat(gate_type,int*,Mat);
void _apply_gate(gate_type,int*,Vec);
void _change_basis_ij_pair(PetscInt*,PetscInt*,PetscInt,PetscInt);
PetscErrorCode _QG_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _QG_PostEventFunction(TS,PetscInt,PetscInt [],PetscReal,Vec,void*);

PetscErrorCode _QC_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _QC_PostEventFunction(TS,PetscInt,PetscInt [],PetscReal,Vec,void*);

void create_circuit(circuit*,PetscInt);
void add_gate_to_circuit(circuit*,PetscReal,gate_type,...);
void start_circuit_at_time(circuit*,PetscReal);

void _get_val_j_from_global_i_gates(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
void combine_circuit_to_mat(Mat*,circuit);
void combine_circuit_to_mat2(Mat*,circuit);

void _get_val_j_from_global_i_super_gates(PetscInt,struct quantum_gate_struct,PetscInt*,
                                          PetscScalar*,PetscInt*,PetscScalar*,PetscInt);
void combine_circuit_to_super_mat(Mat*,circuit);


#define MAX_GATES 100 // Consider not making this a define
struct quantum_gate_struct _quantum_gate_list[MAX_GATES];
extern int _num_quantum_gates;
circuit _circuit_list[MAX_GATES];
extern int _num_circuits;

#endif
