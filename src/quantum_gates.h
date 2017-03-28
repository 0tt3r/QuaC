#ifndef QUANTUM_GATES_H_
#define QUANTUM_GATES_H_

#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include <petscksp.h>
#include <petscts.h>

typedef enum {
  HADAMARD = 0,
  CNOT = 1
} gate_type;


struct quantum_gate_struct{
  PetscReal time;
  gate_type my_gate_type;
  int *qubit_numbers;
};

PetscScalar _get_val_in_subspace_gate(PetscInt,gate_type,PetscInt,PetscInt*,PetscInt*);
void add_gate(PetscReal,gate_type,...);
void _construct_gate_mat(gate_type,int*,Mat);
void _apply_gate(gate_type,int*,Vec);
void _change_basis_ij_pair(PetscInt*,PetscInt*,PetscInt,PetscInt);
PetscErrorCode _QG_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _QG_PostEventFunction(TS,PetscInt,PetscInt event_list[],PetscReal,Vec,void*);

#define MAX_GATES 100 // Consider not making this a define
struct quantum_gate_struct _quantum_gate_list[MAX_GATES];
extern int _num_quantum_gates;

#endif
