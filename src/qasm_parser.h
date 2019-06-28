#include <petsc.h>
#include "quantum_gates.h"

void projectq_qasm_read(char[],PetscInt*,circuit*);
void projectq_vqe_get_expectation(char[],Vec,PetscScalar*);
void _projectq_qasm_add_gate(char*,circuit*,PetscReal);
void projectq_vqe_get_expectation_encoded(char[],Vec,PetscScalar*,PetscInt,...);
void quil_read(char[],PetscInt*,circuit*);
void _quil_add_gate(char*,circuit*,PetscReal);
void _quil_get_angle_pi(char[],PetscReal*);

void qiskit_vqe_get_expectation(char[],Vec,PetscScalar*);
void qiskit_qasm_read(char[],PetscInt*,circuit*);
void _qiskit_qasm_add_gate(char*,circuit*,PetscReal);
