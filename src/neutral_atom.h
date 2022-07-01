#ifndef NEUTRAL_ATOM_H_
#define NEUTRAL_ATOM_H_

#include "qvec_utilities.h"
#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include <stdlib.h>
#include <stdio.h>
#include <petscblaslapack.h>
#include <string.h>
#include "quantum_gates.h"


typedef struct {
  PetscScalar omega,delta;
  PetscReal length,stime,deltat;
} PulseParams;

PetscScalar omega_sp(PetscReal,void*);
PetscScalar delta_arp(PetscReal,void*);
PetscScalar omega_arp(PetscReal,void*);
void apply_projective_measurement_tensor(qvec,PetscScalar*,PetscInt,...);
void apply_projective_measurement_tensor_list(qvec,PetscScalar*,PetscInt,operator*);
void apply_1q_na_gate_to_qvec(qvec,gate_type,operator);
void apply_1q_na_haar_gate_to_qvec(qvec,custom_gate_data*,operator);
void get_probs_pauli_1sys(qvec,operator,PetscScalar[]);
#endif
