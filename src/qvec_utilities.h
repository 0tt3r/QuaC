#ifndef QVEC_UTILITIES_H_
#define QVEC_UTILITIES_H_

#include <petsc.h>
#include <stdarg.h>
#include "operators_p.h"
#include "operators.h"
#include "qsystem_p.h"

void create_qvec_sys(qsystem,qvec*);
void create_dm_sys(qsystem,qvec*);
void create_wf_sys(qsystem,qvec*);

void _create_vec(Vec*,PetscInt,PetscInt);

void assemble_qvec(qvec);
void destroy_qvec(qvec*);

void add_to_qvec_fock_op(PetscScalar,qvec,PetscInt,...);

void add_to_qvec_fock_op_list(PetscScalar,qvec,PetscInt,operator*,PetscInt*);
void add_to_qvec(qvec,PetscScalar,...);
void add_to_qvec_loc(qvec,PetscScalar,PetscInt);
void get_qvec_loc_fock_op(qvec,PetscInt*,PetscInt,...);
void get_qvec_loc_fock_op_list(qvec,PetscInt*,PetscInt,operator[],PetscInt[]);

void get_expectation_value_qvec(qvec,PetscScalar*,PetscInt,...);
void get_expectation_value_qvec_list(qvec,PetscScalar*,PetscInt,operator*);
void _get_expectation_value_wf(qvec,PetscScalar*,PetscInt,operator*);
void _get_expectation_value_dm(qvec,PetscScalar*,PetscInt,operator*);

void qvec_mat_mult(Mat,qvec);

void get_fidelity_qvec(qvec,qvec,PetscReal*);
void _get_fidelity_wf_wf(Vec,Vec,PetscReal*);
void _get_fidelity_dm_dm(Vec,Vec,PetscReal*);
void _get_fidelity_dm_wf(Vec,Vec,PetscReal*);

void print_qvec_file(qvec,char[]);
void print_dm_qvec_file(qvec,char[]);
void print_wf_qvec_file(qvec,char[]);

void print_qvec(qvec);
void print_dm_qvec(qvec);
void print_wf_qvec(qvec);
void get_wf_element_qvec(qvec,PetscInt,PetscScalar*);
void get_dm_element_qvec(qvec,PetscInt,PetscInt,PetscScalar*);
void get_dm_element_qvec_local(qvec,PetscInt,PetscInt,PetscScalar*);

#endif
