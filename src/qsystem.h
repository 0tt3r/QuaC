#ifndef QSYSTEM_H_
#define QSYSTEM_H_

#include <petsc.h>
#include "qvec_utilities.h"
#include "qsystem_p.h"
#include "sprng.h"

void construct_matrix(qsystem);
void construct_op_matrix_wf_list(qsystem,PetscScalar,Mat*,PetscInt,operator*);
void initialize_system(qsystem*);
void destroy_system(qsystem*);
void use_mcwf_solver(qsystem,PetscInt,PetscInt);

void create_op_sys(qsystem,PetscInt,operator*);
void destroy_op_sys(operator*);
void _create_single_op(PetscInt,PetscInt,op_type,operator*);


void create_vec_op_sys(qsystem,PetscInt,vec_op*);
void destroy_vec_op_sys(vec_op*);
void _create_single_vec(PetscInt,PetscInt,PetscInt,operator*);



void add_ham_term(qsystem,PetscScalar,PetscInt,...);
void add_lin_term(qsystem,PetscScalar,PetscInt,...);
void add_lin_term_list(qsystem,PetscScalar,PetscInt,operator*);
void add_ham_term_time_dep(qsystem,PetscScalar,PetscScalar(*)(PetscReal),PetscInt,...);
void add_lin_term_time_dep(qsystem,PetscScalar,PetscScalar(*)(PetscReal),PetscInt,...);

void time_step_sys(qsystem,qvec,PetscReal,PetscReal,PetscReal,PetscInt);
void set_ts_monitor_sys(qsystem,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void*);

void clear_mat_terms_sys(qsystem);
#endif
