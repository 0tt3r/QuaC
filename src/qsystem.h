#ifndef QSYSTEM_H_
#define QSYSTEM_H_

#include <petsc.h>
#include "qvec_utilities.h"
#include "qsystem_p.h"

void construct_matrix(qsystem);
void initialize_system(qsystem*);
void destroy_system(qsystem*);

void create_op_sys(qsystem,PetscInt,operator*);
void destroy_op_sys(operator*);
void _create_single_op(PetscInt,PetscInt,op_type,operator*);

void add_ham_term(qsystem,PetscScalar,PetscInt,...);
void add_lin_term(qsystem,PetscScalar,PetscInt,...);
void add_ham_term_time_dep(qsystem,PetscScalar,PetscScalar(*)(double),PetscInt,...);
void add_lin_term_time_dep(qsystem,PetscScalar,PetscScalar(*)(double),PetscInt,...);

void time_step_sys(qsystem,qvec,PetscReal,PetscReal,PetscReal,PetscInt);
void set_ts_monitor_sys(qsystem,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*));

#endif
