#ifndef DMUTILITIES_P_H_
#define DMUTILITIES_P_H_

#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include <stdarg.h>

void create_dm(Vec*,PetscInt);
void destroy_dm(Vec);
void partial_trace_over_one(Vec,Vec,PetscInt,PetscInt,PetscInt);
void partial_trace_over(Vec,Vec,int,...);
void get_populations(Vec,PetscReal);
void sqrt_mat(Mat);
void get_fidelity(Vec,Vec,double*);
#endif
