
#ifndef DMUTILITIES_P_H_
#define DMUTILITIES_P_H_

#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include "qvec_utilities.h"
#include <stdarg.h>

void create_dm(Vec*,PetscInt);
void create_full_dm(Vec*);
void destroy_dm(Vec);
void get_dm_element(Vec,PetscInt,PetscInt,PetscScalar*);
void get_dm_element_local(Vec,PetscInt,PetscInt,PetscScalar*);
void add_value_to_dm(Vec,PetscInt,PetscInt,PetscScalar);
void set_dm_from_initial_pop(Vec);
void set_initial_dm_2qds_first_plus_pop(Vec,Vec);
void assemble_dm(Vec);
void partial_trace_over_one(Vec,Vec,PetscInt,PetscInt,PetscInt,PetscInt);
void partial_trace_over(Vec,Vec,int,...);
void partial_trace_keep(Vec,Vec,int,...);
void get_populations(Vec,double**);
void get_expectation_value(Vec,PetscScalar*,int,...);
int get_num_populations();
void get_bipartite_concurrence(Vec,double*);
void get_fidelity(Vec,Vec,double*);
void print_psi(Vec,int);
void print_dm(Vec,int);
void print_dm_sparse(Vec,int);
void print_dm_sparse_to_file(Vec,int,char[]);
void _get_expectation_value_psi(Vec,PetscScalar*,int,operator*);
void measure_dm(Vec,operator);
void add_ops_to_mat(Mat,PetscInt,PetscInt,...);
void print_mat_sparse_to_file(Mat,char[]);
void print_mat_sparse(Mat);
void vadd_ops_to_mat(Mat,PetscInt,PetscInt,va_list);
void trace_dm(PetscScalar*,Vec);
#endif
