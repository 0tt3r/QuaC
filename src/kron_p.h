#ifndef KRON_P_H_
#define KRON_P_H_

#include "operators_p.h"
#include "operators.h"

long   _get_loop_limit(op_type,int);
PetscScalar _get_val_in_subspace(long,op_type,int,long*,long*);


void _get_val_j_from_global_i(PetscInt,operator,PetscInt*,PetscScalar*,PetscInt);
void _get_val_j_from_global_i_vec_vec(PetscInt,operator,operator,PetscInt*,PetscScalar*,PetscInt);


void _get_val_j_from_global_i_lin(PetscInt,operator,PetscInt*,PetscScalar*,PetscInt);
void _get_val_j_from_global_i_lin_vec_vec(PetscInt,operator,operator,
                                          PetscInt*,PetscScalar*,PetscInt);



void   _add_to_PETSc_kron(Mat,PetscScalar,int,int,op_type,int,int,int,int);
void   _add_to_PETSc_kron_comb(Mat,PetscScalar,int,int,op_type,int,int,int,
                               op_type,int,int,int,int,int);
void   _add_to_PETSc_kron_lin(Mat,PetscScalar,int,int,op_type,int,int,int,int);
void   _add_to_PETSc_kron_lin_comb(Mat,PetscScalar,int,int,op_type,int);
void   _add_to_PETSc_kron_ij(Mat,PetscScalar,int,int,int,int,int);
void _add_to_PETSc_kron_comb_vec(Mat,PetscScalar,int,int,op_type,int,int,int,int,
                                 int,int,int,int);
void _add_to_PETSc_kron_lin2(Mat,PetscScalar,operator,operator);
void _add_to_PETSc_kron_lin2_comb(Mat,PetscScalar,int,int);

void _add_to_dense_kron(PetscScalar,int,int,op_type,int);
void _add_to_dense_kron_comb(PetscScalar,int,int,op_type,int,int,int,
                               op_type,int);
void _add_to_dense_kron_comb_vec(PetscScalar,int,int,op_type,int,int,int,int);
void _add_to_dense_kron_ij(PetscScalar,int,int,int,int,int);
void _add_PETSc_DM_kron_ij(PetscScalar,Mat,Mat,int,int,int,int,int);
void _mult_PETSc_init_DM(Mat,Mat,double);
void _add_to_PETSc_kron_lin_mat(Mat,PetscScalar,Mat,int,int);



#endif
