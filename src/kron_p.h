#ifndef KRON_P_H_
#define KRON_P_H_

#include "operators_p.h"
#include "operators.h"
#include "qsystem.h"

typedef enum {
              TENSOR_IG=-1,
              TENSOR_GG=0,
              TENSOR_GI=1
} tensor_control_enum;

long   _get_loop_limit(op_type,int);
PetscScalar _get_val_in_subspace(long,op_type,int,long*,long*);

void _get_val_j_ops(PetscScalar*,PetscInt*,PetscInt,operator*,tensor_control_enum);

void _get_val_j_from_global_i(PetscInt,operator,PetscInt*,PetscScalar*,tensor_control_enum);
void _get_val_j_from_global_i_vec_vec(PetscInt,operator,operator,PetscInt*,
                                      PetscScalar*,tensor_control_enum);

void _add_ops_to_mat(PetscScalar,Mat,mat_term_type,PetscInt,PetscInt,operator*);
void _add_ops_to_mat_ham_only(PetscScalar,Mat,PetscInt,operator*);
void _add_ops_to_mat_ham(PetscScalar,Mat,PetscInt,operator*);
void _add_ops_to_mat_lin(PetscScalar,Mat,PetscInt,operator*);

void _count_ops_in_mat(PetscInt*,PetscInt*,PetscInt,PetscInt,Mat,
                       mat_term_type,PetscInt,PetscInt,operator*);
void _count_ops_in_mat_ham_only(PetscInt*,PetscInt*,PetscInt,PetscInt,
                                Mat,PetscInt,operator*);
void _count_ops_in_mat_ham(PetscInt*,PetscInt*,PetscInt,PetscInt,
                           Mat,PetscInt,operator*);
void _count_ops_in_mat_lin(PetscInt*,PetscInt*,PetscInt,PetscInt,
                           Mat,PetscInt,operator*);

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
