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
void create_wf_ens_sys(qsystem,qvec*);
void create_arb_qvec(qvec*,PetscInt,qvec_type);
void create_arb_qvec_dims(qvec*,PetscInt,PetscInt*,qvec_type);
void _create_vec(Vec*,PetscInt,PetscInt);
void change_qvec_dims(qvec,PetscInt,PetscInt*);

void read_qvec_dm_binary(qvec*,const char[]);
void read_qvec_wf_binary(qvec*,const char[]);

void ptrace_over_list_qvec(qvec,PetscInt,PetscInt*,qvec*);
void _ptrace_over_list_qvec_wf(qvec,PetscInt,PetscInt*,PetscInt,PetscInt*,qvec*);
void _ptrace_over_list_qvec_dm(qvec,PetscInt,PetscInt*,PetscInt,PetscInt*,qvec*);

void assemble_qvec(qvec);
void destroy_qvec(qvec*);

void set_qvec_from_init_excited_op(qsystem,qvec);

void add_to_qvec_fock_op(PetscScalar,qvec,PetscInt,...);

void add_to_qvec_fock_op_list(PetscScalar,qvec,PetscInt,operator*,PetscInt*);
void add_to_qvec(qvec,PetscScalar,...);
void add_to_qvec_loc(qvec,PetscScalar,PetscInt);
void get_qvec_loc_fock_op(qvec,PetscInt*,PetscInt,...);
void get_qvec_loc_fock_op_list(qvec,PetscInt*,PetscInt,operator[],PetscInt[]);
void add_to_wf_ens_loc(qvec,PetscInt,PetscScalar,PetscInt);

void get_qvec_local_idxs(qvec,PetscInt,PetscInt*);

void get_expectation_value_qvec(qvec,PetscScalar*,PetscInt,...);
void get_expectation_value_qvec_list(qvec,PetscScalar*,PetscInt,operator*);
void _get_expectation_value_wf(qvec,PetscScalar*,PetscInt,operator*);
void _get_expectation_value_dm(qvec,PetscScalar*,PetscInt,operator*);
void _get_expectation_value_wf_ens(qvec,PetscScalar*,PetscInt,operator*);

void qvec_mat_mult(Mat,qvec);
void loqd_sparse_mat_qvec(char[],Mat*,qvec);
void get_fidelity_qvec(qvec,qvec,PetscReal*,PetscReal*);
void _get_fidelity_wf_wf(Vec,Vec,PetscReal*);
void _get_fidelity_dm_dm(Vec,Vec,PetscReal*);
void _get_fidelity_dm_wf(Vec,Vec,PetscReal*);

void get_superfidelity_qvec(qvec,qvec,PetscReal*);
void  _get_superfidelity_dm_dm(Vec,Vec,PetscReal*);

void print_qvec_file(qvec,char[]);
void print_dm_qvec_file(qvec,char[]);
void print_wf_qvec_file(qvec,char[]);

void get_log_xeb_fidelity(qvec,qvec,PetscReal*,PetscReal*);
void get_linear_xeb_fidelity(qvec,qvec,PetscReal*,PetscReal*);
void get_log_xeb_fidelity_probs(PetscReal*,PetscReal*,PetscInt,PetscReal*,PetscReal*,PetscInt,PetscInt,PetscReal*,PetscReal*);
void get_linear_xeb_fidelity_probs(PetscReal*,PetscReal*,PetscInt,PetscReal*,PetscReal*,PetscInt,PetscInt,PetscReal*,PetscReal*);
void get_bitstring_probs(qvec,PetscInt*,PetscReal**,PetscReal**);
void get_hog_score_fidelity_probs(PetscReal*,PetscReal*,PetscInt,PetscReal*,PetscReal*,PetscInt,PetscInt,PetscReal*,PetscReal*);
void get_hog_score_fidelity(qvec,qvec,PetscReal*,PetscReal*);
void get_hog_score_probs(PetscReal*,PetscReal*,PetscInt,PetscReal*,PetscReal*,PetscInt,PetscInt,PetscReal*,PetscReal*);
void get_hog_score(qvec,qvec,PetscReal*,PetscReal*);
void _get_bitstring_probs_wf(qvec,PetscInt*,PetscReal**);
void _get_bitstring_probs_dm(qvec,PetscInt*,PetscReal**);
void _get_bitstring_probs_wf_ens(qvec,PetscInt*,PetscReal**,PetscReal**);

void print_qvec(qvec);
void print_dm_qvec(qvec);
void print_wf_qvec(qvec);
void print_wf_ens_i_qvec(qvec);

void get_wf_element_qvec(qvec,PetscInt,PetscScalar*);
void get_wf_element_qvec_local(qvec,PetscInt,PetscScalar*);
void get_dm_element_qvec(qvec,PetscInt,PetscInt,PetscScalar*);
void get_dm_element_qvec_local(qvec,PetscInt,PetscInt,PetscScalar*);
void _get_qvec_element_local(qvec,PetscInt,PetscScalar*);

void get_wf_element_ens_i_qvec_local(qvec,PetscInt,PetscInt,PetscScalar*);

void check_qvec_consistent(qvec,qvec);
void check_qvec_equal(qvec,qvec,PetscBool*);
void copy_qvec(qvec,qvec);
void copy_qvec_wf_to_dm(qvec,qvec);
void get_hilbert_schmidt_dist_qvec(qvec,qvec,PetscReal*);
void sqrt_mat(Mat);

#endif
