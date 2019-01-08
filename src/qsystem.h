#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <petsc.h>

struct qsystem;
struct mat_term;

typedef enum {
              TD_LINDBLAD = -2,
              TD_HAM  = -1,
              HAM = 1,
              LINDBLAD  = 2
} mat_term_type;

typedef struct {
  PetscInt num_ops;
  mat_term_type my_term_type;
  PetscScalar a;
  PetscScalar (*time_dep_func)(double);
  operator *ops;
} mat_term;

typedef struct qsystem{
  PetscInt num_time_indep,num_time_dep;
  PetscInt alloc_time_indep,alloc_time_dep;

  mat_term *time_indep,*time_dep;
  PetscInt hspace_frozen;
  PetscBool dm_equations;
  Mat mat_A;

  PetscInt *o_nnz,*d_nnz;

  PetscInt num_subsystems,alloc_subsystems;
  operator *subsystem_list;
  PetscInt total_levels,dim;

  //Distribution info
  PetscInt np,nid;
  PetscInt Istart,Iend,my_num;

  PetscErrorCode (*ts_monitor)(TS,PetscInt,PetscReal,Vec,void*);

} *qsystem;

void construct_matrix(qsystem);
void _setup_distribution(qsystem);
void _preallocate_matrix(qsystem);
void initialize_system(qsystem*);
void destroy_system(qsystem*);

void create_op_sys(qsystem,PetscInt,operator*);
void destroy_op_sys(operator*);
void _create_single_op(PetscInt,PetscInt,op_type,operator*);

void add_ham_term(qsystem,PetscScalar,PetscInt,...);
void add_lin_term(qsystem,PetscScalar,PetscInt,...);
void add_ham_term_time_dep(qsystem,PetscScalar,PetscScalar(*)(double),PetscInt,...);
void add_lin_term_time_dep(qsystem,PetscScalar,PetscScalar(*)(double),PetscInt,...);

void create_vector_sys(qsystem,Vec*);
void create_dm_sys(qsystem,Vec*);
void create_wf_sys(qsystem,Vec*);
void _create_vec(Vec*,PetscInt,PetscInt);

void time_step_sys(qsystem,Vec,PetscReal,PetscReal,PetscReal,PetscInt);
void set_ts_monitor_sys(qsystem,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*));

#endif
