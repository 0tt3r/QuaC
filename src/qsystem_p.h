#ifndef QSYSTEM_P_H_
#define QSYSTEM_P_H_

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


struct qvec;

typedef enum {
              DENSITY_MATRIX = -1,
              WAVEFUNCTION  = 1
} qvec_type;

typedef struct qvec{
  qvec_type my_type;
  PetscInt n,Istart,Iend;
  Vec data;
} *qvec;

void _setup_distribution(qsystem);
void _preallocate_matrix(qsystem);

#endif
