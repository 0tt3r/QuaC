#ifndef QSYSTEM_P_H_
#define QSYSTEM_P_H_

#define MAX_GATES 100 // Consider not making this a define

//Defines for SPRNG
#define SIMPLE_SPRNG		/* simple interface                        */
#define USE_MPI			/* use MPI to find number of processes     */


struct qsystem;
struct mat_term;
struct qvec;
struct custom_gate_data;

typedef enum {
              TD_LINDBLAD = -2,
              TD_HAM  = -1,
              HAM = 1,
              LINDBLAD  = 2
} mat_term_type;

typedef enum {
              DENSITY_MATRIX = -1,
              WAVEFUNCTION  = 1,
              WF_ENSEMBLE   = 2
} qvec_type;

typedef struct {
  PetscInt num_ops;
  mat_term_type my_term_type;
  PetscScalar a;
  Mat mat_A;
  PetscScalar (*time_dep_func)(double);
  operator *ops;
} mat_term;

typedef struct {
  PetscScalar gate_data[4][4];
} custom_gate_data;

typedef struct circuit{
  PetscInt num_gates,gate_list_size,current_gate,num_layers,current_layer;
  PetscReal start_time;
  struct quantum_gate_struct *gate_list;
  struct gate_layer_struct *layer_list;
} circuit;




typedef struct qvec{
  qvec_type my_type;
  PetscBool ens_spawned;
  PetscInt n,Istart,Iend,total_levels,*hspace_dims,ndims_hspace,n_ops,n_ensemble,ens_i;
  Vec data;
  Vec *ens_datas;
} *qvec;


typedef struct qsystem{
  PetscInt num_time_indep,num_time_dep;
  PetscInt alloc_time_indep,alloc_time_dep;

  mat_term *time_indep,*time_dep;
  PetscInt hspace_frozen;
  PetscBool dm_equations,mcwf_solver,time_step_called;

  PetscInt mat_allocated;
  Mat mat_A;

  PetscInt *o_nnz,*d_nnz;

  PetscInt num_subsystems,alloc_subsystems;
  operator *subsystem_list;
  PetscInt total_levels,dim;

  qvec solution_qvec;

  //Distribution info
  PetscInt np,nid;
  PetscInt Istart,Iend,my_num;

  //mcwf related
  PetscReal *rand_number,old_rand;
  PetscInt num_tot_trajs,num_local_trajs,seed,ens_i;
  Vec mcwf_work_vec;

  //For TSEventHandler - 2 is hard coded because we have two events
  PetscErrorCode (*post_event_functions[2])(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*);

  //TS monitor related
  PetscErrorCode (*ts_monitor)(TS,PetscInt,PetscReal,Vec,void*);
  void *ts_ctx;

  //Circuit related
  PetscInt num_circuits,circuit_list_size,current_circuit;
  circuit *circuit_list;

} *qsystem;

PetscErrorCode _sys_MCWF_EventFunction(qsystem,Vec,PetscScalar*);
PetscErrorCode _sys_MCWF_PostEventFunction(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*);

PetscErrorCode _sys_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _sys_PostEventFunction(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*);

void _setup_distribution(PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscInt*);
void _preallocate_qsys_matrix(qsystem);
void _preallocate_op_matrix(Mat*,PetscInt,PetscInt,PetscInt,PetscInt,mat_term_type,PetscBool,PetscBool,PetscInt,operator*);

#endif
