#include "kron_p.h" //Includes petscmat.h and operators_p.h
#include "quac_p.h"
#include "operators.h"
#include "qsystem.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

PetscErrorCode _RHS_time_dep_ham_sys(TS,PetscReal,Vec,Mat,Mat,void*); // Move to header?

void initialize_system(qsystem *qsys){
  qsystem temp = NULL;
  PetscInt num_init_alloc = 25;
  int tmp_nid,tmp_np;
  if (!petsc_initialized){
    if (nid==0){
      printf("ERROR! You need to call QuaC_initialize before creating\n");
      printf("       any systems!\n");
      exit(0);
    }
  }

  temp = malloc(sizeof(struct qsystem));
  temp->hspace_frozen = 0;
  temp->dm_equations   = 0; //Assume no density matrix at beginning
  temp->total_levels   = 1;
  temp->dim = 1;

  temp->num_subsystems = 0;
  temp->alloc_subsystems = num_init_alloc;
  temp->subsystem_list = malloc(num_init_alloc*sizeof(struct operator));
  //temp->stiff_solver could go here if we used it

  //Alloc some space for the mat terms initially
  temp->num_time_indep = 0;
  temp->num_time_dep = 0;

  temp->alloc_time_dep = num_init_alloc;
  temp->alloc_time_indep = num_init_alloc;

  temp->time_indep = malloc(num_init_alloc*sizeof(mat_term));
  temp->time_dep = malloc(num_init_alloc*sizeof(mat_term));

  temp->ts_monitor = NULL;
  //Distribution info
  /* Get core's id */
  MPI_Comm_rank(PETSC_COMM_WORLD,&(tmp_nid));
  /* Get number of processors */
  MPI_Comm_size(PETSC_COMM_WORLD,&(tmp_np));
  temp->nid = tmp_nid;
  temp->np = tmp_np;
  temp->my_num = -1;
  temp->Istart = -1;
  temp->Iend   = -1;


  *qsys = temp;
  return;
}

void destroy_system(qsystem *qsys){
  PetscInt i;
  free((*qsys)->subsystem_list);
  for(i=0;i<(*qsys)->num_time_dep;i++){
    free((*qsys)->time_dep[i].ops);
  }
  free((*qsys)->time_dep);

  for(i=0;i<(*qsys)->num_time_indep;i++){
    free((*qsys)->time_indep[i].ops);
  }
  free((*qsys)->time_indep);
  free((*qsys)->o_nnz);
  free((*qsys)->d_nnz);
  if ((*qsys)->mat_allocated) MatDestroy(&((*qsys)->mat_A));
  free((*qsys));
}

void _create_single_op(PetscInt total_levels,PetscInt number_of_levels,
                       op_type my_op_type,operator *op){
  operator temp = NULL;
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = my_op_type;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;
  *op           = temp;

  return;
}

void create_op_sys(qsystem sys,PetscInt number_of_levels,operator *new_op){

  operator temp = NULL,temp1 = NULL;
  operator *tmp_list;
  PetscInt i=0;
  if (sys->hspace_frozen){
    if (nid==0){
      printf("ERROR! You cannot add more operators after\n");
      printf("       the creating a dm or constructing the matrix!\n");
      exit(0);
    }
  }

  /* First make the annihilation operator */
  _create_single_op(sys->total_levels,number_of_levels,LOWER,&temp);
  *new_op           = temp;
  temp1             = temp;

  /* Make creation operator */
  _create_single_op(sys->total_levels,number_of_levels,RAISE,&temp);
  temp->dag         = temp1; //Point dagger operator to LOWER op
  (*new_op)->dag    = temp;

  /* Make number operator */
  _create_single_op(sys->total_levels,number_of_levels,NUMBER,&temp);
  (*new_op)->n      = temp;

  /* Make identity operator */
  _create_single_op(sys->total_levels,number_of_levels,IDENTITY,&temp);
  (*new_op)->eye    = temp;

  /* Make SIGMA_X operator (only valid for qubits, made for every system) */
  _create_single_op(sys->total_levels,number_of_levels,SIGMA_X,&temp);
  (*new_op)->sig_x      = temp;

  /* Make SIGMA_Z operator (only valid for qubits, made for every system) */
  _create_single_op(sys->total_levels,number_of_levels,SIGMA_Z,&temp);
  (*new_op)->sig_z      = temp;

  /* Make SIGMA_Y operator (only valid for qubits, made for every system) */
  _create_single_op(sys->total_levels,number_of_levels,SIGMA_Y,&temp);
  (*new_op)->sig_y      = temp;

  /* Increase total_levels */
  sys->total_levels = sys->total_levels*number_of_levels;

  /* Add to list */
  if (sys->num_subsystems==sys->alloc_subsystems){
    /* Realloc array */
    sys->alloc_subsystems = 2*sys->alloc_subsystems;
    tmp_list = malloc(sys->num_subsystems*sizeof(struct operator));
    for (i=0;i<sys->num_subsystems;i++){
      tmp_list[i] = sys->subsystem_list[i];
    }
    free(sys->subsystem_list);
    sys->subsystem_list = malloc(sys->alloc_subsystems*sizeof(struct operator));
    for (i=0;i<sys->num_subsystems;i++){
      sys->subsystem_list[i] = tmp_list[i];
    }
    free(tmp_list);
  }

  sys->subsystem_list[sys->num_subsystems] = (*new_op);
  sys->num_subsystems++;

  return;
}


void destroy_op_sys(operator *op){

  free((*op)->dag);
  free((*op)->n);
  free((*op)->eye);
  free((*op)->sig_x);
  free((*op)->sig_z);
  free((*op)->sig_y);
  free(*op);

  return;
}

void add_ham_term(qsystem sys,PetscScalar a,PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  sys->time_indep[sys->num_time_indep].my_term_type = HAM;
  sys->time_indep[sys->num_time_indep].a = a;
  sys->time_indep[sys->num_time_indep].num_ops = num_ops;
  sys->time_indep[sys->num_time_indep].ops = malloc(num_ops*sizeof(struct operator));

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_indep[sys->num_time_indep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);
  sys->num_time_indep = sys->num_time_indep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void add_lin_term(qsystem sys,PetscScalar a,PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  sys->dm_equations = 1;//Lindblad equation
  sys->time_indep[sys->num_time_indep].my_term_type = LINDBLAD;
  sys->time_indep[sys->num_time_indep].a = a;
  sys->time_indep[sys->num_time_indep].num_ops = num_ops;
  sys->time_indep[sys->num_time_indep].ops = malloc(num_ops*sizeof(struct operator));

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_indep[sys->num_time_indep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);
  sys->num_time_indep = sys->num_time_indep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void add_ham_term_time_dep(qsystem sys,PetscScalar a,PetscScalar (*time_dep_func)(double),
                           PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  sys->time_dep[sys->num_time_dep].my_term_type = TD_HAM;
  sys->time_dep[sys->num_time_dep].a = a;
  sys->time_dep[sys->num_time_dep].num_ops = num_ops;
  sys->time_dep[sys->num_time_dep].ops = malloc(num_ops*sizeof(struct operator));
  sys->time_dep[sys->num_time_dep].time_dep_func = time_dep_func;

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_dep[sys->num_time_dep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);
  sys->num_time_dep = sys->num_time_dep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void add_lin_term_time_dep(qsystem sys,PetscScalar a,PetscScalar (*time_dep_func)(double),
                           PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  sys->dm_equations = 1;//Lindblad equation
  sys->time_dep[sys->num_time_dep].my_term_type = TD_LINDBLAD;
  sys->time_dep[sys->num_time_dep].a = a;
  sys->time_dep[sys->num_time_dep].num_ops = num_ops;
  sys->time_dep[sys->num_time_dep].ops = malloc(num_ops*sizeof(struct operator));
  sys->time_dep[sys->num_time_dep].time_dep_func = time_dep_func;

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_dep[sys->num_time_dep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);

  sys->num_time_dep = sys->num_time_dep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void construct_matrix(qsystem sys){
  PetscInt    i;
  PetscScalar tmp_a;
  sys->hspace_frozen = 1;

  /* Check to make sure some operators were created */
  if (sys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to create operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a matrix!\n");
    exit(0);
  }

  /* Check to make sure some terms were added*/
  if ((sys->num_time_dep+sys->num_time_indep)==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to add some terms before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a matrix!\n");
    exit(0);
  }

  if(sys->dm_equations){
    //If we had Lindblads or are wanting to timestep a DM, we need to use the larger space
    sys->dim = sys->total_levels*sys->total_levels;
  } else {
    //Just the schrodinger solver
    sys->dim = sys->total_levels;
  }

  _preallocate_matrix(sys);
  sys->mat_allocated = 1;
  //Loop over time independent terms
  for(i=0;i<sys->num_time_indep;i++){
    _add_ops_to_mat(sys->time_indep[i].a,sys->mat_A,sys->time_indep[i].my_term_type,
                    sys->dm_equations,sys->time_indep[i].num_ops,sys->time_indep[i].ops);
  }

  /*
   * Loop over time dependent terms, using 0 as the scalar. This
   * ensures that the matrix will have the correct nonzero structure
   * when we add those time dependent terms
   */
  tmp_a = 0.0;
  for(i=0;i<sys->num_time_dep;i++){
    _add_ops_to_mat(tmp_a,sys->mat_A,sys->time_dep[i].my_term_type,
                    sys->dm_equations,sys->time_dep[i].num_ops,sys->time_dep[i].ops);
  }

  //Loop over diagonal to specifically add 0 to it.
  tmp_a = 0.0;
  for(i=sys->Istart;i<sys->Iend;i++){
    MatSetValue(sys->mat_A,i,i,tmp_a,ADD_VALUES);
  }
  MatAssemblyBegin(sys->mat_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sys->mat_A,MAT_FINAL_ASSEMBLY);

  return;
}

void _setup_distribution(qsystem sys){
  PetscInt    count,remainder,tmp_nid,tmp_np;

  if (sys->my_num<0){
    count = sys->dim / sys->np;
    remainder = sys->dim % sys->np;

    if (sys->nid < remainder){
      sys->Istart = sys->nid * (count + 1);
      sys->Iend   = sys->Istart + count + 1;
      sys->my_num = count + 1;
    } else {
      sys->Istart = sys->nid * count + remainder;
      sys->Iend   = sys->Istart + (count - 1) + 1;
      sys->my_num = count;
    }
    total_levels = sys->total_levels; //FIXME: Hack to be compatible with old code
  }

  return;
}

void _preallocate_matrix(qsystem sys){
  PetscInt    i,count,remainder,tmp_nid,tmp_np;

  _setup_distribution(sys);
  MatCreate(PETSC_COMM_WORLD,&(sys->mat_A));
  MatSetType(sys->mat_A,MATMPIAIJ);
  MatSetSizes(sys->mat_A,sys->my_num,sys->my_num,sys->dim,sys->dim);
  MatSetFromOptions(sys->mat_A);

  sys->o_nnz = malloc(sys->my_num*sizeof(PetscInt));
  sys->d_nnz = malloc(sys->my_num*sizeof(PetscInt));

  for(i=0;i<sys->my_num;i++){
    sys->o_nnz[i] = 0;
    //Start with assuming the diagonal has values
    //And a bit of a buffer incase we are adding many in the same spot at one time
    sys->d_nnz[i] = 1;
  }


  /*
   * This counting can double count. For instance, if you added
   * several copies of the same exact term, it counts them separately,
   * counting the 1,2 element many times, for instance.
   *
   * FIXME: Possible minor solution: Have a counter which checks if basic
   * operators of a certain nonzero pattern have yet been added. Would fix
   * double counting of basic operators at least.
   */
  //Loop over time independent terms
  for(i=0;i<sys->num_time_indep;i++){
    _count_ops_in_mat(sys->d_nnz,sys->o_nnz,sys->Istart,sys->Iend,
                      sys->mat_A,sys->time_indep[i].my_term_type,
                      sys->dm_equations,sys->time_indep[i].num_ops,sys->time_indep[i].ops);
  }

  //Loop over time dependent terms
  for(i=0;i<sys->num_time_dep;i++){
    _count_ops_in_mat(sys->d_nnz,sys->o_nnz,sys->Istart,sys->Iend,
                      sys->mat_A,sys->time_dep[i].my_term_type,
                      sys->dm_equations,sys->time_dep[i].num_ops,sys->time_dep[i].ops);

  }

  //-1s are ignored
  MatMPIAIJSetPreallocation(sys->mat_A,-1,sys->d_nnz,-1,sys->o_nnz);
  return;
}



/*
 *
 * set_ts_monitor accepts a user function which can calculate observables, print output, etc
 * at each time step.
 *
 * Inputs:
 *      PetscErrorCode *monitor - function pointer for user ts_monitor function
 *
 */
void set_ts_monitor_sys(qsystem sys,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*)){
  sys->ts_monitor = (*monitor);
  return;
}

/*
 * time_step solves for the time_dependence of the system
 * that was previously setup using the add_to_ham and add_lin
 * routines. Solver selection and parameters can be controlled via PETSc
 * command line options. Default solver is TSRK3BS
 *
 * Inputs:
 *       Vec     x:       The density matrix, with appropriate inital conditions
 *       double dt:       initial timestep. For certain explicit methods, this timestep
 *                        can be changed, as those methods have adaptive time steps
 *       double time_max: the maximum time to integrate to
 *       int steps_max:   max number of steps to take
 */
void time_step_sys(qsystem sys,qvec x, PetscReal init_time, PetscReal time_max,
               PetscReal dt,PetscInt steps_max){
  PetscViewer    mat_view;
  TS             ts; /* timestepping context */
  Mat            AA;
  PetscInt       steps;
  PetscLogStagePop();
  PetscLogStagePush(solve_stage);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*
   *       Create the timestepping solver and set various options       *
   *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
   * Create timestepping solver context
   */
  TSCreate(PETSC_COMM_WORLD,&ts);
  TSSetProblemType(ts,TS_LINEAR);

  /*
   * Set function to get information at every timestep
   */
  if (sys->ts_monitor!=NULL){
    TSMonitorSet(ts,sys->ts_monitor,NULL,NULL);
  }

  /*
   * Set up ODE system
   */
  TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);
  if (sys->num_time_dep>0){

    //Duplicate matrix for time dependent runs
    MatDuplicate(sys->mat_A,MAT_COPY_VALUES,&AA);
    MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);
    TSSetRHSJacobian(ts,AA,AA,_RHS_time_dep_ham_sys,sys);

  } else {
    //Time indep
    TSSetRHSJacobian(ts,sys->mat_A,sys->mat_A,TSComputeRHSJacobianConstant,sys);
  }
  /* Print information about the matrix. */
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,NULL,&mat_view);
  PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_INFO);
  /* PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_MATLAB); */
  MatView(sys->mat_A,mat_view);
  PetscViewerPopFormat(mat_view);
  PetscViewerDestroy(&mat_view);



  /*
   * Set default options, can be changed at runtime
   */
  TSSetTimeStep(ts,dt);
  TSSetMaxSteps(ts,steps_max);
  TSSetMaxTime(ts,time_max);
  TSSetTime(ts,init_time);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);

  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK3BS);

  TSSetFromOptions(ts);
  TSSolve(ts,x->data);
  TSGetStepNumber(ts,&steps);

  /* Free work space */
  TSDestroy(&ts);
  if(sys->num_time_dep>0){
    MatDestroy(&AA);
  }

  PetscLogStagePop();
  PetscLogStagePush(post_solve_stage);

  return;
}

/*
 * _RHS_time_dep_ham_p adds the (user created) time dependent functions
 * to the time independent hamiltonian. It is used internally by PETSc
 * during time stepping.
 */

PetscErrorCode _RHS_time_dep_ham_sys(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx){
  double time_dep_val;
  PetscScalar time_dep_scalar;
  int i,j;
  operator op;
  qsystem sys = (qsystem) ctx;

  MatZeroEntries(AA);
  MatCopy(sys->mat_A,AA,SAME_NONZERO_PATTERN);

  for(i=0;i<sys->num_time_dep;i++){
    time_dep_scalar = sys->time_dep[i].a*sys->time_dep[i].time_dep_func(t);
    _add_ops_to_mat(time_dep_scalar,AA,sys->time_dep[i].my_term_type,
                    sys->dm_equations,sys->time_dep[i].num_ops,sys->time_dep[i].ops);
  }

  MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);

  if(AA!=BB) {
    MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);
  }

  PetscFunctionReturn(0);
}


