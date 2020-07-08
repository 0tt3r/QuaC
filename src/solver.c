
#include "operators_p.h"
#include "operators.h"
#include "qsystem.h"
#include "solver.h"
#include "kron_p.h"
#include "quac_p.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "error_correction.h"
#include <stdlib.h>
#include <stdio.h>

static PetscReal default_rtol     = 1e-11;
static PetscInt  default_restart  = 100;
static int       stab_added       = 0;
static int       matrix_assembled = 0;


PetscErrorCode _RHS_time_dep_ham(TS,PetscReal,Vec,Mat,Mat,void*); // Move to header?
PetscErrorCode _RHS_time_dep_ham_p(TS,PetscReal,Vec,Mat,Mat,void*); // Move to header?

PetscErrorCode (*_ts_monitor)(TS,PetscInt,PetscReal,Vec,void*) = NULL;
void          *_tsctx;
PetscErrorCode _Normalize_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
PetscErrorCode _Normalize_PostEventFunction(TS,PetscInt,PetscInt[],PetscReal,Vec,void*);
/*
 * steady_state solves for the steady_state of the system
 * that was previously setup using the add_to_ham and add_lin
 * routines. Solver selection and parameterscan be controlled via PETSc
 * command line options.
 */
void steady_state(Vec x){
  PetscViewer    mat_view;
  PC             pc;
  Vec            b;
  KSP            ksp; /* linear solver context */
  PetscInt       row,col,its,j,i,Istart,Iend;
  PetscScalar    mat_tmp;
  long           dim;
  int            num_pop;
  double         *populations;
  Mat            solve_A;

  if (_lindblad_terms) {
    dim = total_levels*total_levels;
    solve_A = full_A;
    if (nid==0) {
      printf("Lindblad terms found, using Lindblad solver.");
    }
  } else {
    if (nid==0) {
      printf("Warning! Steady state not supported for Schrodinger.\n");
      printf("         Defaulting to (less efficient) Lindblad Solver\n");
      exit(0);
    }
    dim = total_levels*total_levels;
    solve_A = ham_A;
  }
  if (!stab_added){
    if (nid==0) printf("Adding stabilization...\n");
    /*
     * Add elements to the matrix to make the normalization work
     * I have no idea why this works, I am copying it from qutip
     * We add 1.0 in the 0th spot and every n+1 after
     */
    if (nid==0) {
      row = 0;
      for (i=0;i<total_levels;i++){
        col = i*(total_levels+1);
        mat_tmp = 1.0 + 0.*PETSC_i;
        MatSetValue(full_A,row,col,mat_tmp,ADD_VALUES);
      }

      /* Print dense ham, if it was asked for */
      if (_print_dense_ham){
        FILE *fp_ham;

        fp_ham = fopen("ham","w");

        if (nid==0){
          for (i=0;i<total_levels;i++){
            for (j=0;j<total_levels;j++){
              fprintf(fp_ham,"%e %e ",PetscRealPart(_hamiltonian[i][j]),PetscImaginaryPart(_hamiltonian[i][j]));
            }
            fprintf(fp_ham,"\n");
          }
        }
        fclose(fp_ham);
        for (i=0;i<total_levels;i++){
          free(_hamiltonian[i]);
        }
        free(_hamiltonian);
        _print_dense_ham = 0;
      }
    }
    stab_added = 1;
  }

  //  if (!matrix_assembled) {
    MatGetOwnershipRange(full_A,&Istart,&Iend);
    /*
     * Explicitly add 0.0 to all diagonal elements;
     * this fixes a 'matrix in wrong state' message that PETSc
     * gives if the diagonal was never initialized.
     */
    if (nid==0) printf("Adding 0 to diagonal elements...\n");
    for (i=Istart;i<Iend;i++){
      mat_tmp = 0 + 0.*PETSC_i;
      MatSetValue(full_A,i,i,mat_tmp,ADD_VALUES);
    }


    /* Tell PETSc to assemble the matrix */
    MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);
    if (nid==0) printf("Matrix Assembled.\n");
    matrix_assembled = 1;
    //  }
  /* Print information about the matrix. */
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,NULL,&mat_view);
  PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_INFO);
  MatView(full_A,mat_view);
  PetscViewerPopFormat(mat_view);
  PetscViewerDestroy(&mat_view);
  /*
   * Create parallel vectors.
   * - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
   * we specify only the vector's global
   * dimension; the parallel partitioning is determined at runtime.
   * - Note: We form 1 vector from scratch and then duplicate as needed.
   */
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,dim);
  VecSetFromOptions(b);

  //  VecDuplicate(b,&x); Assume x is passed in

  /*
   * Set rhs, b, and solution, x to 1.0 in the first
   * element, 0.0 elsewhere.
   */
  VecSet(b,0.0);
  VecSet(x,0.0);

  if(nid==0) {
    row = 0;
    mat_tmp = 1.0 + 0.0*PETSC_i;
    VecSetValue(x,row,mat_tmp,INSERT_VALUES);
    VecSetValue(b,row,mat_tmp,INSERT_VALUES);
  }

  /* Assemble x and b */
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

  VecAssemblyBegin(b);
  VecAssemblyEnd(b);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*
     *           Create the linear solver and set various options         *
     *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
   * Create linear solver context
   */
  KSPCreate(PETSC_COMM_WORLD,&ksp);

  /*
   * Set operators. Here the matrix that defines the linear system
   * also serves as the preconditioning matrix.
   */
  KSPSetOperators(ksp,full_A,full_A);

  /*
   * Set good default options for solver
   */
  /* relative tolerance */
  KSPSetTolerances(ksp,default_rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

  /* bjacobi preconditioner */
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCASM);

  /* gmres solver with 100 restart*/
  KSPSetType(ksp,KSPGMRES);
  KSPGMRESSetRestart(ksp,default_restart);
  /*
   * Set runtime options, e.g.,
   *     -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
   */
  KSPSetFromOptions(ksp);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (nid==0) printf("KSP set. Solving for steady state...\n");
  KSPSolve(ksp,b,x);

  num_pop = get_num_populations();
  populations = malloc(num_pop*sizeof(double));
  get_populations(x,&populations);
  if(nid==0){
    printf("Final populations: ");
    for(i=0;i<num_pop;i++){
      printf(" %e ",populations[i]);
    }
    printf("\n");
  }

  KSPGetIterationNumber(ksp,&its);

  PetscPrintf(PETSC_COMM_WORLD,"Iterations %D\n",its);

  /* Free work space */
  KSPDestroy(&ksp);
  //  VecDestroy(&x);
  VecDestroy(&b);

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
void time_step(Vec x, PetscReal init_time, PetscReal time_max,PetscReal dt,PetscInt steps_max){
  PetscViewer    mat_view;
  TS             ts; /* timestepping context */
  PetscInt       i,j,Istart,Iend,steps,row,col;
  PetscScalar    mat_tmp;
  PetscReal      tmp_real;
  Mat            AA;
  PetscInt       nevents,direction;
  PetscBool      terminate;
  operator       op;
  int            num_pop;
  double         *populations;
  Mat            solve_A,solve_stiff_A;


  PetscLogStagePop();
  PetscLogStagePush(solve_stage);
  if (_lindblad_terms) {
    if (nid==0) {
      printf("Lindblad terms found, using Lindblad solver.\n");
    }
    solve_A = full_A;
    if (_stiff_solver) {
      if(nid==0) printf("ERROR! Lindblad-stiff solver untested.");
      exit(0);
    }
  } else {
    if (nid==0) {
      printf("No Lindblad terms found, using (more efficient) Schrodinger solver.\n");
    }
    solve_A = ham_A;
    solve_stiff_A = ham_stiff_A;
    if (_num_time_dep&&_stiff_solver) {
      if(nid==0) printf("ERROR! Schrodinger-stiff + timedep solver untested.");
      exit(0);
    }
  }

  /* Possibly print dense ham. No stabilization is needed? */
  if (nid==0) {
    /* Print dense ham, if it was asked for */
    if (_print_dense_ham){
      FILE *fp_ham;
      fp_ham = fopen("ham","w");

      if (nid==0){
        for (i=0;i<total_levels;i++){
          for (j=0;j<total_levels;j++){
            fprintf(fp_ham,"%e %e ",PetscRealPart(_hamiltonian[i][j]),PetscImaginaryPart(_hamiltonian[i][j]));
          }
          fprintf(fp_ham,"\n");
        }
      }
      fclose(fp_ham);
      for (i=0;i<total_levels;i++){
        free(_hamiltonian[i]);
      }
      free(_hamiltonian);
      _print_dense_ham = 0;
    }
  }


  /* Remove stabilization if it was previously added */
  if (stab_added){
    if (nid==0) printf("Removing stabilization...\n");
    /*
     * We add 1.0 in the 0th spot and every n+1 after
     */
    if (nid==0) {
      row = 0;
      for (i=0;i<total_levels;i++){
        col = i*(total_levels+1);
        mat_tmp = -1.0 + 0.*PETSC_i;
        MatSetValue(full_A,row,col,mat_tmp,ADD_VALUES);
      }
    }
  }

  MatGetOwnershipRange(solve_A,&Istart,&Iend);
  /*
   * Explicitly add 0.0 to all diagonal elements;
   * this fixes a 'matrix in wrong state' message that PETSc
   * gives if the diagonal was never initialized.
   */
  //if (nid==0) printf("Adding 0 to diagonal elements...\n");
  for (i=Istart;i<Iend;i++){
    mat_tmp = 0 + 0.*PETSC_i;
    MatSetValue(solve_A,i,i,mat_tmp,ADD_VALUES);
  }
  if(_stiff_solver){
    MatGetOwnershipRange(solve_stiff_A,&Istart,&Iend);
    for (i=Istart;i<Iend;i++){
      mat_tmp = 0 + 0.*PETSC_i;
      MatSetValue(solve_stiff_A,i,i,mat_tmp,ADD_VALUES);
    }

  }

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
  if (_ts_monitor!=NULL){
    TSMonitorSet(ts,_ts_monitor,_tsctx,NULL);
  }
  /*
   * Set up ODE system
   */

  TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);

  if(_stiff_solver) {
    /* TSSetIFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL); */
    if (nid==0) {
      printf("Stiff solver not implemented!\n");
      exit(0);
    }
    if(nid==0) printf("Using stiff solver - TSROSW\n");
  }

  if(_num_time_dep+_num_time_dep_lin) {

    for(i=0;i<_num_time_dep;i++){
      tmp_real = 0.0;
      _add_ops_to_mat_ham(tmp_real,solve_A,total_levels,_time_dep_list[i].num_ops,_time_dep_list[i].ops);
    }

    for(i=0;i<_num_time_dep_lin;i++){
      tmp_real = 0.0;
      _add_ops_to_mat_lin(tmp_real,solve_A,total_levels,_time_dep_list_lin[i].num_ops,_time_dep_list_lin[i].ops);
    }

    /* Tell PETSc to assemble the matrix */
    MatAssemblyBegin(solve_A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(solve_A,MAT_FINAL_ASSEMBLY);
    if (nid==0) printf("Matrix Assembled.\n");

    MatDuplicate(solve_A,MAT_COPY_VALUES,&AA);
    MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);

    TSSetRHSJacobian(ts,AA,AA,_RHS_time_dep_ham_p,NULL);
  } else {
    /* Tell PETSc to assemble the matrix */
    MatAssemblyBegin(solve_A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(solve_A,MAT_FINAL_ASSEMBLY);
    if (_stiff_solver){
      MatAssemblyBegin(solve_stiff_A,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(solve_stiff_A,MAT_FINAL_ASSEMBLY);
      /* TSSetIJacobian(ts,solve_stiff_A,solve_stiff_A,TSComputeRHSJacobianConstant,NULL); */
      if (nid==0) {
        printf("Stiff solver not implemented!\n");
        exit(0);
      }
    }
    if (nid==0) printf("Matrix Assembled.\n");
    TSSetRHSJacobian(ts,solve_A,solve_A,TSComputeRHSJacobianConstant,NULL);
  }

  /* Print information about the matrix. */
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,NULL,&mat_view);
  PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_INFO);
  /* PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_MATLAB); */
  /* MatView(solve_A,mat_view); */

  /* PetscInt          ncols; */
  /* const PetscInt    *cols; */
  /* const PetscScalar *vals; */

  /* for(i=0;i<total_levels*total_levels;i++){ */
  /*   MatGetRow(solve_A,i,&ncols,&cols,&vals); */
  /*   for (j=0;j<ncols;j++){ */

  /*     if(PetscAbsComplex(vals[j])>1e-5){ */
  /*       printf("%d %d %lf %lf\n",i,cols[j],vals[j]); */
  /*     } */
  /*   } */
  /*   MatRestoreRow(solve_A,i,&ncols,&cols,&vals); */
  /* } */

  if(_stiff_solver){
    MatView(solve_stiff_A,mat_view);
  }
  PetscViewerPopFormat(mat_view);
  PetscViewerDestroy(&mat_view);

  TSSetTimeStep(ts,dt);

  /*
   * Set default options, can be changed at runtime
   */

  TSSetMaxSteps(ts,steps_max);
  TSSetMaxTime(ts,time_max);
  TSSetTime(ts,init_time);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
  if (_stiff_solver) {
    TSSetType(ts,TSROSW);
  } else {
    TSSetType(ts,TSRK);
    TSRKSetType(ts,TSRK3BS);
  }

  /* If we have gates to apply, set up the event handler. */
  if (_num_quantum_gates > 0) {
    nevents   =  1; //Only one event for now (did we cross a gate?)
    direction = -1; //We only want to count an event if we go from positive to negative
    terminate = PETSC_FALSE; //Keep time stepping after we passed our event
    /* Arguments are: ts context, nevents, direction of zero crossing, whether to terminate,
     * a function to check event status, a function to apply events, private data context.
     */
    TSSetEventHandler(ts,nevents,&direction,&terminate,_QG_EventFunction,_QG_PostEventFunction,NULL);
  }

  if (_num_circuits > 0) {
    nevents   =  1; //Only one event for now (did we cross a gate?)
    direction = -1; //We only want to count an event if we go from positive to negative
    terminate = PETSC_FALSE; //Keep time stepping after we passed our event
    /* Arguments are: ts context, nevents, direction of zero crossing, whether to terminate,
     * a function to check event status, a function to apply events, private data context.
     */
    TSSetEventHandler(ts,nevents,&direction,&terminate,_QC_EventFunction,_QC_PostEventFunction,NULL);
  }

  if (_discrete_ec > 0) {
    nevents   =  1; //Only one event for now (did we cross an ec step?)
    direction = -1; //We only want to count an event if we go from positive to negative
    terminate = PETSC_FALSE; //Keep time stepping after we passed our event
    /* Arguments are: ts context, nevents, direction of zero crossing, whether to terminate,
     * a function to check event status, a function to apply events, private data context.
     */
    TSSetEventHandler(ts,nevents,&direction,&terminate,_DQEC_EventFunction,_DQEC_PostEventFunction,NULL);
  }

  /* if (_lindblad_terms) { */
  /*   nevents   =  1; //Only one event for now (did we cross a gate?) */
  /*   direction =  0; //We only want to count an event if we go from positive to negative */
  /*   terminate = PETSC_FALSE; //Keep time stepping after we passed our event */
  /*   TSSetEventHandler(ts,nevents,&direction,&terminate,_Normalize_EventFunction,_Normalize_PostEventFunction,NULL); */
  /* } */
  TSSetFromOptions(ts);
  TSSolve(ts,x);
  TSGetStepNumber(ts,&steps);

  num_pop = get_num_populations();
  populations = malloc(num_pop*sizeof(double));
  get_populations(x,&populations);
  /* if(nid==0){ */
  /*   printf("Final populations: "); */
  /*   for(i=0;i<num_pop;i++){ */
  /*     printf(" %e ",populations[i]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  /* PetscPrintf(PETSC_COMM_WORLD,"Steps %D\n",steps); */

  /* Free work space */
  TSDestroy(&ts);
  if(_num_time_dep+_num_time_dep_lin){
    MatDestroy(&AA);
  }
  free(populations);
  PetscLogStagePop();
  PetscLogStagePush(post_solve_stage);

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
void set_ts_monitor(PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*)){
  _ts_monitor = (*monitor);
  _tsctx      = NULL;
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
void set_ts_monitor_ctx(PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void *tsctx){
  _ts_monitor = (*monitor);
  _tsctx = tsctx;
}

/*
 * _RHS_time_dep_ham adds the (user created) time dependent functions
 * to the time independent hamiltonian. It is used internally by PETSc
 * during time stepping.
 */

PetscErrorCode _RHS_time_dep_ham(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx){
  double time_dep_val;
  PetscScalar time_dep_scalar;
  int i,j;
  operator op;

  MatZeroEntries(AA);

  MatCopy(full_A,AA,SAME_NONZERO_PATTERN);

  for (i=0;i<_num_time_dep;i++){
    time_dep_val = _time_dep_list[i].time_dep_func(t);
    for(j=0;j<_time_dep_list[i].num_ops;j++){
      op = _time_dep_list[i].ops[j];

      /* Add -i *(I cross H(t)) */
      time_dep_scalar = 0 - time_dep_val*PETSC_i;
      _add_to_PETSc_kron(AA,time_dep_scalar,op->n_before,op->my_levels,
                         op->my_op_type,op->position,total_levels,1,0);

      /* Add i *(H(t)^T cross I) */
      time_dep_scalar = 0 + time_dep_val*PETSC_i;
      _add_to_PETSc_kron(AA,time_dep_scalar,op->n_before,op->my_levels,
                         op->my_op_type,op->position,1,total_levels,1);

    }
    /* Consider putting _time_dep_func and _time_dep_mats in *ctx? */
  }

  MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);
  if(AA!=BB) {
    MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);
  }

  PetscFunctionReturn(0);
}

/*
 * _RHS_time_dep_ham_p adds the (user created) time dependent functions
 * to the time independent hamiltonian. It is used internally by PETSc
 * during time stepping.
 */

PetscErrorCode _RHS_time_dep_ham_p(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx){
  double time_dep_val;
  PetscScalar time_dep_scalar;
  int i,j;
  operator op;

  MatZeroEntries(AA);

  MatCopy(full_A,AA,SAME_NONZERO_PATTERN);

  for (i=0;i<_num_time_dep;i++){
    time_dep_val = _time_dep_list[i].time_dep_func(t);
    _add_ops_to_mat_ham(time_dep_val,AA,total_levels,_time_dep_list[i].num_ops,_time_dep_list[i].ops);
  }

  for (i=0;i<_num_time_dep_lin;i++){
    time_dep_val = _time_dep_list_lin[i].time_dep_func(t);
    _add_ops_to_mat_lin(time_dep_val,AA,total_levels,_time_dep_list_lin[i].num_ops,_time_dep_list_lin[i].ops);
  }

  MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);

  if(AA!=BB) {
    MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);
  }

  PetscFunctionReturn(0);
}

/*
 * EventFunction is one step in Petsc to apply some action if a statement is true.
 * This function ALWAYS triggers,
 */
PetscErrorCode _Normalize_EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx) {
  /* Return 0 to mean we want to trigger this event*/
  fvalue[0] = 0;
  return(0);
}

/*
 * PostEventFunction is the other step in Petsc. If an event has happend, petsc will call this function
 * to apply that event, which, in this case, normalizes the vector.
*/
PetscErrorCode _Normalize_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,void* ctx) {
  PetscErrorCode  ierr;
  ierr = VecNormalize(U,NULL);CHKERRQ(ierr);
  TSSetSolution(ts,U);
  return(0);
}



void g2_correlation(PetscScalar ***g2_values,Vec dm0,PetscInt n_tau,PetscReal tau_max,PetscInt n_st,PetscReal st_max,PetscInt number_of_ops,...){
  TSCtx tsctx;
  PetscReal st_dt,previous_start_time,this_start_time,dt;
  PetscReal tau_t_max,dt_tau;
  PetscInt i,j,dim,steps_max;
  Mat A_star_A,tmp_mat;
  Vec init_dm;
  va_list ap;
  /*Explicitly construct our jump matrix by adding up all of the operators
   * \rho = A \rho A^\dag
   * Vectorized:
   * \rho = (A* \cross A) \rho
   */
  dim = total_levels*total_levels; //Assumes Lindblad

  MatCreate(PETSC_COMM_WORLD,&tmp_mat);
  MatSetType(tmp_mat,MATMPIAIJ);
  MatSetSizes(tmp_mat,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
  MatSetFromOptions(tmp_mat);
  MatMPIAIJSetPreallocation(tmp_mat,4,NULL,4,NULL);

  va_start(ap,number_of_ops);
  //Get A* \cross I
  vadd_ops_to_mat(tmp_mat,1,number_of_ops,ap);
  va_end(ap);


  MatCreate(PETSC_COMM_WORLD,&tsctx.I_cross_A);
  MatSetType(tsctx.I_cross_A,MATMPIAIJ);
  MatSetSizes(tsctx.I_cross_A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
  MatSetFromOptions(tsctx.I_cross_A);
  MatMPIAIJSetPreallocation(tsctx.I_cross_A,4,NULL,4,NULL);
  va_start(ap,number_of_ops);
  //Get I_cross_A
  vadd_ops_to_mat(tsctx.I_cross_A,-1,number_of_ops,ap);
  va_end(ap);


  //Get (A* \cross I) (I \cross A)
  MatMatMult(tmp_mat,tsctx.I_cross_A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A_star_A);
  MatDestroy(&tmp_mat);
  VecDuplicate(dm0,&(tsctx.tmp_dm));
  VecDuplicate(dm0,&(tsctx.tmp_dm2));
  VecDuplicate(dm0,&init_dm);

  //Arbitrary, set this better
  steps_max = 51000;
  dt        = 0.025;

  //Allocate memory for g2
  (*g2_values) = (PetscScalar **)malloc((n_st+1)*sizeof(PetscScalar *));
  for (i=0;i<n_st+1;i++){
    (*g2_values)[i] = (PetscScalar *)malloc((n_tau+1)*sizeof(PetscScalar));
    for (j=0;j<n_tau+1;j++){
      (*g2_values)[i][j] = 0.0;
    }
  }
  tsctx.g2_values = (*g2_values);

  set_ts_monitor_ctx(_g2_ts_monitor,&tsctx);
  st_dt = st_max/n_st;
  previous_start_time = 0;
  tsctx.i_st = 0;
  tsctx.i_st = tsctx.i_st + 1; //Why?

  for (this_start_time=st_dt;this_start_time<=st_max;this_start_time+=st_dt){
    //Go from previous start time to this_start_time
    tsctx.tau_evolve = 0;
    dt = (this_start_time - previous_start_time)/500; //500 is arbitrary, should be picked better
    time_step(dm0,previous_start_time,this_start_time,dt,steps_max);

    //Timestep through taus
    tau_t_max = this_start_time + tau_max;
    dt_tau = (tau_t_max - this_start_time)/n_tau;

    /*
     * Force an 'emission' to get A \rho A^\dag terms
     * We already have A* \cross A - we just do the multiplication
     */
    tsctx.tau_evolve = 1;
    //Copy the timestepped dm into our init_dm for tau sweep
    VecCopy(dm0,init_dm);
    MatMult(A_star_A,dm0,init_dm); //init_dm = A * dm0
    tsctx.i_tau = 0;
    time_step(init_dm,this_start_time,tau_t_max,dt_tau,steps_max);

    previous_start_time = this_start_time;
    tsctx.i_st = tsctx.i_st + 1;
  }

  return;
}

PetscErrorCode _g2_ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  TSCtx         *tsctx = (TSCtx*) ctx;   /* user-defined application context */
  PetscScalar ev;

  if (tsctx->tau_evolve==1){
    MatMult(tsctx->I_cross_A,dm,tsctx->tmp_dm); // tmp = I \cross A \rho
    MatMultHermitianTranspose(tsctx->I_cross_A,tsctx->tmp_dm,tsctx->tmp_dm2); // tmp2 = I \cross A^\dag tmp
    trace_dm(&ev,tsctx->tmp_dm2);
    tsctx->g2_values[tsctx->i_st][tsctx->i_tau] += ev;
    tsctx->i_tau = tsctx->i_tau + 1;
  }


  PetscFunctionReturn(0);
}


void diagonalize(PetscInt *num_evs,Vec **evecs,PetscScalar **evs){
  EPS eps;
  PetscInt its,nev,maxit,i,nconv;
  PetscReal tol;
  EPSType  type;
  Mat solve_A;

  //FIXME! Clean up evecs memory outside, I guess?
  if (_lindblad_terms) {
    if (nid==0) {
      printf("Lindblad terms found, diagonalizing Lindblad matrix.\n");
    }
    solve_A = full_A;
  } else {
    if (nid==0) {
      printf("No Lindblad terms found, diagonalizing Hamiltonian.\n");
    }
    solve_A = ham_A;
  }

  /* Tell PETSc to assemble the matrix */
  MatAssemblyBegin(solve_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(solve_A,MAT_FINAL_ASSEMBLY);
  MatScale(solve_A,PETSC_i);

   /*
     Create eigensolver context
   */
  EPSCreate(PETSC_COMM_WORLD,&eps);

  /*
    Set operators. In this case, it is a standard eigenvalue problem
  */
  EPSSetOperators(eps,solve_A,NULL);
  EPSSetProblemType(eps,EPS_HEP);
  EPSSetDimensions(eps,*num_evs,PETSC_DEFAULT,PETSC_DEFAULT); //Set number of evs to get
  EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);
  /*
    Set solver parameters at runtime
  */
  EPSSetFromOptions(eps);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  EPSSolve(eps);
  EPSGetIterationNumber(eps,&its);
  PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);
  EPSGetType(eps,&type);
  PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);
  EPSGetDimensions(eps,&nev,NULL,NULL);
  PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);
  EPSGetTolerances(eps,&tol,&maxit);
  PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);

  /*
    Get number of converged approximate eigenpairs
  */
  EPSGetConverged(eps,&nconv);
  PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);
  *num_evs = nconv;

  if (nconv>0) {
    *evecs = malloc(sizeof(Vec)*nconv);
    *evs = malloc(sizeof(PetscScalar)*nconv);

    for(i=0;i<nconv;i++){
      MatCreateVecs(solve_A,NULL,&(*evecs)[i]);
      EPSGetEigenpair(eps,i,&(*evs)[i],NULL,(*evecs)[i],NULL);
    }
  }
  EPSDestroy(&eps);

  return;
}

void destroy_diagonalize(PetscInt num_evs,Vec **evecs,PetscScalar **evs){
  PetscInt i;
  if (num_evs>0){
    free(*evs);
    for(i=0;i<num_evs;i++){
      VecDestroy(&(*evecs)[i]);
    }
    free(*evecs);
  }
}
