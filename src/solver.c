
#include "operators_p.h"
#include "operators.h"
#include "solver.h"
#include "kron_p.h"
#include "dm_utilities.h"
#include <stdlib.h>
#include <stdio.h>

static PetscReal default_rtol    = 1e-11;
static PetscInt  default_restart = 100;
static int       stab_added      = 0;
static int       matrix_assembled = 0;

typedef enum {
  HADAMARD = 0,
  CNOT = 1
} gate_type;

typedef struct time_dep_struct{
  double (*time_dep_func)(double);
  Mat mat;
} time_dep_struct;


typedef struct quantum_gate_struct{
  PetscReal time;
  gate_type my_gate_type;
  int *qubit_numbers;
} quantum_gate_struct;


PetscErrorCode RHSFunction (TS,PetscReal,Vec,Vec,void*); // Move to header?
PetscErrorCode (*_ts_monitor)(TS,PetscInt,PetscReal,Vec,void*) = NULL;
  
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


  dim = total_levels*total_levels;

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
              fprintf(fp_ham,"%e ",_hamiltonian[i][j]);
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
  /* Pass -1.0 to flag the routine to print the final populations to stdout */
  get_populations(x,-1.0);

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
void time_step(Vec x, PetscReal time_max,PetscReal dt,PetscInt steps_max){
  PetscViewer    mat_view;
  //  Vec            x;
  TS             ts; /* timestepping context */
  PetscInt       i,j,Istart,Iend,steps,row,col;
  PetscScalar    mat_tmp;
  /* long           dim; */

  /* dim = total_levels*total_levels; */

  /* Possibly print dense ham. No stabilization is needed? */
  if (nid==0) {
    /* Print dense ham, if it was asked for */
    if (_print_dense_ham){
      FILE *fp_ham;

      fp_ham = fopen("ham","w");
      
      if (nid==0){
        for (i=0;i<total_levels;i++){
          for (j=0;j<total_levels;j++){
            fprintf(fp_ham,"%e ",_hamiltonian[i][j]);
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

  /* Print information about the matrix. */
  PetscViewerASCIIOpen(PETSC_COMM_WORLD,NULL,&mat_view);
  PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_INFO);
  MatView(full_A,mat_view);
  PetscViewerDestroy(&mat_view);
  /*
   * Create parallel vectors.
   * - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
   * we specify only the vector's global
   * dimension; the parallel partitioning is determined at runtime.
   * - Note: We form 1 vector from scratch and then duplicate as needed.
   */
  /* VecCreate(PETSC_COMM_WORLD,&x); */
  /* VecSetSizes(x,PETSC_DECIDE,dim); */

  //  create_dm(&x,total_levels); //Assume 
  /* VecSetFromOptions(x); */

  /* VecSet(x,0.0); */

  //  _set_initial_density_matrix(x);
  
  /* Assemble x and b */
  //  VecAssemblyBegin(x);
  //  VecAssemblyEnd(x);

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
    TSMonitorSet(ts,_ts_monitor,NULL,NULL);
  }
  /*
   * Set up ODE system
   */

   /* TSSetRHSFunct
ion(ts,NULL,RHSFunction,NULL); */
  TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);
  if(_time_dep_ham||_quantum_gates) {
    TSSetRHSJacobian(ts,full_A,full_A,_RHS_time_dep_ham,NULL);
  } else {
    TSSetRHSJacobian(ts,full_A,full_A,TSComputeRHSJacobianConstant,NULL);
  }
  TSSetInitialTimeStep(ts,0.0,dt);

  /*
   * Set default options, can be changed at runtime
   */

  TSSetDuration(ts,steps_max,time_max);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK3BS);
  //  TSSetSolution(ts,x);
  TSSetFromOptions(ts);
  TSSolve(ts,x);
  TSGetTimeStepNumber(ts,&steps);

  /* Pass -1.0 to flag the routine to print the final populations to stdout */
  get_populations(x,-1.0);

  PetscPrintf(PETSC_COMM_WORLD,"Steps %D\n",steps);

  /* Free work space */
  TSDestroy(&ts);
  //  destroy_dm(x);
  /* VecDestroy(&x); */

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
}

<<<<<<< HEAD
=======

PetscErrorCode _RHS_time_dep_ham(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx){
  PetscReal time_dep_scalar;

  /* Copy the time independent H over */
  MatDuplicate(full_A,MAT_COPY_VALUES,&AA);
  /* Add the time dependent parts of H */
  for (i=0;i<_num_time_dep;i++){
    time_dep_scalar = _time_dep_list[i].time_dep_func(t);
    MatAXPY(AA,time_dep_scalar,_time_dep_list[i].mat,DIFFERENT_NONZERO_PATTERN);
    /* Consider putting _time_dep_func and _time_dep_mats in *ctx? */
  }
  /* Add quantum gates, if there are any */
  if (_quantum_gates) {
    if (time > _quantum_gate_list[gate_counter].time) {
      _apply_gate(_quantum_gate_list[gate_counter].my_gate_type,_quantum_gate_list[gate_counter].qubits,AA);
    }
  }
}

/*
 * _set_initial_density_matrix sets the initial condition from the
 * initial conditions provided via the set_initial_pop routine.
 * Currently a sequential routine. May need to be updated in the future.
 *
 * Inputs:
 *      Vec x
 */
void _set_initial_density_matrix(Vec x){
  PetscInt    i,j,init_row_op=0,n_after,i_sub,j_sub;
  PetscScalar mat_tmp_val;
  PetscInt    *index_array;
  Mat         subspace_dm,rho_mat;
  int         simple_init_pop=1;
  MatScalar   *rho_mat_array;
  PetscReal   vec_pop;

  /* 
   * See if there are any vec operators
   */

  for (i=0;i<num_subsystems&&simple_init_pop==1;i++){
    if (subsystem_list[i]->my_op_type==VEC){
      simple_init_pop = 0;
    }
  }

  if (nid==0&&simple_init_pop==1){
    /* 
     * We can only use this simpler initialization if all of the operators
     * are ladder operators, and the user hasn't used any special initialization routine
     */
    for (i=0;i<num_subsystems;i++){ 
      n_after   = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);
      init_row_op += ((int)subsystem_list[i]->initial_pop)*n_after;
    }

    init_row_op = total_levels*init_row_op + init_row_op;
    mat_tmp_val = 1. + 0.0*PETSC_i;
    VecSetValue(x,init_row_op,mat_tmp_val,INSERT_VALUES);

  } else if (nid==0){
    /*
     * This more complicated initialization routine allows for the vec operator
     * to take distributed values (say, 1/3 1/3 1/3)
     */


    /* Create temporary PETSc matrices */
    MatCreate(PETSC_COMM_SELF,&subspace_dm);
    MatSetType(subspace_dm,MATSEQDENSE);
    MatSetSizes(subspace_dm,total_levels,total_levels,total_levels,total_levels);
    MatSetUp(subspace_dm);

    MatCreate(PETSC_COMM_SELF,&rho_mat);
    MatSetType(rho_mat,MATSEQDENSE);
    MatSetSizes(rho_mat,total_levels,total_levels,total_levels,total_levels);
    MatSetUp(rho_mat);
    /* 
     * Set initial density matrix to the identity matrix, because
     * A' cross B' cross C' = I (A cross I_b cross I_c) (I_a cross B cross I_c) 
     */
    for (i=0;i<total_levels;i++){
      MatSetValue(rho_mat,i,i,1.0,INSERT_VALUES);
    }
    MatAssemblyBegin(rho_mat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(rho_mat,MAT_FINAL_ASSEMBLY);
    /* Loop through subsystems */
    for (i=0;i<num_subsystems;i++){ 
      n_after = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);

      /* 
       * If the subsystem is a ladder operator, the population will be just on a 
       * diagonal element within the subspace.
       * LOWER is in the if because that is the op in the subsystem list for ladder operators
       */
      if (subsystem_list[i]->my_op_type==LOWER){

        /* Zero out the subspace density matrix */
        MatZeroEntries(subspace_dm);
        i_sub = (int)subsystem_list[i]->initial_pop;
        j_sub = i_sub;
        mat_tmp_val = 1. + 0.0*PETSC_i;
        _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
        _mult_PETSc_init_DM(subspace_dm,rho_mat,(double)1.0);

      } else if (subsystem_list[i]->my_op_type==VEC){
        /*
         * If the subsystem is a vec type operator, we loop over each
         * state and set the initial population.
         */
        n_after = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);
        /* Zero out the subspace density matrix */
        MatZeroEntries(subspace_dm);
        vec_pop = 0.0;
        for (j=0;j<subsystem_list[i]->my_levels;j++){
          i_sub   = subsystem_list[i]->vec_op_list[j]->position;
          j_sub   = i_sub;
          vec_pop += subsystem_list[i]->vec_op_list[j]->initial_pop;
          mat_tmp_val = subsystem_list[i]->vec_op_list[j]->initial_pop + 0.0*PETSC_i;
          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
        }

        if (vec_pop==(double)0.0){
          printf("WARNING! No initial population set for a vector operator!\n");
          printf("         Defaulting to all population in the 0th element\n");
          mat_tmp_val = 1.0 + 0.0*PETSC_i;
          vec_pop     = 1.0;
          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,0,0,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
        }
        /* 
         * Now that the subspace_dm is fully constructed, we multiply it into the full
         * initial DM
         */
        _mult_PETSc_init_DM(subspace_dm,rho_mat,vec_pop);
      }
    }
    /* vectorize the density matrix */

    /* First, get the array where matdense is stored */
    MatDenseGetArray(rho_mat,&rho_mat_array);

    /* Create an index list for, needed for VecSetValues */
    PetscMalloc1(total_levels*total_levels,&index_array);
    for (i=0;i<total_levels;i++){
      for (j=0;j<total_levels;j++){
        index_array[i+j*total_levels] = i+j*total_levels;
      }
    }

    VecSetValues(x,total_levels*total_levels,index_array,rho_mat_array,INSERT_VALUES);
    MatDenseRestoreArray(rho_mat,&rho_mat_array);
    MatDestroy(&subspace_dm);
    MatDestroy(&rho_mat);
    PetscFree(index_array);
  }

  return;
}
>>>>>>> Working on adding time dependent hamiltonians and quantum gates.

PetscErrorCode RHSFunction (TS ts, PetscReal t, Vec array_in, Vec array_out, void *s){
  MatMult(full_A,array_in,array_out);
  return 0;
}

/*
 * ts_monitor is the catchall routine which will look at the data
 * at every time step, such as printing observables/populations.
 *
 * Inputs:
 *    ts     - the timestep context
 *    step   - the count of the current step (with 0 meaning the
 *             initial condition)
 *    time   - the current time
 *    u      - the solution at this timestep
 *    ctx    - the user-provided context for this monitoring routine.
 */

/* PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx) { */
  
/*   get_populations(u); */
/*   PetscFunctionReturn(0); */
/* } */

