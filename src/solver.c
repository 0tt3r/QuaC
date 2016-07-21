
#include "operators_p.h"
#include "operators.h"
#include "solver.h"
#include "kron_p.h"
#include <stdlib.h>
#include <stdio.h>

static PetscReal default_rtol    = 1e-11;
static PetscInt  default_restart = 100;
static int       stab_added      = 0;

PetscErrorCode RHSFunction (TS,PetscReal,Vec,Vec,void*);
void _set_initial_density_matrix(Vec);

/*
 * steady_state solves for the steady_state of the system
 * that was previously setup using the add_to_ham and add_lin
 * routines. Solver selection and parameterscan be controlled via PETSc
 * command line options.
 */
void steady_state(){
  PetscViewer    mat_view;
  PC             pc;
  Vec            x,b;
  KSP            ksp; /* linear solver context */
  int            row,col,its,j;
  PetscInt       i,Istart,Iend;
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
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,dim);
  VecSetFromOptions(b);
  VecDuplicate(b,&x);

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
  get_populations(x);

  KSPGetIterationNumber(ksp,&its);

  PetscPrintf(PETSC_COMM_WORLD,"Iterations %D\n",its);

  /* Free work space */
  KSPDestroy(&ksp);
  VecDestroy(&x);
  VecDestroy(&b);

  return;
}

/*
 * time_step solves for the time_dependence of the system
 * that was previously setup using the add_to_ham and add_lin
 * routines. Solver selection and parameterscan be controlled via PETSc
 * command line options.
 */
void time_step(){
  PetscViewer    mat_view;
  Vec            x;
  TS             ts; /* timestepping context */
  PetscInt       i,j,Istart,Iend,time_steps_max = 1000,steps;
  PetscReal      time_total_max = 100.0,dt = 10;
  PetscScalar    mat_tmp;
  long           dim;

  dim = total_levels*total_levels;

  if (!stab_added){
    /* row = 0; */
    /* for (i=0;i<total_levels;i++){ */
    /*   col = i*(total_levels+1); */
    /*   mat_tmp = 1.0 + 0.*PETSC_i; */
    /*   MatSetValue(full_A,row,col,mat_tmp,ADD_VALUES); */
    /* } */

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
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,dim);
  VecSetFromOptions(x);

  VecSet(x,0.0);

  _set_initial_density_matrix(x);
  
  /* Assemble x and b */
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

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
  TSMonitorSet(ts,ts_monitor,NULL,NULL);

  /*
   * Set up ODE system
   */

   /* TSSetRHSFunction(ts,NULL,RHSFunction,NULL); */
  TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);
  TSSetRHSJacobian(ts,full_A,full_A,TSComputeRHSJacobianConstant,NULL);

  TSSetInitialTimeStep(ts,0.0,dt);

  /*
   * Set default options, can be changed at runtime
   */

  TSSetDuration(ts,time_steps_max,time_total_max);
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);
  TSSetType(ts,TSBEULER);
  TSSetSolution(ts,x);
  TSSetFromOptions(ts);
  TSSolve(ts,x);
  TSGetTimeStepNumber(ts,&steps);

  get_populations(x);

  PetscPrintf(PETSC_COMM_WORLD,"Steps %D\n",steps);

  /* Free work space */
  TSDestroy(&ts);
  VecDestroy(&x);

  return;
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
      init_row_op += subsystem_list[i]->initial_pop*n_after;
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

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx) {
  /* Print populations for this time step */
  if (nid==0) printf("Time: %e ",time);
  get_populations(u);
  //get_concurrence(u);
  PetscFunctionReturn(0);
}

/*
 * Get populations for density matrix x
 * Inputs:
 *       Vec x - petsc vector representing density matrix
 * Outputs: 
 *       None, but prints populations to file
 */

void get_populations(Vec x) {
  int               j,my_levels,n_after,cur_state,num_pop;
  int               *i_sub_to_i_pop;
  PetscInt          x_low,x_high,i;
  const PetscScalar *xa;
  PetscReal         tmp_real;
  double            *populations;
  VecGetOwnershipRange(x,&x_low,&x_high);
  VecGetArrayRead(x,&xa); 

  /*
   * Loop through operators to see how many populations we need to 
   * calculate, because VEC need a population for each level.
   * We also set an array that translates i_subsystem to i_pop for normal ops.
   */
  i_sub_to_i_pop = malloc(num_subsystems*sizeof(int));
  num_pop = 0;
  for (i=0;i<num_subsystems;i++){
    if (subsystem_list[i]->my_op_type==VEC){
      i_sub_to_i_pop[i] = num_pop;
      num_pop += subsystem_list[i]->my_levels;
    } else {
      i_sub_to_i_pop[i] = num_pop;
      num_pop += 1;      
    }
  }

  /* Initialize population arrays */
  populations = malloc(num_pop*sizeof(double));
  for (i=0;i<num_pop;i++){
    populations[i] = 0.0;
  }


  for (i=0;i<total_levels;i++){
    if ((i*total_levels+i)>=x_low&&(i*total_levels+i)<x_high) {
      /* Get the diagonal entry of rho */
      tmp_real = (double)PetscRealPart(xa[i*(total_levels)+i-x_low]);
      //      printf("%e \n",(double)PetscRealPart(xa[i*(total_levels)+i-x_low]));
      for(j=0;j<num_subsystems;j++){
        /*
         * We want to calculate the populations. To do that, we need 
         * to know what the state of the number operator for a specific
         * subsystem is for a given i. To accomplish this, we make use
         * of the fact that we can take the diagonal index i from the
         * full space and get which diagonal index it is in the subspace
         * by calculating:
         * i_subspace = mod(mod(floor(i/n_a),l*n_a),l)
         * For regular operators, these are just number operators, and we count from 0,
         * so, cur_state = i_subspace
         *
         * For VEC ops, we can use the same technique. Once we get cur_state (i_subspace),
         * we use that to go the appropriate location in the population array.
         */
        if (subsystem_list[j]->my_op_type==VEC){
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]+cur_state] += tmp_real;
        } else {
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]] += tmp_real*cur_state;
        }
      }
    }
  } 

  /* Reduce results across cores */
  if(nid==0) {
    MPI_Reduce(MPI_IN_PLACE,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(populations,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }

  /* Print results */
  if(nid==0) {
    printf("Populations: ");
    for(i=0;i<num_pop;i++){
      printf(" %e ",populations[i]);
    }
    printf("\n");
  }

  /* Put the array back in Petsc's hands */
  VecRestoreArrayRead(x,&xa);
  /* Free memory */
  free(i_sub_to_i_pop);
  free(populations);
  return;
}

void get_concurrence(Vec x) {
  int               j,my_levels,n_after,cur_state,num_pop;
  int               *i_sub_to_i_pop;
  PetscInt          x_low,x_high,i;
  const PetscScalar *xa;
  PetscReal         tmp_real;
  double            *populations;
  VecGetOwnershipRange(x,&x_low,&x_high);
  VecGetArrayRead(x,&xa); 

  /*
   * Loop through operators to see how many populations we need to 
   * calculate, because VEC need a population for each level.
   * We also set an array that translates i_subsystem to i_pop for normal ops.
   */
  i_sub_to_i_pop = malloc(num_subsystems*sizeof(int));
  num_pop = 0;
  for (i=0;i<num_subsystems;i++){
    if (subsystem_list[i]->my_op_type==VEC){
      i_sub_to_i_pop[i] = num_pop;
      num_pop += subsystem_list[i]->my_levels;
    } else {
      i_sub_to_i_pop[i] = num_pop;
      num_pop += 1;      
    }
  }

  /* Initialize population arrays */
  populations = malloc(num_pop*sizeof(double));
  for (i=0;i<num_pop;i++){
    populations[i] = 0.0;
  }


  for (i=0;i<total_levels;i++){
    if ((i*total_levels+i)>=x_low&&(i*total_levels+i)<x_high) {
      /* Get the diagonal entry of rho */
      tmp_real = (double)PetscRealPart(xa[i*(total_levels)+i-x_low]);
      //      printf("%e \n",(double)PetscRealPart(xa[i*(total_levels)+i-x_low]));
      for(j=0;j<num_subsystems;j++){
        /*
         * We want to calculate the populations. To do that, we need 
         * to know what the state of the number operator for a specific
         * subsystem is for a given i. To accomplish this, we make use
         * of the fact that we can take the diagonal index i from the
         * full space and get which diagonal index it is in the subspace
         * by calculating:
         * i_subspace = mod(mod(floor(i/n_a),l*n_a),l)
         * For regular operators, these are just number operators, and we count from 0,
         * so, cur_state = i_subspace
         *
         * For VEC ops, we can use the same technique. Once we get cur_state (i_subspace),
         * we use that to go the appropriate location in the population array.
         */
        if (subsystem_list[j]->my_op_type==VEC){
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]+cur_state] += tmp_real;
        } else {
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]] += tmp_real*cur_state;
        }
      }
    }
  } 



  /* Reduce results across cores */
  if(nid==0) {
    MPI_Reduce(MPI_IN_PLACE,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(populations,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }

  /* Print results */
  if(nid==0) {
    printf("Populations: ");
    for(i=0;i<num_pop;i++){
      printf(" %e ",populations[i]);
    }
    printf("\n");
  }

  /* Put the array back in Petsc's hands */
  VecRestoreArrayRead(x,&xa);
  /* Free memory */
  free(i_sub_to_i_pop);
  free(populations);
  return;
}
