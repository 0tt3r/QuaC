#include "operators_p.h"
#include "operators.h"
#include "solver.h"
#include <stdlib.h>
#include <stdio.h>

static PetscReal default_rtol    = 1e-11;
static PetscInt  default_restart = 100;
static int       stab_added      = 0;

void steady_state(){
  PetscErrorCode ierr;
  PC             pc;
  Vec            x,b;
  KSP            ksp; /* linear solver context */
  int            row,col,its;
  PetscInt       i,Istart,Iend;
  PetscScalar    mat_tmp;
  long           dim;

  dim = total_levels*total_levels;

  if (!stab_added){
    /*
     * Add elements to the matrix to make the normalization work
     * I have no idea why this works, I am copying it from qutip
     * We add 1.0 in the 0th spot and every n+1 after
     */
    if(nid==0) {
      row = 0;
      for(i=0;i<total_levels;i++){
        col = i*(total_levels+1);
        mat_tmp = 1.0 + 0.*PETSC_i;
        ierr = MatSetValue(full_A,row,col,mat_tmp,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  
  ierr = MatGetOwnershipRange(full_A,&Istart,&Iend);CHKERRQ(ierr);
  /*
   * Explicitly add 0.0 to all diagonal elements;
   * this fixes a 'matrix in wrong state' message that PETSc
   * gives if the diagonal was never initialized.
   */
  for (i=Istart;i<Iend;i++){
    mat_tmp = 0 + 0.*PETSC_i;
    ierr = MatSetValue(full_A,i,i,mat_tmp,ADD_VALUES);CHKERRQ(ierr);
  }

  
  /* Tell PETSc to assemble the matrix */
  ierr = MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
   * Create parallel vectors.
   * - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
   * we specify only the vector's global
   * dimension; the parallel partitioning is determined at runtime.
   * - Note: We form 1 vector from scratch and then duplicate as needed.
   */
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,dim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /*
   * Set rhs, b, and solution, x to 1.0 in the first
   * element, 0.0 elsewhere.
   */
  ierr = VecSet(b,0.0);
  ierr = VecSet(x,0.0);
  
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
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /*
   * Set operators. Here the matrix that defines the linear system
   * also serves as the preconditioning matrix.
   */
  ierr = KSPSetOperators(ksp,full_A,full_A);CHKERRQ(ierr);
  
  /*
   * Set good default options for solver
   */
  /* relative tolerance */
  ierr = KSPSetTolerances(ksp,default_rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* bjacobi preconditioner */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);

  /* gmres solver with 100 restart*/
  ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPGMRESSetRestart(ksp,default_restart);
  /* 
   * Set runtime options, e.g.,
   *     -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
   */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  get_populations(x);

  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Iterations %D\n",its);CHKERRQ(ierr);

  /* Free work space */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);

  return;
}

void get_populations(Vec x) {
  int         j,my_levels,n_after,cur_state;
  PetscInt    x_low,x_high,i;
  PetscScalar *xa;
  PetscReal   tmp_real;
  double      *populations;
  PetscErrorCode ierr;
  ierr = VecGetOwnershipRange(x,&x_low,&x_high);
  ierr = VecGetArray(x,&xa);CHKERRQ(ierr); 



  /* Initialize population arrays */
  populations = malloc(num_subsystems*sizeof(double));
  for(i=0;i<num_subsystems;i++){
    populations[i] = 0.0;
  }


  for(i=0;i<total_levels;i++){
    if((i*total_levels+i)>=x_low&&(i*total_levels+i)<x_high) {
      /* Get the diagonal entry of rho */
      tmp_real = (double)PetscRealPart(xa[i*(total_levels)+i-x_low]);
      printf("%f \n",(double)PetscRealPart(xa[i*(total_levels)+i-x_low]));
      for(j=0;j<num_subsystems;j++){
        /*
         * We want to calculate the populations. To do that, we need 
         * to know what the state of the number operator for a specific
         * subsystem is for a given i. To accomplish this, we make use
         * of the fact that we can take the diagonal index i from the
         * full space and get which diagonal index it is in the subspace
         * by calculating:
         * i_subspace = mod(mod(floor(i/n_a),l*n_a),l)
         * because these are number operators, and we count from 0,
         * i_subspace = cur_state
         *
         * NOTE: DOESN'T WORK FOR VEC OPS!
         */
        /* my_levels = subsystem_list[j]->my_levels; */
        /* n_after   = total_levels/(my_levels*subsystem_list[j]->n_before); */
        /* cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels; */
        cur_state    = 1;
        populations[j] += tmp_real*cur_state;
      }
    }
  } 



  /* Reduce results across cores */
  if(nid==0) {
    MPI_Reduce(MPI_IN_PLACE,populations,num_subsystems,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(populations,populations,num_subsystems,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }

  /* Print results */
  if(nid==0) {
    printf("Populations: ");
    for(i=0;i<num_subsystems;i++){
      printf(" %f ",populations[i]);
    }
    printf("\n");
  }

  /* Put the array back in Petsc's hands */
  ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);
  /* Free memory */
  free(populations);
  return;
}
