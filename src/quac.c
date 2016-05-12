
#include "quac_p.h"
#include "quac.h"
#include "operators_p.h"
#include "operators.h"
#include <petsc.h>

int petsc_initialized = 0;
int nid;
int np;

/* 
 * QuaC_initialize initializes petsc, gets each core's nid, and lets the
 * rest of the program know that it has been initialized.
 * Inputs:
 *       int argc, char **args - command line input, for PETSc
 */
void QuaC_initialize(int argc,char **args){
  PetscErrorCode ierr;
  /* Initialize Petsc */
  ierr             = PetscInitialize(&argc,&args,(char*)0,NULL);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
  /* Get core's id */
  ierr              = MPI_Comm_rank(PETSC_COMM_WORLD,&nid);CHKERRQ(ierr);
  /* Get number of processors */
  ierr              = MPI_Comm_size(PETSC_COMM_WORLD,&np);CHKERRQ(ierr);

  petsc_initialized = 1;
}

/* 
 * QuaC_finalize finalizes petsc and destroys full_A.
 * The user is responsible for freeing all of the objects
 * using destroy_*
 */

void QuaC_finalize(){
  PetscErrorCode ierr;
  int            i;
  /* Destroy Matrix */
  ierr = MatDestroy(&full_A);CHKERRQ(ierr);

  /* Finalize Petsc */
  ierr             = PetscFinalize();CHKERRQ(ierr);
  return;
}

/*
 * destroy_op frees the memory from op.
 * Inputs:
 *       operator *op - pointer to operator to be freed
 */

void destroy_op(operator *op){

  free((*op)->dag);
  free((*op)->n);
  free(*op);
}

/*
 * destroy_vec frees the memory from a vec.
 * Inputs:
 *       vec_op *op - pointer to vec_op to be freed
 */

void destroy_vec(vec_op *op){
  int num_levels,i;
  num_levels = (*op)[0]->my_levels;
  
  /* Free each up in the array */
  for (i=0;i<num_levels;i++){
    free((*op)[i]);
  }
  free(*op);
}
