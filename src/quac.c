
#include "quac_p.h"
#include "quac.h"
#include "operators_p.h"
#include <petsc.h>

int petsc_initialized = 0;
int nid;

void QuaC_initialize(int argc,char **args){
  PetscErrorCode ierr;
  /* Initialize Petsc */
  ierr             = PetscInitialize(&argc,&args,(char*)0,NULL);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
  /* Get core's id */
  ierr              = MPI_Comm_rank(PETSC_COMM_WORLD,&nid);CHKERRQ(ierr);
  petsc_initialized = 1;
}

void QuaC_finalize(){
  PetscErrorCode ierr;
  int            i;
  /* Destroy Matrix */
  ierr = MatDestroy(&full_A);CHKERRQ(ierr);

  /* Finalize Petsc */
  ierr             = PetscFinalize();CHKERRQ(ierr);

  /* Free dense hamiltonian */
  for (i=0;i<total_levels;i++){
    free(_hamiltonian[i]);
  }
  free(_hamiltonian);
}
