
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

  /* Initialize Petsc */
  PetscInitialize(&argc,&args,(char*)0,NULL);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
  /* Get core's id */
  MPI_Comm_rank(PETSC_COMM_WORLD,&nid);
  /* Get number of processors */
  MPI_Comm_size(PETSC_COMM_WORLD,&np);

  petsc_initialized = 1;
}

/*
 * QuaC_clear clears the internal state of many of QuaC's
 * variables so that multiple systems can be run in one file.
 */

void QuaC_clear(){
  int i;
  /* Destroy Matrix */
  MatDestroy(&full_A);
  MatDestroy(&ham_A);
  MatDestroy(&full_stiff_A);
  MatDestroy(&ham_stiff_A);

  for (i=0;i<_num_time_dep;i++){
    MatDestroy(&_time_dep_list[i].mat);
  }
  //stab_added       = 0;
  _print_dense_ham = 0;
  _num_time_dep = 0;
  op_initialized = 0;
}


/*
 * QuaC_finalize finalizes petsc and destroys full_A.
 * The user is responsible for freeing all of the objects
 * using destroy_*
 */

void QuaC_finalize(){
  int i;
  /* Destroy Matrix */
  MatDestroy(&full_A);
  MatDestroy(&ham_A);
  MatDestroy(&full_stiff_A);
  MatDestroy(&ham_stiff_A);

  for (i=0;i<_num_time_dep;i++){
    MatDestroy(&_time_dep_list[i].mat);
  }
  /* Finalize Petsc */
  PetscFinalize();
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
