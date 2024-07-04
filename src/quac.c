
#include "quac_p.h"
#include "quac.h"
#include "operators_p.h"
#include "operators.h"

#include <petsc.h>
#include <slepc.h>

int petsc_initialized = 0;
int nid;
int np;

PetscLogEvent add_lin_event,add_to_ham_event,add_lin_recovery_event,add_encoded_gate_to_circuit_event;
PetscLogEvent _qc_event_function_event,_qc_postevent_function_event,_apply_gate_event,_RHS_time_dep_event;
PetscClassId quac_class_id;
PetscLogStage pre_solve_stage,solve_stage,post_solve_stage;

/*
 * QuaC_initialize initializes petsc, gets each core's nid, and lets the
 * rest of the program know that it has been initialized.
 * Inputs:
 *       int argc, char **args - command line input, for PETSc
 */
void QuaC_initialize(int argc,char **args){
  PetscInt seed=1;
  /* Initialize Petsc */
  //PetscInitialize(&argc,&args,(char*)0,NULL);
  /* Initialize SLEPc */
  SlepcInitialize(&argc,&args,(char*)0,NULL);

#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
  /* Get core's id */
  MPI_Comm_rank(PETSC_COMM_WORLD,&nid);
  /* Get number of processors */
  MPI_Comm_size(PETSC_COMM_WORLD,&np);

  petsc_initialized = 1;
  PetscLogStageRegister("Pre-solve",&pre_solve_stage);
  PetscLogStageRegister("Solve",&solve_stage);
  PetscLogStageRegister("Post-solve",&post_solve_stage);
  /* Register events */
  PetscClassIdRegister("QuaC Class",&quac_class_id);
  PetscLogEventRegister("add_lin",quac_class_id,&add_lin_event);
  PetscLogEventRegister("add_to_ham",quac_class_id,&add_to_ham_event);
  PetscLogEventRegister("add_lin_recovery",quac_class_id,&add_lin_recovery_event);
  PetscLogEventRegister("_qc_event",quac_class_id,&_qc_event_function_event);
  PetscLogEventRegister("_qc_postevent",quac_class_id,&_qc_postevent_function_event);
  PetscLogEventRegister("_apply_gate",quac_class_id,&_apply_gate_event);
  PetscLogEventRegister("_RHS_time_dep",quac_class_id,&_RHS_time_dep_event);

  seed = make_sprng_seed();

  PetscLogStagePush(pre_solve_stage);
  init_sprng(seed,SPRNG_DEFAULT);
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
  /**********************************************************
   * FIXME: The code below legacy code which should be removed!!
   ***********************************************************/

  /* Destroy Matrix */
  MatDestroy(&full_A);
  MatDestroy(&ham_A);
  MatDestroy(&full_stiff_A);
  MatDestroy(&ham_stiff_A);

  for (i=0;i<_num_time_dep;i++){
    MatDestroy(&_time_dep_list[i].mat);
  }
  /**********************************************************
   * FIXME: The code below legacy code which should be removed!!
   ***********************************************************/
  /* Finalize Petsc */
  PetscLogStagePop();
  SlepcFinalize();
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
