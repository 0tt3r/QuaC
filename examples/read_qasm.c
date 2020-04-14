#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "error_correction.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "quantum_circuits.h"
#include "petsc.h"
#include "qasm_parser.h"
#include "qsystem.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);

int main(int argc,char **args){
  qvec rho;
  circuit circ;
  qsystem system;
  operator *qubits;
  PetscReal *gamma_1,*gamma_2,gamma_base,dt,time_max;
  PetscInt num_qubits,num_stpes,steps_max,i;
  PetscScalar mat_val;
  char filename[256],outfile[256];
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 2;
  strcpy(filename,"NULL");
  strcpy(outfile,"wf_out.dat");
  PetscOptionsGetString(NULL,NULL,"-file_circ",filename,sizeof(filename),NULL);
  PetscOptionsGetString(NULL,NULL,"-file_out",outfile,sizeof(outfile),NULL);
  qiskit_qasm_read(filename,&num_qubits,&circ);
  printf("Num_qubits: %d\n",num_qubits);
  printf("gates: %d\n",circ.num_gates);

  qubits  = malloc(num_qubits*sizeof(struct operator)); //Allocate structure to store qubits
  gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Allocate array for qubit error rates
  gamma_2 = malloc(num_qubits*sizeof(PetscReal));

  initialize_system(&system);
  gamma_base = 0;
  for (i=0;i<num_qubits;i++){
    create_op_sys(system,2,&qubits[i]); //create qubit
    gamma_1[i] = gamma_base;
    gamma_2[i] = gamma_base/2;
  }


  //add errors to lindblad term
  /* for (i=0;i<num_qubits;i++){ */
  /*   add_lin_term(system,gamma_1[i],1,qubits[i]); */
  /*   add_lin_term(system,gamma_2[i]*2,1,qubits[i]->n); */
  /* } */
  /* add_ham_term(system,gamma_1[0],1,qubits[0]->n); */
  /* //Construct the matrix now that we are done adding to it */
  /* construct_matrix(system); */

  //Time step until 1 after the last gate; gates are applied every 1.0 time unit
  time_max  = circ.num_gates+1;
  dt        = 1;
  steps_max = 100000;

  /* Set the ts_monitor to print results at each time step, if desired */
  set_ts_monitor_sys(system,ts_monitor,NULL);

  create_qvec_sys(system,&rho);
  mat_val = 1.0;
  add_to_qvec_loc(rho,mat_val,0);
  assemble_qvec(rho);

  Mat circ_mat;
  Vec result;
  combine_circuit_to_mat_sys(system,&circ_mat,circ);
  qvec_mat_mult(circ_mat,rho);
  //Start out circuit at time 0.0, first gate will be at 0
  //  apply_circuit_to_sys(system,&circ,0.0);

  //Run the evolution, with error and with the circuit
  //  time_step_sys(system,rho,0.0,time_max,dt,steps_max);

  print_qvec_file(rho,outfile);
  //Clean up memory
  /* PetscScalar pop; */
  /* printf("Pops: \n"); */
  /* for (i=0;i<num_qubits;i++){ */
  /*   get_expectation_value_qvec(rho,&pop,1,qubits[i]->n); */
  /*   printf(" %f ",PetscRealPart(pop)); */
  /* } */
  /* printf("\n"); */
  for (i=0;i<num_qubits;i++){
    destroy_op(&qubits[i]);
  }
  destroy_qvec(&rho);
  QuaC_finalize();
  return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho,void *ctx){
  //Print out things at each time step, if desired
  PetscPrintf(PETSC_COMM_WORLD,"Step: %d, time: %f\n",step,time);
  /* print_qvec(rho); */
  PetscFunctionReturn(0);
}

