

#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "error_correction.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"
#include "qasm_parser.h"
#include "qsystem.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);

/* Declared globally so that we can access this in ts_monitor */
operator *qubits;
FILE *f_pop;

int main(int argc,char **args){
  PetscReal time_max,dt,*gamma_1,*gamma_2;
  PetscScalar mat_val;
  PetscInt  steps_max,num_qubits;
  int i;
  Vec rho;
  circuit projectq_read;
  char           string[10],filename[128],file_dm[128];

  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 0;
  /* Get the encoding strategies for each qubit */

  strcpy(filename,"NULL");
  strcpy(file_dm,"dm.dat");
  PetscOptionsGetString(NULL,NULL,"-file_circ",filename,sizeof(filename),NULL);
  PetscOptionsGetString(NULL,NULL,"-file_dm",file_dm,sizeof(file_dm),NULL);
  quil_read(filename,&num_qubits,&projectq_read); // Read the file

  qubits  = malloc(num_qubits*sizeof(struct operator)); //Allocate structure to store qubits
  gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Allocate array for qubit error rates
  gamma_2 = malloc(num_qubits*sizeof(PetscReal));

  for (i=0;i<num_qubits;i++){
    create_op(2,&qubits[i]); //create qubit
    gamma_1[i] = 0;
    gamma_2[i] = 0;
  }

  for (i=0; i<num_qubits;i++){
    //Read errors from command line using -gam0, -gam1, etc
    sprintf(string, "-gam%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&gamma_1[i],NULL);
    sprintf(string, "-dep%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&gamma_2[i],NULL);
  }

  //Add errors to lindblad term
  for (i=0;i<num_qubits;i++){
    add_lin(gamma_1[i],qubits[i]);
    add_lin(gamma_2[i]*2,qubits[i]->n);
  }

  if (nid==0){
    // Print read in variables
    printf("qubit gam dep\n");
    for (i=0;i<num_qubits;i++){
      printf("%d %f %f \n",i,gamma_1[i],gamma_2[i]);
    }
  }
  //Time step until 1 after the last gate; gates are applied every 1.0 time unit
  time_max  = projectq_read.num_gates+1;
  dt        = 0.01;
  steps_max = 100000;

  /* Set the ts_monitor to print results at each time step, if desired */
  set_ts_monitor(ts_monitor);
  /* Open file that we will print to in ts_monitor */
  /* if (nid==0){ */
  /*   f_pop = fopen("pop","w"); */
  /*   fprintf(f_pop,"#Time Populations\n"); */
  /* } */

  create_full_dm(&rho); //Allocate and set our initial density matrix
  mat_val = 1.0;
  add_value_to_dm(rho,0,0,mat_val);
  assemble_dm(rho);
  if (nid==0){
    //Print number of gates
    printf("num_gates: %ld\n",projectq_read.num_gates);
  }

  //Start out circuit at time 0.0, first gate will be at 1.0
  start_circuit_at_time(&projectq_read,0.0);
  //Run the evolution, with error and with the circuit
  time_step(rho,0.0,time_max,dt,steps_max);

  print_dm_sparse_to_file(rho,pow(2,num_qubits),file_dm);


  //Clean up memory
  PetscScalar pop;
  for (i=0;i<num_qubits;i++){
    get_expectation_value(rho,&pop,1,qubits[i]->n);
    printf("pop: %d %f\n",i,PetscRealPart(pop));
    destroy_op(&qubits[i]);
  }
  destroy_dm(rho);
  QuaC_finalize();
  return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho,void *ctx){
  //Print out things at each time step, if desired
  PetscFunctionReturn(0);
}

