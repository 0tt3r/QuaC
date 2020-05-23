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
  qvec rho,rho2;
  circuit circ;
  Mat circ_mat;
  qsystem system,sys2;
  operator *qubits,*qubits2;
  PetscReal *gamma_1,*gamma_2,gamma_base,gamma_base1,gamma_base2,dt,time_max,fidelity;
  PetscInt num_qubits,steps_max,i;
  PetscScalar mat_val;
  char filename[256],outfile[256],outfile2[256];
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 2;
  strcpy(filename,"NULL");
  strcpy(outfile,"wf_out.dat");
  strcpy(outfile2,"wf_out2.dat");
  gamma_base1 = 0;
  gamma_base2 = 0;

  PetscOptionsGetString(NULL,NULL,"-file_circ",filename,sizeof(filename),NULL);
  PetscOptionsGetString(NULL,NULL,"-file_out",outfile,sizeof(outfile),NULL);
  PetscOptionsGetReal(NULL,NULL,"-gam1",&gamma_base1,NULL);
  PetscOptionsGetReal(NULL,NULL,"-gam2",&gamma_base2,NULL);
  qiskit_qasm_read(filename,&num_qubits,&circ);
  PetscPrintf(PETSC_COMM_WORLD,"Num_qubits: %d\n",num_qubits);
  PetscPrintf(PETSC_COMM_WORLD,"gates: %d\n",circ.num_gates);

  qubits  = malloc(num_qubits*sizeof(struct operator)); //Allocate structure to store qubits
  qubits2  = malloc(num_qubits*sizeof(struct operator)); //Allocate structure to store qubits
  gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Allocate array for qubit error rates
  gamma_2 = malloc(num_qubits*sizeof(PetscReal));

  initialize_system(&system);
  initialize_system(&sys2);
  gamma_base = 1e-9;
  for (i=0;i<num_qubits;i++){
    create_op_sys(system,2,&qubits[i]); //create qubit
    create_op_sys(sys2,2,&qubits2[i]); //create qubit
    gamma_1[i] = gamma_base1;
    gamma_2[i] = 2*gamma_base2;
  }

  //add errors to lindblad term
  for (i=0;i<num_qubits;i++){
    add_lin_term(system,gamma_1[i],1,qubits[i]);
    add_lin_term(system,gamma_2[i],1,qubits[i]->n);
  }
  add_ham_term(system,0*gamma_2[0],1,qubits[0]->n);


  /* //Construct the matrix now that we are done adding to it */
  construct_matrix(system);


  //Time step until 1 after the last gate; gates are applied every 1.0 time unit
  time_max  = circ.num_gates+1;
  dt        = 100;
  steps_max = 100000;

  /* Set the ts_monitor to print results at each time step, if desired */
  //  set_ts_monitor_sys(system,ts_monitor,NULL);

  /* create_wf_sys(sys2,&rho); */
  /* mat_val = 1.0; */
  /* add_to_qvec_loc(rho,mat_val,0); */
  /* assemble_qvec(rho); */

  create_qvec_sys(system,&rho2);
  mat_val = 1.0;
  add_to_qvec_loc(rho2,mat_val,0);
  assemble_qvec(rho2);

  /*
   * Set gate run_times
   */
  for (i=0;i<circ.num_gates;i++){
    if (circ.gate_list[i].my_gate_type==U1){
      //U1 is essentially a 'free' gate
      circ.gate_list[i].run_time = 0;//ns
    } else if (circ.gate_list[i].num_qubits==1){
      circ.gate_list[i].run_time = 200;//ns
    } else if (circ.gate_list[i].num_qubits==2){
      circ.gate_list[i].run_time = 1000;//ns
    }
  }

  /* combine_circuit_to_mat_sys(system,&circ_mat,circ); */
  /* qvec_mat_mult(circ_mat,rho); */
  /* print_qvec_file(rho,outfile2); */


  schedule_circuit_layers(system,&circ);
  PetscPrintf(PETSC_COMM_WORLD,"num_layers: %d\n",circ.num_layers);


  /* //Start out circuit at time 0.0, first gate will be at 0 */
  apply_circuit_to_sys(system,&circ,0.0);
  time_max  = circ.layer_list[circ.num_layers-1].time+1; //-1 because counting from 0
  PetscPrintf(PETSC_COMM_WORLD,"Max time: %f %d \n",time_max, circ.num_layers );
  //Run the evolution, with error and with the circuit
  time_step_sys(system,rho2,0.0,time_max,dt,steps_max);

  /* get_fidelity_qvec(rho,rho2,&fidelity); */
  /* printf("Fidelity: %f\n",fidelity); */

  print_qvec_file(rho2,outfile);
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
  destroy_qvec(&rho2);
  QuaC_finalize();
  return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho,void *ctx){
  //Print out things at each time step, if desired
  /* PetscPrintf(PETSC_COMM_WORLD,"Step: %d, time: %f\n",step,time); */
  /* print_qvec(rho); */
  PetscFunctionReturn(0);
}

