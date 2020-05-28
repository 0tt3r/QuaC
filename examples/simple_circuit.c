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
  PetscReal single_qubit_gate_time=1,two_qubit_gate_time=1;
  circuit circ;
  qsystem system;
  operator *qubits;
  PetscReal *gamma_1,*gamma_2,gamma_base,dt,time_max;
  PetscInt num_qubits,num_stpes,steps_max,i;
  PetscScalar mat_val;
  PetscInt nloc;
  PetscReal *probs;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 2;

  qubits  = malloc(num_qubits*sizeof(struct operator)); //Allocate structure to store qubits
  gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Allocate array for qubit error rates
  gamma_2 = malloc(num_qubits*sizeof(PetscReal));

  initialize_system(&system);
  gamma_base = 0*0.000001;
  for (i=0;i<num_qubits;i++){
    create_op_sys(system,2,&qubits[i]); //create qubit
    gamma_1[i] = gamma_base;
    gamma_2[i] = gamma_base/2;
  }


  //Add errors to lindblad term
  for (i=0;i<num_qubits;i++){
    add_lin_term(system,gamma_1[i],1,qubits[i]);
    add_lin_term(system,gamma_2[i]*2,1,qubits[i]->n);
  }
  //Construct the matrix now that we are done adding to it
  construct_matrix(system);

  create_circuit(&circ,5);
  //Add some gates
  add_gate_to_circuit_sys(&circ,0.0,SIGMAX,0);
  add_gate_to_circuit_sys(&circ,3.0,RY,0.43,1);
  add_gate_to_circuit_sys(&circ,4.0,CNOT,0,1);


  //Time step until 1 after the last gate; gates are applied every 1.0 time unit
  time_max  = 10000;
  dt        = 0.01;
  steps_max = 100;

  /* Set the ts_monitor to print results at each time step, if desired */
  set_ts_monitor_sys(system,ts_monitor,NULL);

  create_qvec_sys(system,&rho);
  mat_val = 1.0;
  add_to_qvec_loc(rho,mat_val,0);
  assemble_qvec(rho);

  /*
   * Set gate run_times
   */
  time_max = 0;
  single_qubit_gate_time = 20;
  two_qubit_gate_time = 100;
  for (i=0;i<circ.num_gates;i++){
    if (circ.gate_list[i].num_qubits==1){
      circ.gate_list[i].run_time = single_qubit_gate_time;
      time_max += single_qubit_gate_time;
    } else if (circ.gate_list[i].num_qubits==2){
      circ.gate_list[i].run_time = two_qubit_gate_time;
      time_max += two_qubit_gate_time;
    }
  }
  schedule_circuit_layers(system,&circ);

  //Start out circuit at time 0.0, first gate will be at 0
  apply_circuit_to_sys(system,&circ,0.0);

  //Run the evolution, with error and with the circuit
  time_step_sys(system,rho,0.0,time_max,dt,steps_max);

  get_bitstring_probs(rho,&nloc,&probs);
  for (i=0;i<nloc;i++){
    printf("probs[%d] = %f\n",i,probs[i]);
  }
  print_qvec(rho);

  //Clean up memory
  PetscScalar pop;
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

