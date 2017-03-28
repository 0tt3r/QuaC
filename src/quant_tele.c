
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);

/* Declared globally so that we can access this in ts_monitor */
operator *qubits;
Vec rho_init,rho_tmp;




int main(int argc,char **args){
  PetscReal time_max,dt,*gamma_1,*omega,gate_time_step;
  PetscInt  steps_max;
  Vec rho;
  int num_qubits,i,j,h_dim,system; 
 
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 3;
  qubits  = malloc(num_qubits*sizeof(struct operator)); //Only need 3 qubits for teleportation
  gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  omega   = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  for (i=0;i<num_qubits;i++){
    create_op(2,&qubits[i]);
    omega[i]   = 0.0;
    gamma_1[i] = 0;
  }

  PetscOptionsGetReal(NULL,NULL,"-gam1",&gamma_1[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-gam2",&gamma_1[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-gam3",&gamma_1[2],NULL);
  PetscOptionsGetReal(NULL,NULL,"-om1",&omega[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-om2",&omega[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-om3",&omega[2],NULL);


  /* Add terms to the hamiltonian */
  for (i=0;i<num_qubits;i++){
    add_to_ham(omega[i],qubits[i]->n);
    /* qubit decay */
    add_lin(gamma_1[i],qubits[i]);
  }


  printf("gam: %f %f %f\n",gamma_1[0],gamma_1[1],gamma_1[2]);
  printf("om: %f %f %f\n",omega[0],omega[1],omega[2]);

  time_max  = 20;
  dt        = 0.01;
  steps_max = 1000;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor);

  create_full_dm(&rho);

  set_initial_pop(qubits[0],1);
  //  set_initial_pop(qubits[1],1);
  //  set_initial_pop(qubits[2],1);
  set_dm_from_initial_pop(rho);


  create_dm(&rho_init,2);
  create_dm(&rho_tmp,2);



  h_dim = pow(2,num_qubits);

  system = 0;
  _apply_gate(HADAMARD,&system,rho); //Hadamard on 0 for superposed starting state

 
  partial_trace_over(rho,rho_init,2,qubits[1],qubits[2]);

  //  print_dm(rho,hdim); //Print starting state
  //  systems[0] = 1;
  //  systems[1] = 2;

  //  _apply_gate(HADAMARD,systems,rho); //Hadamard on 1
  //  print_dm(rho,h_dim);

  //  _apply_gate(CNOT,systems,rho); //CNOT
  //  print_dm(rho,h_dim);

  //  systems[0]=0; systems[1]=1;
  //  _apply_gate(CNOT,systems,rho); //CNOT 
  //  print_dm(rho,h_dim);

  //  _apply_gate(HADAMARD,systems,rho); //Hadamard on 0
  //  print_dm(rho,h_dim);

  //  systems[0]=1; systems[1]=2;
  //  _apply_gate(CNOT,systems,rho); //CNOT 
  //  print_dm(rho,h_dim);

  //  systems[0] = 2;
  //  _apply_gate(HADAMARD,systems,rho); //Hadamard on 2
  //  print_dm(rho,h_dim);

  //  systems[0]=0; systems[1]=2;
  //  _apply_gate(CNOT,systems,rho); //CNOT 
  //  print_dm(rho,h_dim);

  //  systems[0] = 2;
  //  _apply_gate(HADAMARD,systems,rho); //Hadamard on 2
  //  print_dm(rho,h_dim);

  gate_time_step = 1.0;
  add_gate(1*gate_time_step,HADAMARD,1); //Hadamard on 1
  add_gate(2*gate_time_step,CNOT,1,2); //CNOT(1,2)
  add_gate(3*gate_time_step,CNOT,0,1); //CNOT(0,1)
  add_gate(4*gate_time_step,HADAMARD,0); //Hadamard on 0
  add_gate(5*gate_time_step,CNOT,1,2); //CNOT(1,2)
  add_gate(6*gate_time_step,HADAMARD,2); //Hadamard on 2
  add_gate(7*gate_time_step,CNOT,0,2); //CNOT(0,2)
  add_gate(8*gate_time_step,HADAMARD,2); //Hadamard on 2


  
  time_step(rho,time_max,dt,steps_max);
  //  steady_state(rho);

  //  partial_trace_over(rho,rho_init,2,qubits[0],qubits[1]);
  print_dm(rho,h_dim);
  //  print_dm(rho_init,2);
  destroy_dm(rho_init);
  destroy_dm(rho_tmp);
  for (i=0;i<num_qubits;i++){
    destroy_op(&qubits[i]);
  }
  free(qubits);
  destroy_dm(rho);
  QuaC_finalize();
  return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho,void *ctx){
  double fidelity;
  /* get_populations prints to pop file */
  get_populations(rho,time);
  partial_trace_over(rho,rho_tmp,2,qubits[0],qubits[1]);
  get_fidelity(rho_init,rho_tmp,&fidelity);
  if(nid==0) printf("%f %f\n",time,fidelity);
  
  //  print_dm(dm,2);
  PetscFunctionReturn(0);

}
