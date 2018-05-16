
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
FILE *f_pop;


int main(int argc,char **args){
  PetscReal time_max,dt,*gamma_1,*gamma_2,*omega,*sigma_x,*sigma_y,*sigma_z;
  PetscReal gate_time_step,theta,fidelity,t1,t2;
  PetscScalar mat_val;
  PetscInt  steps_max;
  Vec rho;
  int num_qubits,i,j,h_dim,system;
  circuit teleportation,teleportation2;
  Mat circ_mat,circ_mat2;
  PetscViewer    mat_view;

  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 3;
  qubits  = malloc(num_qubits*sizeof(struct operator)); //Only need 3 qubits for teleportation
  gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  gamma_2 = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_x = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_y = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_z = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  omega   = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  for (i=0;i<num_qubits;i++){
    create_op(2,&qubits[i]);
    omega[i]   = 0.0;
    gamma_1[i] = 0;
    gamma_2[i] = 0;
    sigma_x[i] = 0;
    sigma_y[i] = 0;
    sigma_z[i] = 0;
  }

  theta = 0.0;
  PetscOptionsGetReal(NULL,NULL,"-gam1",&gamma_1[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-gam2",&gamma_1[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-gam3",&gamma_1[2],NULL);

  PetscOptionsGetReal(NULL,NULL,"-dep1",&gamma_2[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-dep2",&gamma_2[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-dep3",&gamma_2[2],NULL);

  PetscOptionsGetReal(NULL,NULL,"-sigx1",&sigma_x[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-sigx2",&sigma_x[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-sigx3",&sigma_x[2],NULL);

  PetscOptionsGetReal(NULL,NULL,"-sigy1",&sigma_y[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-sigy2",&sigma_y[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-sigy3",&sigma_y[2],NULL);

  PetscOptionsGetReal(NULL,NULL,"-sigz1",&sigma_z[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-sigz2",&sigma_z[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-sigz3",&sigma_z[2],NULL);

  PetscOptionsGetReal(NULL,NULL,"-om1",&omega[0],NULL);
  PetscOptionsGetReal(NULL,NULL,"-om2",&omega[1],NULL);
  PetscOptionsGetReal(NULL,NULL,"-om3",&omega[2],NULL);

  PetscOptionsGetReal(NULL,NULL,"-theta",&theta,NULL);

  /* Add terms to the hamiltonian */
  for (i=0;i<num_qubits;i++){
    add_to_ham(omega[i],qubits[i]->n);
    /* qubit decay */
    add_lin(gamma_1[i],qubits[i]);
    add_lin(gamma_2[i],qubits[i]->n);
    add_lin(sigma_x[i],qubits[i]->sig_x);
    add_lin(sigma_y[i],qubits[i]->sig_y);
    add_lin(sigma_z[i],qubits[i]->sig_z);
  }


  printf("gam: %f %f %f\n",gamma_1[0],gamma_1[1],gamma_1[2]);
  printf("dep: %f %f %f\n",gamma_2[0],gamma_2[1],gamma_2[2]);
  printf("sigx: %f %f %f\n",sigma_x[0],sigma_x[1],sigma_x[2]);
  printf("sigy: %f %f %f\n",sigma_y[0],sigma_y[1],sigma_y[2]);
  printf("sigz: %f %f %f\n",sigma_z[0],sigma_z[1],sigma_z[2]);
  printf("om: %f %f %f\n",omega[0],omega[1],omega[2]);
  printf("theta: %f\n",theta);

  time_max  = 9;
  dt        = 0.01;
  steps_max = 1000;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop","w");
    fprintf(f_pop,"#Time Populations\n");
  }

  create_full_dm(&rho);

  create_dm(&rho_init,2);
  create_dm(&rho_tmp,2);

  create_circuit(&teleportation,20);

  //Set the initial DM
  mat_val = cos(theta)*cos(theta);
  mat_val = 0.5;
  add_value_to_dm(rho,0,0,mat_val);
  /* mat_val = cos(theta)*sin(theta); */
  mat_val = -0.5;
  add_value_to_dm(rho,4,0,mat_val);
  add_value_to_dm(rho,0,4,mat_val);
  /* mat_val = sin(theta)*sin(theta); */
  mat_val = 0.5;
  add_value_to_dm(rho,4,4,mat_val);
  assemble_dm(rho);

  /* print_dm(rho,8); */
  h_dim = pow(2,num_qubits);

  partial_trace_over(rho,rho_init,2,qubits[1],qubits[2]);
  // 1 = Identity
  // 2 = H on A
  // 3 = H on B
  // 4 = H on C
  // 5 = CNOT(A,B) [A control]
  // 6 = CNOT(B,C) [B control]
  // 7 = CNOT(A,C) [A control]
  /* gate_time_step = 1.0; */
  /* add_gate(1*gate_time_step,HADAMARD,1); //Hadamard on 1 */
  /* add_gate(2*gate_time_step,CNOT,1,2); //CNOT(1,2) */
  /* add_gate(3*gate_time_step,CNOT,0,1); //CNOT(0,1) */
  /* add_gate(4*gate_time_step,HADAMARD,0); //Hadamard on 0 */
  /* add_gate(5*gate_time_step,CNOT,1,2); //CNOT(1,2) */
  /* add_gate(6*gate_time_step,HADAMARD,2); //Hadamard on 2 */
  /* add_gate(7*gate_time_step,CNOT,0,2); //CNOT(0,2) */
  /* add_gate(8*gate_time_step,HADAMARD,2); //Hadamard on 2 */

  gate_time_step = 1.0;
  /* add_gate_to_circuit(&teleportation,1*gate_time_step,HADAMARD,1); //Hadamard on 1 */
  /* add_gate_to_circuit(&teleportation,1*gate_time_step,CNOT,1,2); //CNOT(1,2) */
  /* add_gate_to_circuit(&teleportation,1*gate_time_step,CNOT,0,1); //CNOT(0,1) */
  /* add_gate_to_circuit(&teleportation,1*gate_time_step,HADAMARD,0); //Hadamard on 0 */
  /* add_gate_to_circuit(&teleportation,1*gate_time_step,CNOT,1,2); //CNOT(1,2) */
  add_gate_to_circuit(&teleportation,1*gate_time_step,HADAMARD,2); //Hadamard on 2
  /* add_gate_to_circuit(&teleportation,1*gate_time_step,CNOT,0,2); //CNOT(0,2) */
  /* add_gate_to_circuit(&teleportation,8*gate_time_step,HADAMARD,2); //Hadamard on 2 */

  combine_circuit_to_mat(&circ_mat,teleportation);
  start_circuit_at_time(&teleportation,0.0);

  /* add_gate(8*gate_time_step,CNOT,1,2); //CNOT(1,2) */
  time_step(rho,0.0,time_max,dt,steps_max);
  //  steady_state(rho);:
  partial_trace_over(rho,rho_tmp,2,qubits[0],qubits[1]);
  get_fidelity(rho_init,rho_tmp,&fidelity);
  if(nid==0) printf("Final fidelity: %f\n",fidelity);

  //  partial_trace_over(rho,rho_init,2,qubits[0],qubits[1]);
  print_dm(rho,h_dim);
  if(nid==0) printf("Final PTraced DM: C\n");
  print_dm(rho_tmp,2);
  if(nid==0) printf("Final PTraced DM: A\n");
  partial_trace_over(rho,rho_tmp,2,qubits[1],qubits[2]);
  print_dm(rho_tmp,2);
  if(nid==0) printf("Final PTraced DM: B\n");
  partial_trace_over(rho,rho_tmp,2,qubits[0],qubits[2]);
  print_dm(rho_tmp,2);

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
  double fidelity,*populations;
  int num_pop,i;

  num_pop = get_num_populations();
  populations = malloc(num_pop*sizeof(double));
  get_populations(rho,&populations);
  if (nid==0){
    /* Print populations to file */
    fprintf(f_pop,"%e",time);
    for(i=0;i<num_pop;i++){
      fprintf(f_pop," %e ",populations[i]);
    }
    fprintf(f_pop,"\n");
  }

  partial_trace_over(rho,rho_tmp,2,qubits[0],qubits[1]);
  get_fidelity(rho_init,rho_tmp,&fidelity);
  if(nid==0) printf("%f %f\n",time,fidelity);
  free(populations);
  //  print_dm(dm,2);
  PetscFunctionReturn(0);

}
