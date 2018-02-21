
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "error_correction.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);

/* Declared globally so that we can access this in ts_monitor */
operator *qubits;
Vec rho_init,rho_tmp,rho_tmp2;
FILE *f_pop;
PetscInt final_qubit;
encoded_qubit L0,L1,L2;

int main(int argc,char **args){
  PetscReal time_max,dt,*gamma_1,*gamma_2,*omega,*sigma_x,*sigma_y,*sigma_z;
  PetscReal *gamma_1L,*gamma_2L,*sigma_xL,*sigma_yL,*sigma_zL;
  PetscReal gate_time_step,theta,fidelity,t1,t2;
  PetscScalar mat_val;
  PetscInt  steps_max;
  Vec rho,rho_base,rho_base2,rho_base3;
  int num_qubits,i,j,h_dim,system,dm_place,logical_qubits,prev_qb,prev_qb2;
  circuit teleportation,encoder1;
  Mat     encoder_mat;
  char           string[10];
  stabilizer     S1,S2,S3,S4;
  PetscReal *r_str;
  char           encoder_str[128];
  encoder_type   encoder_type0,encoder_type1,encoder_type2;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 0;
  /* Get the encoding strategies for each qubit */

  strcpy(encoder_str,"NULL");
  PetscOptionsGetString(NULL,NULL,"-encoder0",encoder_str,sizeof(encoder_str),NULL);
  if (strcmp(encoder_str,"BIT")==0){
    if (nid==0) printf("Using BIT encoding for logical qubit 0!\n");
    num_qubits = num_qubits + 3;
    encoder_type0 = BIT;
  } else if (strcmp(encoder_str,"FIVE")==0){
    if (nid==0) printf("Using FIVE encoding for logical qubit 0!\n");
    num_qubits = num_qubits + 5;
    encoder_type0 = FIVE;
  } else if (strcmp(encoder_str,"PHASE")==0){
    if (nid==0) printf("Using PHASE encoding for logical qubit 0!\n");
    num_qubits = num_qubits + 3;
    encoder_type0 = PHASE;
  } else if (strcmp(encoder_str,"NONE")==0){
    if (nid==0) printf("Using NONE encoding for logical qubit 0!\n");
    num_qubits = num_qubits + 1;
    encoder_type0 = NONE;
  } else {
    if (nid==0) printf("Encoding not understood or provided, defaulting to NONE\n");
    num_qubits = num_qubits + 1;
    encoder_type0 = NONE;
  }

  strcpy(encoder_str,"NULL");
  PetscOptionsGetString(NULL,NULL,"-encoder1",encoder_str,sizeof(encoder_str),NULL);
  if (strcmp(encoder_str,"BIT")==0){
    if (nid==0) printf("Using BIT encoding for logical qubit 1!\n");
    num_qubits = num_qubits + 3;
    encoder_type1 = BIT;
  } else if (strcmp(encoder_str,"FIVE")==0){
    if (nid==0) printf("Using FIVE encoding for logical qubit 1!\n");
    num_qubits = num_qubits + 5;
    encoder_type1 = FIVE;
  } else if (strcmp(encoder_str,"PHASE")==0){
    if (nid==0) printf("Using PHASE encoding for logical qubit 1!\n");
    num_qubits = num_qubits + 3;
    encoder_type1 = PHASE;
  } else if (strcmp(encoder_str,"NONE")==0){
    if (nid==0) printf("Using NONE encoding for logical qubit 1!\n");
    num_qubits = num_qubits + 1;
    encoder_type1 = NONE;
  } else {
    if (nid==0) printf("Encoding not understood or provided, defaulting to NONE\n");
    num_qubits = num_qubits + 1;
    encoder_type1 = NONE;
  }
  //Set the final qubit to be the first of the final encoding
  final_qubit = num_qubits;

  strcpy(encoder_str,"NULL");
  PetscOptionsGetString(NULL,NULL,"-encoder2",encoder_str,sizeof(encoder_str),NULL);
  if (strcmp(encoder_str,"BIT")==0){
    if (nid==0) printf("Using BIT encoding for logical qubit 2!\n");
    num_qubits = num_qubits + 3;
    encoder_type2 = BIT;
  } else if (strcmp(encoder_str,"FIVE")==0){
    if (nid==0) printf("Using FIVE encoding for logical qubit 2!\n");
    num_qubits = num_qubits + 5;
    encoder_type2 = FIVE;
  } else if (strcmp(encoder_str,"PHASE")==0){
    if (nid==0) printf("Using PHASE encoding for logical qubit 2!\n");
    num_qubits = num_qubits + 3;
    encoder_type2 = PHASE;
  } else if (strcmp(encoder_str,"NONE")==0){
    if (nid==0) printf("Using NONE encoding for logical qubit 2!\n");
    num_qubits = num_qubits + 1;
    encoder_type2 = NONE;
  } else {
    if (nid==0) printf("Encoding not understood or provided, defaulting to NONE\n");
    num_qubits = num_qubits + 1;
    encoder_type2 = NONE;
  }

  logical_qubits = 3;
  qubits  = malloc(num_qubits*sizeof(struct operator)); //Only need 3 qubits for teleportation
  gamma_1 = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  gamma_2 = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_x = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_y = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_z = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  omega   = malloc(num_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation

  gamma_1L = malloc(logical_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  gamma_2L = malloc(logical_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_xL = malloc(logical_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_yL = malloc(logical_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  sigma_zL = malloc(logical_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation
  r_str   = malloc(logical_qubits*sizeof(PetscReal)); //Only need 3 qubits for teleportation

  for (i=0;i<num_qubits;i++){
    create_op(2,&qubits[i]);
    omega[i]   = 0.0;
    gamma_1[i] = 0;
    gamma_2[i] = 0;
    sigma_x[i] = 0;
    sigma_y[i] = 0;
    sigma_z[i] = 0;
  }

  for (i=0;i<logical_qubits;i++){
    r_str[i]   = 0.0;
    gamma_1L[i] = 0;
    gamma_2L[i] = 0;
    sigma_xL[i] = 0;
    sigma_yL[i] = 0;
    sigma_zL[i] = 0;
  }
  theta = 0.0;

  for (i=0; i<num_qubits;i++){
    sprintf(string, "-gam%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&gamma_1[i],NULL);
    sprintf(string, "-dep%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&gamma_2[i],NULL);
    sprintf(string, "-sigx%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&sigma_x[i],NULL);
    sprintf(string, "-sigy%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&sigma_y[i],NULL);
    sprintf(string, "-sigz%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&sigma_z[i],NULL);
    sprintf(string, "-om%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&omega[i],NULL);
  }
  for (i=0; i<logical_qubits;i++){
    sprintf(string, "-r%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&r_str[i],NULL);
    sprintf(string, "-gamL%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&gamma_1L[i],NULL);
    sprintf(string, "-depL%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&gamma_2L[i],NULL);
    sprintf(string, "-sigxL%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&sigma_xL[i],NULL);
    sprintf(string, "-sigyL%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&sigma_yL[i],NULL);
    sprintf(string, "-sigzL%d",i);
    PetscOptionsGetReal(NULL,NULL,string,&sigma_zL[i],NULL);
  }

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

  if (nid==0){
    printf("Before logical error adding: \n");
    printf("qubit gam dep sigx sigy sigz om r1 r2 r3\n");
    for (i=0;i<num_qubits;i++){
      printf("%d %f %f %f %f %f %f \n",i,gamma_1[i],gamma_2[i],sigma_x[i],sigma_y[i],sigma_z[i],omega[i]);
    }
  }

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



  dm_place = pow(2,num_qubits-1);
  //Set the initial DM
  mat_val = cos(theta)*cos(theta);
  add_value_to_dm(rho,0,0,mat_val);
  mat_val = cos(theta)*sin(theta);
  add_value_to_dm(rho,dm_place,0,mat_val);
  add_value_to_dm(rho,0,dm_place,mat_val);
  mat_val = sin(theta)*sin(theta);
  add_value_to_dm(rho,dm_place,dm_place,mat_val);
  assemble_dm(rho);

  /* h_dim = pow(2,num_qubits); */

  //partial_trace_over(rho_base,rho_init,5,qubits[1],qubits[2],qubits[3],qubits[4],qubits[5]);
  partial_trace_keep(rho,rho_init,1,qubits[0]);
  print_dm(rho_init,2);

  gate_time_step = 1.0;

  // Set the encodings
  prev_qb = 0;
  prev_qb2 = 0;
  if (encoder_type0==BIT||encoder_type0==PHASE){
    //Three qubit encoding
    create_encoded_qubit(&L0,encoder_type0,prev_qb,prev_qb+1,prev_qb+2);
    prev_qb = prev_qb + 3;
  } else if (encoder_type0==FIVE) {
    //FIVE qubit encoding
    create_encoded_qubit(&L0,encoder_type0,prev_qb,prev_qb+1,prev_qb+2,prev_qb+3,prev_qb+4);
    prev_qb = prev_qb + 5;
  } else {
    //NONE qubit encoding
    create_encoded_qubit(&L0,encoder_type0,prev_qb);
    prev_qb = prev_qb + 1;
  }

  //Add same error terms to all physical qubits within a logical qubit
  for (i=prev_qb2;i<prev_qb;i++){
    /* qubit decay */
    gamma_1[i] = gamma_1L[0];
    gamma_2[i] = gamma_2L[0];
    sigma_x[i]  = sigma_xL[0];
    sigma_y[i]  = sigma_yL[0];
    sigma_z[i]  = sigma_zL[0];
    add_lin(gamma_1[i],qubits[i]);
    add_lin(gamma_2[i],qubits[i]->n);
    add_lin(sigma_x[i],qubits[i]->sig_x);
    add_lin(sigma_y[i],qubits[i]->sig_y);
    add_lin(sigma_z[i],qubits[i]->sig_z);
  }
  prev_qb2 = prev_qb;

  if (encoder_type1==BIT||encoder_type1==PHASE){
    //Three qubit encoding
    create_encoded_qubit(&L1,encoder_type1,prev_qb,prev_qb+1,prev_qb+2);
    prev_qb = prev_qb + 3;
  } else if (encoder_type1==FIVE) {
    //FIVE qubit encoding
    create_encoded_qubit(&L1,encoder_type1,prev_qb,prev_qb+1,prev_qb+2,prev_qb+3,prev_qb+4);
    prev_qb = prev_qb + 5;
  } else {
    //NONE qubit encoding
    create_encoded_qubit(&L1,encoder_type1,prev_qb);
    prev_qb = prev_qb + 1;
  }

  //Add same error terms to all physical qubits within a logical qubit
  for (i=prev_qb2;i<prev_qb;i++){
    /* qubit decay */
    gamma_1[i] = gamma_1L[1];
    gamma_2[i] = gamma_2L[1];
    sigma_x[i]  = sigma_xL[1];
    sigma_y[i]  = sigma_yL[1];
    sigma_z[i]  = sigma_zL[1];
    add_lin(gamma_1[i],qubits[i]);
    add_lin(gamma_2[i],qubits[i]->n);
    add_lin(sigma_x[i],qubits[i]->sig_x);
    add_lin(sigma_y[i],qubits[i]->sig_y);
    add_lin(sigma_z[i],qubits[i]->sig_z);
  }
  prev_qb2 = prev_qb;

  if (encoder_type2==BIT||encoder_type2==PHASE){
    //Three qubit encoding
    create_encoded_qubit(&L2,encoder_type2,prev_qb,prev_qb+1,prev_qb+2);
    prev_qb = prev_qb + 3;
  } else if (encoder_type2==FIVE) {
    //FIVE qubit encoding
    create_encoded_qubit(&L2,encoder_type2,prev_qb,prev_qb+1,prev_qb+2,prev_qb+3,prev_qb+4);
    prev_qb = prev_qb + 5;
  } else {
    //NONE qubit encoding
    create_encoded_qubit(&L2,encoder_type2,prev_qb);
    prev_qb = prev_qb + 1;
  }

  //Add same error terms to all physical qubits within a logical qubit
  for (i=prev_qb2;i<prev_qb;i++){
    /* qubit decay */
    gamma_1[i] = gamma_1L[2];
    gamma_2[i] = gamma_2L[2];
    sigma_x[i]  = sigma_xL[2];
    sigma_y[i]  = sigma_yL[2];
    sigma_z[i]  = sigma_zL[2];
    add_lin(gamma_1[i],qubits[i]);
    add_lin(gamma_2[i],qubits[i]->n);
    add_lin(sigma_x[i],qubits[i]->sig_x);
    add_lin(sigma_y[i],qubits[i]->sig_y);
    add_lin(sigma_z[i],qubits[i]->sig_z);
  }
  prev_qb2 = prev_qb;


  if (nid==0){
    printf("After logical error adding: \n");
    printf("qubit gam dep sigx sigy sigz om\n");
    for (i=0;i<num_qubits;i++){
      printf("%d %f %f %f %f %f %f \n",i,gamma_1[i],gamma_2[i],sigma_x[i],sigma_y[i],sigma_z[i],omega[i]);
    }
  }

  encode_state(rho,3,L0,L1,L2);

  /* print_dm_sparse(rho,pow(2,num_qubits)); */
  /* exit(0); */
  create_circuit(&teleportation,1000);

  add_encoded_gate_to_circuit(&teleportation,1*gate_time_step,HADAMARD,L1); //Hadamard on 1
  add_encoded_gate_to_circuit(&teleportation,2*gate_time_step,CNOT,L1,L2); //CNOT(1,2)
  add_encoded_gate_to_circuit(&teleportation,3*gate_time_step,CNOT,L0,L1); //CNOT(0,1)
  add_encoded_gate_to_circuit(&teleportation,4*gate_time_step,HADAMARD,L0); //Hadamard on 0
  add_encoded_gate_to_circuit(&teleportation,5*gate_time_step,CNOT,L1,L2); //CNOT(1,2)
  add_encoded_gate_to_circuit(&teleportation,6*gate_time_step,HADAMARD,L2); //Hadamard on 2
  add_encoded_gate_to_circuit(&teleportation,7*gate_time_step,CNOT,L0,L2); //CNOT(0,2)
  add_encoded_gate_to_circuit(&teleportation,8*gate_time_step,HADAMARD,L2); //Hadamard on 2

  if (nid==0){
    printf("num_gates: %d\n",teleportation.num_gates);
  }



  start_circuit_at_time(&teleportation,0.0);

  add_continuous_error_correction(L0,r_str[0]);
  add_continuous_error_correction(L1,r_str[1]);
  add_continuous_error_correction(L2,r_str[2]);

  time_step(rho,time_max,dt,steps_max);

  //  osteady_state(rho);:
  //partial_trace_over(rho,rho_tmp,5,qubits[0],qubits[1],qubits[2],qubits[3],qubits[5]);
  decode_state(rho,3,L0,L1,L2);
  partial_trace_keep(rho,rho_tmp,1,qubits[final_qubit]);
  /* partial_trace_keep(rho,rho_tmp,1,qubits[0]); */
  get_fidelity(rho_init,rho_tmp,&fidelity);
  if(nid==0) printf("Final fidelity: %f\n",fidelity);

  //  partial_trace_over(rho,rho_init,2,qubits[0],qubits[1]);
  print_dm(rho,h_dim);
  if(nid==0) printf("Final PTraced DM: C\n");
  print_dm(rho_tmp,2);
  QuaC_finalize();
  return 0;
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
  double fidelity,*populations;  int num_pop,i;
 ///num_pop = get_num_populations();
  //populations = malloc(num_pop*sizeof(double));
  //get_populations(rho,&populations);
  //if (nid==0){
    /* Print populations to file */
  //  fprintf(f_pop,"%e",time);
  //  for(i=0;i<num_pop;i++){
  //    fprintf(f_pop," %e ",populations[i]);
  //  }
   // fprintf(f_pop,"\n");
  //}
  //partial_trace_keep(rho,rho_tmp,1,qubits[final_qubit]);
  //get_fidelity(rho_init,rho_tmp,&fidelity);
  if(nid==0) printf("%f \n",time);
  //free(populations);
  /* destroy_dm(&rho_tmp2); */
  /* printf("Sparse dm: \n"); */
  /* printf("\n"); */
  /* print_dm_sparse(rho,pow(2,7)); */
  /* printf("\n"); */
  PetscFunctionReturn(0);

}
