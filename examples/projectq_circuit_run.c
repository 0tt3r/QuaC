

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

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);

/* Declared globally so that we can access this in ts_monitor */
operator *qubits;
Vec rho_init,rho_tmp,rho_tmp2;
FILE *f_pop;
PetscInt final_qubit;
encoded_qubit L0,L1,L2,L3;

int main(int argc,char **args){
  PetscReal time_max,dt,*gamma_1,*gamma_2,*omega,*sigma_x,*sigma_y,*sigma_z;
  PetscReal *gamma_1L,*gamma_2L,*sigma_xL,*sigma_yL,*sigma_zL;
  PetscReal gate_time_step,theta,fidelity,t1,t2;
  PetscScalar mat_val;
  PetscInt  steps_max,num_qubits2;
  Vec rho,rho_base,rho_base2,rho_base3;
  int num_qubits,i,j,h_dim,system,dm_place,logical_qubits,prev_qb,prev_qb2;
  circuit projectq_read,encoded_projectq;
  Mat     encoder_mat;
  char           string[10],filename[128];
  stabilizer     S1,S2,S3,S4;
  PetscReal *r_str;
  char           encoder_str[128];
  encoder_type   encoder_type0,encoder_type1,encoder_type2,encoder_type3;
  struct quantum_gate_struct *gate_list;
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



  strcpy(encoder_str,"NULL");
  PetscOptionsGetString(NULL,NULL,"-encoder3",encoder_str,sizeof(encoder_str),NULL);
  if (strcmp(encoder_str,"BIT")==0){
    if (nid==0) printf("Using BIT encoding for logical qubit 3!\n");
    num_qubits = num_qubits + 3;
    encoder_type3 = BIT;
  } else if (strcmp(encoder_str,"FIVE")==0){
    if (nid==0) printf("Using FIVE encoding for logical qubit 3!\n");
    num_qubits = num_qubits + 5;
    encoder_type3 = FIVE;
  } else if (strcmp(encoder_str,"PHASE")==0){
    if (nid==0) printf("Using PHASE encoding for logical qubit 3!\n");
    num_qubits = num_qubits + 3;
    encoder_type3 = PHASE;
  } else if (strcmp(encoder_str,"NONE")==0){
    if (nid==0) printf("Using NONE encoding for logical qubit 3!\n");
    num_qubits = num_qubits + 1;
    encoder_type3 = NONE;
  } else {
    if (nid==0) printf("Encoding not understood or provided, defaulting to NONE\n");
    num_qubits = num_qubits + 1;
    encoder_type3 = NONE;
  }

  strcpy(filename,"NULL");
  PetscOptionsGetString(NULL,NULL,"-file_circ",filename,sizeof(filename),NULL);
  projectq_qasm_read(filename,&num_qubits2,&projectq_read);
  logical_qubits = 4;

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

  if (encoder_type3==BIT||encoder_type3==PHASE){
    //Three qubit encoding
    create_encoded_qubit(&L3,encoder_type3,prev_qb,prev_qb+1,prev_qb+2);
    prev_qb = prev_qb + 3;
  } else if (encoder_type3==FIVE) {
    //FIVE qubit encoding
    create_encoded_qubit(&L3,encoder_type3,prev_qb,prev_qb+1,prev_qb+2,prev_qb+3,prev_qb+4);
    prev_qb = prev_qb + 5;
  } else {
    //NONE qubit encoding
    create_encoded_qubit(&L3,encoder_type3,prev_qb);
    prev_qb = prev_qb + 1;
  }

  //Add same error terms to all physical qubits within a logical qubit
  for (i=prev_qb2;i<prev_qb;i++){
    /* qubit decay */
    gamma_1[i] = gamma_1L[3];
    gamma_2[i] = gamma_2L[3];
    sigma_x[i]  = sigma_xL[3];
    sigma_y[i]  = sigma_yL[3];
    sigma_z[i]  = sigma_zL[3];
    add_lin(gamma_1[i],qubits[i]);
    add_lin(gamma_2[i],qubits[i]->n);
    add_lin(sigma_x[i],qubits[i]->sig_x);
    add_lin(sigma_y[i],qubits[i]->sig_y);
    add_lin(sigma_z[i],qubits[i]->sig_z);
  }
  prev_qb2 = prev_qb;


  if (nid==0){
    printf("After logical error adding: \n");
    printf("qubit gam dep sigx sigy sigz om r1 r2 r3\n");
    for (i=0;i<num_qubits;i++){
      printf("%d %f %f %f %f %f %f \n",i,gamma_1[i],gamma_2[i],sigma_x[i],sigma_y[i],sigma_z[i],omega[i]);
    }
  }
  create_circuit(&encoded_projectq,10000);
  encode_circuit(projectq_read,&encoded_projectq,4,L0,L1,L2,L3);

  time_max  = projectq_read.num_gates + 1;
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
  mat_val = 1.0;
  add_value_to_dm(rho,0,0,mat_val);
  assemble_dm(rho);
  encode_state(rho,4,L0,L1,L2,L3);
  if (nid==0){
    printf("num_gates: %ld %ld\n",encoded_projectq.num_gates,projectq_read.num_gates);
  }

  printf("correction_strength: %f %f %f %f\n",r_str[0],r_str[1],r_str[2],r_str[3]);
  add_continuous_error_correction(L0,r_str[0]);
  add_continuous_error_correction(L1,r_str[1]);
  add_continuous_error_correction(L2,r_str[2]);
  add_continuous_error_correction(L3,r_str[3]);

  start_circuit_at_time(&encoded_projectq,0.0);
  //start_circuit_at_time(&projectq_read,0.0);

  time_step(rho,0.0,time_max,dt,steps_max);
  decode_state(rho,4,L0,L1,L2,L3);
  /* projectq_vqe_get_expectation((char *)"vqe_ev",rho,&mat_val); */
  /* printf("Energy: %.18lf\n",PetscRealPart(mat_val)); */
  projectq_vqe_get_expectation_encoded((char *)"vqe_ev",rho,&mat_val,4,L0,L1,L2,L3);
  if (nid==0) printf("Energy: %.18lf\n",PetscRealPart(mat_val));
  //print_dm_sparse(rho,16);
  for (i=0;i<num_qubits;i++){
    destroy_op(&qubits[i]);
  }
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
  /* if(nid==0) printf("%f \n",time); */
  //free(populations);
  /* destroy_dm(&rho_tmp2); */
  /* printf("Sparse dm: \n"); */
  /* printf("\n"); */
  /* print_dm_sparse(rho,pow(2,7)); */
  /* printf("\n"); */
  PetscFunctionReturn(0);

}

