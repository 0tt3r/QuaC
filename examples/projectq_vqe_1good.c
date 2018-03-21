

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
  PetscInt  steps_max;
  Vec rho,rho_base,rho_base2,rho_base3;
  int num_qubits,i,j,h_dim,system,dm_place,logical_qubits,prev_qb,prev_qb2,num_qubits2;
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

  strcpy(filename,"NULL");
  PetscOptionsGetString(NULL,NULL,"-file_circ",filename,sizeof(filename),NULL);
  projectq_qasm_read(filename,&num_qubits,&projectq_read);

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

  if (nid==0){
    printf("qubit gam dep sigx sigy sigz om r1 r2 r3\n");
    for (i=0;i<num_qubits;i++){
      printf("%d %f %f %f %f %f %f \n",i,gamma_1[i],gamma_2[i],sigma_x[i],sigma_y[i],sigma_z[i],omega[i]);
    }
  }
  //Add lindblad terms
  for (i=0;i<num_qubits;i++){
    add_lin(gamma_1[i],qubits[i]);
    add_lin(gamma_2[i],qubits[i]->n);
  }
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

  if (nid==0){
    printf("num_gates: %d \n",projectq_read.num_gates);
  }

  start_circuit_at_time(&projectq_read,0.0);

  time_step(rho,time_max,dt,steps_max);
  projectq_vqe_get_expectation("vqe_ev",rho,&mat_val);
  /* printf("Energy: %.18lf\n",PetscRealPart(mat_val)); */
  if (nid==0) printf("Energy: %.18lf\n",PetscRealPart(mat_val));
  //print_dm_sparse(rho,16);
  for (i=0;i<num_qubits;i++){
    destroy_op(&qubits[i]);
  }
  destroy_dm(&rho);
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

