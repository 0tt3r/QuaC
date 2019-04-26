

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
  PetscReal time_max,dt,*gamma_1,*gamma_2;
  PetscReal gate_time_step,theta,fidelity,t1,t2;
  PetscScalar mat_val;
  PetscInt  steps_max,num_qubits;
  Vec rho,rho_base,rho_base2,rho_base3;
  int i,j,h_dim,system,dm_place,logical_qubits,prev_qb,prev_qb2,num_qubits2;
  circuit qiskit_read;
  Mat     encoder_mat;
  char           string[10],filename[128];
  stabilizer     S1,S2,S3,S4;
  PetscReal *r_str,gam,dep,noise;
  char           encoder_str[128];
  encoder_type   encoder_type0,encoder_type1,encoder_type2,encoder_type3;
  struct quantum_gate_struct *gate_list;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  num_qubits = 0;

  strcpy(filename,"NULL");
  PetscOptionsGetString(NULL,NULL,"-file_circ",filename,sizeof(filename),NULL);
  qiskit_qasm_read(filename,&num_qubits,&qiskit_read);

  strcpy(filename,"NULL");
  PetscOptionsGetString(NULL,NULL,"-file_ev",filename,sizeof(filename),NULL);

  qubits  = malloc(num_qubits*sizeof(struct operator));
  gamma_1 = malloc(num_qubits*sizeof(PetscReal));
  gamma_2 = malloc(num_qubits*sizeof(PetscReal));
  for (i=0;i<num_qubits;i++){
    create_op(2,&qubits[i]);
    gamma_1[i] = 0;
    gamma_2[i] = 0;
    printf("MY_OP_TYPE op %d qubits[i]->dag %d qubits[i]->dag->dag %d\n",qubits[i]->my_op_type,qubits[i]->dag->my_op_type,qubits[i]->dag->dag->my_op_type);
  }

  gam = 1.0;
  dep = 1.0;
  noise = 0;
  PetscOptionsGetReal(NULL,NULL,"-noise",&noise,NULL);

  for (i=0; i<num_qubits;i++){
    gamma_1[i] = gam*noise;
    gamma_2[i] = dep*noise;
  }

  //Add lindblad terms
  for (i=0;i<num_qubits;i++){
    if (gamma_1[i]>0){
      //Spontaneous emission
      add_lin(gamma_1[i],qubits[i]);
    }
    if (gamma_2[i]>0){
      //Dephasing
      add_lin(gamma_2[i],qubits[i]->n);
    }
  }

  time_max  = qiskit_read.num_gates + 1;

  dt        = 1;
  steps_max = 10000000;

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
    printf("num_gates: %ld \n",qiskit_read.num_gates);
  }

  start_circuit_at_time(&qiskit_read,0.0);

  time_step(rho,0.0,time_max,dt,steps_max);
  /* get_expectation_value(rho,&mat_val,2,qubits[0]->sig_z,qubits[0]->sig_z); */
  qiskit_vqe_get_expectation(filename,rho,&mat_val);

  if (nid==0) printf("Energy: %.18lf\n",PetscRealPart(mat_val));
  //print_psi(rho,16);
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

