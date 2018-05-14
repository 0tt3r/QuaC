
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
FILE *f_fid,*f_pop;
operator *qd;
Vec antisym_bell_dm;

int main(int argc,char **args){
  double omega,gamma_pi,gamma_di;
  double eV;
  PetscReal time_max,dt;
  PetscScalar val;
  PetscInt  steps_max;
  PetscInt num_qd=2,i;

  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  PetscOptionsGetInt(NULL,NULL,"-num_qd",&num_qd,NULL);

  /* Define units, in AU */
  eV = 1/27.21140;

  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency for plasmon, qd
  gamma_pi   = 1.0e-7*eV; // qd dephasing
  gamma_di   = 2.0e-3*eV; // qd decay


  qd = malloc(num_qd*sizeof(struct operator));
  for (i=0;i<num_qd;i++){
    create_op(2,&qd[i]);
  }

  /* Add terms to the hamiltonian */
  for (i=0;i<num_qd;i++){
    add_to_ham(omega,qd[i]->n); // omega qdt qd
    /* qd decay */
    /* add_lin(gamma_pi/2,qd[i]); */
    /* add_lin(gamma_di,qd[i]->n); */
    add_lin(gamma_di,qd[i]->sig_x);
    /* add_lin(gamma_di,qd[i]->n); */
    /* add_lin(gamma_di,qd[i]->n); */
  }


  /* Create a reference dm (the antisym bell state) for fidelity calculations */
  create_dm(&antisym_bell_dm,4);

  val = 0.5;
  add_value_to_dm(antisym_bell_dm,1,1,val);
  add_value_to_dm(antisym_bell_dm,2,2,val);
  val = 0.5;
  add_value_to_dm(antisym_bell_dm,1,2,val);
  add_value_to_dm(antisym_bell_dm,2,1,val);

  assemble_dm(antisym_bell_dm);
  /* set_initial_pop(qd[0],1); */
  /* set_initial_pop(qd[1],1); */
  /* set_dm_from_initial_pop(antisym_bell_dm); */

  /*
   * Also create a place to store the partial trace
   * No assembly is necessary here, as we will be ptracing into this dm
   */

  /* These units are 1/eV, because we used eV as our base unit */
  time_max  = 100000;
  dt        = 0.01;
  steps_max = 100;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_fid = fopen("fid","w");
    fprintf(f_fid,"#Time Fidelity Concurrence\n");
    f_pop = fopen("pop","w");
    fprintf(f_pop,"#Time Populations\n");

  }

  time_step(antisym_bell_dm,0.0,time_max,dt,steps_max);
  /* steady_state(antisym_bell_dm); */

  for (i=0;i<num_qd;i++){
    destroy_op(&qd[i]);
  }
  free(qd);

  destroy_dm(antisym_bell_dm);
  if (nid==0) fclose(f_fid);
  QuaC_finalize();
  return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  double fidelity,concurrence,fidelity2,*populations;
  PetscScalar dm_element;
  int num_pop,i;

  num_pop = get_num_populations();
  populations = malloc(num_pop*sizeof(double));
  get_populations(dm,&populations);

  /* Partial trace away the oscillator */

  get_fidelity(antisym_bell_dm,dm,&fidelity);
  get_bipartite_concurrence(dm,&concurrence);
  if (nid==0){
    /* Print populations to file */
    fprintf(f_pop,"%e",time);
    for(i=0;i<num_pop;i++){
      fprintf(f_pop," %e ",populations[i]);
    }
    fprintf(f_pop,"\n");
  }

  /*
   * Fidelity: F = 0.5 * (rho(01,01) + rho(10,10) -
   *                      rho(01,10) - rho(10,01))
   */
  get_dm_element(dm,1,1,&dm_element);
  fidelity2 = dm_element;
  get_dm_element(dm,2,2,&dm_element);
  fidelity2 += dm_element;
  get_dm_element(dm,1,2,&dm_element);
  fidelity2 -= dm_element;
  get_dm_element(dm,2,1,&dm_element);
  fidelity2 -= dm_element;
  fidelity2 *= 0.5;

  if (nid==0){
    /* Print fidelity and concurrence to file */
    fprintf(f_fid,"%e %e %e %e\n",time,fidelity*fidelity,concurrence,fidelity2);
  }
  free(populations);
  PetscFunctionReturn(0);

}
