
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
double pulse(double);

/* Declared globally so that we can access this in ts_monitor */
FILE *f_fid,*f_pop;
operator a,*qd;
Vec antisym_bell_dm,ptraced_dm;

int main(int argc,char **args){
  double omega,gamma_pi,gamma_di,gamma_s,g_couple;
  double eV;
  PetscReal time_max,dt;
  PetscScalar val;
  PetscInt  steps_max;
  PetscInt num_plasmon=2,num_qd=2,i;
  Vec      rho;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);


  PetscOptionsGetInt(NULL,NULL,"-num_plasmon",&num_plasmon,NULL);
  PetscOptionsGetInt(NULL,NULL,"-num_qd",&num_qd,NULL);

  /* Define units, in AU */
  eV = 1/27.21140;

  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency for plasmon, qd
  gamma_pi   = 0*1.0e-7*eV; // qd dephasing
  gamma_di   = 2.0e-3*eV; // qd decay
  gamma_s    = 1.5e-1*eV; // plasmon decay
  g_couple   = 30e-3*eV; //qd-plasmon coupling


  qd = malloc(num_qd*sizeof(struct operator));
  for (i=0;i<num_qd;i++){
    create_op(2,&qd[i]);
  }
  create_op(num_plasmon,&a);
  /* Add terms to the hamiltonian */
  add_to_ham(omega,a->n); // omega at a
  for (i=0;i<num_qd;i++){
    add_to_ham(omega,qd[i]->n); // omega qdt qd

    add_to_ham_mult2(g_couple,qd[i]->dag,a);  //qdt a
    add_to_ham_mult2(g_couple,qd[i],a->dag);  //qd at

    /* qd decay */
    add_lin(gamma_pi/2,qd[i]);
    add_lin(gamma_di,qd[i]->n);
  }
  /* plasmon decay */
  add_lin(gamma_s,a);

  /* add_to_ham(gamma_di,a->n); */
  /* add_to_ham(gamma_di,a->dag); */
  add_to_ham_time_dep(pulse,1,a->n);

  create_full_dm(&rho);
  set_initial_pop(a,0);
  set_initial_pop(qd[0],0);
  set_initial_pop(qd[1],1);
  set_dm_from_initial_pop(rho);

  /* Create a reference dm (the antisym bell state) for fidelity calculations */
  create_dm(&antisym_bell_dm,4);

  val = 0.5;
  add_value_to_dm(antisym_bell_dm,1,1,val);
  add_value_to_dm(antisym_bell_dm,2,2,val);
  val = -0.5;
  add_value_to_dm(antisym_bell_dm,1,2,val);
  add_value_to_dm(antisym_bell_dm,2,1,val);

  assemble_dm(antisym_bell_dm);

  /*
   * Also create a place to store the partial trace
   * No assembly is necessary here, as we will be ptracing into this dm
   */
  create_dm(&ptraced_dm,4);

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

  time_step(rho,time_max,dt,steps_max);
  /* steady_state(rho); */

  destroy_op(&a);
  for (i=0;i<num_qd;i++){
    destroy_op(&qd[i]);
  }
  free(qd);
  destroy_dm(ptraced_dm);
  destroy_dm(antisym_bell_dm);
  destroy_dm(rho);
  if (nid==0) fclose(f_fid);
  QuaC_finalize();
  return 0;
}


double pulse(double time){
  double pulse_value,eV,gamma_di;
  /* Define units, in AU */
  eV = 1/27.21140;

  /* Define scalars to add to Ham */
  gamma_di   = 2.0e-3*eV; // qd decay

  pulse_value = gamma_di;

  return pulse_value;

}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  double fidelity,concurrence,*populations;
  int num_pop,i;

  num_pop = get_num_populations();
  populations = malloc(num_pop*sizeof(double));
  get_populations(dm,&populations);

  if (nid==0){
    /* Print populations to file */
    fprintf(f_pop,"%e",time);
    for(i=0;i<num_pop;i++){
      fprintf(f_pop," %e ",populations[i]);
    }
    fprintf(f_pop,"\n");
  }

  /* Partial trace away the oscillator */

  partial_trace_over(dm,ptraced_dm,1,a);

  get_fidelity(antisym_bell_dm,ptraced_dm,&fidelity);
  get_bipartite_concurrence(ptraced_dm,&concurrence);

  if (nid==0){
    /* Print fidelity and concurrence to file */
    fprintf(f_fid,"%e %e %e\n",time,fidelity,concurrence);
  }
  PetscFunctionReturn(0);

}
