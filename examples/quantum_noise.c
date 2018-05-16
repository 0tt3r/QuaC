#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"
#include "dm_utilities.h"

/*
 * This simple test adds several operators, combines them in ways
 * similar to Jaynes-Cummings for the Hamiltonian and includes
 * thermal terms for the Lindblad.
 *
 * Should test that all of the kroneckor products with I, including
 * I_between, are working.
 *
 */
PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
FILE *f_pop;

int main(int argc,char **args){
  double w0,wtls,wtls_base,GHz,MHz,g,g_mult,dt,time_max;
  PetscInt num_cavity,i,num_tls,steps_max,init_cavity,levels;
  operator a,*tls;
  Vec      rho;

  MHz        = 1;
  GHz        = .001;

  w0         = 5*GHz;
  wtls_base  = 10*MHz;
  g_mult     = 1;

  init_cavity = 5;
  num_cavity = 6;
  num_tls    = 2;
  QuaC_initialize(argc,args);

  PetscOptionsGetInt(NULL,NULL,"-num_tls",&num_tls,NULL);
  PetscOptionsGetInt(NULL,NULL,"-num_cavity",&num_cavity,NULL);
  PetscOptionsGetReal(NULL,NULL,"-g_mult",&g_mult,NULL);
  PetscOptionsGetInt(NULL,NULL,"-init_cavity",&init_cavity,NULL);
  g = g_mult*wtls_base;

  create_op(num_cavity,&a);

  tls = malloc(num_tls*sizeof(struct operator));
  for (i=0;i<num_tls;i++){
    create_op(2,&tls[i]);
  }
  print_dense_ham();
  /* Setup simple JC-like Hamiltonian */
  add_to_ham(w0,a->n); //w0 * (at*a)
  for (i=0;i<num_tls;i++){
    wtls = i*50*MHz + wtls_base;
    add_to_ham_stiff(wtls,tls[i]->n); //wtls * (tls^t*tls)
    add_to_ham_stiff_mult2(g,a->dag,tls[i]); //g * (a^t*tls)
    add_to_ham_stiff_mult2(g,a,tls[i]->dag); //g * (a*tls^t)
  }
  /* g = 1.0*GHz*0; */
  /* add_lin(g,a); */
  create_full_dm(&rho);

  set_initial_pop(a,init_cavity);
  set_dm_from_initial_pop(rho);

  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop","w");
    fprintf(f_pop,"#Time (us) Populations\n");
  }

  time_max = 10; //Units of MHz (in eV) / hbar = 0.1592 microseconds
  dt       = 0.0001;
  steps_max = 1000;

  set_ts_monitor(ts_monitor);
  //  steady_state(rho);
  time_step(rho,0.0,time_max,dt,steps_max);

  for (i=0;i<num_tls;i++){
    destroy_op(&tls[i]);
  }
  free(tls);

  destroy_op(&a);
  destroy_dm(rho);

  QuaC_finalize();
  return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  double *populations;
  PetscReal norm, time_us;
  int num_pop,i;

  num_pop = get_num_populations();

  populations = malloc(num_pop*sizeof(double));
  VecNorm(dm,NORM_2,&norm);
  //print_psi(dm,3);
  get_populations(dm,&populations);
  time_us = time*0.1592; // Based on MHz (in eV) / hbar
  if (nid==0){
    /* Print populations to file */
    fprintf(f_pop,"%e",time_us);
    for(i=0;i<num_pop;i++){
      fprintf(f_pop," %e ",populations[i]);
    }
    fprintf(f_pop,"\n");
  }
  free(populations);
  PetscFunctionReturn(0);

}
