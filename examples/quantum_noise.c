#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"

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
  PetscInt num_cavity,i,num_tls,steps_max,init_cavity;
  operator a,*tls;
  Vec      rho;

  MHz        = 1000;
  GHz        = 1;

  w0         = 7*GHz;
  wtls_base  = 10*MHz;
  g_mult     = 0.1;

  init_cavity = 5;
  num_cavity = 15;
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
    wtls = i*5 + wtls_base;
    add_to_ham(wtls,tls[i]->n); //wtls * (tls^t*tls)
    add_to_ham_mult2(g,a->dag,tls[i]); //g * (a^t*tls)
    add_to_ham_mult2(g,a,tls[i]->dag); //g * (a*tls^t)
  }

  create_full_dm(&rho);

  set_initial_pop(a,init_cavity);
  set_dm_from_initial_pop(rho);

  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop","w");
    fprintf(f_pop,"#Time Populations\n");
  }

  time_max = 10; //Units of 1/MHz = microseconds
  dt       = 0.1;
  steps_max = 1000;

  set_ts_monitor(ts_monitor);
  //  steady_state(rho);
  time_step(rho,time_max,dt,steps_max);

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

  PetscFunctionReturn(0);

}
