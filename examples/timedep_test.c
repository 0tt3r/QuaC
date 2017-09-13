
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
double pulse(double);
FILE *f_pop;
/* Declared globally so that we can access this in ts_monitor */

void timedep_test(double **final_populations,int *num_pop){
  operator a,b;
  int n1,n2;
  double g,kappa1,kappa2;
  PetscReal time_max,dt;
  PetscInt steps_max;
  Vec rho;
  n1 = 2;
  n2 = 3;
  create_op(n1,&b);
  create_op(n2,&a);

  g = 5;
  add_to_ham_mult2(-g,b->dag,a);
  add_to_ham_mult2(-g,a->dag,b);
  /* add_to_ham(1.0,a->dag); */
  /* add_to_ham(1.0,a); */
   add_to_ham_time_dep(pulse,2,a->dag,a);
  kappa1 = 1.5;
  add_lin(kappa1,a);

  kappa2 = 3;
  add_lin(kappa2,b);

  create_full_dm(&rho);

  set_initial_pop(a,0);
  set_initial_pop(b,0);
  set_dm_from_initial_pop(rho);

  time_max = 30;
  dt = 0.1;
  steps_max = 300;
  set_ts_monitor(ts_monitor);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop","w");
    fprintf(f_pop,"#Time Populations\n");
  }

  time_step(rho,time_max,dt,steps_max);
  *num_pop = get_num_populations();
  (*final_populations) = malloc((*num_pop)*sizeof(double));
  get_populations(rho,&(*final_populations));
  destroy_op(&a);
  destroy_op(&b);
  destroy_dm(rho);

  return;
}

#ifndef UNIT_TEST
int main(int argc,char **args){
  double *populations;
  int num_pop,i;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);
  timedep_test(&populations,&num_pop);
  printf("Final Populations: ");
  for(i=0;i<num_pop;i++){
    printf(" %e ",populations[i]);
  }
  printf("\n");
  QuaC_finalize();
  return 0;

}
#endif

double pulse(double time){
  double pulse_value;

  pulse_value = 1 * exp(-pow(time/5,2));
  return pulse_value;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  double pulse_value,*populations;
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

  pulse_value = pulse(time);
  free(populations);
  PetscFunctionReturn(0);
}
