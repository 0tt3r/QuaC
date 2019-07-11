
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
double pulse_plasmon(double,void*);
double pulse_qd(double,void*);

/* Declared globally so that we can access this in ts_monitor */
FILE *f_fid,*f_pop;
operator a,*qd;
Vec antisym_bell_dm,ptraced_dm;
double energy0_pls,pulse_duration,timeunit,pulse_t_0,mu_q,mu_s,omega;

int main(int argc,char **args){
  double gamma_pi,gamma_di,gamma_s,g_couple1,g_couple2,g_couple[2];
  double eV,debye,fluence=500,c_speed,eesu_per_au,tmp_doub,eps_med;
  PetscReal time_max,dt;
  PetscScalar val;
  PetscInt  steps_max;
  PetscInt num_plasmon=2,num_qd=2,i;
  Vec      rho;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);


  PetscOptionsGetInt(NULL,NULL,"-num_plasmon",&num_plasmon,NULL);
  PetscOptionsGetInt(NULL,NULL,"-num_qd",&num_qd,NULL);
  PetscOptionsGetReal(NULL,NULL,"-fluence",&fluence,NULL);
  fluence = 263.4;
  if(fluence<160){
    num_plasmon = 6;
  } else {
    num_plasmon = 1.1*fluence/10;
  }
  if(num_plasmon%2==1) num_plasmon = num_plasmon - 1;
  
  /* Define units, in AU */
  eV = 1/27.21140;
  debye = 0.3934303070;
  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency for plasmon, qd
  gamma_pi   = 0*1.9e-7*eV; // qd dephasing
  gamma_di   = 2.0e-3*eV; // qd decay
  gamma_s    = 186e-3*eV; // plasmon decay
  g_couple1   = 12.8e-3*eV; //qd-plasmon coupling
  g_couple2   = 24.9e-3*eV; //qd-plasmon coupling
  mu_s = 4e3*debye;
  mu_q = 1.3e1*debye;
  fluence = fluence*1e-9*1e7;
  c_speed = 2.99792458e10;
  eesu_per_au = 5.14220652e17 / c_speed;
  timeunit= 2.418884326505e-17;
  pulse_t_0     = 1e-13/timeunit;
  eps_med     = 2.25;
  pulse_duration = 12.5e-15/timeunit;
  tmp_doub = 5e-1 * sqrt(5e-1 * PETSC_PI/(2.0*log(2.0)/pow(pulse_duration,2)))*(1.0 + exp(-0.5*pow(omega,2)/
                                                                            (2.0*log(2.0)/pow(pulse_duration,2))))*timeunit;
  energy0_pls =sqrt(4*PETSC_PI*fluence/(c_speed*sqrt(eps_med)*tmp_doub));
  energy0_pls = energy0_pls / eesu_per_au;

  if(nid==0) printf("Pulse properties: t0: %f energy0: %f duration: %f \n",pulse_t_0,energy0_pls,pulse_duration);
  qd = malloc(num_qd*sizeof(struct operator));
  for (i=0;i<num_qd;i++){
    create_op(2,&qd[i]);
  }
  create_op(num_plasmon,&a);
  //  print_dense_ham();
  /* Add terms to the hamiltonian */
  add_to_ham(omega,a->n); // omega at a
  g_couple[0] = g_couple1;
  g_couple[1] = g_couple2;
  for (i=0;i<num_qd;i++){
    add_to_ham(omega,qd[i]->n); // omega qdt qd

    add_to_ham_mult2(g_couple[i],qd[i]->dag,a);  //qdt a
    add_to_ham_mult2(g_couple[i],qd[i],a->dag);  //qd at

    /* qd decay */
    add_lin(gamma_pi/2,qd[i]);
    add_lin(gamma_di,qd[i]->n);
  }
  /* plasmon decay */
  add_lin(gamma_s,a);

  /* add_to_ham(gamma_di,a->n); */
  /* add_to_ham(gamma_di,a->dag); */
  add_to_ham_time_dep(pulse_plasmon,NULL,2,a,a->dag);
  add_to_ham_time_dep(pulse_qd,NULL,2,qd[0],qd[0]->dag);
  add_to_ham_time_dep(pulse_qd,NULL,2,qd[1],qd[1]->dag);

  create_full_dm(&rho);
  set_initial_pop(a,0);
  set_initial_pop(qd[0],0);
  set_initial_pop(qd[1],0);
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

  time_step(rho,0.0,time_max,dt,steps_max);
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


double pulse_plasmon(double time,void *ctx){
  double pulse_value,alpha;
  /* Define units, in AU */

  alpha = 2 * log(2.0)/ pow(pulse_duration,2);
  pulse_value = -mu_s*energy0_pls*exp(-alpha*pow(time-pulse_t_0,2))*cos(omega*time);

  return pulse_value;

}

double pulse_qd(double time,void *ctx){
  double pulse_value,alpha;

  alpha = 2 * log(2.0)/ pow(pulse_duration,2);
  pulse_value = -mu_q*energy0_pls*exp(-alpha*pow(time-pulse_t_0,2))*cos(omega*time);

  return pulse_value;

}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  double fidelity,concurrence,*populations,pulse_val;
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
  pulse_val = pulse_qd(time,NULL)/(-mu_q);
  if (nid==0){
    /* Print fidelity and concurrence to file */
    fprintf(f_fid,"%e %e %e %e\n",time*timeunit,fidelity,concurrence,pulse_val);
  }
  free(populations);
  PetscFunctionReturn(0);

}
