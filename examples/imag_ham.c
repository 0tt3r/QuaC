/*
* Plasmons, two 2-level quantum dots, 2 photon fields coupled with dots.
* Modified from qd_plasmons.c
*/

#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "petsc.h"
//#include <complex.h>

PetscErrorCode ts_monitor_ih(TS,PetscInt,PetscReal,Vec,void*);

/* Declared globally so that we can access this in ts_monitor */
FILE *f_pop;
operator qd,ph;

void imag_ham_dm_test(double **final_populations,int *num_pop){
  double omega,gamma_pi,gamma_di,gamma_s,gamma_ph,g_couple, g_couple_ph;
  double eV, meV;
  PetscReal time_max,dt;
  PetscScalar val;
  PetscInt  steps_max;

  PetscInt i;
  Vec      rho;

  /* Define units, in AU */ /* time in (hbar/1eV)=0.66 fs*/
  eV = 1/27.21140;
  meV = 0.001*eV;

  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency
  gamma_pi   = 0*20*meV;//1.0e-7*eV; // qd decay+dephasing
  gamma_ph   = 0*20*meV; //photon deday+dephasing
  g_couple_ph= 30.*meV;  //qd-photon coupling

  /* creating operators */
  create_op(2,&qd);            /* 2-level QDs */
  create_op(2,&ph); /* (num_photon=2)-level photons */

  /* Add terms to the hamiltonian */
  add_to_ham(omega,qd->n); // QD quadratic part
  /* qd decay and dephasing*/
  //add_lin(gamma_pi,qd);

  add_to_ham(omega,ph->n); // photon quadratic part
  //NOT EXACT, BUT GIVES GOOD RESULT :)
  /* add_to_ham_mult2(g_couple_ph,qd->dag,ph); */
  /* add_to_ham_mult2(g_couple_ph,qd,ph->dag); */

  //this GIVES NEGATIVE PHOTON POLUTATION :(
  add_to_ham_mult2(-PETSC_i*g_couple_ph,ph,qd->dag);
  add_to_ham_mult2(PETSC_i*g_couple_ph,ph->dag,qd);

    /* photon decay */
  add_lin(0*gamma_ph,ph);

  /*initial population*/
  create_full_dm(&rho);

  set_initial_pop(qd,1);
  set_initial_pop(ph,0);

  set_dm_from_initial_pop(rho);

  /* These units are hbar/eV, because we used eV as our base unit */
  time_max  = 100000;
  dt        = 0.1;
  steps_max = 14246;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor_ih);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop.dat","w");
    fprintf(f_pop,"#Time Populations: qd, ph\n");
  }

  time_step(rho,0.0,time_max,dt,steps_max);
  *num_pop = get_num_populations();
  (*final_populations) = malloc((*num_pop)*sizeof(double));
  get_populations(rho,&(*final_populations));

  destroy_op(&qd);
  destroy_op(&ph);
  destroy_dm(rho);
  if (nid==0) {fclose(f_pop);}
  return;
}

void imag_ham_psi_test(double **final_populations,int *num_pop){
  double omega,gamma_pi,gamma_di,gamma_s,gamma_ph,g_couple, g_couple_ph;
  double eV, meV;
  PetscReal time_max,dt;
  PetscScalar val;
  PetscInt  steps_max;

  PetscInt i;
  Vec      rho;

  /* Define units, in AU */ /* time in (hbar/1eV)=0.66 fs*/
  eV = 1/27.21140;
  meV = 0.001*eV;

  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency
  gamma_pi   = 0*20*meV;//1.0e-7*eV; // qd decay+dephasing
  gamma_ph   = 0*20*meV; //photon deday+dephasing
  g_couple_ph= 30.*meV;  //qd-photon coupling

  /* creating operators */
  create_op(2,&qd);            /* 2-level QDs */
  create_op(2,&ph); /* (num_photon=2)-level photons */

  /* Add terms to the hamiltonian */
  add_to_ham(omega,qd->n); // QD quadratic part
  /* qd decay and dephasing*/
  //add_lin(gamma_pi,qd);

  add_to_ham(omega,ph->n); // photon quadratic part
  //NOT EXACT, BUT GIVES GOOD RESULT :)
  /* add_to_ham_mult2(g_couple_ph,qd->dag,ph); */
  /* add_to_ham_mult2(g_couple_ph,qd,ph->dag); */

  //this GIVES NEGATIVE PHOTON POLUTATION :(
  add_to_ham_mult2(-PETSC_i*g_couple_ph,ph,qd->dag);
  add_to_ham_mult2(PETSC_i*g_couple_ph,ph->dag,qd);

    /* photon decay */
  /* add_lin(0*gamma_ph,ph); */

  /*initial population*/
  create_full_dm(&rho);

  set_initial_pop(qd,1);
  set_initial_pop(ph,0);

  set_dm_from_initial_pop(rho);

  /* These units are hbar/eV, because we used eV as our base unit */
  time_max  = 100000;
  dt        = 0.1;
  steps_max = 14246;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor_ih);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop.dat","w");
    fprintf(f_pop,"#Time Populations: qd, ph\n");
  }

  time_step(rho,0.0,time_max,dt,steps_max);
  *num_pop = get_num_populations();
  (*final_populations) = malloc((*num_pop)*sizeof(double));
  get_populations(rho,&(*final_populations));

  destroy_op(&qd);
  destroy_op(&ph);
  destroy_dm(rho);
  if (nid==0) {fclose(f_pop);}
  return;
}

void real_ham_dm_test(double **final_populations,int *num_pop){
  double omega,gamma_pi,gamma_di,gamma_s,gamma_ph,g_couple, g_couple_ph;
  double eV, meV;
  PetscReal time_max,dt;
  PetscScalar val;
  PetscInt  steps_max;

  PetscInt i;
  Vec      rho;

  /* Define units, in AU */ /* time in (hbar/1eV)=0.66 fs*/
  eV = 1/27.21140;
  meV = 0.001*eV;

  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency
  gamma_pi   = 0*20*meV;//1.0e-7*eV; // qd decay+dephasing
  gamma_ph   = 0*20*meV; //photon deday+dephasing
  g_couple_ph= 30.*meV;  //qd-photon coupling

  /* creating operators */
  create_op(2,&qd);            /* 2-level QDs */
  create_op(2,&ph); /* (num_photon=2)-level photons */

  /* Add terms to the hamiltonian */
  add_to_ham(omega,qd->n); // QD quadratic part
  /* qd decay and dephasing*/
  //add_lin(gamma_pi,qd);

  add_to_ham(omega,ph->n); // photon quadratic part

  add_to_ham_mult2(g_couple_ph,qd->dag,ph);
  add_to_ham_mult2(g_couple_ph,qd,ph->dag);

    /* photon decay */
  add_lin(0*gamma_ph,ph);

  /*initial population*/
  create_full_dm(&rho);

  set_initial_pop(qd,1);
  set_initial_pop(ph,0);

  set_dm_from_initial_pop(rho);

  /* These units are hbar/eV, because we used eV as our base unit */
  time_max  = 100000;
  dt        = 0.1;
  steps_max = 14246;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor_ih);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop.dat","w");
    fprintf(f_pop,"#Time Populations: qd, ph\n");
  }

  time_step(rho,0.0,time_max,dt,steps_max);
  *num_pop = get_num_populations();
  (*final_populations) = malloc((*num_pop)*sizeof(double));
  get_populations(rho,&(*final_populations));

  destroy_op(&qd);
  destroy_op(&ph);
  destroy_dm(rho);
  if (nid==0) {fclose(f_pop);}
  return;
}

void real_ham_psi_test(double **final_populations,int *num_pop){
  double omega,gamma_pi,gamma_di,gamma_s,gamma_ph,g_couple, g_couple_ph;
  double eV, meV;
  PetscReal time_max,dt;
  PetscScalar val;
  PetscInt  steps_max;

  PetscInt i;
  Vec      rho;

  /* Define units, in AU */ /* time in (hbar/1eV)=0.66 fs*/
  eV = 1/27.21140;
  meV = 0.001*eV;

  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency
  gamma_pi   = 0*20*meV;//1.0e-7*eV; // qd decay+dephasing
  gamma_ph   = 0*20*meV; //photon deday+dephasing
  g_couple_ph= 30.*meV;  //qd-photon coupling

  /* creating operators */
  create_op(2,&qd);            /* 2-level QDs */
  create_op(2,&ph); /* (num_photon=2)-level photons */

  /* Add terms to the hamiltonian */
  add_to_ham(omega,qd->n); // QD quadratic part
  /* qd decay and dephasing*/
  //add_lin(gamma_pi,qd);

  add_to_ham(omega,ph->n); // photon quadratic part

  add_to_ham_mult2(g_couple_ph,qd->dag,ph);
  add_to_ham_mult2(g_couple_ph,qd,ph->dag);

    /* photon decay */
  /* add_lin(0*gamma_ph,ph); */

  /*initial population*/
  create_full_dm(&rho);

  set_initial_pop(qd,1);
  set_initial_pop(ph,0);

  set_dm_from_initial_pop(rho);

  /* These units are hbar/eV, because we used eV as our base unit */
  time_max  = 100000;
  dt        = 0.1;
  steps_max = 14246;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor_ih);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop.dat","w");
    fprintf(f_pop,"#Time Populations: qd, ph\n");
  }

  time_step(rho,0.0,time_max,dt,steps_max);
  *num_pop = get_num_populations();
  (*final_populations) = malloc((*num_pop)*sizeof(double));
  get_populations(rho,&(*final_populations));

  destroy_op(&qd);
  destroy_op(&ph);
  destroy_dm(rho);
  if (nid==0) {fclose(f_pop);}
  return;
}


#ifndef UNIT_TEST
int main(int argc,char **args){
  double *populations;
  int num_pop,i;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);
  /* imag_ham_dm_test(&populations,&num_pop); */
  /* QuaC_clear(); */
  /* imag_ham_psi_test(&populations,&num_pop); */
  /* QuaC_clear(); */
  /* real_ham_dm_test(&populations,&num_pop); */
  /* QuaC_clear(); */
  real_ham_psi_test(&populations,&num_pop);

  printf("Final Populations: ");
  for(i=0;i<num_pop;i++){
    printf(" %e ",populations[i]);
  }
  printf("\n");
  QuaC_finalize();
  return 0;

}
#endif

PetscErrorCode ts_monitor_ih(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  double fidelity,concurrence,*populations;
  int num_pop,i;

  num_pop = get_num_populations();
  //printf("ts_monitor:: num_pop = %d\n",num_pop); abort(); //test
  populations = malloc(num_pop*sizeof(double));
  get_populations(dm,&populations);
  //print_dm(dm,4);
  if (nid==0){
    /* Print populations to file */
    fprintf(f_pop,"%e",time);
    for(i=0;i<num_pop;i++){
      fprintf(f_pop," %e ",populations[i]);
    }
    fprintf(f_pop,"\n");
  }
  free(populations);
  return(0);
}
