#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"

int main(int argc,char **args){
  /* tc is whether to do Tavis Cummings or not */
  PetscInt n_th=2,num_phonon=5,num_nv=2,tc=0; //Default values set here
  PetscInt i,nv_levels=2,full_H_space;
  PetscReal w_m,D_e,Omega,gamma_eff,lambda_eff,lambda_s,gamma_par;
  PetscReal Q,alpha,kHz,MHz,GHz,THz,Hz,rate;
  operator a,*nv;
  Vec rho;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  /* Define units, in AU */
  GHz = 1e3;//1.519827e-7;
  MHz = GHz*1e-3;
  kHz = MHz*1e-3;
  THz = GHz*1e3;
  Hz  = kHz*1e-3;

  /* Get arguments from command line */
  PetscOptionsGetInt(NULL,NULL,"-num_nv",&num_nv,NULL);
  PetscOptionsGetInt(NULL,NULL,"-num_phonon",&num_phonon,NULL);
  PetscOptionsGetInt(NULL,NULL,"-n_th",&n_th,NULL);
  PetscOptionsGetInt(NULL,NULL,"-tc",&tc,NULL);
  PetscOptionsGetInt(NULL,NULL,"-nv_levels",&nv_levels,NULL);

  if (nid==0) printf("Num_phonon: %d n_th: %d num_nv: %d nv_levels: %d Tavis Cummings?: %d\n",num_phonon,n_th,num_nv,nv_levels,tc);
  /* Define scalars to add to Ham */
  w_m        = 475*MHz*2*M_PI; //Mechanical resonator frequency
  gamma_eff  = 145.1*MHz; //Effective dissipation rate
  lambda_s   = 10*1.06*kHz*2*M_PI;
  lambda_s   = 0.1*MHz*2*M_PI;

  gamma_par  = 166.666666666*MHz;
  Q          = pow(10,6);  //Mechanical resonator quality factor


  print_dense_ham();


  if (tc) {
    nv = malloc(1*sizeof(struct operator));
    /* Use Tavis Cummings approximation */
    create_op(nv_levels,&nv[0]);
    create_op(num_phonon,&a);
    lambda_eff = lambda_s*sqrt(num_nv);
    /* Add terms to the hamiltonian */
    add_to_ham(w_m,a->n); // w_m at a
    add_to_ham(w_m,nv[0]->n); // w_m nvt nv

    /* Below 4 terms represent lambda_eff (nvt + nv)(at + a) */
    add_to_ham_mult2(lambda_eff,a->dag,nv[0]->dag); //nvt at
    add_to_ham_mult2(lambda_eff,nv[0]->dag,a);  //nvt a
    add_to_ham_mult2(lambda_eff,nv[0],a->dag);  //nt at
    add_to_ham_mult2(lambda_eff,nv[0],a);   //nv a

    /* nv center lindblad terms */
    add_lin(gamma_eff,nv[0]);
    //add_lin(gamma_par,nv[0]->n);
    //    add_lin_mult2(gamma_par,nv[0],nv[0]->dag);
  } else {
    /* Do not use Tavis Cummings */
    nv = malloc(num_nv*sizeof(struct operator));
    for (i=0;i<num_nv;i++){
      create_op(nv_levels,&nv[i]);
    }
    create_op(num_phonon,&a);
    add_to_ham(w_m,a->n); // w_m at a

    lambda_eff = lambda_s;
    for (i=0;i<num_nv;i++){
      add_to_ham(w_m,nv[i]->n); // w_m nvt nv

      /* Below 4 terms represent lambda_eff (nvt + nv)(at + a) */
      /* add_to_ham_mult2(lambda_eff,a->dag,nv[i]->dag); //nvt at */
      add_to_ham_mult2(lambda_eff,nv[i]->dag,a);  //nvt a
      add_to_ham_mult2(lambda_eff,nv[i],a->dag);  //nt at
      /* add_to_ham_mult2(lambda_eff,nv[i],a);   //nv a */

      /* nv center lindblad terms */
      add_lin(gamma_eff,nv[i]);
      /* add_lin(gamma_par,nv[i]->n); */
      /* add_lin_mult2(gamma_par,nv[i],nv[i]->dag); */
    }
  }




  /* phonon bath thermal terms */
  rate = w_m/(Q)*(n_th+1);
  add_lin(rate,a);

  rate = w_m/(Q)*(n_th);
  add_lin(rate,a->dag);

  create_full_dm(&rho);
  steady_state(rho);

  destroy_op(&a);
  if (tc) {
    destroy_op(&nv[0]);
  } else {
    for (i=0;i<num_nv;i++){
      destroy_op(&nv[i]);
    }
  }
  free(nv);
  destroy_dm(rho);
  QuaC_finalize();
  return 0;
}
