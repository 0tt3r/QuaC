#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"

int main(int argc,char **args){
  int N_th,num_phonon;
  double w_m,D_e,Omega,gamma_eff,lambda_eff,lambda_s,gamma_par;
  double Q,alpha,kHz,MHz,GHz,THz,Hz,rate;
  operator a,nv;

  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  
  /* Define units, in AU */
  GHz = 1.519827e-7;
  MHz = GHz*1e-3;
  kHz = MHz*1e-3;
  THz = GHz*1e3;
  Hz  = kHz*1e-3;

  /* Define scalars to add to Ham */
  w_m        = 475*MHz*2*M_PI; //Mechanical resonator frequency
  gamma_eff  = 145.1*MHz; //Effective dissipation rate
  lambda_s   = 100*1.06*kHz*2*M_PI;
  alpha      = 0.01663;//0.0163486//0.0159748//0.0149146//0.0135
  lambda_eff = lambda_s*sqrt(alpha);//*sqrt(number_nv)
  gamma_par  = 166.666666666*MHz;
  Q          = 10^6;  //Mechanical resonator quality factor
  N_th       = 10;
  /* Create basic operators */
  num_phonon = 3;


  //  create_op(num_phonon,&a);
  create_op(2,&nv);
  
  /* Add terms to the hamiltonian */
  //  add_to_ham(w_m,a->n); // w_m at a
  
  add_to_ham(w_m,nv->n); // w_m nvt n

  /* Below 4 terms represent lambda_eff (nvt + nv)(at + a) */
  /* add_to_ham_mult2(lambda_eff,a->dag,nv->dag); //nvt at */
  /* add_to_ham_mult2(lambda_eff,nv->dag,a);  //nvt a */
  /* add_to_ham_mult2(lambda_eff,nv,a->dag);  //nt at */
  /* add_to_ham_mult2(lambda_eff,nv,a);   //nv a */

  /* /\* nv center lindblad terms *\/ */
  /* add_lin(gamma_eff/2,nv); */
  /* add_lin(gamma_par/2,nv->n); */
  /* add_lin_mult2(gamma_par/2,nv,nv->dag); */

  /* /\* phonon bath thermal terms *\/ */
  /* rate = w_m/(2*Q)*(N_th+1); */
  /* add_lin(rate,a); */

  /* rate = w_m/(2*Q)*(N_th); */
  /* add_lin(rate,a->dag); */
  
  steady_state();

  //  destroy_op(&a);
  destroy_op(&nv);

  QuaC_finalize();
  return 0;
}
