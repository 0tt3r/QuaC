#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"

int main(int argc,char **args){
  int N_th,num_phonon,num_nv;
  double w_m,D_e,Omega,gamma_eff,lambda_eff,lambda_s,gamma_par;
  double Q,alpha,kHz,MHz,GHz,THz,Hz,rate;
  operator a,nv;
  FILE     *fp;

  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  
  /* Define units, in AU */
  GHz = 1.519827e-7;
  MHz = GHz*1e-3;
  kHz = MHz*1e-3;
  THz = GHz*1e3;
  Hz  = kHz*1e-3;
  alpha      = 0.01663;
  N_th       = 10;
  num_phonon = 3;
  num_nv     = 1;
  /* Read in important parameters */
  fp         = fopen("parameters","r");
  fscanf(fp, "%*[^\n]\n", NULL); //Skip first line
  fscanf(fp,"%d %d %d %f",&num_phonon,&N_th,&num_nv,&alpha);
  fclose(fp);
  if (nid==0) printf("Num_phonon: %d N_th: %d num_nv: %d alpha: %f\n",num_phonon,N_th,num_nv,alpha);
  /* Define scalars to add to Ham */
  w_m        = 475*MHz*2*M_PI; //Mechanical resonator frequency
  gamma_eff  = 145.1*MHz; //Effective dissipation rate
  lambda_s   = 100*1.06*kHz*2*M_PI;

  lambda_eff = lambda_s*sqrt(alpha)*sqrt(num_nv);
  gamma_par  = 166.666666666*MHz;
  Q          = pow(10,6);  //Mechanical resonator quality factor


  print_dense_ham();

  create_op(num_phonon,&a);
  create_op(2,&nv);
  
  /* Add terms to the hamiltonian */
  add_to_ham(w_m,a->n); // w_m at a
  
  add_to_ham(w_m,nv->n); // w_m nvt n

  /* Below 4 terms represent lambda_eff (nvt + nv)(at + a) */
  add_to_ham_mult2(lambda_eff,a->dag,nv->dag); //nvt at
  add_to_ham_mult2(lambda_eff,nv->dag,a);  //nvt a
  add_to_ham_mult2(lambda_eff,nv,a->dag);  //nt at
  add_to_ham_mult2(lambda_eff,nv,a);   //nv a

  /* nv center lindblad terms */
  add_lin(gamma_eff,nv);
  add_lin(gamma_par,nv->n);
  add_lin_mult2(gamma_par,nv,nv->dag);

  /* phonon bath thermal terms */
  rate = w_m/(Q)*(N_th+1);
  add_lin(rate,a);

  rate = w_m/(Q)*(N_th);
  add_lin(rate,a->dag);
  
  steady_state();

  destroy_op(&a);
  destroy_op(&nv);

  QuaC_finalize();
  return 0;
}
