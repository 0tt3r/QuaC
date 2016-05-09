#include <math.h>
#include <stdlib.h>
#include "operators.h"

int main(int argc,char **args){
  int N_th,num_phonon;
  double w_m,D_e,Omega,gamma_eff,lambda_eff,lambda_s,gamma_par;
  double Q,alpha,kHz,MHz,GHz,THz,Hz;
  int at,a,an,nvt,nv,nvn;

  /* Initialize Petsc */
  PetscInitialize(&argc,&args,(char*)0,NULL);
  
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
  alpha      = 0.0166350247;//0.0163486//0.0159748//0.0149146//0.0135
  lambda_eff = lambda_s*sqrt(alpha);//*sqrt(number_nv)
  gamma_par  = 166.666666666*MHz;
  Q          = 10^6;  //Mechanical resonator quality factor

  /* Create basic operators */
  num_phonon = 3;

  create_op(num_phonon,&at,&a,&an);
  create_op(2,&nvt,&nv,&nvn);
  
  printf("ops: %d %d %d\n",at,a,an);
  printf("ops: %d %d %d\n",nvt,nv,nvn);
  /* Add terms to the hamiltonian */
  add_to_ham(w_m,an); // w_m at a
  
  add_to_ham(w_m,nvn); // w_m nvt n

  /* Below 4 terms represent lambda_eff (nvt + nv)(at + a) */
  add_to_ham_comb(lambda_eff,at,nvt); //nvt at
  add_to_ham_comb(lambda_eff,nvt,a);  //nvt a
  add_to_ham_comb(lambda_eff,nv,at);  //nt at
  add_to_ham_comb(lambda_eff,nv,a);   //nv a

  print_ham();
  steady_state();
  
}
