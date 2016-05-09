#include <math.h>
#include <stdlib.h>
#include "operators.h"

int main(int argc,char **args){
  double wc,wa,g,kappa,gamma,rate;
  int num_cavity,N_th;
  int at,a,an,sm,smt,smn;

  /* Initialize Petsc */
  PetscInitialize(&argc,&args,(char*)0,NULL);

  wc         = 1.0*2*M_PI;
  wa         = 1.0*2*M_PI;
  g          = 0.05*2*M_PI;
  kappa      = 0.005;
  gamma      = 0.05;
  num_cavity = 5;
  N_th       = 1;
  
  create_op(num_cavity,&at,&a,&an);
  create_op(2,&smt,&sm,&smn);

  /* Setup simple JC Hamiltonian */
  add_to_ham(wc,an); //wc * (at*a)
  add_to_ham(wa,smn); //wa * (smt*sm)

  add_to_ham_comb(g,at,sm); //g * (at*sm)
  add_to_ham_comb(g,a,smt); //g * (a*smt)

  /* Setup Lindblad operators */

  /* Cavity thermal term */
  /* rate = kappa*(1+N_th); */
  /* add_lin(rate,a); */

  /* rate = kappa*N_th; */
  /* add_lin(rate,at); */

  /* /\* Atom decay *\/ */
  /* add_lin(gamma,sm); */

  print_ham();
  steady_state();

}
