#include <math.h>
#include <stdlib.h>
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

int main(int argc,char **args){
  double wc,wa,g,kappa,gamma,rate;
  int num_cavity,N_th;
  int at,a,an,sm,smt,smn;
  int sm2,sm2t,sm2n;
  int a2t,a2,a2n;
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
  create_op(2,&sm2t,&sm2,&sm2n);
  create_op(2,&a2t,&a2,&a2n);

  /* Setup simple JC Hamiltonian */
  add_to_ham(wc,an); //wc * (at*a)
  add_to_ham(wa,smn); //wa * (smt*sm)

  add_to_ham_comb(g,at,sm); //g * (at*sm)
  add_to_ham_comb(g,a,smt); //g * (a*smt)
  add_to_ham_comb(2*g,at,sm2);
  add_to_ham_comb(2*g,a,sm2t);
  add_to_ham_comb(3*g,smt,sm2);
  add_to_ham_comb(3*g,sm,sm2t);
  
  add_to_ham_comb(5*g,a2t,sm2);
  add_to_ham_comb(5*g,a2,sm2t);

  /* Setup Lindblad operators */

  /* Cavity thermal term */
  rate = kappa*(1+N_th);
  add_lin(rate,a);

  rate = kappa*N_th;
  add_lin(rate,at);

  rate = kappa*(1+N_th);
  add_lin(rate,a2);

  rate = kappa*(N_th);
  add_lin(rate,a2t);

  /* Atom decay */
  add_lin(gamma,sm);
  add_lin(gamma,sm2t);

  print_ham();
  steady_state();

}
