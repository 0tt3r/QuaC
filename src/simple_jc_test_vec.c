#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"

/*
 * This simple test adds does a JC like Hamiltonian with a vec op, combines them in ways
 * similar to Jaynes-Cummings for the Hamiltonian and includes
 * thermal terms for the Lindblad. 
 *
 * Should test that all of the kroneckor products with I, including
 * I_between, are working with vec_operators.
 *
 */

int main(int argc,char **args){
  double wc,wa,g,kappa,gamma,rate;
  int num_cavity,N_th,i;
  operator a;
  vec_op   sm,sm2,a2;
  
  wc         = 1.0*2*M_PI;
  wa         = 1.0*2*M_PI;
  g          = 0.05*2*M_PI;
  kappa      = 0.005;
  gamma      = 0.05;
  num_cavity = 5;
  N_th       = 1;

  QuaC_initialize(argc,args);

  create_op(num_cavity,&a);
  create_vec(3,&sm);

  /* Setup simple JC-like Hamiltonian */
   add_to_ham(wc,a->n); //wc * (at*a)

  add_to_ham(wa,sm[1]); //wa * (|sm1><sm1| - add_to_ham assumes one vec means outer product
  add_to_ham(2*wa,sm[2]); //wa * (|sm1><sm1| - add_to_ham assumes one vec means outer product

  add_to_ham_mult3(g,a->dag,sm[0],sm[1]); //g * (at*|0><1|)
  add_to_ham_mult3(g,a,sm[1],sm[0]); //g * (a*smt)

  add_to_ham_mult3(sqrt(2)*g,a->dag,sm[1],sm[2]); //g * (at*|1><2|)
  add_to_ham_mult3(sqrt(2)*g,a,sm[2],sm[1]); //g * (a*smt)


  /* Setup Lindblad operators */

  /* Cavity thermal term */
  rate = kappa*(1+N_th);
  add_lin(rate,a);

  rate = kappa*N_th;
  add_lin(rate,a->dag);

  /* Atom decay */
  add_lin_mult2(gamma,sm[0],sm[1]);
  add_lin_mult2(sqrt(2)*gamma,sm[1],sm[2]);

  steady_state();

  destroy_vec(&sm);
  destroy_op(&a);

  QuaC_finalize();
  return 0;
}
