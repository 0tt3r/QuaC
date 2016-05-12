#include <math.h>
#include <stdlib.h>
#include "quac.h"
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
  int num_cavity,N_th,i;
  operator a,sm,sm2,a2;
  vec_op   nv;
  
  wc         = 1.0*2*M_PI;
  wa         = 1.0*2*M_PI;
  g          = 0.05*2*M_PI;
  kappa      = 0.005;
  gamma      = 0.05;
  num_cavity = 5;
  N_th       = 1;

  QuaC_initialize(argc,args);

  create_op(num_cavity,&a);
  create_op(2,&sm);
  create_op(2,&sm2);
  create_op(2,&a2);

  /* Setup simple JC-like Hamiltonian */
  add_to_ham(wc,a->n); //wc * (at*a)

  add_to_ham(wa,sm->n); //wa * (smt*sm)

  add_to_ham_mult2(g,sm,a->dag); //g * (at*sm)
  add_to_ham_mult2(g,sm->dag,a); //g * (a*smt)


  add_to_ham_mult2(2*g,a->dag,sm2);
  add_to_ham_mult2(2*g,a,sm2->dag);
  add_to_ham_mult2(3*g,sm->dag,sm2);
  add_to_ham_mult2(3*g,sm,sm2->dag);
  
  add_to_ham_mult2(5*g,a2->dag,sm2);
  add_to_ham_mult2(5*g,a2,sm2->dag);


  /* Setup Lindblad operators */

  /* Cavity thermal term */
  rate = kappa*(1+N_th);
  add_lin(rate,a);

  rate = kappa*N_th;
  add_lin(rate,a->dag);

  rate = kappa*(1+N_th);
  add_lin(rate,a2);

  rate = kappa*(N_th);
  add_lin(rate,a2->dag);

  /* Atom decay */
  add_lin(gamma,sm);
  add_lin(gamma,sm2->dag);

  steady_state();

  destroy_op(&a);
  destroy_op(&sm);
  destroy_op(&sm2);
  destroy_op(&a2);

  QuaC_finalize();
  return 0;
}
