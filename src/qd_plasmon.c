
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"

int main(int argc,char **args){
  double omega,gamma_pi,gamma_di,gamma_s,g_couple;
  double eV;
  PetscReal time_max,dt;
  PetscInt  steps_max;
  int num_plasmon=10,num_qd=2,i;
  operator a,*qd;

  
  /* Initialize QuaC */
  QuaC_initialize(argc,args);


  PetscOptionsGetInt(NULL,NULL,"-num_plasmon",&num_plasmon,NULL);
  PetscOptionsGetInt(NULL,NULL,"-num_qd",&num_qd,NULL);
  
  /* Define units, in AU */
  eV = 1/27.21140;

  /* Define scalars to add to Ham */
  omega      = 2.05*eV; //natural frequency for plasmon, qd
  gamma_pi   = 1.0e-7*eV; // qd dephasing
  gamma_di   = 2.0e-3*eV; // qd decay
  gamma_s    = 1.5e-1*eV; // plasmon decay
  g_couple   = 30e-3*eV; //qd-plasmon coupling


  qd = malloc(num_qd*sizeof(struct operator));
  for (i=0;i<num_qd;i++){
    create_op(2,&qd[i]);
  }
  create_op(num_plasmon,&a);  
  /* Add terms to the hamiltonian */
  add_to_ham(omega,a->n); // omega at a
  for (i=0;i<num_qd;i++){
    add_to_ham(omega,qd[i]->n); // omega qdt qd
    
    add_to_ham_mult2(g_couple,qd[i]->dag,a);  //qdt a
    add_to_ham_mult2(g_couple,qd[i],a->dag);  //qd at
    
    /* qd decay */
    add_lin(gamma_pi/2,qd[i]);
    add_lin(gamma_di,qd[i]->n);
  }
  /* plasmon decay */
  add_lin(gamma_s,a);

  set_initial_pop(a,8);
  set_initial_pop(qd[0],1);
  set_initial_pop(qd[1],1);

  /* What units are these?! */
  time_max  = 10000;
  dt        = 0.01;
  steps_max = 100;

  time_step(time_max,dt,steps_max);
  /* steady_state(); */

  destroy_op(&a);
  destroy_op(&qd);

  QuaC_finalize();
  return 0;
}
