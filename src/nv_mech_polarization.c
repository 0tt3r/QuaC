#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"

int main(int argc,char **args){
  /* tc is whether to do Tavis Cummings or not */
  PetscInt n_th=10,num_phonon=200; //Default values set here
  PetscInt i;
  PetscReal lambda,T2star,resFreq,magDrvM,magDrvP,gamOpto;
  PetscReal Q,kHz,MHz,GHz,THz,Hz,rate;
  operator a;
  vec_op   nv;

  enum STATE {gm=0,g0,gp};
  
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  
  /* Define units, in AU */
  GHz = 1.519827e-7;
  MHz = 1;//GHz*1e-3;
  kHz = MHz*1e-3;
  THz = GHz*1e3;
  Hz  = kHz*1e-3;

  /* Get arguments from command line */
  PetscOptionsGetInt(NULL,NULL,"-num_phonon",&num_phonon,NULL);
  PetscOptionsGetInt(NULL,NULL,"-n_th",&n_th,NULL);

  if (nid==0) printf("Num_phonon: %d n_th: %d \n",num_phonon,n_th);
  /* Define scalars to add to Ham */
  lambda     = 0.1*MHz*2*M_PI;
  T2star     = 1*MHz;
  resFreq    = 5*MHz*2*M_PI;
  magDrvM    = 0;
  gamOpto    = 2*MHz;
  magDrvP    = lambda;
  Q          = pow(10,6);

  //  print_dense_ham();

  create_vec(3,&nv);
  create_op(num_phonon,&a);



  /* Add terms to the hamiltonian */
  add_to_ham(resFreq,a->n); // w_m at a
  add_to_ham_mult2(magDrvP/2,nv[gp],nv[g0]); //magDrvP/2 |+1><0|
  add_to_ham_mult2(magDrvP/2,nv[g0],nv[gp]);//magDrvP/2 |0><+1|

  add_to_ham_mult2(magDrvM/2,nv[gm],nv[g0]); //magDrvM/2 |-1><0|
  add_to_ham_mult2(magDrvM/2,nv[g0],nv[gm]);//magDrvM/2 |0><-1|

  add_to_ham(-resFreq,nv[gm]); //-resFreq |-1><-1|

  /* Lindblad terms */
  /* 1/T2star */
  rate = 1/T2star;
  add_lin_mult2(rate,nv[gp],nv[gp]); //L gp gp
  add_lin_mult2(rate,nv[gm],nv[gm]); //L gm gm

  
  /* /\* Below 4 terms represent coupling *\/ */
  add_to_ham_mult3(lambda,nv[gm],nv[gp],a); // |e-><e+| a
  add_to_ham_mult3(lambda,nv[gp],nv[gm],a->dag); // |e+><e-| at
  
  /* phonon bath thermal terms */
  rate = resFreq/(Q)*(n_th+1);
  add_lin(rate,a);

  rate = resFreq/(Q)*(n_th);
  add_lin(rate,a->dag);

  rate = gamOpto*2;
  add_lin(rate,a);

  set_initial_pop(a,8);
  set_initial_pop(nv[gm],1.);
  set_initial_pop(nv[gp],1.);
  set_initial_pop(nv[g0],1.);

  time_step();
  /* time_step_euler(); */
  /* steady_state(); */

  destroy_op(&a);
  destroy_vec(&nv);
  QuaC_finalize();

  return 0;
}
