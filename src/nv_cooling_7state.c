#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "petsc.h"


void add_nv_terms(vec_op,operator,double);

int main(int argc,char **args){
  int      N_th,num_phonon,num_nv,i;
  double   w_m,lambda_s;
  double   Q,kHz,MHz,GHz,rate;
  operator a;
  vec_op   *nv;
  Vec      rho;

  /* enumerate state names so we don't to remember them */
  enum STATE {gp=0,g0,gm,ep,e0,em,s};

  /* Initialize QuaC */
  QuaC_initialize(argc,args);


  
  GHz = 1.519827e-7;
  MHz = GHz*1e-3;
  kHz = MHz*1e-3;

  N_th       = 10;
  num_phonon = 152;
  num_nv     = 1;

  /* Get arguments from command line */
  num_nv     = atoi(args[1]);
  num_phonon = atoi(args[2]);
  N_th       = atoi(args[3]);

  if (nid==0) printf("Num_phonon: %d N_th %d num_nv: %d\n",num_phonon,N_th,num_nv);
  lambda_s   = 100*1.06*kHz*2*M_PI;

  nv = malloc(num_nv*sizeof(vec_op));

  /* Define scalars to add to Ham */
  w_m        = 475*MHz*2*M_PI; //Mechanical resonator frequency
  Q          = pow(10,6);  //Mechanical resonator quality factor

  /* print_dense_ham(); */
  /* NV = {g+,g0,g-,e+,e0,e-,s} */
  /*       0  1  2  3  4  5  6 */
  create_op(num_phonon,&a);  

  for (i=0;i<num_nv;i++){
    create_vec(7,&nv[i]);
  }

  /* Add terms to the hamiltonian */
  add_to_ham(w_m,a->n); // w_m at a
  for (i=0;i<num_nv;i++){
    /* All nv terms are in a function below */
    add_nv_terms(nv[i],a,lambda_s);
  }
  if (nid==0) printf("Added Hamiltonian.");
  /* phonon bath thermal terms */
  rate = w_m/(Q)*(N_th+1); //CHECK A FACTOR OF 2 HERE!
  add_lin(rate,a);

  rate = w_m/(Q)*(N_th);
  add_lin(rate,a->dag);
  
  create_full_dm(&rho);
  //  time_step();
  steady_state(rho);

  
  destroy_op(&a);
  for (i=0;i<num_nv;i++){
    destroy_vec(&nv[i]);
  }
  free(nv);
  destroy_dm(rho);
  QuaC_finalize();
  return 0;
}

/* Add all of the terms for an NV */
void add_nv_terms(vec_op nv,operator a,double lambda_s){
  double MHz,GHz,rate,T2g,T2e,omega,w_m;
  double k42,k31,k45,k35,k52,k51,gamma_opt,ns,timeunit;
  /* enumerate state names so we don't to remember them */
  enum STATE {gp=0,g0,gm,ep,e0,em,s};
  /* Define units, in AU */
  timeunit= 2.418884326505e-17;
  ns = 1e-9/timeunit;

  GHz = 1.519827e-7;
  MHz = GHz*1e-3;

  w_m        = 475*MHz*2*M_PI; //Mechanical resonator frequency
  omega      = 100*2*M_PI*MHz;

  gamma_opt  = 200*MHz;
  k42        = 65.3*MHz;
  k31        = 64.9*MHz;
  k45        = 79.8*MHz;
  k35        = 10.6*MHz;
  k52        = 2.61*MHz;
  k51        = 3.00*MHz;
  T2g        = 59*ns;
  T2e        = 6*ns;

  add_to_ham_mult2(omega/2,nv[e0],nv[em]); // |e0><e-|
  add_to_ham_mult2(omega/2,nv[em],nv[e0]); // |e-><e0

  add_to_ham_mult3(lambda_s,nv[em],nv[ep],a); // |e-><e+| a
  add_to_ham_mult3(lambda_s,nv[em],nv[ep],a->dag); // |e-><e+| at

  add_to_ham_mult3(lambda_s,nv[ep],nv[em],a); // |e+><e-| a
  add_to_ham_mult3(lambda_s,nv[ep],nv[em],a->dag); // |e+><e-| at


  add_to_ham(w_m,nv[ep]); // |e+><e+|


  /* Lindblad terms for NV 
   * Note that lince we define L as L(C) = L(|f><|i|), this
   * is slightly different than Evan's paper, which defines
   * L_i,j. I have put Evan's definition to the side of each
   * term, but the values entered are reversed.
   */

  /* gamma_opt terms */
  add_lin_mult2(gamma_opt,nv[ep],nv[gp]); // L g+ e+
  add_lin_mult2(gamma_opt,nv[e0],nv[g0]); // L g0 e0
  add_lin_mult2(gamma_opt,nv[em],nv[gm]); // L g- e-

  /* k42 terms */
  add_lin_mult2(k42,nv[gp],nv[ep]); // L e+ g+
  add_lin_mult2(k42,nv[gm],nv[em]); // L e- g-

  /* k45 terms */
  add_lin_mult2(k45,nv[s],nv[ep]); //L e+ s
  add_lin_mult2(k45,nv[s],nv[em]); //L e- s

  /* k52 terms */
  add_lin_mult2(k52,nv[gp],nv[s]); //L s g+
  add_lin_mult2(k52,nv[gm],nv[s]); //L s g-

  /* k31 term */
  add_lin_mult2(k31,nv[g0],nv[e0]); //L e0 g0
 
  /* k35 term */
  add_lin_mult2(k35,nv[s],nv[e0]); //L e0 s

  /* k51 term */
  add_lin_mult2(k51,nv[g0],nv[s]); //L s g0
  
  /* 1/T2g */
  rate = 1/T2g;
  add_lin_mult2(rate,nv[gp],nv[gp]); //L gp gp
  add_lin_mult2(rate,nv[gm],nv[gm]); //L gm gm
  
  /* 1/T2e */
  rate = 1/T2e;
  add_lin_mult2(rate,nv[ep],nv[ep]); //L ep ep
  add_lin_mult2(rate,nv[e0],nv[e0]); //L e0 e0
  add_lin_mult2(rate,nv[em],nv[em]); //L em em



}
