#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

int main(int argc,char **args){
  PetscScalar *evs,val;
  PetscReal omega_q,omega_b,g,dicke_sz,osci_sz;
  PetscInt num_dicke=1,num_osci=10,i,j,num_evs;
  operator *dicke,*osci;
  Vec *evecs;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  PetscOptionsGetInt(NULL,NULL,"-num_dicke",&num_dicke,NULL);
  PetscOptionsGetInt(NULL,NULL,"-num_osci",&num_osci,NULL);

  /* Define scalars to add to Ham */
  omega_q    = 1; //Dicke spin frequency
  omega_b    = 1; //Oscillator spin frequency
  g          = 1; //Spin - oscillator coupling

  dicke = malloc(num_dicke*sizeof(struct operator));
  for (i=0;i<num_dicke;i++){
    create_op(2,&dicke[i]);
  }

  osci = malloc(num_osci*sizeof(struct operator));
  for (i=0;i<num_osci;i++){
    create_op(2,&osci[i]);
  }
  no_lindblad_terms();
  /* Add terms to the hamiltonian */
  for (i=0;i<num_dicke;i++){
    add_to_ham_p(omega_q,1,dicke[i]->sig_z); // omega sig_z
  }

  for (i=0;i<num_osci;i++){
    add_to_ham_p(omega_b,1,osci[i]->sig_z); // omega sig_z
  }

  //Interaction terms
  for (i=0;i<num_dicke;i++){
    for (j=0;j<num_osci;j++){
      add_to_ham_p(g,2,dicke[i]->sig_x,osci[j]->sig_x);
    }
  }
  num_evs = 10;
  diagonalize(&num_evs,&evecs,&evs);

  //Now we have the eigenvectors. We can get the expectation values of operators
  for (i=0;i<num_evs;i++){
    dicke_sz = 0;
    for (j=0;j<num_dicke;j++){
      get_expectation_value(evecs[i],&val,1,dicke[j]->sig_z);
      dicke_sz = dicke_sz + PetscRealPart(val);
    }

    osci_sz = 0;
    for (j=0;j<num_osci;j++){
      get_expectation_value(evecs[i],&val,1,osci[j]->sig_z);
      osci_sz = osci_sz + PetscRealPart(val);
    }
    PetscPrintf(PETSC_COMM_WORLD,"EV, DSz, OSz: %f %f %f\n",PetscRealPart(evs[i]),dicke_sz,osci_sz);
  }


  destroy_diagonalize(num_evs,&evecs,&evs);
  for (i=0;i<num_dicke;i++){
    destroy_op(&dicke[i]);
  }
  free(dicke);

  for (i=0;i<num_osci;i++){
    destroy_op(&osci[i]);
  }
  free(osci);


  QuaC_finalize();
  return 0;
}
