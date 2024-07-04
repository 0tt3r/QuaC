#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "error_correction.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "quantum_circuits.h"
#include "petsc.h"
#include "qasm_parser.h"
#include "qsystem.h"


qvec dm_dummy; // Needed to effectively integrate with PETSc call backs

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
operator sigma,b;
int main(int argc,char **args){
  qvec rho;
  PetscReal wc,w0,g,kappa,gamma;
  qsystem qsys;
  PetscReal dt,time_max;
  PetscInt steps_max;
  PetscScalar mat_val;

  wc         = 1.0*2*M_PI;
  w0         = 1.0*2*M_PI;
  g          = 0.05*2*M_PI;
  kappa      = 0.005;
  gamma      = 0.05;

  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  initialize_system(&qsys);
  create_op_sys(qsys,2,&sigma); //create qubit
  create_op_sys(qsys,7,&b); //create oscillator


  add_ham_term(qsys,w0,1,sigma->n); // qubit frequency
  add_ham_term(qsys,wc,1,b->n);     // cavity frequency
  add_ham_term(qsys,g,2,sigma->dag,b);     // coupling
  add_ham_term(qsys,g,2,b->dag,sigma);     // coupling

  add_lin_term(qsys,gamma,1,sigma); // qubit decay
  add_lin_term(qsys,kappa,1,b); // cavity decay
  
  //Construct the matrix now that we are done adding to it
  construct_matrix(qsys);

  //Time step until 1 after the last gate; gates are applied every 1.0 time unit
  time_max  = 100;
  dt        = 0.1;
  steps_max = 1000;

  /* Set the ts_monitor to print results at each time step, if desired */
  set_ts_monitor_sys(qsys,ts_monitor,NULL);

  create_qvec_sys(qsys,&rho); // create density matrix
  create_qvec_sys(qsys,&(dm_dummy));
  mat_val = 1.0;
  add_to_qvec(rho,mat_val,1,1); // add initial value
  assemble_qvec(rho); // assemble

  //Run the evolution
  time_step_sys(qsys,rho,0.0,time_max,dt,steps_max);

  destroy_op(&sigma);
  destroy_op(&b);
  destroy_qvec(&rho);
  QuaC_finalize();
  return 0;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho,void *ctx){
  Vec tmp_data;
  PetscScalar trace_val1,trace_val2;
  //Print out things at each time step, if desired
  // putting the petsc Vec rho into a dummy qvec so we can use QuaC functions
  tmp_data = dm_dummy->data;
  dm_dummy->data = rho; 
  get_expectation_value_qvec(dm_dummy,&trace_val1,1,sigma->n);
  get_expectation_value_qvec(dm_dummy,&trace_val2,1,b->n);
  PetscPrintf(PETSC_COMM_WORLD,"%d %f %f %f \n",step,time,PetscRealPart(trace_val1),PetscRealPart(trace_val2));

  dm_dummy->data = tmp_data; // Remove referene to PETSc's Vec rho
  /* print_qvec(rho); */
  PetscFunctionReturn(0);
}

