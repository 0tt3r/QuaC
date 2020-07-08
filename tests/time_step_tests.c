#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"
#include "t_helpers.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho,void *ctx){
  qsystem qsys = (qsystem) ctx;
  //Print out things at each time step, if desired
  PetscPrintf(PETSC_COMM_WORLD,"Step: %d, time: %f\n",step,time);
  print_qvec(qsys->solution_qvec);
  PetscPrintf(PETSC_COMM_WORLD,"\n");
  PetscFunctionReturn(0);
}


void test_time_step_t1_decay_dm_vec_1sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val;
  vec_op vop2;
  PetscInt steps_max;
  PetscReal dt,time_max;
  qvec dm;
  enum STATE {gnd=0,exc};

  initialize_system(&qsys);
  //Create some operators
  create_vec_op_sys(qsys,2,&vop2);

  //  add_ham_term(qsys,omega2,1,vop2->n);
  add_lin_term(qsys,gamma,2,vop2[gnd],vop2[exc]); // sigma = |0><1|

  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,1,1);
  assemble_qvec(dm);

  construct_matrix(qsys);

  time_max  = 1;
  dt        = 0.05;
  steps_max = 200;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);

  get_expectation_value_qvec(dm,&trace_val,2,vop2[exc],vop2[exc]);

  //within because adaptive timestepping
  TEST_ASSERT_FLOAT_WITHIN(0.0001,1/exp(1),PetscRealPart(trace_val));
  destroy_vec_op_sys(&vop2);
  destroy_system(&qsys);
  destroy_qvec(&dm);

  return;
}



void test_time_step_t1_decay_dm_1sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val;
  operator op2;
  PetscInt steps_max;
  PetscReal dt,time_max;
  qvec dm;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);

  //  add_ham_term(qsys,omega2,1,op2->n);
  add_lin_term(qsys,gamma,1,op2);

  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,1,1);
  assemble_qvec(dm);

  construct_matrix(qsys);
  print_mat_sparse(qsys->mat_A);

  time_max  = 1;
  dt        = 0.05;
  steps_max = 200;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);
  get_expectation_value_qvec(dm,&trace_val,1,op2->n);

  //Within because adaptive timestepping
  TEST_ASSERT_FLOAT_WITHIN(0.0001,1/exp(1),PetscRealPart(trace_val));
  destroy_op_sys(&op2);
  destroy_system(&qsys);
  destroy_qvec(&dm);

  return;
}

void test_time_step_t1_decay_mcwf_1sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val;
  operator op2;
  PetscInt steps_max,n_samples;
  PetscReal dt,time_max;
  qvec wf_ens;

  n_samples = 1000;
  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);

  add_ham_term(qsys,omega2,1,op2->n);
  add_lin_term(qsys,gamma,1,op2);

  use_mcwf_solver(qsys,n_samples,NULL);

  create_qvec_sys(qsys,&wf_ens);
  add_to_qvec(wf_ens,1.0,1);

  assemble_qvec(wf_ens);

  construct_matrix(qsys);

  time_max  = 1;
  dt        = 0.01;
  steps_max = 200;

  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,wf_ens,0.0,time_max,dt,steps_max);
  get_expectation_value_qvec(wf_ens,&trace_val,1,op2->n);
  TEST_ASSERT_FLOAT_WITHIN(2/sqrt(n_samples),1/exp(1),PetscRealPart(trace_val));


  destroy_op_sys(&op2);
  destroy_system(&qsys);
  destroy_qvec(&wf_ens);

  return;
}


void test_time_step_t1_decay_dm_vec_2sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val,trace_val2;
  vec_op vop2,vop3;
  PetscInt steps_max;
  PetscReal dt,time_max;
  qvec dm;
  enum STATE {gnd=0,exc,thrd};
  initialize_system(&qsys);
  //Create some operators
  create_vec_op_sys(qsys,3,&vop3);
  create_vec_op_sys(qsys,2,&vop2);

  add_ham_term(qsys,omega2,2,vop2[exc],vop2[exc]);
  add_lin_term(qsys,gamma,2,vop2[gnd],vop2[exc]);


  add_lin_term(qsys,gamma,2,vop3[gnd],vop3[exc]);//1->0
  add_lin_term(qsys,2*gamma,2,vop3[exc],vop3[thrd]);//2->1

  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,5,5);
  assemble_qvec(dm);

  construct_matrix(qsys);

  time_max  = 1;
  dt        = 0.05;
  steps_max = 200;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);

  get_expectation_value_qvec(dm,&trace_val,2,vop2[exc],vop2[exc]);

  TEST_ASSERT_FLOAT_WITHIN(0.0001,1/exp(1),PetscRealPart(trace_val));

  get_expectation_value_qvec(dm,&trace_val,2,vop3[exc],vop3[exc]);
  get_expectation_value_qvec(dm,&trace_val2,2,vop3[thrd],vop3[thrd]);
  TEST_ASSERT_FLOAT_WITHIN(0.0002,2/exp(1),PetscRealPart(trace_val+2*trace_val2));

  destroy_vec_op_sys(&vop2);
  destroy_vec_op_sys(&vop3);
  destroy_system(&qsys);
  destroy_qvec(&dm);

  return;
}


void test_time_step_t1_decay_dm_2sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val;
  operator op2,op3;
  PetscInt steps_max;
  PetscReal dt,time_max;
  qvec dm;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,2,&op2);

  add_ham_term(qsys,omega2,1,op2->n);
  add_lin_term(qsys,gamma,1,op2);
  add_lin_term(qsys,gamma,1,op3);

  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,5,5);
  assemble_qvec(dm);

  construct_matrix(qsys);

  time_max  = 1;
  dt        = 0.05;
  steps_max = 200;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);

  get_expectation_value_qvec(dm,&trace_val,1,op2->n);

  TEST_ASSERT_FLOAT_WITHIN(0.0001,1/exp(1),PetscRealPart(trace_val));
  get_expectation_value_qvec(dm,&trace_val,1,op3->n);

  TEST_ASSERT_FLOAT_WITHIN(0.0002,2/exp(1),PetscRealPart(trace_val));

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_system(&qsys);
  destroy_qvec(&dm);

  return;
}

void test_time_step_t1_decay_mcwf_2sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val;
  operator op2,op3;
  PetscInt steps_max,n_samples;
  PetscReal dt,time_max;
  qvec wf_ens;

  n_samples=1000;
  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,2,&op2);

  add_ham_term(qsys,omega2,1,op2->n);
  add_lin_term(qsys,gamma,1,op2);
  add_lin_term(qsys,gamma,1,op3);

  use_mcwf_solver(qsys,n_samples,NULL);

  create_qvec_sys(qsys,&wf_ens);
  add_to_qvec(wf_ens,1.0,5);
  assemble_qvec(wf_ens);

  get_expectation_value_qvec(wf_ens,&trace_val,1,op2->n);


  construct_matrix(qsys);

  time_max  = 1;
  dt        = 0.05;
  steps_max = 200;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,wf_ens,0.0,time_max,dt,steps_max);

  get_expectation_value_qvec(wf_ens,&trace_val,1,op2->n);

  TEST_ASSERT_FLOAT_WITHIN(2/sqrt(n_samples),1/exp(1),PetscRealPart(trace_val));
  get_expectation_value_qvec(wf_ens,&trace_val,1,op3->n);

  TEST_ASSERT_FLOAT_WITHIN(2/sqrt(n_samples),2/exp(1),PetscRealPart(trace_val));

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_system(&qsys);
  destroy_qvec(&wf_ens);

  return;
}

void test_time_step_circuit_decay_dm_2sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val;
  operator op2,op3;
  PetscInt steps_max,i;
  PetscReal dt,time_max;
  circuit circ;
  qvec dm;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op3);
  create_op_sys(qsys,2,&op2);

  add_ham_term(qsys,omega2,1,op2->n);
  add_lin_term(qsys,gamma,1,op2);
  add_lin_term(qsys,gamma,1,op3);

  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,0,0);
  assemble_qvec(dm);

  get_expectation_value_qvec(dm,&trace_val,1,op2->n);

  construct_matrix(qsys);


  create_circuit(&circ,5);
  //Add some gates
  add_gate_to_circuit_sys(&circ,0.0,SIGMAX,0);
  add_gate_to_circuit_sys(&circ,0.1,SIGMAX,1);
  //Start out circuit at time 0.0, first gate will be at 0
  apply_circuit_to_sys(qsys,&circ,0.1);//Circuit at 0 and gates at 0 still a problem


  time_max  = 1.1;
  dt        = 0.05;
  steps_max = 200;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);

  //the within is because adaptive timesteps are hard

  get_expectation_value_qvec(dm,&trace_val,1,op2->n);
  TEST_ASSERT_FLOAT_WITHIN(0.0005,1/exp(0.9),PetscRealPart(trace_val));

  get_expectation_value_qvec(dm,&trace_val,1,op3->n);
  TEST_ASSERT_FLOAT_WITHIN(0.0005,1/exp(1),PetscRealPart(trace_val));

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_system(&qsys);
  destroy_qvec(&dm);
  destroy_circuit(&circ);

  return;
}

void test_time_step_circuit_decay_wf_ens_2sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0,gamma=1.0,trace_val;
  operator op2,op3;
  PetscInt steps_max,i,n_samples;
  PetscReal dt,time_max;
  circuit circ;
  qvec wf_ens;

  n_samples = 1000;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op3);
  create_op_sys(qsys,2,&op2);

  add_lin_term(qsys,gamma,1,op2);
  add_lin_term(qsys,gamma,1,op3);

  use_mcwf_solver(qsys,n_samples,NULL);

  create_qvec_sys(qsys,&wf_ens);
  add_to_qvec(wf_ens,1.0,0);
  assemble_qvec(wf_ens);

  get_expectation_value_qvec(wf_ens,&trace_val,1,op2->n);

  construct_matrix(qsys);


  create_circuit(&circ,5);
  //Add some gates
  add_gate_to_circuit_sys(&circ,0.0,SIGMAX,0);
  add_gate_to_circuit_sys(&circ,0.1,SIGMAX,1);
  //Start out circuit at time 0.0, first gate will be at 0
  apply_circuit_to_sys(qsys,&circ,0.1);//Circuit at 0 and gates at 0 still a problem


  time_max  = 1.1;
  dt        = 0.1;
  steps_max = 25;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,wf_ens,0.0,time_max,dt,steps_max);

  get_expectation_value_qvec(wf_ens,&trace_val,1,op2->n);

  TEST_ASSERT_FLOAT_WITHIN(2/sqrt(n_samples),1/exp(0.9),PetscRealPart(trace_val));
  get_expectation_value_qvec(wf_ens,&trace_val,1,op3->n);

  TEST_ASSERT_FLOAT_WITHIN(2/sqrt(n_samples),1/exp(1),PetscRealPart(trace_val));

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_system(&qsys);
  destroy_qvec(&wf_ens);
  destroy_circuit(&circ);

  return;
}

void test_time_step_circuit_wf_ens_dm_gam0(void){
  qsystem qsys,qsys2;
  PetscScalar omega2=1.0,gamma=0.0,trace_val0,trace_val1,trace_val2,trace_val3;
  operator op0,op1,op2,op3;
  PetscInt steps_max,i,n_samples,nloc_wf_ens,nloc_dm;
  PetscReal dt,time_max,*probs_dm,*probs_wf_ens,*vars_dm,*vars_wf_ens;
  circuit circ;
  qvec wf_ens,dm;

  n_samples = 100;

  //DM system
  initialize_system(&qsys);

  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);

  add_lin_term(qsys,gamma,1,op0);
  add_lin_term(qsys,gamma,1,op1);

  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,0,0);
  assemble_qvec(dm);

  construct_matrix(qsys);
  //wf ens system
  initialize_system(&qsys2);

  create_op_sys(qsys2,2,&op2);
  create_op_sys(qsys2,2,&op3);

  add_lin_term(qsys2,gamma,1,op2);
  add_lin_term(qsys2,gamma,1,op3);

  use_mcwf_solver(qsys2,n_samples,NULL);

  create_qvec_sys(qsys2,&wf_ens);
  add_to_qvec(wf_ens,1.0,0);
  assemble_qvec(wf_ens);

  construct_matrix(qsys2);


  create_circuit(&circ,10);
  //Add some gates
  add_gate_to_circuit_sys(&circ,0.0,SIGMAX,0);
  add_gate_to_circuit_sys(&circ,0.1,SIGMAX,1);
  add_gate_to_circuit_sys(&circ,0.2,CNOT,0,1);
  add_gate_to_circuit_sys(&circ,0.3,RX,0,0.1);
  add_gate_to_circuit_sys(&circ,0.4,RY,1,0.1);
  add_gate_to_circuit_sys(&circ,0.5,HADAMARD,1,0.1);
  add_gate_to_circuit_sys(&circ,1.0,CNOT,0,1);
  /* start out circuit at time 0.0, first gate will be at 0 */
  apply_circuit_to_sys(qsys,&circ,0.0001);//Circuit at 0 and gates at 0 still a problem


  apply_circuit_to_sys(qsys2,&circ,0.0001);//Circuit at 0 and gates at 0 still a problem

  time_max  = 1.0001;
  dt        = 0.05;
  steps_max = 200;

  time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);

  time_step_sys(qsys2,wf_ens,0.0,time_max,dt,steps_max);

  get_expectation_value_qvec(dm,&trace_val0,1,op0->n);
  get_expectation_value_qvec(dm,&trace_val1,1,op1->n);

  get_expectation_value_qvec(wf_ens,&trace_val2,1,op2->n);
  get_expectation_value_qvec(wf_ens,&trace_val3,1,op3->n);

  //  printf("trace: %f %f\n");
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(trace_val0),PetscRealPart(trace_val2));
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(trace_val1),PetscRealPart(trace_val3));


  get_bitstring_probs(wf_ens,&nloc_wf_ens,&probs_wf_ens,&vars_wf_ens);
  get_bitstring_probs(dm,&nloc_dm,&probs_dm,&vars_dm);

  for(i=0;i<4;i++){
    TEST_ASSERT_EQUAL_FLOAT(probs_wf_ens[i],probs_dm[i]);
  }

  destroy_op_sys(&op0);
  destroy_op_sys(&op1);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);

  destroy_system(&qsys);

  destroy_system(&qsys2);

  destroy_qvec(&wf_ens);

  destroy_qvec(&dm);

  destroy_circuit(&circ);

  free(vars_dm);
  free(vars_wf_ens);
  free(probs_wf_ens);
  free(probs_dm);
  return;
}

void test_time_step_circuit_wf_ens_dm_gam1(void){
  qsystem qsys,qsys2;
  PetscScalar omega2=1.0,gamma=1.0,trace_val0,trace_val1,trace_val2,trace_val3;
  operator op0,op1,op2,op3;
  PetscInt steps_max,i,n_samples,nloc_wf_ens,nloc_dm;
  PetscReal dt,time_max,*probs_dm,*probs_wf_ens,*vars_dm,*vars_wf_ens;
  circuit circ;
  qvec wf_ens,dm;

  n_samples = 1000;

  //DM system
  initialize_system(&qsys);

  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);

  add_lin_term(qsys,gamma,1,op0);
  add_lin_term(qsys,gamma,1,op1);
  add_lin_term(qsys,gamma,1,op0->n);
  add_lin_term(qsys,gamma,1,op1->n);

  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,0,0);
  assemble_qvec(dm);

  construct_matrix(qsys);
  //wf ens system
  initialize_system(&qsys2);

  create_op_sys(qsys2,2,&op2);
  create_op_sys(qsys2,2,&op3);

  add_lin_term(qsys2,gamma,1,op2);
  add_lin_term(qsys2,gamma,1,op3);

  use_mcwf_solver(qsys2,n_samples,NULL);

  create_qvec_sys(qsys2,&wf_ens);
  add_to_qvec(wf_ens,1.0,0);
  assemble_qvec(wf_ens);

  construct_matrix(qsys2);


  create_circuit(&circ,10);
  //Add some gates
  add_gate_to_circuit_sys(&circ,0.0,SIGMAX,0);
  add_gate_to_circuit_sys(&circ,0.1,SIGMAX,1);
  add_gate_to_circuit_sys(&circ,0.15,HADAMARD,0);
  add_gate_to_circuit_sys(&circ,0.2,CNOT,0,1);
  add_gate_to_circuit_sys(&circ,0.3,RX,0,0.1);
  add_gate_to_circuit_sys(&circ,0.4,RY,1,0.1);
  add_gate_to_circuit_sys(&circ,0.5,HADAMARD,1);
  add_gate_to_circuit_sys(&circ,1.0,CNOT,0,1);
  /* start out circuit at time 0.0, first gate will be at 0 */
  apply_circuit_to_sys(qsys,&circ,0.1);//Circuit at 0 and gates at 0 still a problem


  apply_circuit_to_sys(qsys2,&circ,0.1);//Circuit at 0 and gates at 0 still a problem

  time_max  = 1.0001;
  dt        = 0.05;
  steps_max = 200;

  time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);

  time_step_sys(qsys2,wf_ens,0.0,time_max,dt,steps_max);

  get_expectation_value_qvec(dm,&trace_val0,1,op0->n);
  get_expectation_value_qvec(dm,&trace_val1,1,op1->n);

  get_expectation_value_qvec(wf_ens,&trace_val2,1,op2->n);
  get_expectation_value_qvec(wf_ens,&trace_val3,1,op3->n);

  //  printf("trace: %f %f\n");
  TEST_ASSERT_FLOAT_WITHIN(2/sqrt(n_samples),PetscRealPart(trace_val0),PetscRealPart(trace_val2));
  TEST_ASSERT_FLOAT_WITHIN(2/sqrt(n_samples),PetscRealPart(trace_val1),PetscRealPart(trace_val3));


  get_bitstring_probs(wf_ens,&nloc_wf_ens,&probs_wf_ens,&vars_wf_ens);
  get_bitstring_probs(dm,&nloc_dm,&probs_dm,&vars_dm);

  for(i=0;i<4;i++){
    TEST_ASSERT_FLOAT_WITHIN(1/sqrt(n_samples),probs_wf_ens[i],probs_dm[i]);
  }

  destroy_op_sys(&op0);
  destroy_op_sys(&op1);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);

  destroy_system(&qsys);

  destroy_system(&qsys2);

  destroy_qvec(&wf_ens);

  destroy_qvec(&dm);

  destroy_circuit(&circ);

  free(vars_dm);
  free(vars_wf_ens);
  free(probs_wf_ens);
  free(probs_dm);
  return;
}



void test_time_step_circuit_restart_dm_2sys(void){
  qsystem qsys,qsys2,qsys3;
  PetscScalar omega2=0.0,gamma=0.5;
  PetscReal fidelity,fid2,var,var2;
  operator op2,op3,op0,op1,op4,op5;
  PetscInt steps_max,i,n_samples;
  PetscReal dt,time_max;
  circuit circ;
  qvec dm,wf_ens,wf;

  n_samples=1000;

  initialize_system(&qsys);
  initialize_system(&qsys2);
  initialize_system(&qsys3);
  //Create some operators
  create_op_sys(qsys,2,&op3);
  create_op_sys(qsys,2,&op2);

  create_op_sys(qsys2,2,&op0);
  create_op_sys(qsys2,2,&op1);

  create_op_sys(qsys3,2,&op4);
  create_op_sys(qsys3,2,&op5);

  add_ham_term(qsys3,omega2,1,op4);

  add_lin_term(qsys,gamma,1,op2);
  add_lin_term(qsys,gamma,1,op3);

  add_lin_term(qsys2,gamma,1,op0);
  add_lin_term(qsys2,gamma,1,op1);


  create_qvec_sys(qsys,&dm);
  add_to_qvec(dm,1.0,0,0);
  assemble_qvec(dm);

  use_mcwf_solver(qsys2,n_samples,NULL);
  create_qvec_sys(qsys2,&wf_ens);
  add_to_qvec(wf_ens,1.0,0);
  assemble_qvec(wf_ens);

  create_qvec_sys(qsys3,&wf);
  add_to_qvec(wf,1.0,0);
  assemble_qvec(wf);

  construct_matrix(qsys);
  construct_matrix(qsys2);
  construct_matrix(qsys3);

  create_circuit(&circ,10);
  //Add some gates
  add_gate_to_circuit_sys(&circ,0.5,HADAMARD,0);
  add_gate_to_circuit_sys(&circ,0.5,HADAMARD,1);
  add_gate_to_circuit_sys(&circ,0.3,RX,0,0.1);
  add_gate_to_circuit_sys(&circ,0.4,RY,1,0.3);
  add_gate_to_circuit_sys(&circ,1.0,CNOT,0,1);
  add_gate_to_circuit_sys(&circ,0.4,RZ,1,0.9);
  add_gate_to_circuit_sys(&circ,0.4,RZ,1,0.4);

  time_max  = 1.0;
  dt        = 0.01;
  steps_max = 200;

  for(i=0;i<10;i++){
    apply_circuit_to_qvec(qsys,circ,dm);
    apply_circuit_to_qvec(qsys2,circ,wf_ens);
    apply_circuit_to_qvec(qsys3,circ,wf);

    time_step_sys(qsys,dm,0.0,time_max,dt,steps_max);
    time_step_sys(qsys2,wf_ens,0.0,time_max,dt,steps_max);

    get_fidelity_qvec(dm,wf,&fidelity,&var);
    get_fidelity_qvec(wf,wf_ens,&fid2,&var2);
    TEST_ASSERT_FLOAT_WITHIN(1/sqrt(n_samples),fidelity,fid2);
  }

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_system(&qsys);

  destroy_op_sys(&op0);
  destroy_op_sys(&op1);
  destroy_system(&qsys2);

  destroy_op_sys(&op4);
  destroy_op_sys(&op5);
  destroy_system(&qsys3);

  destroy_qvec(&dm);
  destroy_qvec(&wf_ens);
  destroy_qvec(&wf);;
  destroy_circuit(&circ);

  return;
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);

  RUN_TEST(test_time_step_t1_decay_dm_vec_1sys);
  /* RUN_TEST(test_time_step_t1_decay_dm_1sys); */
  /* RUN_TEST(test_time_step_t1_decay_mcwf_1sys); */

  RUN_TEST(test_time_step_t1_decay_dm_vec_2sys);
  RUN_TEST(test_time_step_t1_decay_dm_2sys);
  /* RUN_TEST(test_time_step_t1_decay_dm_2sys); */
  /* RUN_TEST(test_time_step_t1_decay_mcwf_2sys); */

  /* RUN_TEST(test_time_step_circuit_decay_dm_2sys); */
  /* RUN_TEST(test_time_step_circuit_decay_wf_ens_2sys); */

  /* RUN_TEST(test_time_step_circuit_wf_ens_dm_gam0); */
  /* RUN_TEST(test_time_step_circuit_wf_ens_dm_gam1); */

  /* RUN_TEST(test_time_step_circuit_restart_dm_2sys); */
  QuaC_finalize();
  return UNITY_END();
}
