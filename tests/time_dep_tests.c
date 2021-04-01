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

#define MAX_STEPS 1000000
#define TOLERANCE 1e-4

typedef struct {
  PetscScalar alpha,omega;
  PetscScalar sz[MAX_STEPS],sx[MAX_STEPS],sy[MAX_STEPS];
  PetscScalar sz_analytic[MAX_STEPS],sx_analytic[MAX_STEPS],sy_analytic[MAX_STEPS];
} PulseParams;

operator op2;

qvec qvec_dummy; //Really need a better solution for this - wrap ts_monitor?
PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
PetscScalar time_dep_func(PetscReal,void*);
PetscScalar analytic_func(PetscReal,PetscScalar);


PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec rho_data,void *ctx){
  PulseParams *pulse_params = (PulseParams*) ctx;
  PetscScalar trace_val;
  Vec tmp_data;

  //Kluge fix because rho_data is not a qvec, but rather a PETSc Vec
  tmp_data = qvec_dummy->data;
  qvec_dummy->data = rho_data;
  //get time-stepped values
  get_expectation_value_qvec(qvec_dummy,&trace_val,1,op2->sig_z);
  pulse_params->sz[step] = trace_val;
  get_expectation_value_qvec(qvec_dummy,&trace_val,1,op2->sig_y);
  pulse_params->sy[step] = trace_val;
  get_expectation_value_qvec(qvec_dummy,&trace_val,1,op2->sig_x);
  pulse_params->sx[step] = trace_val;

  //get analytic value
  pulse_params->sz_analytic[step] = cos(pulse_params->omega*analytic_func(time,pulse_params->alpha));
  pulse_params->sy_analytic[step] = -sin(pulse_params->omega*analytic_func(time,pulse_params->alpha));
  pulse_params->sx_analytic[step] = 0.0;
  //  printf("time %f SZ: %f %f %f %f\n",time,pulse_params->sz[step],pulse_params->sz_analytic[step]);
  qvec_dummy-> data = tmp_data;
   /* print_qvec(qsys->solution_qvec); */
  PetscFunctionReturn(0);
}

PetscScalar analytic_func(PetscReal time,PetscScalar alpha){
  return (1-exp(-alpha*time))/alpha;
}

PetscScalar time_dep_func(PetscReal time,void *ctx){
  PetscScalar pulse_value;
  PulseParams *pulse_params = (PulseParams*) ctx;   /* user-defined struct */
  pulse_value = exp(-pulse_params->alpha*time);
  return pulse_value;
}


void test_time_dep_1sys(void){
  qsystem qsys;
  PetscScalar omega2=1.0*2*PETSC_PI,gamma,trace_val,omega;
  vec_op vop2;
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

  time_max  = 20;
  dt        = 0.05;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,dm,0.0,time_max,dt,MAX_STEPS);

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


  time_max  = 1;
  dt        = 0.05;
  /* set_ts_monitor_sys(qsys,ts_monitor,qsys); */
  time_step_sys(qsys,dm,0.0,time_max,dt,MAX_STEPS);
  get_expectation_value_qvec(dm,&trace_val,1,op2->n);

  //Within because adaptive timestepping
  TEST_ASSERT_FLOAT_WITHIN(0.0001,1/exp(1),PetscRealPart(trace_val));
  destroy_op_sys(&op2);
  destroy_system(&qsys);
  destroy_qvec(&dm);

  return;
}

void test_time_dep_wf_1sys(void){
  qsystem qsys;
  PetscScalar omega2=2*PETSC_PI,max_diff_sx,max_diff_sy,max_diff_sz,diff_sx,diff_sy,diff_sz,tmp_scalar;
  PetscInt i;
  PetscReal dt,time_max;
  qvec wf;
  PulseParams pulse_params;

  /*
   * Test based off of https://github.com/qutip/qutip/blob/master/qutip/tests/test_sesolve.py
   */

  pulse_params.alpha = 0.1;
  pulse_params.omega = omega2;

  for(i=0;i<MAX_STEPS;i++){
    pulse_params.sx[i] = 0;
    pulse_params.sy[i] = 0;
    pulse_params.sz[i] = 0;
    pulse_params.sx_analytic[i] = 0;
    pulse_params.sy_analytic[i] = 0;
    pulse_params.sz_analytic[i] = 0;
  }

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);

  //0.5*2*pi * exp(-alpha*t)*(a+a')
  tmp_scalar = 0.5*omega2;
  add_ham_term_time_dep(qsys,tmp_scalar,&pulse_params,time_dep_func,1,op2);
  add_ham_term_time_dep(qsys,tmp_scalar,&pulse_params,time_dep_func,1,op2->dag);
  create_qvec_sys(qsys,&wf);
  create_qvec_sys(qsys,&qvec_dummy);
  add_to_qvec(wf,1.0,0);
  assemble_qvec(wf);

  construct_matrix(qsys);

  time_max  = 20;
  dt        = 0.01;

  set_ts_monitor_sys(qsys,ts_monitor,&pulse_params);
  time_step_sys(qsys,wf,0.0,time_max,dt,MAX_STEPS);

  max_diff_sx = 0;
  max_diff_sy = 0;
  max_diff_sz = 0;

  for(i=0;i<MAX_STEPS;i++){
    //Loop through and find the max diff
    diff_sx = PetscAbsScalar(pulse_params.sx[i]-pulse_params.sx_analytic[i]);
    diff_sy = PetscAbsScalar(pulse_params.sy[i]-pulse_params.sy_analytic[i]);
    diff_sz = PetscAbsScalar(pulse_params.sz[i]-pulse_params.sz_analytic[i]);
    if(PetscAbsScalar(diff_sz)>PetscAbsScalar(max_diff_sz)){
      max_diff_sz = diff_sz;
    }

    if(PetscAbsScalar(diff_sy)>PetscAbsScalar(max_diff_sy)){
      max_diff_sy = diff_sy;
    }

    if(PetscAbsScalar(diff_sx)>PetscAbsScalar(max_diff_sx)){
      max_diff_sx = diff_sx;
    }
  }
  printf("max_diff_sx: %e %f\n",max_diff_sx);
  printf("max_diff_sy: %e %f\n",max_diff_sy);
  printf("max_diff_sz: %e %f\n",max_diff_sz);
  TEST_ASSERT_FLOAT_WITHIN(TOLERANCE,0.0,PetscRealPart(max_diff_sx));
  TEST_ASSERT_FLOAT_WITHIN(TOLERANCE,0.0,PetscRealPart(max_diff_sy));
  TEST_ASSERT_FLOAT_WITHIN(TOLERANCE,0.0,PetscRealPart(max_diff_sz));
  destroy_op_sys(&op2);
  destroy_system(&qsys);
  destroy_qvec(&wf);
  destroy_qvec(&qvec_dummy);

  return;
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);

  RUN_TEST(test_time_dep_wf_1sys);

  QuaC_finalize();
  return UNITY_END();
}
