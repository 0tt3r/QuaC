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
/*---------------------------------------------
 * 1OP
 *---------------------------------------------*/

void test_add_lin_1op_basic_real(void)
{
  PetscScalar omega2,omega3,omega4;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2,op3,op4;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;

  add_lin_term(qsys,omega2,1,op2);
  add_lin_term(qsys,omega2,1,op2->dag);
  add_lin_term(qsys,omega2,1,op2->n);

  add_lin_term(qsys,omega3,1,op3);
  add_lin_term(qsys,omega3,1,op3->dag);
  add_lin_term(qsys,omega3,1,op3->n);

  add_lin_term(qsys,omega4,1,op4);
  add_lin_term(qsys,omega4,1,op4->dag);
  add_lin_term(qsys,omega4,1,op4->n);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_1op_br");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;
}

void test_add_lin_1op_pauli_real(void)
{
  PetscScalar omega2;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2;

  initialize_system(&qsys);

  create_op_sys(qsys,2,&op2);

  omega2 = 2.0;

  add_lin_term(qsys,omega2,1,op2->sig_x);
  add_lin_term(qsys,omega2,1,op2->sig_y);
  add_lin_term(qsys,omega2,1,op2->sig_z);
  add_lin_term(qsys,omega2,1,op2->eye);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_1op_pr");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}


void test_add_lin_1op_basic_complex(void)
{
  PetscScalar omega2,omega3,omega4;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2,op3,op4;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;

  add_lin_term(qsys,omega2,1,op2);
  add_lin_term(qsys,omega2,1,op2->dag);
  add_lin_term(qsys,omega2,1,op2->n);

  add_lin_term(qsys,omega3,1,op3);
  add_lin_term(qsys,omega3,1,op3->dag);
  add_lin_term(qsys,omega3,1,op3->n);

  add_lin_term(qsys,omega4,1,op4);
  add_lin_term(qsys,omega4,1,op4->dag);
  add_lin_term(qsys,omega4,1,op4->n);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_1op_bc");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}



void test_add_lin_1op_pauli_complex(void)
{
  PetscScalar omega2;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2;

  initialize_system(&qsys);
  create_op_sys(qsys,2,&op2);
  omega2 = 2.0*PETSC_i;

  add_lin_term(qsys,omega2,1,op2->sig_x);
  add_lin_term(qsys,omega2,1,op2->sig_y);
  add_lin_term(qsys,omega2,1,op2->sig_z);
  add_lin_term(qsys,omega2,1,op2->eye);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_1op_pc");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}


/*---------------------------------------------
 * 2OP
 *---------------------------------------------*/

void test_add_lin_2op_basic_real(void)
{
  PetscScalar omega2,omega3,omega4,omega5;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2,op3,op4;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;
  omega5 = 0.25;

  add_lin_term(qsys,omega2,2,op2,op2->dag);
  add_lin_term(qsys,omega2,2,op2->dag,op2);
  add_lin_term(qsys,omega2,2,op2->n,op2->dag);

  add_lin_term(qsys,omega3,2,op3,op3);
  add_lin_term(qsys,omega3,2,op3->dag,op3->dag);
  add_lin_term(qsys,omega3,2,op3->n,op3);

  add_lin_term(qsys,omega4,2,op4,op4->n);
  add_lin_term(qsys,omega4,2,op4->dag,op4->n);
  add_lin_term(qsys,omega4,2,op4->n,op4->n);

  add_lin_term(qsys,omega5,2,op4,op3->n);
  add_lin_term(qsys,omega5,2,op3->dag,op2->n);
  add_lin_term(qsys,omega5,2,op2->n,op4->n);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_2op_br");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}

void test_add_lin_2op_pauli_real(void)
{
  PetscScalar omega2;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2;

  initialize_system(&qsys);
  create_op_sys(qsys,2,&op2);
  omega2 = 2.0;

  add_lin_term(qsys,omega2,2,op2->sig_x,op2->sig_y);
  add_lin_term(qsys,omega2,2,op2->sig_y,op2->sig_z);
  add_lin_term(qsys,omega2,2,op2->sig_z,op2->sig_x);
  add_lin_term(qsys,omega2,2,op2->eye,op2->eye);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_2op_pr");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}


void test_add_lin_2op_basic_complex(void)
{
  PetscScalar omega2,omega3,omega4,omega5;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2,op3,op4;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;
  omega5 = 0.25*PETSC_i;

  add_lin_term(qsys,omega2,2,op2,op2->dag);
  add_lin_term(qsys,omega2,2,op2->dag,op2);
  add_lin_term(qsys,omega2,2,op2->n,op2->dag);

  add_lin_term(qsys,omega3,2,op3,op3);
  add_lin_term(qsys,omega3,2,op3->dag,op3->dag);
  add_lin_term(qsys,omega3,2,op3->n,op3);

  add_lin_term(qsys,omega4,2,op4,op4->n);
  add_lin_term(qsys,omega4,2,op4->dag,op4->n);
  add_lin_term(qsys,omega4,2,op4->n,op4->n);

  add_lin_term(qsys,omega5,2,op4,op3->n);
  add_lin_term(qsys,omega5,2,op3->dag,op2->n);
  add_lin_term(qsys,omega5,2,op2->n,op4->n);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_2op_bc");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}


void test_add_lin_2op_pauli_complex(void)
{
  PetscScalar omega2;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2;

  initialize_system(&qsys);
  create_op_sys(qsys,2,&op2);
  omega2 = 2.0*PETSC_i;

  add_lin_term(qsys,omega2,2,op2->sig_x,op2->sig_y);
  add_lin_term(qsys,omega2,2,op2->sig_y,op2->sig_z);
  add_lin_term(qsys,omega2,2,op2->sig_z,op2->sig_x);
  add_lin_term(qsys,omega2,2,op2->eye,op2->eye);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_2op_pc");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;
}



/*---------------------------------------------
 * 3OP
 *---------------------------------------------*/

void test_add_lin_3op_basic_real(void)
{
  PetscScalar omega2,omega3,omega4,omega5;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2,op3,op4;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;
  omega5 = 0.25;

  add_lin_term(qsys,omega2,3,op2,op2->dag,op2->n);
  add_lin_term(qsys,omega2,3,op2->dag,op2,op2->n);
  add_lin_term(qsys,omega2,3,op2->n,op2->dag,op2);

  add_lin_term(qsys,omega3,3,op3,op3,op3);
  add_lin_term(qsys,omega3,3,op3->dag,op3->dag,op3->dag);
  add_lin_term(qsys,omega3,3,op3->n,op3,op3->n);

  add_lin_term(qsys,omega4,3,op4,op4->n,op4->dag);
  add_lin_term(qsys,omega4,3,op4->dag,op4->n,op4);
  add_lin_term(qsys,omega4,3,op4->n,op4->n,op4->n);

  add_lin_term(qsys,omega5,3,op4,op3->n,op2->dag);
  add_lin_term(qsys,omega5,3,op3->dag,op2->n,op4);
  add_lin_term(qsys,omega5,3,op2->n,op4->n,op3->n);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_3op_br");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}

void test_add_lin_3op_pauli_real(void)
{
  PetscScalar omega2;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2;

  initialize_system(&qsys);
  create_op_sys(qsys,2,&op2);
  omega2 = 2.0;

  add_lin_term(qsys,omega2,3,op2->sig_x,op2->sig_x,op2->sig_z);
  add_lin_term(qsys,omega2,3,op2->sig_y,op2->sig_y,op2->sig_x);
  add_lin_term(qsys,omega2,3,op2->sig_z,op2->sig_z,op2->sig_y);
  add_lin_term(qsys,omega2,3,op2->eye,op2->eye,op2->eye);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_3op_pr");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}


void test_add_lin_3op_basic_complex(void)
{
  PetscScalar omega2,omega3,omega4,omega5;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2,op3,op4;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;
  omega5 = 0.25*PETSC_i;

  add_lin_term(qsys,omega2,3,op2,op2->dag,op2->n);
  add_lin_term(qsys,omega2,3,op2->dag,op2,op2->n);
  add_lin_term(qsys,omega2,3,op2->n,op2->dag,op2);

  add_lin_term(qsys,omega3,3,op3,op3,op3);
  add_lin_term(qsys,omega3,3,op3->dag,op3->dag,op3->dag);
  add_lin_term(qsys,omega3,3,op3->n,op3,op3->n);

  add_lin_term(qsys,omega4,3,op4,op4->n,op4->dag);
  add_lin_term(qsys,omega4,3,op4->dag,op4->n,op4);
  add_lin_term(qsys,omega4,3,op4->n,op4->n,op4->n);

  add_lin_term(qsys,omega5,3,op4,op3->n,op2->dag);
  add_lin_term(qsys,omega5,3,op3->dag,op2->n,op4);
  add_lin_term(qsys,omega5,3,op2->n,op4->n,op3->n);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_3op_bc");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;

}


void test_add_lin_3op_pauli_complex(void)
{
  PetscScalar omega2;
  qsystem qsys;
  PetscReal norm;
  char fname[255];
  operator op2;

  initialize_system(&qsys);
  create_op_sys(qsys,2,&op2);
  omega2 = 2.0*PETSC_i;

  add_lin_term(qsys,omega2,3,op2->sig_x,op2->sig_x,op2->sig_z);
  add_lin_term(qsys,omega2,3,op2->sig_y,op2->sig_y,op2->sig_x);
  add_lin_term(qsys,omega2,3,op2->sig_z,op2->sig_z,op2->sig_y);
  add_lin_term(qsys,omega2,3,op2->eye,op2->eye,op2->eye);

  construct_matrix(qsys);

  strcpy(fname,"tests/pristine_matrices/lin_3op_pc");
  _get_mat_and_diff_norm(fname,qsys->mat_A,&norm);

  destroy_op_sys(&op2);

  destroy_system(&qsys);
  TEST_ASSERT_FLOAT_WITHIN(DELTA,0,norm);
  return;
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);

  RUN_TEST(test_add_lin_1op_pauli_real);
  RUN_TEST(test_add_lin_1op_pauli_complex);
  RUN_TEST(test_add_lin_2op_pauli_real);
  RUN_TEST(test_add_lin_2op_pauli_complex);
  RUN_TEST(test_add_lin_3op_pauli_real);
  RUN_TEST(test_add_lin_3op_pauli_complex);
  RUN_TEST(test_add_lin_1op_basic_real); //Qutip
  RUN_TEST(test_add_lin_1op_basic_complex); //Qutip
  RUN_TEST(test_add_lin_2op_basic_real); //Qutip
  RUN_TEST(test_add_lin_2op_basic_complex); //Qutip
  RUN_TEST(test_add_lin_3op_basic_real); //Qutip
  RUN_TEST(test_add_lin_3op_basic_complex); //Qutip

  QuaC_finalize();
  return UNITY_END();
}

