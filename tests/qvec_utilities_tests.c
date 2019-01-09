#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"
#include "qsystem.h"
#include "qvec_utilities.h"
/*
 * Test get_expectation_value_wf_real
 */
void test_get_expectation_value_wf_real(void)
{
  operator qd1,qd2;
  PetscScalar ev1,ev2,ev3,ev4,val;
  qvec wf;
  qsystem qsys;

  initialize_system(&qsys);

  create_op_sys(qsys,2,&qd1);
  create_op_sys(qsys,2,&qd2);
  val = 0;

  create_wf_sys(qsys,&wf);
  ev1 = 1.0;

  add_to_qvec_fock_op(ev1,wf,2,qd1,1,qd2,1);

  assemble_qvec(wf);

  get_expectation_value_qvec(wf,&ev1,1,qd1->n);
  get_expectation_value_qvec(wf,&ev2,1,qd2->n);
  get_expectation_value_qvec(wf,&ev3,2,qd1->dag,qd2);
  get_expectation_value_qvec(wf,&ev4,2,qd2->dag,qd1);

  TEST_ASSERT_EQUAL_FLOAT(1.0,PetscRealPart(ev1));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev1));

  TEST_ASSERT_EQUAL_FLOAT(1.0,PetscRealPart(ev2));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev2));

  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(ev3));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev3));

  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(ev4));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev4));

  destroy_system(&qsys);
  return;
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);
  RUN_TEST(test_get_expectation_value_wf_real);
  QuaC_finalize();
  return UNITY_END();
}

