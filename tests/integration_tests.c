#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"
#include "tests.h"

void test_timedep(void)
{
  double *populations;
  int num_pop;
  /* Initialize QuaC */

  timedep_test(&populations,&num_pop);

  TEST_ASSERT_EQUAL_FLOAT(populations[0],-1.487990e-04);
  TEST_ASSERT_EQUAL_FLOAT(populations[1],1.799424e-04);
}



int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);
  RUN_TEST(test_timedep);
  QuaC_finalize();
  return UNITY_END();
}

