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

  /* These values assume TSRK3BS */
  /* TEST_ASSERT_EQUAL_FLOAT(0.0,populations[0]); */
  /* TEST_ASSERT_EQUAL_FLOAT(0.0,populations[1]); */
}



int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);
  RUN_TEST(test_timedep);
  QuaC_finalize();
  return UNITY_END();
}

