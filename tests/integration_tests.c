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
  double *populations,eps;
  int num_pop;
  eps = 1e-14;
  /* Initialize QuaC */
  timedep_test(&populations,&num_pop);
  /* These values assume TSRK3BS */
  TEST_ASSERT_FLOAT_WITHIN(eps,0.0,populations[0]);
  TEST_ASSERT_FLOAT_WITHIN(eps,0.0,populations[1]);
}

void test_imag_ham_dm(void)
{
  double *populations,eps;
  int num_pop;
  eps = 1e-7;
  /* Initialize QuaC */
  imag_ham_dm_test(&populations,&num_pop);
  /* These values assume TSRK3BS */
  TEST_ASSERT_FLOAT_WITHIN(eps,1.0,populations[1]);
  TEST_ASSERT_FLOAT_WITHIN(eps,4.0e-8,populations[0]);

}

void test_imag_ham_psi(void)
{
  double *populations,eps;
  int num_pop;
  eps = 1e-4;
  /* Initialize QuaC */
  imag_ham_psi_test(&populations,&num_pop);
  /* These values assume TSRK3BS */
  TEST_ASSERT_FLOAT_WITHIN(eps,1.0,populations[1]);
  TEST_ASSERT_FLOAT_WITHIN(eps,4.0e-8,populations[0]);

}

void test_real_ham_dm(void)
{
  double *populations,eps;
  int num_pop;
  eps = 1e-7;
  /* Initialize QuaC */
  real_ham_dm_test(&populations,&num_pop);
  /* These values assume TSRK3BS */
  TEST_ASSERT_FLOAT_WITHIN(eps,1.0,populations[1]);
  TEST_ASSERT_FLOAT_WITHIN(eps,4.0e-8,populations[0]);

}

void test_real_ham_psi(void)
{
  double *populations,eps;
  int num_pop;
  eps = 1e-4;
  /* Initialize QuaC */
  real_ham_psi_test(&populations,&num_pop);
  /* These values assume TSRK3BS */
  TEST_ASSERT_FLOAT_WITHIN(eps,1.0,populations[1]);
  TEST_ASSERT_FLOAT_WITHIN(eps,4.0e-8,populations[0]);

}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);
  RUN_TEST(test_timedep);
  QuaC_clear();
  RUN_TEST(test_imag_ham_dm);
  QuaC_clear();
  RUN_TEST(test_imag_ham_psi);
  QuaC_clear();
  RUN_TEST(test_real_ham_dm);
  QuaC_clear();
  RUN_TEST(test_real_ham_psi);
  QuaC_clear();
  QuaC_finalize();
  return UNITY_END();
}

