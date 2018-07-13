#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

/*
 * Test get_bipartite_concurrence with bell states
 * should give concurrence of 1.
 */
void test_bipartite_bell(void)
{
  PetscScalar val;
  double concurrence;
  Vec bell_dm;

  // Antisymmetric bell state
  create_dm(&bell_dm,4);
  val = 0.5;
  add_value_to_dm(bell_dm,1,1,val);
  add_value_to_dm(bell_dm,2,2,val);
  val = -0.5;
  add_value_to_dm(bell_dm,1,2,val);
  add_value_to_dm(bell_dm,2,1,val);

  assemble_dm(bell_dm);

  get_bipartite_concurrence(bell_dm,&concurrence);
  destroy_dm(bell_dm);

  TEST_ASSERT_EQUAL_FLOAT(1.0,concurrence);


  // Symmetric bell state
  create_dm(&bell_dm,4);
  val = 0.5;
  add_value_to_dm(bell_dm,1,1,val);
  add_value_to_dm(bell_dm,2,2,val);
  val = 0.5;
  add_value_to_dm(bell_dm,1,2,val);
  add_value_to_dm(bell_dm,2,1,val);

  assemble_dm(bell_dm);

  get_bipartite_concurrence(bell_dm,&concurrence);
  destroy_dm(bell_dm);

  TEST_ASSERT_EQUAL_FLOAT(1.0,concurrence);


  // Antisymmetric2 bell state
  create_dm(&bell_dm,4);
  val = 0.5;
  add_value_to_dm(bell_dm,0,0,val);
  add_value_to_dm(bell_dm,3,3,val);
  val = -0.5;
  add_value_to_dm(bell_dm,0,3,val);
  add_value_to_dm(bell_dm,3,0,val);

  assemble_dm(bell_dm);

  get_bipartite_concurrence(bell_dm,&concurrence);
  destroy_dm(bell_dm);

  TEST_ASSERT_EQUAL_FLOAT(1.0,concurrence);


  // Symmetric2 bell state
  create_dm(&bell_dm,4);
  val = 0.5;
  add_value_to_dm(bell_dm,0,0,val);
  add_value_to_dm(bell_dm,3,3,val);
  val = 0.5;
  add_value_to_dm(bell_dm,0,3,val);
  add_value_to_dm(bell_dm,3,0,val);

  assemble_dm(bell_dm);

  get_bipartite_concurrence(bell_dm,&concurrence);
  destroy_dm(bell_dm);
  TEST_ASSERT_EQUAL_FLOAT(concurrence,1.00);
}


/*
 * Test get_bipartite_concurrence on separable states.
 */
void test_bipartite_separable(void)
{
  PetscScalar val;
  double concurrence;
  Vec separable_dm;

  create_dm(&separable_dm,4);
  val = 1.0;
  add_value_to_dm(separable_dm,0,0,val);
  assemble_dm(separable_dm);

  get_bipartite_concurrence(separable_dm,&concurrence);
  destroy_dm(separable_dm);
  TEST_ASSERT_EQUAL_FLOAT(0.0,concurrence);


  create_dm(&separable_dm,4);
  val = 1.0;
  add_value_to_dm(separable_dm,1,1,val);
  assemble_dm(separable_dm);

  get_bipartite_concurrence(separable_dm,&concurrence);
  destroy_dm(separable_dm);
  TEST_ASSERT_EQUAL_FLOAT(0.0,concurrence);


  create_dm(&separable_dm,4);
  val = 1.0;
  add_value_to_dm(separable_dm,2,2,val);
  assemble_dm(separable_dm);

  get_bipartite_concurrence(separable_dm,&concurrence);
  destroy_dm(separable_dm);
  TEST_ASSERT_EQUAL_FLOAT(0.0,concurrence);


  create_dm(&separable_dm,4);
  val = 1.0;
  add_value_to_dm(separable_dm,3,3,val);
  assemble_dm(separable_dm);

  get_bipartite_concurrence(separable_dm,&concurrence);
  destroy_dm(separable_dm);
  TEST_ASSERT_EQUAL_FLOAT(0.0,concurrence);

  create_dm(&separable_dm,4);
  val = 0.5;
  add_value_to_dm(separable_dm,3,3,val);
  val = 0.5;
  add_value_to_dm(separable_dm,2,2,val);

  assemble_dm(separable_dm);

  get_bipartite_concurrence(separable_dm,&concurrence);
  destroy_dm(separable_dm);
  TEST_ASSERT_EQUAL_FLOAT(0.0,concurrence);


  create_dm(&separable_dm,4);
  val = 0.25;
  add_value_to_dm(separable_dm,3,3,val);
  val = 0.25;
  add_value_to_dm(separable_dm,2,2,val);
  val = 0.25;
  add_value_to_dm(separable_dm,1,1,val);
  val = 0.25;
  add_value_to_dm(separable_dm,0,0,val);

  assemble_dm(separable_dm);

  get_bipartite_concurrence(separable_dm,&concurrence);
  destroy_dm(separable_dm);
  /*
   * Though concurrence is strictly max(0,concurrence), we compare here
   * against the negative value.
   */
  TEST_ASSERT_EQUAL_FLOAT(-0.5,concurrence);
}


/*
 * Test get_expectation_value
 */
void test_get_expectation_value(void)
{
  operator qd1,qd2;
  PetscScalar ev1,ev2,ev3,ev4,val;
  double concurrence;
  Vec dm0;

  create_op(2,&qd1);
  create_op(2,&qd2);
  val = 0;
  add_lin_p(val,1,qd1->n); //Have to add_lin to trick QuaC into thinking we are done creating ops
  create_full_dm(&dm0);
  val = 0.5;
  add_value_to_dm(dm0,0,0,val);
  val = 0.5;
  add_value_to_dm(dm0,3,3,val);

  val = 0.5 + 0.5*PETSC_i;
  add_value_to_dm(dm0,0,3,val);

  val = 0.5 - 0.5*PETSC_i;
  add_value_to_dm(dm0,3,0,val);

  val = 0.5 + 0.5*PETSC_i;
  add_value_to_dm(dm0,1,3,val);

  val = 0.5 - 0.5*PETSC_i;
  add_value_to_dm(dm0,3,1,val);

  val = 0.5 + 0.5*PETSC_i;
  add_value_to_dm(dm0,1,2,val);

  val = 0.5 - 0.5*PETSC_i;
  add_value_to_dm(dm0,2,1,val);

  val = 0.5 + 0.5*PETSC_i;
  add_value_to_dm(dm0,2,3,val);

  val = 0.5 - 0.5*PETSC_i;
  add_value_to_dm(dm0,3,2,val);

  assemble_dm(dm0);


  get_expectation_value(dm0,&ev1,1,qd1->n);
  get_expectation_value(dm0,&ev2,1,qd2->n);
  get_expectation_value(dm0,&ev3,2,qd1->dag,qd2);
  get_expectation_value(dm0,&ev4,2,qd2->dag,qd1);

  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(ev1));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev1));

  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(ev2));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev2));

  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(ev3));
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscImaginaryPart(ev3));

  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(ev4));
  TEST_ASSERT_EQUAL_FLOAT(-0.5,PetscImaginaryPart(ev4));

  return;
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);
  RUN_TEST(test_bipartite_bell);
  RUN_TEST(test_bipartite_separable);
  RUN_TEST(test_get_expectation_value);
  QuaC_finalize();
  return UNITY_END();
}

