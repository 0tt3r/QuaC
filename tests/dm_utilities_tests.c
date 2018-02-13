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


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);
  RUN_TEST(test_bipartite_bell);
  RUN_TEST(test_bipartite_separable);
  QuaC_finalize();
  return UNITY_END();
}

