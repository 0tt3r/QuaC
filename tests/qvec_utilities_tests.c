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

void test_create_arb_qvec(void){
  qvec test_wf,test_dm;
  PetscInt nstates=144;

  create_arb_qvec(&test_wf,nstates,WAVEFUNCTION);
  create_arb_qvec(&test_dm,nstates,DENSITY_MATRIX);

  TEST_ASSERT_EQUAL(test_wf->n,nstates);
  TEST_ASSERT_EQUAL(test_wf->total_levels,nstates);

  TEST_ASSERT_EQUAL(test_dm->n,nstates);
  TEST_ASSERT_EQUAL(test_dm->total_levels,sqrt(nstates));

  destroy_qvec(&test_wf);
  destroy_qvec(&test_dm);


  return;
}

void test_create_arb_qvec_dims(void){
  qvec test_wf,test_dm;
  PetscInt n,i,ndims=4,dims[4] = {2,3,4,2};
  PetscInt ndims_dm=8,dims_dm[8] = {2,3,4,2,2,3,4,2};

  create_arb_qvec_dims(&test_wf,ndims,dims,WAVEFUNCTION);
  create_arb_qvec_dims(&test_dm,ndims_dm,dims_dm,DENSITY_MATRIX);

  n=1;
  for(i=0;i<ndims;i++){
    n=n*dims[i];
  }

  TEST_ASSERT_EQUAL(test_wf->n,n);
  TEST_ASSERT_EQUAL(test_wf->total_levels,n);

  TEST_ASSERT_EQUAL(test_dm->n,pow(n,2));
  TEST_ASSERT_EQUAL(test_dm->total_levels,n);

  destroy_qvec(&test_wf);
  destroy_qvec(&test_dm);

  return;
}

void test_create_qvec_sys_wf(void){
  qsystem qsys;
  operator op2,op3,op4;
  qvec test_wf;
  PetscInt n=24;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  create_qvec_sys(qsys,&test_wf);

  TEST_ASSERT_EQUAL(test_wf->n,n);
  TEST_ASSERT_EQUAL(test_wf->total_levels,n);
  TEST_ASSERT_EQUAL(test_wf->my_type,WAVEFUNCTION);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);
  destroy_qvec(&test_wf);
  destroy_system(&qsys);

}

void test_create_qvec_sys_wf_ens(void){
  qsystem qsys;
  operator op2,op3,op4;
  qvec test_wf;
  PetscInt n=24,n_samples=10;
  PetscScalar alpha=1.0;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  add_lin_term(qsys,alpha,1,op3);

  use_mcwf_solver(qsys,n_samples,NULL);

  create_qvec_sys(qsys,&test_wf);

  TEST_ASSERT_EQUAL(n,test_wf->n);
  TEST_ASSERT_EQUAL(n,test_wf->total_levels);
  TEST_ASSERT_EQUAL(WF_ENSEMBLE,test_wf->my_type);
  TEST_ASSERT_EQUAL(n_samples,test_wf->n_ensemble);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);
  destroy_qvec(&test_wf);
  destroy_system(&qsys);

}

void test_create_wf_sys(void){
  qsystem qsys;
  operator op2,op3,op4;
  qvec test_wf;
  PetscInt n=24;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  create_wf_sys(qsys,&test_wf);

  TEST_ASSERT_EQUAL(test_wf->n,n);
  TEST_ASSERT_EQUAL(test_wf->total_levels,n);
  TEST_ASSERT_EQUAL(test_wf->my_type,WAVEFUNCTION);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);
  destroy_qvec(&test_wf);
  destroy_system(&qsys);

}

void test_create_qvec_sys_dm(void){
  qsystem qsys;
  operator op2,op3,op4;
  qvec test_dm;
  PetscInt n=24;
  PetscScalar alpha=0.0;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  add_lin_term(qsys,alpha,1,op3);

  create_qvec_sys(qsys,&test_dm);

  TEST_ASSERT_EQUAL(test_dm->n,n*n);
  TEST_ASSERT_EQUAL(test_dm->total_levels,n);
  TEST_ASSERT_EQUAL(test_dm->my_type,DENSITY_MATRIX);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);
  destroy_qvec(&test_dm);
  destroy_system(&qsys);
  return;
}

void test_create_dm_sys(void){
  qsystem qsys;
  operator op2,op3,op4;
  qvec test_dm;
  PetscInt n=24;
  PetscScalar alpha=0.0;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op2);
  create_op_sys(qsys,3,&op3);
  create_op_sys(qsys,4,&op4);

  create_dm_sys(qsys,&test_dm);

  TEST_ASSERT_EQUAL(test_dm->n,n*n);
  TEST_ASSERT_EQUAL(test_dm->total_levels,n);
  TEST_ASSERT_EQUAL(test_dm->my_type,DENSITY_MATRIX);

  destroy_op_sys(&op2);
  destroy_op_sys(&op3);
  destroy_op_sys(&op4);
  destroy_qvec(&test_dm);
  destroy_system(&qsys);
  return;
}


void test_change_qvec_dims(void){
  qvec test_wf,test_dm;
  PetscInt n,i,nstates=48,ndims=4,dims[4] = {2,3,4,2};
  PetscInt ndims_dm=8,dims_dm[8] = {2,3,4,2,2,3,4,2};

  //Create qvec with only one big dimension
  create_arb_qvec(&test_wf,nstates,WAVEFUNCTION);

  change_qvec_dims(test_wf,ndims,dims);
  TEST_ASSERT_EQUAL(test_wf->n_ops,ndims);
  TEST_ASSERT_EQUAL(test_wf->ndims_hspace,ndims);
  for(i=0;i<ndims;i++){
    TEST_ASSERT_EQUAL(test_wf->hspace_dims[i],dims[i]);
  }


  create_arb_qvec(&test_dm,nstates*nstates,DENSITY_MATRIX);

  change_qvec_dims(test_dm,ndims_dm,dims_dm);
  TEST_ASSERT_EQUAL(test_dm->n_ops,ndims);
  TEST_ASSERT_EQUAL(test_dm->ndims_hspace,ndims_dm);
  for(i=0;i<ndims_dm;i++){
    TEST_ASSERT_EQUAL(test_dm->hspace_dims[i],dims_dm[i]);
  }

  destroy_qvec(&test_wf);
  destroy_qvec(&test_dm);
  return;
}

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

  destroy_op_sys(&qd1);
  destroy_op_sys(&qd2);
  destroy_system(&qsys);
  destroy_qvec(&wf);
  return;
}

/*
 * Test get_expectation_value for a dm with real numbers
 */
void test_get_expectation_value_dm_real(void)
{
  operator qd1,qd2;
  PetscScalar ev1,ev2,ev3,ev4,val;
  qvec dm;
  qsystem qsys;

  initialize_system(&qsys);

  create_op_sys(qsys,2,&qd1);
  create_op_sys(qsys,2,&qd2);
  val = 0;

  create_dm_sys(qsys,&dm);
  ev1 = 1.0;

  add_to_qvec_fock_op(ev1,dm,2,qd1,1,qd2,1);

  assemble_qvec(dm);

  get_expectation_value_qvec(dm,&ev1,1,qd1->n);
  get_expectation_value_qvec(dm,&ev2,1,qd2->n);
  get_expectation_value_qvec(dm,&ev3,2,qd1->dag,qd2);
  get_expectation_value_qvec(dm,&ev4,2,qd2->dag,qd1);

  TEST_ASSERT_EQUAL_FLOAT(1.0,PetscRealPart(ev1));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev1));

  TEST_ASSERT_EQUAL_FLOAT(1.0,PetscRealPart(ev2));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev2));

  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(ev3));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev3));

  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(ev4));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(ev4));

  destroy_system(&qsys);
  destroy_op_sys(&qd1);
  destroy_op_sys(&qd2);
  destroy_qvec(&dm);
  return;
}

/*
 * Test ptrace
 */
void test_ptrace_wf_22_bs(void)
{
  qvec wf,ptrace0_dm,ptrace1_dm;
  PetscInt ndims_wf=2,dims_wf[2] = {2,2},op_list[1]={0};
  PetscScalar val=pow(2,-0.5);
  PetscBool flag;

  create_arb_qvec_dims(&wf,ndims_wf,dims_wf,WAVEFUNCTION);

  //Create a Bell state
  // 1/sqrt(2)
  // 0
  // 0
  // 1/sqrt(2)

  add_to_qvec(wf,val,0);
  add_to_qvec(wf,val,3);


  assemble_qvec(wf);

  //Partial trace the first qubit away
  ptrace_over_list_qvec(wf,1,op_list,&ptrace0_dm);

  get_dm_element_qvec(ptrace0_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  //Partial trace the first qubit away
  op_list[0] = 1;
  ptrace_over_list_qvec(wf,1,op_list,&ptrace1_dm);

  check_qvec_equal(ptrace0_dm,ptrace1_dm,&flag);

  TEST_ASSERT_EQUAL_INT(flag,PETSC_TRUE);

  destroy_qvec(&wf);
  destroy_qvec(&ptrace0_dm);
  destroy_qvec(&ptrace1_dm);
  return;
}


/*
 * Test ptrace
 */
void test_ptrace_wf_22_rand(void)
{
  qvec wf,ptrace0_dm,ptrace1_dm;
  PetscInt ndims_wf=2,dims_wf[2] = {2,2},op_list[1]={0};
  PetscScalar val=pow(2,-0.5);
  PetscBool flag;

  create_arb_qvec_dims(&wf,ndims_wf,dims_wf,WAVEFUNCTION);

  //Create a Wf
  val = 0.46076709- 0.32805127*PETSC_i;
  add_to_qvec(wf,val,0);
  val = -0.34063756+0.22835621*PETSC_i;
  add_to_qvec(wf,val,1);
  val = 0.28068659-0.12410744*PETSC_i;
  add_to_qvec(wf,val,2);
  val =  0.62797233+0.15283547*PETSC_i;
  add_to_qvec(wf,val,3);

  assemble_qvec(wf);

  //Partial trace the qubit 0 away - keep qubit 1
  ptrace_over_list_qvec(wf,1,op_list,&ptrace0_dm);

  get_dm_element_qvec(ptrace0_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.41411156544486266,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace0_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.07457172896962921,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.11430734736991019,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace0_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.07457172896962921,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.11430734736991019,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace0_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5858884340640674,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  //Partial trace qubit 1 away - keep qubit 0
  op_list[0] = 1;
  ptrace_over_list_qvec(wf,1,op_list,&ptrace1_dm);
  get_dm_element_qvec(ptrace1_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.48810445290399873,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace1_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.008965286971174186,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.1605679145006028,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace1_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.008965286971174186,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.1605679145006028,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace1_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5118955466049314,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));


  destroy_qvec(&wf);
  destroy_qvec(&ptrace0_dm);
  destroy_qvec(&ptrace1_dm);
  return;
}

/*
 * Test ptrace
 * This tests the ordering [qt0,qb1,qb2]
 * where qt is a qutrit, qb is a qubit
 */
void test_ptrace_wf_322_rand(void)
{
  qvec wf,ptrace0_dm,ptrace1_dm,ptrace2_dm;
  PetscInt i,j,ndims_wf=3,dims_wf[3] = {3,2,2},op_list[2]={0,2};
  PetscScalar val=pow(2,-0.5);
  PetscBool flag;

  create_arb_qvec_dims(&wf,ndims_wf,dims_wf,WAVEFUNCTION);

  //Create a Wf
  val = 0.46076709- 0.32805127*PETSC_i;
  add_to_qvec(wf,val,0);
  val = -0.34063756+0.22835621*PETSC_i;
  add_to_qvec(wf,val,1);
  val = 0.28068659-0.12410744*PETSC_i;
  add_to_qvec(wf,val,2);
  val =  0.62797233+0.15283547*PETSC_i;
  add_to_qvec(wf,val,3);

  assemble_qvec(wf);

  //Partial trace qb2 and qt0 away, keep qb1
  ptrace_over_list_qvec(wf,2,op_list,&ptrace0_dm);
  print_qvec(ptrace0_dm);
  get_dm_element_qvec(ptrace0_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.48810445290399873,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace0_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.008965286971174186,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.1605679145006028,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace0_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.008965286971174186,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.1605679145006028,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace0_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5118955466049314,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  //Partial trace the qt0 and qb1 away, keep qb2
  op_list[1] = 1;
  ptrace_over_list_qvec(wf,2,op_list,&ptrace1_dm);

  get_dm_element_qvec(ptrace1_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.41411156544486266,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace1_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.07457172896962921,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.11430734736991019,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace1_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.07457172896962921,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.11430734736991019,PetscImaginaryPart(val));

  get_dm_element_qvec(ptrace1_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5858884340640674,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  //Partial trace the qb1 and qb2 away, keeping qt0
  op_list[0] = 1;
  op_list[1] = 2;
  ptrace_over_list_qvec(wf,2,op_list,&ptrace2_dm);

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      get_dm_element_qvec(ptrace2_dm,i,j,&val);
      if(i==0&&j==0){
        TEST_ASSERT_EQUAL_FLOAT(1.0,PetscRealPart(val));
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));
      } else {
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));
      }
    }
  }

  destroy_qvec(&wf);
  destroy_qvec(&ptrace0_dm);
  destroy_qvec(&ptrace1_dm);
  destroy_qvec(&ptrace2_dm);
  return;
}


/*
 * Test ptrace
 */
void test_ptrace_dm_22(void)
{
  qvec dm,ptrace0_dm,ptrace1_dm;
  PetscInt ndims_dm=4,dims_dm[4] = {2,2,2,2},op_list[1]={0};
  PetscScalar val=0.5;
  PetscBool flag;
  create_arb_qvec_dims(&dm,ndims_dm,dims_dm,DENSITY_MATRIX);

  //Create a Bell state
  // 0.5 0 0 0.5
  // 0 0 0 0
  // 0 0 0 0
  // 0.5 0 0 0.5

  add_to_qvec(dm,val,0,0);
  add_to_qvec(dm,val,0,3);
  add_to_qvec(dm,val,3,0);
  add_to_qvec(dm,val,3,3);

  assemble_qvec(dm);

  //Partial trace the first qubit away
  ptrace_over_list_qvec(dm,1,op_list,&ptrace0_dm);

  get_dm_element_qvec(ptrace0_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  //Partial trace the second qubit away
  op_list[0] = 1;
  ptrace_over_list_qvec(dm,1,op_list,&ptrace1_dm);

  check_qvec_equal(ptrace0_dm,ptrace1_dm,&flag);

  TEST_ASSERT_EQUAL_INT(flag,PETSC_TRUE);

  destroy_qvec(&dm);
  destroy_qvec(&ptrace0_dm);
  destroy_qvec(&ptrace1_dm);
  return;
}

/*
 * Test ptrace
 * This tests the ordering [qt0,qb1,qb2]
 * where qt is a qutrit, qb is qubit
 */
void test_ptrace_dm_322(void)
{
  qvec dm,ptrace0_dm,ptrace1_dm,ptrace2_dm;
  PetscInt i,j,ndims_dm=6,dims_dm[6] = {3,2,2,3,2,2},op_list[2]={0,2};
  PetscScalar val=0.5;
  PetscBool flag;
  create_arb_qvec_dims(&dm,ndims_dm,dims_dm,DENSITY_MATRIX);

  //Create a Bell state
  //           0.5 0 0 0.5
  //           0 0 0 0
  //I_3 \cross 0 0 0 0
  //           0.5 0 0 0.5

  add_to_qvec(dm,val,0,0);
  add_to_qvec(dm,val,0,3);
  add_to_qvec(dm,val,3,0);
  add_to_qvec(dm,val,3,3);

  assemble_qvec(dm);

  //Partial trace the qutrit (0) and qubit (2) away - keeping qubit 1
  ptrace_over_list_qvec(dm,2,op_list,&ptrace0_dm);

  get_dm_element_qvec(ptrace0_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  //Partial trace the qubit 1 and the qutrit (0) away - keeping qubit 2
  op_list[1] = 1;
  ptrace_over_list_qvec(dm,2,op_list,&ptrace1_dm);

  check_qvec_equal(ptrace0_dm,ptrace1_dm,&flag);

  TEST_ASSERT_EQUAL_INT(flag,PETSC_TRUE);

  //Partial trace the qubits away - keeping qutrit (0)
  op_list[0] = 1;
  op_list[1] = 2;
  ptrace_over_list_qvec(dm,2,op_list,&ptrace2_dm);

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      get_dm_element_qvec(ptrace2_dm,i,j,&val);
      if(i==0&&j==0){
        TEST_ASSERT_EQUAL_FLOAT(1.0,PetscRealPart(val));
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));
      } else {
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));
      }
    }
  }

  destroy_qvec(&dm);
  destroy_qvec(&ptrace0_dm);
  destroy_qvec(&ptrace1_dm);
  destroy_qvec(&ptrace2_dm);
  return;
}

/*
 * Test ptrace
 */
void test_ptrace_dm_232(void)
{
  qvec dm,ptrace0_dm,ptrace1_dm;
  PetscInt ndims_dm=6,dims_dm[6] = {2,3,2,2,3,2},op_list[2]={0,1};
  PetscScalar val=0.5;
  PetscBool flag;
  create_arb_qvec_dims(&dm,ndims_dm,dims_dm,DENSITY_MATRIX);

  //Create a Bell state for the two qubits

  add_to_qvec(dm,val,0,0);
  add_to_qvec(dm,val,0,11);
  add_to_qvec(dm,val,11,0);
  add_to_qvec(dm,val,11,11);

  assemble_qvec(dm);

  //Partial trace the first qubit away
  ptrace_over_list_qvec(dm,2,op_list,&ptrace0_dm);

  get_dm_element_qvec(ptrace0_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  //Partial trace the second qubit away
  op_list[0] = 2;
  ptrace_over_list_qvec(dm,2,op_list,&ptrace1_dm);

  check_qvec_equal(ptrace0_dm,ptrace1_dm,&flag);

  TEST_ASSERT_EQUAL_INT(flag,PETSC_TRUE);

  destroy_qvec(&dm);
  destroy_qvec(&ptrace0_dm);
  destroy_qvec(&ptrace1_dm);
  return;
}


/*
 * Test ptrace
 * This tests the ordering [qb0,qb1,qt2]
 * where qt is a qutrit, qb is qubit

 */
void test_ptrace_dm_223(void)
{
  qvec dm,ptrace0_dm,ptrace1_dm,ptrace2_dm;
  PetscInt i,j,ndims_dm=6,dims_dm[6] = {2,2,3,2,2,3},op_list[2]={0,2};
  PetscScalar val=0.5;
  PetscBool flag;
  create_arb_qvec_dims(&dm,ndims_dm,dims_dm,DENSITY_MATRIX);

  //Create a Bell state for the two qubits

  add_to_qvec(dm,val,0,0);
  add_to_qvec(dm,val,0,9);
  add_to_qvec(dm,val,9,0);
  add_to_qvec(dm,val,9,9);

  assemble_qvec(dm);

  //Partial trace qb0 and qt2 away, leaving qb1
  ptrace_over_list_qvec(dm,2,op_list,&ptrace0_dm);

  get_dm_element_qvec(ptrace0_dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,0,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));

  get_dm_element_qvec(ptrace0_dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));

  //Partial trace qb1 and qt2 away, leaving qb0
  op_list[0] = 1;
  ptrace_over_list_qvec(dm,2,op_list,&ptrace1_dm);

  check_qvec_equal(ptrace0_dm,ptrace1_dm,&flag);

  TEST_ASSERT_EQUAL_INT(flag,PETSC_TRUE);


  //Partial trace the qubits away - keeping qutrit (2)
  op_list[0] = 0;
  op_list[1] = 1;
  ptrace_over_list_qvec(dm,2,op_list,&ptrace2_dm);

  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      get_dm_element_qvec(ptrace2_dm,i,j,&val);
      if(i==0&&j==0){
        TEST_ASSERT_EQUAL_FLOAT(1.0,PetscRealPart(val));
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));
      } else {
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscRealPart(val));
        TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));
      }
    }
  }


  destroy_qvec(&dm);
  destroy_qvec(&ptrace0_dm);
  destroy_qvec(&ptrace1_dm);
  destroy_qvec(&ptrace2_dm);
  return;
}


void test_read_dm_binary(void){
  qvec dm;
  char fname[255];
  PetscScalar val,val2;

  strcpy(fname,"tests/pristine_matrices/rand_dm.dat");
  read_qvec_dm_binary(&dm,fname);

  get_dm_element_qvec(dm,0,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.21986779173439994,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_dm_element_qvec(dm,1,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.16336963153116132,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.05584012449039134,PetscImaginaryPart(val));

  //Should be hermitian,check for that
  get_dm_element_qvec(dm,0,1,&val2);
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(val),PetscRealPart(val2));
  TEST_ASSERT_EQUAL_FLOAT(PetscImaginaryPart(val),-PetscImaginaryPart(val2));

  get_dm_element_qvec(dm,2,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.04361830309636767,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.023593974808361683,PetscImaginaryPart(val));

  //Should be hermitian,check for that
  get_dm_element_qvec(dm,0,2,&val2);
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(val),PetscRealPart(val2));
  TEST_ASSERT_EQUAL_FLOAT(PetscImaginaryPart(val),-PetscImaginaryPart(val2));

  get_dm_element_qvec(dm,3,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.021305934381833865,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.08383562473168714,PetscImaginaryPart(val));

  //Should be hermitian,check for that
  get_dm_element_qvec(dm,0,3,&val2);
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(val),PetscRealPart(val2));
  TEST_ASSERT_EQUAL_FLOAT(PetscImaginaryPart(val),-PetscImaginaryPart(val2));

  get_dm_element_qvec(dm,1,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.1836281750676842,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_dm_element_qvec(dm,2,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.03482138755734677,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.01344691070638579,PetscImaginaryPart(val));

  //Should be hermitian,check for that
  get_dm_element_qvec(dm,1,2,&val2);
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(val),PetscRealPart(val2));
  TEST_ASSERT_EQUAL_FLOAT(PetscImaginaryPart(val),-PetscImaginaryPart(val2));

  get_dm_element_qvec(dm,3,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.0056141822506085995,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.041200982904222694,PetscImaginaryPart(val));

  //Should be hermitian, show check for that
  get_dm_element_qvec(dm,1,3,&val2);
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(val),PetscRealPart(val2));
  TEST_ASSERT_EQUAL_FLOAT(PetscImaginaryPart(val),-PetscImaginaryPart(val2));

  get_dm_element_qvec(dm,2,2,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.3375913321814747,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_dm_element_qvec(dm,3,2,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.010080847069452996,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.06315549591134322,PetscImaginaryPart(val));

  //Should be hermitian, show check for that
  get_dm_element_qvec(dm,2,3,&val2);
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(val),PetscRealPart(val2));
  TEST_ASSERT_EQUAL_FLOAT(PetscImaginaryPart(val),-PetscImaginaryPart(val2));

  get_dm_element_qvec(dm,3,3,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.25891270101644115,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));


  destroy_qvec(&dm);
  return;
}


void test_read_wf_binary(void){
  qvec wf;
  char fname[255];
  PetscScalar val;

  strcpy(fname,"tests/pristine_matrices/rand_wf.dat");
  read_qvec_wf_binary(&wf,fname);

  get_wf_element_qvec(wf,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(-0.637991927277723,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.3126100254765921,PetscImaginaryPart(val));

  get_wf_element_qvec(wf,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.08900411349251001,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(-0.19371936261879386,PetscImaginaryPart(val));

  get_wf_element_qvec(wf,2,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.45877685279777203,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.06902325924823785,PetscImaginaryPart(val));

  get_wf_element_qvec(wf,3,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.1497653947732421,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.4605673290368073,PetscImaginaryPart(val));


  destroy_qvec(&wf);
  return;
}

void test_get_bitstring_probs_wf(void){
    qvec test_wf;
    qsystem qsys;
    operator op0,op1,op2;
    PetscInt i,nloc;
    PetscScalar val=1/sqrt(8);
    PetscReal *probs,*vars,exp_prob=1./8.;
    char message[255];
    initialize_system(&qsys);
    //Create some operators
    create_op_sys(qsys,2,&op0);
    create_op_sys(qsys,2,&op1);
    create_op_sys(qsys,2,&op2);

    create_qvec_sys(qsys,&test_wf);

    for(i=0;i<8;i++){
      add_to_qvec(test_wf,val,i);
    }

    assemble_qvec(test_wf);

    get_bitstring_probs(test_wf,&nloc,&probs,&vars);

    for(i=0;i<8;i++){
      sprintf(message,"Index %d\n",i);
      TEST_ASSERT_EQUAL_FLOAT_MESSAGE(exp_prob,probs[i],message);
    }

    free(probs);
    free(vars);
    destroy_op_sys(&op0);
    destroy_op_sys(&op1);
    destroy_op_sys(&op2);
    destroy_system(&qsys);
    return;
}


void test_get_bitstring_probs_dm(void){
  qvec test_dm;
  qsystem qsys;
  operator op0,op1,op2;
  PetscInt i,nloc;
  PetscScalar val=1./8.;
  PetscReal *probs,*vars,exp_prob=1./8.;
  char message[255];
  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);
  create_op_sys(qsys,2,&op2);

  add_lin_term(qsys,val,1,op0);

  create_qvec_sys(qsys,&test_dm);

  for(i=0;i<8;i++){
    add_to_qvec(test_dm,val,i,i);
  }

  assemble_qvec(test_dm);

  get_bitstring_probs(test_dm,&nloc,&probs,&vars);

  for(i=0;i<8;i++){
    sprintf(message,"Index %d\n",i);
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(exp_prob,probs[i],message);
  }

  free(probs);
  free(vars);
  destroy_op_sys(&op0);
  destroy_op_sys(&op1);
  destroy_op_sys(&op2);
  destroy_system(&qsys);
  return;
}


void test_get_bitstring_probs_wf_ens(void){
  qvec test_wf;
  qsystem qsys;
  operator op0,op1,op2;
  PetscInt i,nloc;
  PetscScalar val=1.0;
  PetscReal *probs,*vars,exp_prob=1./8.;
  char message[255];

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);
  create_op_sys(qsys,2,&op2);

  add_lin_term(qsys,val,1,op0);
  use_mcwf_solver(qsys,8,NULL);
  create_qvec_sys(qsys,&test_wf);

  for(i=0;i<8;i++){
    add_to_wf_ens_loc(test_wf,i,val,i);
  }

  assemble_qvec(test_wf);

  get_bitstring_probs(test_wf,&nloc,&probs,&vars);

  for(i=0;i<8;i++){
    sprintf(message,"Index %d",i);
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(exp_prob,probs[i],message);
  }

  free(probs);
  free(vars);
  destroy_op_sys(&op0);
  destroy_op_sys(&op1);
  destroy_op_sys(&op2);
  destroy_system(&qsys);
  return;
}


void test_get_fidelity_qvec_wf_wf(void){
  qvec wf1,wf2;
  PetscInt nstates = 4;
  PetscScalar val = 1/sqrt(2);
  PetscReal fid,var;

  create_arb_qvec(&wf1,nstates,WAVEFUNCTION);
  create_arb_qvec(&wf2,nstates,WAVEFUNCTION);

  add_to_qvec_loc(wf1,val,0);
  add_to_qvec_loc(wf1,val,3);

  add_to_qvec_loc(wf2,1.0,0);

  get_fidelity_qvec(wf1,wf2,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&wf1);
  destroy_qvec(&wf2);

  return;
}

void test_get_fidelity_qvec_wf_dm(void){
  qvec wf1,dm;
  PetscInt nstates = 4;
  PetscScalar val = 1/sqrt(2);
  PetscReal fid,var;

  create_arb_qvec(&wf1,nstates,WAVEFUNCTION);
  create_arb_qvec(&dm,nstates*nstates,DENSITY_MATRIX);

  add_to_qvec_loc(wf1,val,0);
  add_to_qvec_loc(wf1,val,3);

  add_to_qvec_loc(dm,1.0,0);

  get_fidelity_qvec(wf1,dm,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&wf1);
  destroy_qvec(&dm);

  return;
}

void test_get_fidelity_qvec_dm_wf(void){
  qvec dm1,wf2;
  PetscInt nstates = 4;
  PetscScalar val = 0.5;
  PetscReal fid,var;

  create_arb_qvec(&dm1,nstates*nstates,DENSITY_MATRIX);
  create_arb_qvec(&wf2,nstates,WAVEFUNCTION);

  add_to_qvec(dm1,val,0,0);
  add_to_qvec(dm1,val,0,3);
  add_to_qvec(dm1,val,3,0);
  add_to_qvec(dm1,val,3,3);

  add_to_qvec_loc(wf2,1.0,0);

  get_fidelity_qvec(dm1,wf2,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&dm1);
  destroy_qvec(&wf2);

  return;
}

void test_get_fidelity_qvec_dm_dm(void){
  qvec dm1,dm2;
  PetscInt nstates = 4;
  PetscScalar val = 0.5;
  PetscReal fid,var;

  create_arb_qvec(&dm1,nstates*nstates,DENSITY_MATRIX);
  create_arb_qvec(&dm2,nstates*nstates,DENSITY_MATRIX);

  add_to_qvec(dm1,val,0,0);
  add_to_qvec(dm1,val,0,3);
  add_to_qvec(dm1,val,3,0);
  add_to_qvec(dm1,val,3,3);

  add_to_qvec(dm2,1.0,0,0);

  get_fidelity_qvec(dm1,dm2,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&dm1);
  destroy_qvec(&dm2);

  return;
}

void test_get_fidelity_qvec_wf_wf_ens(void){
  qvec wf1,wf_ens2;
  PetscInt nstates = 4,n_ens = 4,i;
  PetscScalar val = 1/sqrt(2);
  PetscReal fid,var;
  operator op0,op1;
  qsystem qsys;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);

  add_lin_term(qsys,val,1,op0);
  use_mcwf_solver(qsys,n_ens,NULL);


  create_arb_qvec(&wf1,nstates,WAVEFUNCTION);
  create_qvec_sys(qsys,&wf_ens2);

  add_to_qvec(wf1,val,0);
  add_to_qvec(wf1,val,3);

  for(i=0;i<n_ens;i++){
    add_to_wf_ens_loc(wf_ens2,i,1.0,0);
  }
  get_fidelity_qvec(wf1,wf_ens2,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&wf1);
  destroy_qvec(&wf_ens2);

  destroy_op_sys(&op0);
  destroy_op_sys(&op1);
  destroy_system(&qsys);

  return;
}

void test_get_fidelity_qvec_wf_ens_wf(void){
  qvec wf_ens1,wf2;
  PetscInt nstates = 4,n_ens = 2;
  PetscScalar val = 1;
  PetscReal fid,var;
  operator op0,op1;
  qsystem qsys;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);

  add_lin_term(qsys,val,1,op0);
  use_mcwf_solver(qsys,n_ens,NULL);

  create_qvec_sys(qsys,&wf_ens1);
  create_arb_qvec(&wf2,nstates,WAVEFUNCTION);


  add_to_qvec(wf2,val,0);

  add_to_wf_ens_loc(wf_ens1,0,val,0);
  add_to_wf_ens_loc(wf_ens1,1,val,3);

  get_fidelity_qvec(wf_ens1,wf2,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&wf_ens1);
  destroy_qvec(&wf2);


  destroy_op_sys(&op0);
  destroy_op_sys(&op1);
  destroy_system(&qsys);

  return;
}

//Not a valid test at the moment
void test_get_fidelity_qvec_wf_ens_dm(void){
  qvec wf_ens1,dm2;
  PetscInt nstates = 4,n_ens = 2;
  PetscScalar val = 1;
  PetscReal fid,var;
  operator op0,op1;
  qsystem qsys;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);

  add_lin_term(qsys,val,1,op0);
  use_mcwf_solver(qsys,n_ens,NULL);

  create_qvec_sys(qsys,&wf_ens1);
  create_arb_qvec(&dm2,nstates*nstates,DENSITY_MATRIX);


  add_to_qvec(dm2,val,0,0);

  add_to_wf_ens_loc(wf_ens1,0,val,0);
  add_to_wf_ens_loc(wf_ens1,1,val,3);

  get_fidelity_qvec(wf_ens1,dm2,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&wf_ens1);
  destroy_qvec(&dm2);


  destroy_op_sys(&op0);
  destroy_op_sys(&op1);
  destroy_system(&qsys);

  return;
}

//Not a valid test at the moment
void test_get_fidelity_qvec_dm_wf_ens(void){
  qvec wf_ens2,dm1;
  PetscInt nstates = 4,n_ens = 2;
  PetscScalar val = 1;
  PetscReal fid,var;
  operator op0,op1;
  qsystem qsys;

  initialize_system(&qsys);
  //Create some operators
  create_op_sys(qsys,2,&op0);
  create_op_sys(qsys,2,&op1);

  add_lin_term(qsys,val,1,op0);
  use_mcwf_solver(qsys,n_ens,NULL);

  create_qvec_sys(qsys,&wf_ens2);
  create_arb_qvec(&dm1,nstates*nstates,DENSITY_MATRIX);

  val=0.5;
  add_to_qvec(dm1,val,0,0);
  add_to_qvec(dm1,val,0,3);
  add_to_qvec(dm1,val,3,0);
  add_to_qvec(dm1,val,3,3);

  val=1.0;
  add_to_wf_ens_loc(wf_ens2,0,val,0);
  add_to_wf_ens_loc(wf_ens2,1,val,0);

  get_fidelity_qvec(dm1,wf_ens2,&fid,&var);

  TEST_ASSERT_EQUAL_FLOAT(0.5,fid);

  destroy_qvec(&wf_ens2);
  destroy_qvec(&dm1);


  destroy_op_sys(&op0);
  destroy_op_sys(&op1);
  destroy_system(&qsys);

  return;
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);

  RUN_TEST(test_get_expectation_value_wf_real);
  RUN_TEST(test_get_expectation_value_dm_real);

  RUN_TEST(test_create_arb_qvec);
  RUN_TEST(test_create_arb_qvec_dims);
  RUN_TEST(test_change_qvec_dims);
  RUN_TEST(test_create_qvec_sys_wf);
  RUN_TEST(test_create_qvec_sys_wf_ens);
  RUN_TEST(test_create_qvec_sys_dm);

  RUN_TEST(test_create_wf_sys);
  RUN_TEST(test_create_dm_sys);

  RUN_TEST(test_ptrace_dm_22);
  RUN_TEST(test_ptrace_dm_223);
  RUN_TEST(test_ptrace_dm_232);
  RUN_TEST(test_ptrace_dm_322);

  RUN_TEST(test_ptrace_wf_22_bs);
  RUN_TEST(test_ptrace_wf_22_rand);
  RUN_TEST(test_ptrace_wf_322_rand);

  RUN_TEST(test_read_dm_binary);
  RUN_TEST(test_read_wf_binary);

  RUN_TEST(test_get_bitstring_probs_wf);
  RUN_TEST(test_get_bitstring_probs_wf_ens);
  RUN_TEST(test_get_bitstring_probs_dm);

  RUN_TEST(test_get_fidelity_qvec_wf_wf);
  RUN_TEST(test_get_fidelity_qvec_wf_dm);
  RUN_TEST(test_get_fidelity_qvec_dm_wf);
  RUN_TEST(test_get_fidelity_qvec_dm_dm);
  RUN_TEST(test_get_fidelity_qvec_wf_wf_ens);
  RUN_TEST(test_get_fidelity_qvec_wf_ens_wf);

  //Below two are not good tests, but are kept because the
  //code might be used to make a good test
  /* RUN_TEST(test_get_fidelity_qvec_wf_ens_dm); */
  /* RUN_TEST(test_get_fidelity_qvec_dm_wf_ens); */

  QuaC_finalize();
  return UNITY_END();
}

