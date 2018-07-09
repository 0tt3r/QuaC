#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

operator op2,op3,op4;
Mat mat_pristine;


/*---------------------------------------------
 * 1OP
 *---------------------------------------------*/

void test_add_to_ham_1op_basic_real(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4;
  PetscBool equal;
  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;

  add_to_ham_p(omega2,1,op2);
  add_to_ham_p(omega2,1,op2->dag);
  add_to_ham_p(omega2,1,op2->n);

  add_to_ham_p(omega3,1,op3);
  add_to_ham_p(omega3,1,op3->dag);
  add_to_ham_p(omega3,1,op3->n);

  add_to_ham_p(omega4,1,op4);
  add_to_ham_p(omega4,1,op4->dag);
  add_to_ham_p(omega4,1,op4->n);


  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_br",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_br",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);

  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}

void test_add_to_ham_1op_pauli_real(void)
{
  PetscViewer ham;
  PetscScalar omega2;

  PetscBool equal;
  omega2 = 2.0;

  add_to_ham_p(omega2,1,op2->sig_x);
  add_to_ham_p(omega2,1,op2->sig_y);
  add_to_ham_p(omega2,1,op2->sig_z);
  add_to_ham_p(omega2,1,op2->eye);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_pr",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_pr",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);

  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}


void test_add_to_ham_1op_basic_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4;
  PetscBool equal;

  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;

  add_to_ham_p(omega2,1,op2);
  add_to_ham_p(omega2,1,op2->dag);
  add_to_ham_p(omega2,1,op2->n);

  add_to_ham_p(omega3,1,op3);
  add_to_ham_p(omega3,1,op3->dag);
  add_to_ham_p(omega3,1,op3->n);

  add_to_ham_p(omega4,1,op4);
  add_to_ham_p(omega4,1,op4->dag);
  add_to_ham_p(omega4,1,op4->n);


  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_bc",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_bc",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}



void test_add_to_ham_1op_pauli_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;

  add_to_ham_p(omega2,1,op2->sig_x);
  add_to_ham_p(omega2,1,op2->sig_y);
  add_to_ham_p(omega2,1,op2->sig_z);
  add_to_ham_p(omega2,1,op2->eye);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_pc",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_1op_pc",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}


/*---------------------------------------------
 * 2OP
 *---------------------------------------------*/

void test_add_to_ham_2op_basic_real(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;

  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;
  omega5 = 0.25;

  add_to_ham_p(omega2,2,op2,op2->dag);
  add_to_ham_p(omega2,2,op2->dag,op2);
  add_to_ham_p(omega2,2,op2->n,op2->dag);

  add_to_ham_p(omega3,2,op3,op3);
  add_to_ham_p(omega3,2,op3->dag,op3->dag);
  add_to_ham_p(omega3,2,op3->n,op3);

  add_to_ham_p(omega4,2,op4,op4->n);
  add_to_ham_p(omega4,2,op4->dag,op4->n);
  add_to_ham_p(omega4,2,op4->n,op4->n);

  add_to_ham_p(omega5,2,op4,op3->n);
  add_to_ham_p(omega5,2,op3->dag,op2->n);
  add_to_ham_p(omega5,2,op2->n,op4->n);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_br",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_br",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}

void test_add_to_ham_2op_pauli_real(void)
{
  PetscViewer ham;
  PetscScalar omega2;
  PetscBool equal;
  omega2 = 2.0;

  add_to_ham_p(omega2,2,op2->sig_x,op2->sig_y);
  add_to_ham_p(omega2,2,op2->sig_y,op2->sig_z);
  add_to_ham_p(omega2,2,op2->sig_z,op2->sig_x);
  add_to_ham_p(omega2,2,op2->eye,op2->eye);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_pr",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_pr",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}


void test_add_to_ham_2op_basic_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;
  omega5 = 0.25*PETSC_i;

  add_to_ham_p(omega2,2,op2,op2->dag);
  add_to_ham_p(omega2,2,op2->dag,op2);
  add_to_ham_p(omega2,2,op2->n,op2->dag);

  add_to_ham_p(omega3,2,op3,op3);
  add_to_ham_p(omega3,2,op3->dag,op3->dag);
  add_to_ham_p(omega3,2,op3->n,op3);

  add_to_ham_p(omega4,2,op4,op4->n);
  add_to_ham_p(omega4,2,op4->dag,op4->n);
  add_to_ham_p(omega4,2,op4->n,op4->n);

  add_to_ham_p(omega5,2,op4,op3->n);
  add_to_ham_p(omega5,2,op3->dag,op2->n);
  add_to_ham_p(omega5,2,op2->n,op4->n);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);



#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_bc",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_bc",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}


void test_add_to_ham_2op_pauli_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;

  add_to_ham_p(omega2,2,op2->sig_x,op2->sig_y);
  add_to_ham_p(omega2,2,op2->sig_y,op2->sig_z);
  add_to_ham_p(omega2,2,op2->sig_z,op2->sig_x);
  add_to_ham_p(omega2,2,op2->eye,op2->eye);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_pc",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_2op_pc",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}



/*---------------------------------------------
 * 3OP
 *---------------------------------------------*/

void test_add_to_ham_3op_basic_real(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;
  omega5 = 0.25;

  add_to_ham_p(omega2,3,op2,op2->dag,op2->n);
  add_to_ham_p(omega2,3,op2->dag,op2,op2->n);
  add_to_ham_p(omega2,3,op2->n,op2->dag,op2);

  add_to_ham_p(omega3,3,op3,op3,op3);
  add_to_ham_p(omega3,3,op3->dag,op3->dag,op3->dag);
  add_to_ham_p(omega3,3,op3->n,op3,op3->n);

  add_to_ham_p(omega4,3,op4,op4->n,op4->dag);
  add_to_ham_p(omega4,3,op4->dag,op4->n,op4);
  add_to_ham_p(omega4,3,op4->n,op4->n,op4->n);

  add_to_ham_p(omega5,3,op4,op3->n,op2->dag);
  add_to_ham_p(omega5,3,op3->dag,op2->n,op4);
  add_to_ham_p(omega5,3,op2->n,op4->n,op3->n);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_br",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_br",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}

void test_add_to_ham_3op_pauli_real(void)
{
  PetscViewer ham;
  PetscScalar omega2;

  PetscBool equal;
  omega2 = 2.0;

  add_to_ham_p(omega2,3,op2->sig_x,op2->sig_x,op2->sig_z);
  add_to_ham_p(omega2,3,op2->sig_y,op2->sig_y,op2->sig_x);
  add_to_ham_p(omega2,3,op2->sig_z,op2->sig_z,op2->sig_y);
  add_to_ham_p(omega2,3,op2->eye,op2->eye,op2->eye);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_pr",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif

  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_pr",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);

  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}


void test_add_to_ham_3op_basic_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;
  omega5 = 0.25*PETSC_i;

  add_to_ham_p(omega2,3,op2,op2->dag,op2->n);
  add_to_ham_p(omega2,3,op2->dag,op2,op2->n);
  add_to_ham_p(omega2,3,op2->n,op2->dag,op2);

  add_to_ham_p(omega3,3,op3,op3,op3);
  add_to_ham_p(omega3,3,op3->dag,op3->dag,op3->dag);
  add_to_ham_p(omega3,3,op3->n,op3,op3->n);

  add_to_ham_p(omega4,3,op4,op4->n,op4->dag);
  add_to_ham_p(omega4,3,op4->dag,op4->n,op4);
  add_to_ham_p(omega4,3,op4->n,op4->n,op4->n);

  add_to_ham_p(omega5,3,op4,op3->n,op2->dag);
  add_to_ham_p(omega5,3,op3->dag,op2->n,op4);
  add_to_ham_p(omega5,3,op2->n,op4->n,op3->n);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);


#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_bc",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif

  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_bc",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}


void test_add_to_ham_3op_pauli_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;

  add_to_ham_p(omega2,3,op2->sig_x,op2->sig_x,op2->sig_z);
  add_to_ham_p(omega2,3,op2->sig_y,op2->sig_y,op2->sig_x);
  add_to_ham_p(omega2,3,op2->sig_z,op2->sig_z,op2->sig_y);
  add_to_ham_p(omega2,3,op2->eye,op2->eye,op2->eye);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_pc",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/ham_3op_pc",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);

  //Create some operators
  create_op(2,&op2);

  RUN_TEST(test_add_to_ham_1op_pauli_real);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);

  RUN_TEST(test_add_to_ham_1op_pauli_complex);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);

  RUN_TEST(test_add_to_ham_2op_pauli_real);

  QuaC_clear();
  //create some operators
  create_op(2,&op2);

  RUN_TEST(test_add_to_ham_2op_pauli_complex);

  QuaC_clear();
  // Create some operators
  create_op(2,&op2);

  RUN_TEST(test_add_to_ham_3op_pauli_real);

  QuaC_clear();
  //create some operators
  create_op(2,&op2);

  RUN_TEST(test_add_to_ham_3op_pauli_complex);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_1op_basic_real);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_1op_basic_complex);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_2op_basic_real);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_2op_basic_complex);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_3op_basic_real);

  QuaC_clear();
  //Create some operators
  create_op(2,&op2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_3op_basic_complex);


  QuaC_finalize();
  return UNITY_END();
}
