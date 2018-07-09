#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

operator op3,op4;
vec_op   vop2,vop3,vop4;
Mat mat_pristine;


/*---------------------------------------------
 * 2VOP
 *---------------------------------------------*/

void test_add_to_ham_2vop_basic_real(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4;
  PetscBool equal;
  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;

  add_to_ham_p(omega2,2,vop2[1],vop2[0]);
  add_to_ham_p(omega2,2,vop2[0],vop2[1]);
  add_to_ham_p(omega2,2,vop2[1],vop2[1]);


  add_to_ham_p(omega3,2,vop3[1],vop3[0]);
  add_to_ham_p(sqrt(2)*omega3,2,vop3[2],vop3[1]);

  add_to_ham_p(omega3,2,vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,2,vop3[1],vop3[2]);

  add_to_ham_p(omega3,2,vop3[1],vop3[1]);
  add_to_ham_p(2*omega3,2,vop3[2],vop3[2]);

  add_to_ham_p(omega4,2,vop4[1],vop4[0]);
  add_to_ham_p(sqrt(2)*omega4,2,vop4[2],vop4[1]);
  add_to_ham_p(sqrt(3)*omega4,2,vop4[3],vop4[2]);

  add_to_ham_p(omega4,2,vop4[0],vop4[1]);
  add_to_ham_p(sqrt(2)*omega4,2,vop4[1],vop4[2]);
  add_to_ham_p(sqrt(3)*omega4,2,vop4[2],vop4[3]);

  add_to_ham_p(omega4,2,vop4[1],vop4[1]);
  add_to_ham_p(2*omega4,2,vop4[2],vop4[2]);
  add_to_ham_p(3*omega4,2,vop4[3],vop4[3]);



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

void test_add_to_ham_2vop_basic_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4;
  PetscBool equal;

  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;

  add_to_ham_p(omega2,2,vop2[1],vop2[0]);
  add_to_ham_p(omega2,2,vop2[0],vop2[1]);
  add_to_ham_p(omega2,2,vop2[1],vop2[1]);


  add_to_ham_p(omega3,2,vop3[1],vop3[0]);
  add_to_ham_p(sqrt(2)*omega3,2,vop3[2],vop3[1]);

  add_to_ham_p(omega3,2,vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,2,vop3[1],vop3[2]);

  add_to_ham_p(omega3,2,vop3[1],vop3[1]);
  add_to_ham_p(2*omega3,2,vop3[2],vop3[2]);

  add_to_ham_p(omega4,2,vop4[1],vop4[0]);
  add_to_ham_p(sqrt(2)*omega4,2,vop4[2],vop4[1]);
  add_to_ham_p(sqrt(3)*omega4,2,vop4[3],vop4[2]);

  add_to_ham_p(omega4,2,vop4[0],vop4[1]);
  add_to_ham_p(sqrt(2)*omega4,2,vop4[1],vop4[2]);
  add_to_ham_p(sqrt(3)*omega4,2,vop4[2],vop4[3]);

  add_to_ham_p(omega4,2,vop4[1],vop4[1]);
  add_to_ham_p(2*omega4,2,vop4[2],vop4[2]);
  add_to_ham_p(3*omega4,2,vop4[3],vop4[3]);

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


/* /\*--------------------------------------------- */
/*  * 2OP */
/*  *---------------------------------------------*\/ */

void test_add_to_ham_4vop_basic_real(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;

  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;
  omega5 = 0.25;

  add_to_ham_p(omega2,4,vop2[0],vop2[1],vop2[1],vop2[0]);
  add_to_ham_p(omega2,4,vop2[1],vop2[0],vop2[0],vop2[1]);
  add_to_ham_p(omega2,4,vop2[1],vop2[1],vop2[1],vop2[0]);

  /* add_to_ham_p(omega3,2,op3,op3); */
  add_to_ham_p(omega3,4,vop3[0],vop3[1],vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[0],vop3[1],vop3[1],vop3[2]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[1],vop3[2],vop3[0],vop3[1]);
  add_to_ham_p(2*omega3,4,vop3[1],vop3[2],vop3[1],vop3[2]);

  /* add_to_ham_p(omega3,2,op3->dag,op3->dag); */
  add_to_ham_p(omega3,4,vop3[1],vop3[0],vop3[1],vop3[0]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[1],vop3[0],vop3[2],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[2],vop3[1],vop3[1],vop3[0]);
  add_to_ham_p(2*omega3,4,vop3[2],vop3[1],vop3[2],vop3[1]);

  /* add_to_ham_p(omega3,2,op3->n,op3); */
  add_to_ham_p(omega3,4,vop3[1],vop3[1],vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[1],vop3[1],vop3[1],vop3[2]);
  add_to_ham_p(2*omega3,4,vop3[2],vop3[2],vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*2*omega3,4,vop3[2],vop3[2],vop3[1],vop3[2]);

  /* add_to_ham_p(omega4,2,op4,op4->n); */
  add_to_ham_p(omega4,4,vop4[0],vop4[1],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(2)*omega4,4,vop4[1],vop4[2],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(3)*omega4,4,vop4[2],vop4[3],vop4[1],vop4[1]);

  add_to_ham_p(2*omega4,4,vop4[0],vop4[1],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(2)*omega4,4,vop4[1],vop4[2],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(3)*omega4,4,vop4[2],vop4[3],vop4[2],vop4[2]);

  add_to_ham_p(3*omega4,4,vop4[0],vop4[1],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(2)*omega4,4,vop4[1],vop4[2],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(3)*omega4,4,vop4[2],vop4[3],vop4[3],vop4[3]);


  /* add_to_ham_p(omega4,2,op4->dag,op4->n); */
  add_to_ham_p(omega4,4,vop4[1],vop4[0],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(2)*omega4,4,vop4[2],vop4[1],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(3)*omega4,4,vop4[3],vop4[2],vop4[1],vop4[1]);

  add_to_ham_p(2*omega4,4,vop4[1],vop4[0],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(2)*omega4,4,vop4[2],vop4[1],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(3)*omega4,4,vop4[3],vop4[2],vop4[2],vop4[2]);

  add_to_ham_p(3*omega4,4,vop4[1],vop4[0],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(2)*omega4,4,vop4[2],vop4[1],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(3)*omega4,4,vop4[3],vop4[2],vop4[3],vop4[3]);


  /* add_to_ham_p(omega4,2,op4->n,op4->n); */
  add_to_ham_p(omega4,4,vop4[1],vop4[1],vop4[1],vop4[1]);
  add_to_ham_p(2*omega4,4,vop4[2],vop4[2],vop4[1],vop4[1]);
  add_to_ham_p(3*omega4,4,vop4[3],vop4[3],vop4[1],vop4[1]);

  add_to_ham_p(2*omega4,4,vop4[1],vop4[1],vop4[2],vop4[2]);
  add_to_ham_p(2*2*omega4,4,vop4[2],vop4[2],vop4[2],vop4[2]);
  add_to_ham_p(2*3*omega4,4,vop4[3],vop4[3],vop4[2],vop4[2]);

  add_to_ham_p(3*omega4,4,vop4[1],vop4[1],vop4[3],vop4[3]);
  add_to_ham_p(3*2*omega4,4,vop4[2],vop4[2],vop4[3],vop4[3]);
  add_to_ham_p(3*3*omega4,4,vop4[3],vop4[3],vop4[3],vop4[3]);


  /* add_to_ham_p(omega5,2,op4,op3->n); */
  add_to_ham_p(omega5,4,vop4[0],vop4[1],vop3[1],vop3[1]);
  add_to_ham_p(sqrt(2)*omega5,4,vop4[1],vop4[2],vop3[1],vop3[1]);
  add_to_ham_p(sqrt(3)*omega5,4,vop4[2],vop4[3],vop3[1],vop3[1]);

  add_to_ham_p(2*omega5,4,vop4[0],vop4[1],vop3[2],vop3[2]);
  add_to_ham_p(2*sqrt(2)*omega5,4,vop4[1],vop4[2],vop3[2],vop3[2]);
  add_to_ham_p(2*sqrt(3)*omega5,4,vop4[2],vop4[3],vop3[2],vop3[2]);


  /* add_to_ham_p(omega5,2,op3->dag,op2->n); */
  add_to_ham_p(omega5,4,vop3[1],vop3[0],vop2[1],vop2[1]);
  add_to_ham_p(sqrt(2)*omega5,4,vop3[2],vop3[1],vop2[1],vop2[1]);

  /* add_to_ham_p(omega5,2,op2->n,op4->n); */
  add_to_ham_p(omega5,4,vop2[1],vop2[1],vop4[1],vop4[1]);
  add_to_ham_p(2*omega5,4,vop2[1],vop2[1],vop4[2],vop4[2]);
  add_to_ham_p(3*omega5,4,vop2[1],vop2[1],vop4[3],vop4[3]);

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

void test_add_to_ham_4vop_basic_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;
  omega5 = 0.25*PETSC_i;

  add_to_ham_p(omega2,4,vop2[0],vop2[1],vop2[1],vop2[0]);
  add_to_ham_p(omega2,4,vop2[1],vop2[0],vop2[0],vop2[1]);
  add_to_ham_p(omega2,4,vop2[1],vop2[1],vop2[1],vop2[0]);

  /* add_to_ham_p(omega3,2,op3,op3); */
  add_to_ham_p(omega3,4,vop3[0],vop3[1],vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[0],vop3[1],vop3[1],vop3[2]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[1],vop3[2],vop3[0],vop3[1]);
  add_to_ham_p(2*omega3,4,vop3[1],vop3[2],vop3[1],vop3[2]);

  /* add_to_ham_p(omega3,2,op3->dag,op3->dag); */
  add_to_ham_p(omega3,4,vop3[1],vop3[0],vop3[1],vop3[0]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[1],vop3[0],vop3[2],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[2],vop3[1],vop3[1],vop3[0]);
  add_to_ham_p(2*omega3,4,vop3[2],vop3[1],vop3[2],vop3[1]);

  /* add_to_ham_p(omega3,2,op3->n,op3); */
  add_to_ham_p(omega3,4,vop3[1],vop3[1],vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*omega3,4,vop3[1],vop3[1],vop3[1],vop3[2]);
  add_to_ham_p(2*omega3,4,vop3[2],vop3[2],vop3[0],vop3[1]);
  add_to_ham_p(sqrt(2)*2*omega3,4,vop3[2],vop3[2],vop3[1],vop3[2]);

  /* add_to_ham_p(omega4,2,op4,op4->n); */
  add_to_ham_p(omega4,4,vop4[0],vop4[1],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(2)*omega4,4,vop4[1],vop4[2],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(3)*omega4,4,vop4[2],vop4[3],vop4[1],vop4[1]);

  add_to_ham_p(2*omega4,4,vop4[0],vop4[1],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(2)*omega4,4,vop4[1],vop4[2],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(3)*omega4,4,vop4[2],vop4[3],vop4[2],vop4[2]);

  add_to_ham_p(3*omega4,4,vop4[0],vop4[1],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(2)*omega4,4,vop4[1],vop4[2],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(3)*omega4,4,vop4[2],vop4[3],vop4[3],vop4[3]);


  /* add_to_ham_p(omega4,2,op4->dag,op4->n); */
  add_to_ham_p(omega4,4,vop4[1],vop4[0],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(2)*omega4,4,vop4[2],vop4[1],vop4[1],vop4[1]);
  add_to_ham_p(sqrt(3)*omega4,4,vop4[3],vop4[2],vop4[1],vop4[1]);

  add_to_ham_p(2*omega4,4,vop4[1],vop4[0],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(2)*omega4,4,vop4[2],vop4[1],vop4[2],vop4[2]);
  add_to_ham_p(2*sqrt(3)*omega4,4,vop4[3],vop4[2],vop4[2],vop4[2]);

  add_to_ham_p(3*omega4,4,vop4[1],vop4[0],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(2)*omega4,4,vop4[2],vop4[1],vop4[3],vop4[3]);
  add_to_ham_p(3*sqrt(3)*omega4,4,vop4[3],vop4[2],vop4[3],vop4[3]);


  /* add_to_ham_p(omega4,2,op4->n,op4->n); */
  add_to_ham_p(omega4,4,vop4[1],vop4[1],vop4[1],vop4[1]);
  add_to_ham_p(2*omega4,4,vop4[2],vop4[2],vop4[1],vop4[1]);
  add_to_ham_p(3*omega4,4,vop4[3],vop4[3],vop4[1],vop4[1]);

  add_to_ham_p(2*omega4,4,vop4[1],vop4[1],vop4[2],vop4[2]);
  add_to_ham_p(2*2*omega4,4,vop4[2],vop4[2],vop4[2],vop4[2]);
  add_to_ham_p(2*3*omega4,4,vop4[3],vop4[3],vop4[2],vop4[2]);

  add_to_ham_p(3*omega4,4,vop4[1],vop4[1],vop4[3],vop4[3]);
  add_to_ham_p(3*2*omega4,4,vop4[2],vop4[2],vop4[3],vop4[3]);
  add_to_ham_p(3*3*omega4,4,vop4[3],vop4[3],vop4[3],vop4[3]);


  /* add_to_ham_p(omega5,2,op4,op3->n); */
  add_to_ham_p(omega5,4,vop4[0],vop4[1],vop3[1],vop3[1]);
  add_to_ham_p(sqrt(2)*omega5,4,vop4[1],vop4[2],vop3[1],vop3[1]);
  add_to_ham_p(sqrt(3)*omega5,4,vop4[2],vop4[3],vop3[1],vop3[1]);

  add_to_ham_p(2*omega5,4,vop4[0],vop4[1],vop3[2],vop3[2]);
  add_to_ham_p(2*sqrt(2)*omega5,4,vop4[1],vop4[2],vop3[2],vop3[2]);
  add_to_ham_p(2*sqrt(3)*omega5,4,vop4[2],vop4[3],vop3[2],vop3[2]);


  /* add_to_ham_p(omega5,2,op3->dag,op2->n); */
  add_to_ham_p(omega5,4,vop3[1],vop3[0],vop2[1],vop2[1]);
  add_to_ham_p(sqrt(2)*omega5,4,vop3[2],vop3[1],vop2[1],vop2[1]);

  /* add_to_ham_p(omega5,2,op2->n,op4->n); */
  add_to_ham_p(omega5,4,vop2[1],vop2[1],vop4[1],vop4[1]);
  add_to_ham_p(2*omega5,4,vop2[1],vop2[1],vop4[2],vop4[2]);
  add_to_ham_p(3*omega5,4,vop2[1],vop2[1],vop4[3],vop4[3]);

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


/*---------------------------------------------
 * MIX
 *---------------------------------------------*/

void test_add_to_ham_mix_basic_real(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;
  omega5 = 0.25;

  /* add_to_ham_p(omega2,3,op2,op2->dag,op2->n); */
  add_to_ham_p(omega2,6,vop2[0],vop2[1],vop2[1],vop2[0],vop2[1],vop2[1]);

  /* add_to_ham_p(omega2,3,op2->dag,op2,op2->n); */
  add_to_ham_p(omega2,6,vop2[1],vop2[0],vop2[0],vop2[1],vop2[1],vop2[1]);

  /* add_to_ham_p(omega2,3,op2->n,op2->dag,op2); */
  add_to_ham_p(omega2,6,vop2[1],vop2[1],vop2[1],vop2[0],vop2[0],vop2[1]);

  add_to_ham_p(omega3,3,op3,op3,op3);
  add_to_ham_p(omega3,3,op3->dag,op3->dag,op3->dag);
  add_to_ham_p(omega3,3,op3->n,op3,op3->n);

  add_to_ham_p(omega4,3,op4,op4->n,op4->dag);
  add_to_ham_p(omega4,3,op4->dag,op4->n,op4);
  add_to_ham_p(omega4,3,op4->n,op4->n,op4->n);


  /* add_to_ham_p(omega5,3,op4,op3->n,op2->dag); */
  add_to_ham_p(omega5,4,op4,op3->n,vop2[1],vop2[0]);

  /* add_to_ham_p(omega5,3,op3->dag,op2->n,op4); */
  add_to_ham_p(omega5,4,op3->dag,vop2[1],vop2[1],op4);

  /* add_to_ham_p(omega5,3,op2->n,op4->n,op3->n); */
  add_to_ham_p(omega5,4,vop2[1],vop2[1],op4->n,op3->n);

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

void test_add_to_ham_mix_basic_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;
  omega5 = 0.25*PETSC_i;

  /* add_to_ham_p(omega2,3,op2,op2->dag,op2->n); */
  add_to_ham_p(omega2,6,vop2[0],vop2[1],vop2[1],vop2[0],vop2[1],vop2[1]);

  /* add_to_ham_p(omega2,3,op2->dag,op2,op2->n); */
  add_to_ham_p(omega2,6,vop2[1],vop2[0],vop2[0],vop2[1],vop2[1],vop2[1]);

  /* add_to_ham_p(omega2,3,op2->n,op2->dag,op2); */
  add_to_ham_p(omega2,6,vop2[1],vop2[1],vop2[1],vop2[0],vop2[0],vop2[1]);

  add_to_ham_p(omega3,3,op3,op3,op3);
  add_to_ham_p(omega3,3,op3->dag,op3->dag,op3->dag);
  add_to_ham_p(omega3,3,op3->n,op3,op3->n);

  add_to_ham_p(omega4,3,op4,op4->n,op4->dag);
  add_to_ham_p(omega4,3,op4->dag,op4->n,op4);
  add_to_ham_p(omega4,3,op4->n,op4->n,op4->n);


  /* add_to_ham_p(omega5,3,op4,op3->n,op2->dag); */
  add_to_ham_p(omega5,4,op4,op3->n,vop2[1],vop2[0]);

  /* add_to_ham_p(omega5,3,op3->dag,op2->n,op4); */
  add_to_ham_p(omega5,4,op3->dag,vop2[1],vop2[1],op4);

  /* add_to_ham_p(omega5,3,op2->n,op4->n,op3->n); */
  add_to_ham_p(omega5,4,vop2[1],vop2[1],op4->n,op3->n);

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


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);

  //Create some operators
  create_vec(2,&vop2);
  create_vec(3,&vop3);
  create_vec(4,&vop4);

  RUN_TEST(test_add_to_ham_2vop_basic_real);

  QuaC_clear();
  //Create some operators
  create_vec(2,&vop2);
  create_vec(3,&vop3);
  create_vec(4,&vop4);

  RUN_TEST(test_add_to_ham_2vop_basic_complex);

  QuaC_clear();
  //Create some operators
  create_vec(2,&vop2);
  create_vec(3,&vop3);
  create_vec(4,&vop4);

  RUN_TEST(test_add_to_ham_4vop_basic_real);

  QuaC_clear();
  //Create some operators
  create_vec(2,&vop2);
  create_vec(3,&vop3);
  create_vec(4,&vop4);

  RUN_TEST(test_add_to_ham_4vop_basic_complex);

  QuaC_clear();
  //Create some operators
  create_vec(2,&vop2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_mix_basic_real);

  QuaC_clear();
  //Create some operators
  create_vec(2,&vop2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_to_ham_mix_basic_complex);


  QuaC_finalize();
  return UNITY_END();
}
