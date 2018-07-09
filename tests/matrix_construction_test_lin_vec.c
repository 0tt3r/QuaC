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
 * MIX
 *---------------------------------------------*/

void test_add_lin_mix_basic_real(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0;
  omega3 = 1.0;
  omega4 = 0.5;
  omega5 = 0.25;

  /* add_lin_p(omega2,3,op2,op2->dag,op2->n); */
  add_lin_p(omega2,6,vop2[0],vop2[1],vop2[1],vop2[0],vop2[1],vop2[1]);

  /* add_lin_p(omega2,3,op2->dag,op2,op2->n); */
  add_lin_p(omega2,6,vop2[1],vop2[0],vop2[0],vop2[1],vop2[1],vop2[1]);

  /* add_lin_p(omega2,3,op2->n,op2->dag,op2); */
  add_lin_p(omega2,6,vop2[1],vop2[1],vop2[1],vop2[0],vop2[0],vop2[1]);

  add_lin_p(omega3,3,op3,op3,op3);
  add_lin_p(omega3,3,op3->dag,op3->dag,op3->dag);
  add_lin_p(omega3,3,op3->n,op3,op3->n);

  add_lin_p(omega4,3,op4,op4->n,op4->dag);
  add_lin_p(omega4,3,op4->dag,op4->n,op4);
  add_lin_p(omega4,3,op4->n,op4->n,op4->n);


  /* add_lin_p(omega5,3,op4,op3->n,op2->dag); */
  add_lin_p(omega5,4,op4,op3->n,vop2[1],vop2[0]);

  /* add_lin_p(omega5,3,op3->dag,op2->n,op4); */
  add_lin_p(omega5,4,op3->dag,vop2[1],vop2[1],op4);

  /* add_lin_p(omega5,3,op2->n,op4->n,op3->n); */
  add_lin_p(omega5,4,vop2[1],vop2[1],op4->n,op3->n);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/lin_3op_br",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/lin_3op_br",FILE_MODE_READ,&ham);
  MatDuplicate(full_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine);
  MatLoad(mat_pristine,ham);
  PetscViewerDestroy(&ham);
  MatEqual(mat_pristine,full_A,&equal);
  MatDestroy(&mat_pristine);

  TEST_ASSERT(equal==PETSC_TRUE);
}

void test_add_lin_mix_basic_complex(void)
{
  PetscViewer ham;
  PetscScalar omega2,omega3,omega4,omega5;
  PetscBool equal;
  omega2 = 2.0*PETSC_i;
  omega3 = 1.0*PETSC_i;
  omega4 = 0.5*PETSC_i;
  omega5 = 0.25*PETSC_i;

  /* add_lin_p(omega2,3,op2,op2->dag,op2->n); */
  add_lin_p(omega2,6,vop2[0],vop2[1],vop2[1],vop2[0],vop2[1],vop2[1]);

  /* add_lin_p(omega2,3,op2->dag,op2,op2->n); */
  add_lin_p(omega2,6,vop2[1],vop2[0],vop2[0],vop2[1],vop2[1],vop2[1]);

  /* add_lin_p(omega2,3,op2->n,op2->dag,op2); */
  add_lin_p(omega2,6,vop2[1],vop2[1],vop2[1],vop2[0],vop2[0],vop2[1]);

  add_lin_p(omega3,3,op3,op3,op3);
  add_lin_p(omega3,3,op3->dag,op3->dag,op3->dag);
  add_lin_p(omega3,3,op3->n,op3,op3->n);

  add_lin_p(omega4,3,op4,op4->n,op4->dag);
  add_lin_p(omega4,3,op4->dag,op4->n,op4);
  add_lin_p(omega4,3,op4->n,op4->n,op4->n);


  /* add_lin_p(omega5,3,op4,op3->n,op2->dag); */
  add_lin_p(omega5,4,op4,op3->n,vop2[1],vop2[0]);

  /* add_lin_p(omega5,3,op3->dag,op2->n,op4); */
  add_lin_p(omega5,4,op3->dag,vop2[1],vop2[1],op4);

  /* add_lin_p(omega5,3,op2->n,op4->n,op3->n); */
  add_lin_p(omega5,4,vop2[1],vop2[1],op4->n,op3->n);

  MatAssemblyBegin(full_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(full_A,MAT_FINAL_ASSEMBLY);

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/lin_3op_bc",FILE_MODE_WRITE,&ham);
  MatView(full_A,ham);
  PetscViewerDestroy(&ham);
#endif

  PetscViewerBinaryOpen(PETSC_COMM_WORLD,"tests/lin_3op_bc",FILE_MODE_READ,&ham);
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

  //Pure VOP tests will be written later;
  //Since L(A)+L(B) != L(A+B), we can't just reuse
  //The lin_op tests, as we did with ham
  //Create some operators
  create_vec(2,&vop2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_lin_mix_basic_real);

  QuaC_clear();
  //Create some operators
  create_vec(2,&vop2);
  create_op(3,&op3);
  create_op(4,&op4);

  RUN_TEST(test_add_lin_mix_basic_complex);


  QuaC_finalize();
  return UNITY_END();
}
