#include "t_helpers.h"

void _get_mat_and_diff_norm(char* mat_string,Mat mat_A,PetscReal *norm){
  Mat mat_pristine2;
  PetscViewer ham;

#ifdef SAVE_MATS
  //Save the matrix
  printf("Saving matrix...\n");
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat_string,FILE_MODE_WRITE,&ham);
  MatView(mat_A,ham);
  PetscViewerDestroy(&ham);
#endif
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat_string,FILE_MODE_READ,&ham);
  MatDuplicate(mat_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine2);
  MatLoad(mat_pristine2,ham);
  PetscViewerDestroy(&ham);

  MatAXPY(mat_pristine2,-1,mat_A,DIFFERENT_NONZERO_PATTERN);
  MatNorm(mat_pristine2,NORM_FROBENIUS,norm);

  MatDestroy(&mat_pristine2);
  return;
}

void _get_mat_combine_and_diff_norm(char* mat_string,Mat mat_A,PetscReal *norm){
  Mat mat_pristine_lin,mat_pristine_ham;
  PetscViewer ham;
  char fname[255];

  // Load Lindblad matrix
  strcpy(fname,"tests/pristine_matrices/lin_");
  strcat(fname,mat_string);

  PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&ham);
  MatDuplicate(mat_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine_lin);
  MatLoad(mat_pristine_lin,ham);
  PetscViewerDestroy(&ham);

  // Load Lindblad matrix
  strcpy(fname,"tests/pristine_matrices/ham_");
  strcat(fname,mat_string);

  PetscViewerBinaryOpen(PETSC_COMM_WORLD,fname,FILE_MODE_READ,&ham);
  MatDuplicate(mat_A,MAT_DO_NOT_COPY_VALUES,&mat_pristine_ham);
  MatLoad(mat_pristine_ham,ham);
  PetscViewerDestroy(&ham);

  //Get Ham + Lin -- already included -i part for Ham
  MatAXPY(mat_pristine_ham,1,mat_pristine_lin,DIFFERENT_NONZERO_PATTERN);

  //Compare to mat_A
  MatAXPY(mat_pristine_ham,-1,mat_A,DIFFERENT_NONZERO_PATTERN);
  MatNorm(mat_pristine_ham,NORM_FROBENIUS,norm);

  MatDestroy(&mat_pristine_ham);
  MatDestroy(&mat_pristine_lin);
  return;
}

