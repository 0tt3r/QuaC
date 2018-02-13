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
 * Test sigmaz
 */
void test_sigmaz(void)
{
  circuit  circ;
  Mat      circ_mat;
  operator qubit;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,SIGMAZ,0);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);
  destroy_op(&qubit);
  MatDestroy(&circ_mat);
}

/*
 * Test sigmaz for two qubits, on both qubit 1 and qubit 2
 */
void test_sigmaz2(void)
{
  circuit  circ,circ2,circ3;
  Mat      circ_mat;
  operator qubit,qubit2;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);
  create_op(2,&qubit2);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,SIGMAZ,0);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==2){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==3){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ2,1);
  add_gate_to_circuit(&circ2,1.0,SIGMAZ,1);
  combine_circuit_to_mat(&circ_mat,circ2);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==3){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ3,2);
  add_gate_to_circuit(&circ3,1.0,SIGMAZ,0);
  add_gate_to_circuit(&circ3,2.0,SIGMAZ,1);
  combine_circuit_to_mat(&circ_mat,circ3);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==2){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==3){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  destroy_op(&qubit);
  destroy_op(&qubit2);
}

/*
 * Test CNOT for two qubits
 */
void test_cnot(void)
{
  circuit  circ,circ2;
  Mat      circ_mat;
  operator qubit,qubit2;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);
  create_op(2,&qubit2);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,CNOT,0,1);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==3){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ2,1);
  add_gate_to_circuit(&circ2,1.0,CNOT,1,0);
  combine_circuit_to_mat(&circ_mat,circ2);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==3){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);
  destroy_op(&qubit);
  destroy_op(&qubit2);
}

/*
 * Test CXZ for two qubits
 */
void test_cxz(void)
{
  circuit  circ,circ2;
  Mat      circ_mat;
  operator qubit,qubit2;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);
  create_op(2,&qubit2);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,CXZ,0,1);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==3){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ2,1);
  add_gate_to_circuit(&circ2,1.0,CXZ,1,0);
  combine_circuit_to_mat(&circ_mat,circ2);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==3){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);
  destroy_op(&qubit);
  destroy_op(&qubit2);
}

/*
 * Test CZ for two qubits
 */
void test_cz(void)
{
  circuit  circ,circ2;
  Mat      circ_mat;
  operator qubit,qubit2;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);
  create_op(2,&qubit2);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,CZ,0,1);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==3){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ2,1);
  add_gate_to_circuit(&circ2,1.0,CZ,1,0);
  combine_circuit_to_mat(&circ_mat,circ2);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==3){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);
  destroy_op(&qubit);
  destroy_op(&qubit2);
}


/*
 * Test sigmax for two qubits, on both qubit 1 and qubit 2
 */
void test_sigmax2(void)
{
  circuit  circ,circ2,circ3;
  Mat      circ_mat;
  operator qubit,qubit2;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);
  create_op(2,&qubit2);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,SIGMAX,0);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==3){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ2,1);
  add_gate_to_circuit(&circ2,1.0,SIGMAX,1);
  combine_circuit_to_mat(&circ_mat,circ2);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==3){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ3,2);
  add_gate_to_circuit(&circ3,1.0,SIGMAX,0);
  add_gate_to_circuit(&circ3,2.0,SIGMAX,1);
  combine_circuit_to_mat(&circ_mat,circ3);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==3){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  destroy_op(&qubit);
  destroy_op(&qubit2);
}

/*
 * Test sigmay for two qubits, on both qubit 1 and qubit 2
 */
void test_sigmay2(void)
{
  circuit  circ,circ2,circ3;
  Mat      circ_mat;
  operator qubit,qubit2;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);
  create_op(2,&qubit2);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,SIGMAY,0);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==2){
      val = -PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==3){
      val = -PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==0){
      val = PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==1){
      val = PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ2,1);
  add_gate_to_circuit(&circ2,1.0,SIGMAY,1);
  combine_circuit_to_mat(&circ_mat,circ2);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==1){
      val = -PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==0){
      val = PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==3){
      val = -PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==2){
      val = PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ3,2);
  add_gate_to_circuit(&circ3,1.0,SIGMAY,0);
  add_gate_to_circuit(&circ3,2.0,SIGMAY,1);
  combine_circuit_to_mat(&circ_mat,circ3);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==3){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==2){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==2&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==3&&cols[0]==0){
      val = -1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else {
      equal_int = 0;
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  destroy_op(&qubit);
  destroy_op(&qubit2);
}

/*
 * Test hadamard for two qubits, on both qubit 1 and qubit 2
 */
void test_hadamard2(void)
{
  circuit  circ,circ2,circ3;
  Mat      circ_mat;
  operator qubit,qubit2;
  PetscInt Istart,Iend,i,j,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);
  create_op(2,&qubit2);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,HADAMARD,0);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols!=2) {
      equal_int = 0;
      break;
    }
    if (i==0){
      if (cols[0]==0) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==2) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    } else if(i==1){
      if (cols[0]==1) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==3) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    } else if(i==2){
      if (cols[0]==0) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==2) {
        val = -1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    } else if(i==3){
      if (cols[0]==1) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==3) {
        val = -1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ2,1);
  add_gate_to_circuit(&circ2,1.0,HADAMARD,1);
  combine_circuit_to_mat(&circ_mat,circ2);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols!=2) {
      equal_int = 0;
      break;
    }
    if (i==0){
      if (cols[0]==0) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==1) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    } else if(i==1){
      if (cols[0]==0) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==1) {
        val = -1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    } else if(i==2){
      if (cols[0]==2) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==3) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    } else if(i==3){
      if (cols[0]==2) {
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else if (cols[1]==3) {
        val = -1/sqrt(2);
        if(PetscAbsComplex(vals[0]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      } else {
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  create_circuit(&circ3,2);
  add_gate_to_circuit(&circ3,1.0,HADAMARD,0);
  add_gate_to_circuit(&circ3,2.0,HADAMARD,1);
  combine_circuit_to_mat(&circ_mat,circ3);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
    for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols!=4) {
      equal_int = 0;
      break;
    }
    if (i==0){
      for (j=0;j<ncols;j++){
        val = 0.5;
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }
    } else if(i==1){
      for (j=0;j<ncols;j++){
        if (cols[j]==1||cols[j]==3){
          val = -0.5;
        } else {
          val = 0.5;
        }
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }
    } else if(i==2){
      for (j=0;j<ncols;j++){
        if (cols[j]==2||cols[j]==3){
          val = -0.5;
        } else {
          val = 0.5;
        }
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }
    } else if(i==3){
      for (j=0;j<ncols;j++){
        if (cols[j]==1||cols[j]==2){
          val = -0.5;
        } else {
          val = 0.5;
        }
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);

  MatDestroy(&circ_mat);

  destroy_op(&qubit);
  destroy_op(&qubit2);
}




/*
 * Test sigmay
 */
void test_sigmay(void)
{
  circuit  circ;
  Mat      circ_mat;
  operator qubit;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,SIGMAZ,0);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==1){
      val = -PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==0){
      val = PETSC_i;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);
  destroy_op(&qubit);
  MatDestroy(&circ_mat);
}


/*
 * Test sigmax
 */
void test_sigmax(void)
{
  circuit  circ;
  Mat      circ_mat;
  operator qubit;
  PetscInt Istart,Iend,i,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,SIGMAX,0);
  combine_circuit_to_mat(&circ_mat,circ);

  equal_int = 1;

  //Compare to known result
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    if(ncols>1) equal_int = 0;
    if(i==0&&cols[0]==1){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    } else if(i==1&&cols[0]==0){
      val = 1.0;
      if(PetscAbsComplex(vals[0]-val)>1e-10){
        //They are different!
        equal_int = 0;
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }

  TEST_ASSERT_EQUAL_INT(1,equal_int);
  destroy_op(&qubit);
  MatDestroy(&circ_mat);
}



/*
 * Test hadamard
 */
void test_hadamard(void)
{
  circuit  circ;
  Mat      circ_mat;
  operator qubit;
  PetscInt Istart,Iend,i,j,equal_int;
  PetscScalar val;

  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  //Create 1 two level system
  create_op(2,&qubit);

  //Build single SIGMAX matrix
  create_circuit(&circ,1);
  add_gate_to_circuit(&circ,1.0,HADAMARD,0);
  combine_circuit_to_mat(&circ_mat,circ);

  //Explicitly create the matrix
  MatGetOwnershipRange(circ_mat,&Istart,&Iend);
  equal_int = 1;
  //Compare to known result
  for (i=Istart;i<Iend;i++){
    MatGetRow(circ_mat,i,&ncols,&cols,&vals);
    for (j=0;j<ncols;j++){
      if (i==0&&cols[j]==0){
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }
      if (i==0&&cols[j]==1){
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }

      if (i==1&&cols[j]==1){
        val = -1/sqrt(2);
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }

      if (i==1&&cols[j]==0){
        val = 1/sqrt(2);
        if(PetscAbsComplex(vals[j]-val)>1e-10){
          //They are different!
          equal_int = 0;
        }
      }
    }
    MatRestoreRow(circ_mat,i,&ncols,&cols,&vals);
  }
  TEST_ASSERT_EQUAL_INT(1,equal_int);
  destroy_op(&qubit);
  MatDestroy(&circ_mat);
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);
  RUN_TEST(test_sigmax);
  QuaC_clear();
  RUN_TEST(test_sigmax2);
  QuaC_clear();
  RUN_TEST(test_sigmay);
  QuaC_clear();
  RUN_TEST(test_sigmay2);
  QuaC_clear();
  RUN_TEST(test_sigmaz);
  QuaC_clear();
  RUN_TEST(test_sigmaz2);
  QuaC_clear();
  RUN_TEST(test_hadamard);
  QuaC_clear();
  RUN_TEST(test_hadamard2);
  QuaC_clear();
  RUN_TEST(test_cnot);
  QuaC_clear();
  RUN_TEST(test_cxz);
  QuaC_finalize();
  return UNITY_END();
}

