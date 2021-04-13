#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"
#include "t_helpers.h"
#include "neutral_atom.h"


void test_measure_3sys(void){
  PetscInt n_atoms=3,n_levels,i,dmpos;
  qsystem qsys;
  PetscScalar tmp_scalar=1.0,meas_val,meas_val2;
  vec_op *atoms;
  operator op_list[3];
  qvec wf;
  char bitstr[3] = "111";
  enum STATE {zero=0,one,r};

  dmpos = 0;
  //Convert from the bitstr to the dmpos and dmstdpos
  for(i=0;i<n_atoms;i++){
    //We use length-1-i to go through the list in reverse, because we want 00001 to be dmpos=2
    if(bitstr[n_atoms-1-i]=='0'){ //Must use single apostrophe for character equality
      dmpos += 0*pow(3,i);
    } else if(bitstr[n_atoms-1-i]=='1') {
      dmpos += 1*pow(3,i);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: Must be 0 or 1\n");
      exit(0);
    }
  }

  //Initialize the qsystem
  initialize_system(&qsys);

  n_levels = 3;

  atoms = malloc(n_atoms*sizeof(vec_op));

  for(i=0;i<n_atoms;i++){
    //Create an n_level system which is stored in atoms[i]
    create_vec_op_sys(qsys,n_levels,&(atoms[i]));
  }


  //  for(i=0;i<n_atoms;i++){
    //add some dummy terms
    add_ham_term(qsys,tmp_scalar,2,atoms[0][r],atoms[0][one]);
    //  }

  construct_matrix(qsys);

  create_qvec_sys(qsys,&(wf));

  add_to_qvec(wf,1.0,dmpos,dmpos);

  for(i=0;i<n_atoms;i++){
    op_list[i] = atoms[i][zero]->sig_x;
  }

  apply_projective_measurement_tensor_list(wf,&meas_val,1,op_list);
  apply_projective_measurement_tensor_list(wf,&meas_val2,1,op_list);

  //Repeated measurements should give the same result
  TEST_ASSERT_EQUAL_FLOAT(PetscRealPart(meas_val),PetscRealPart(meas_val2));

  for(i=0;i<n_atoms;i++){
    destroy_vec_op_sys(&atoms[i]);
  }

  destroy_qvec(&wf);
  destroy_system(&qsys);


  return;
}

void test_hadamard_2sys(void){
  PetscInt n_atoms=2,n_levels,i,dmpos;
  qsystem qsys;
  PetscScalar tmp_scalar=1.0,val;
  vec_op *atoms;
  qvec wf;
  char bitstr[2] = "00";
  enum STATE {zero=0,one,r};

  dmpos = 0;
  //Convert from the bitstr to the dmpos and dmstdpos
  for(i=0;i<n_atoms;i++){
    //We use length-1-i to go through the list in reverse, because we want 00001 to be dmpos=2
    if(bitstr[n_atoms-1-i]=='0'){ //Must use single apostrophe for character equality
      dmpos += 0*pow(3,i);
    } else if(bitstr[n_atoms-1-i]=='1') {
      dmpos += 1*pow(3,i);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: Must be 0 or 1\n");
      exit(0);
    }
  }

  //Initialize the qsystem
  initialize_system(&qsys);

  n_levels = 3;

  atoms = malloc(n_atoms*sizeof(vec_op));

  for(i=0;i<n_atoms;i++){
    //Create an n_level system which is stored in atoms[i]
    create_vec_op_sys(qsys,n_levels,&(atoms[i]));
  }


  //  for(i=0;i<n_atoms;i++){
    //add some dummy terms
    add_ham_term(qsys,tmp_scalar,2,atoms[0][r],atoms[0][one]);
    //  }

  construct_matrix(qsys);

  create_qvec_sys(qsys,&(wf));

  add_to_qvec(wf,1.0,dmpos,dmpos);

  apply_1q_na_gate_to_qvec(wf,HADAMARD,atoms[1][zero]);
  apply_1q_na_gate_to_qvec(wf,HADAMARD,atoms[0][zero]);

  get_wf_element_qvec(wf,0,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_wf_element_qvec(wf,1,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_wf_element_qvec(wf,3,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  get_wf_element_qvec(wf,4,&val);
  TEST_ASSERT_EQUAL_FLOAT(0.5,PetscRealPart(val));
  TEST_ASSERT_EQUAL_FLOAT(0.0,PetscImaginaryPart(val));

  for(i=0;i<n_atoms;i++){
    destroy_vec_op_sys(&atoms[i]);
  }

  destroy_qvec(&wf);
  destroy_system(&qsys);


  return;
}

void test_imag_ham(void){
  PetscInt n_atoms=1,n_levels,i,dmpos;
  qsystem qsys;
  PetscScalar tmp_scalar=1.0,val;
  vec_op *atoms;
  qvec wf;
  char bitstr[2] = "00";
  enum STATE {zero=0,one,r};

  dmpos = 0;
  //Convert from the bitstr to the dmpos and dmstdpos
  for(i=0;i<n_atoms;i++){
    //We use length-1-i to go through the list in reverse, because we want 00001 to be dmpos=2
    if(bitstr[n_atoms-1-i]=='0'){ //Must use single apostrophe for character equality
      dmpos += 0*pow(3,i);
    } else if(bitstr[n_atoms-1-i]=='1') {
      dmpos += 1*pow(3,i);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: Must be 0 or 1\n");
      exit(0);
    }
  }

  //Initialize the qsystem
  initialize_system(&qsys);

  n_levels = 3;

  atoms = malloc(n_atoms*sizeof(vec_op));

  for(i=0;i<n_atoms;i++){
    //Create an n_level system which is stored in atoms[i]
    create_vec_op_sys(qsys,n_levels,&(atoms[i]));
  }


  for(i=0;i<n_atoms;i++){
    //add some dummy terms
    /* tmp_scalar = 0; */
    /* add_ham_term(qsys,tmp_scalar,2,atoms[i][one],atoms[i][one]); */
    tmp_scalar = -1*PETSC_i;
    add_ham_term(qsys,tmp_scalar,2,atoms[i][r],atoms[i][r]);
  }
  tmp_scalar = 0;
  add_lin_term(qsys,tmp_scalar,2,atoms[0][zero],atoms[0][zero]);
  construct_matrix(qsys);
  create_qvec_sys(qsys,&(wf));
  add_to_qvec(wf,1.0,r,r);
  assemble_qvec(wf);

  time_step_sys(qsys,wf,0.0,10,0.001,1000);
  print_qvec(wf);
  for(i=0;i<n_atoms;i++){
    destroy_vec_op_sys(&atoms[i]);
  }

  //  destroy_qvec(&wf);
  destroy_system(&qsys);


  return;
}

void test_t1_decay(void){
  PetscInt n_atoms=1,n_levels,i,dmpos;
  qsystem qsys;
  PetscScalar tmp_scalar=1.0,val;
  vec_op *atoms;
  qvec wf;
  char bitstr[2] = "1";
  enum STATE {zero=0,one,r};

  dmpos = 0;
  //Convert from the bitstr to the dmpos and dmstdpos
  for(i=0;i<n_atoms;i++){
    //We use length-1-i to go through the list in reverse, because we want 00001 to be dmpos=2
    if(bitstr[n_atoms-1-i]=='0'){ //Must use single apostrophe for character equality
      dmpos += 0*pow(3,i);
    } else if(bitstr[n_atoms-1-i]=='1') {
      dmpos += 1*pow(3,i);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR: Must be 0 or 1\n");
      exit(0);
    }
  }

  //Initialize the qsystem
  initialize_system(&qsys);

  n_levels = 3;

  atoms = malloc(n_atoms*sizeof(vec_op));

  create_vec_op_sys(qsys,n_levels,&(atoms[0]));

  tmp_scalar = 0.1;
  add_lin_term(qsys,tmp_scalar,2,atoms[0][zero],atoms[0][one]);
  construct_matrix(qsys);
  create_qvec_sys(qsys,&(wf));
  add_to_qvec(wf,1.0,one,one);
  assemble_qvec(wf);

  time_step_sys(qsys,wf,0.0,10,0.001,1000);
  print_qvec(wf);
  for(i=0;i<n_atoms;i++){
    destroy_vec_op_sys(&atoms[i]);
  }

  //  destroy_qvec(&wf);
  destroy_system(&qsys);


  return;
}


int main(int argc, char** argv)
{
  UNITY_BEGIN();
  QuaC_initialize(argc,argv);

  /* RUN_TEST(test_measure_3sys); */
  /* RUN_TEST(test_hadamard_2sys); */
  /* RUN_TEST(test_imag_ham); */
  RUN_TEST(test_t1_decay);
  QuaC_finalize();
  return UNITY_END();
}
