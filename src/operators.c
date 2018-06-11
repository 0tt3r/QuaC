#include "kron_p.h" //Includes petscmat.h and operators_p.h
#include "quac_p.h"
#include "operators.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>


/* TODO? :
 * - put wrappers into quac.h
 * - variable number of arguments to add_to_ham and add_lin
 * - add_to_ham_mult4 for coupling between two vec subsystems
 * - add PetscLog for getting setup time
 * - check if input DM is a valid DM (trace, hermitian, etc)
 */

#define MAX_NNZ_PER_ROW 100

int              op_initialized = 0;
/* Declare private, library variables. Externed in operators_p.h */
int op_finalized;
int _stiff_solver;
int _lindblad_terms;
Mat full_A,full_stiff_A;
Mat ham_A,ham_stiff_A;
PetscInt total_levels;
int num_subsystems;
operator subsystem_list[MAX_SUB];
int _print_dense_ham = 0;
int _num_time_dep = 0;
time_dep_struct _time_dep_list[MAX_SUB];
PetscScalar **_hamiltonian;

/*
 * print_dense_ham tells the program to print the dense hamiltonian when it is constructed.
 */
void print_dense_ham(){
  if (op_finalized) {
    if (nid==0){
      printf("ERROR! You need to call print_dense_ham before adding anything to the hamiltonian!");
      exit(0);
    }
  }
  printf("Printing dense Hamiltonian in file 'ham'.\n");
  _print_dense_ham = 1;
}

/*
 * create_op creates a basic set of operators, namely the creation, annihilation, and
 * number operator.
 * Inputs:
 *        int number_of_levels: number of levels for this basic set
 * Outputs:
 *       operator *new_op: lowering op (op), raising op (op->dag), and number op (op->n)
 */

void create_op(int number_of_levels,operator *new_op) {
  operator temp = NULL,temp1 = NULL;

  _check_initialized_op();

  /* First make the annihilation operator */
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = LOWER;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;
  *new_op           = temp;
  temp1             = temp;

  /* Make creation operator */
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = RAISE;
  temp->dag         = temp1; //Point dagger operator to LOWER op
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;
  (*new_op)->dag    = temp;

  /* Make number operator */
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = NUMBER;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;

  (*new_op)->n      = temp;

  /* Make identity operator */
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = IDENTITY;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;

  (*new_op)->eye    = temp;

  /* Make SIGMA_X operator (only valid for qubits, made for every system) */
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = SIGMA_X;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;

  (*new_op)->sig_x      = temp;

  /* Make SIGMA_Z operator (only valid for qubits, made for every system) */
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = SIGMA_Z;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;

  (*new_op)->sig_z      = temp;

  /* Make SIGMA_Y operator (only valid for qubits, made for every system) */
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = SIGMA_Y;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;

  (*new_op)->sig_y      = temp;



  /* Increase total_levels */
  total_levels = total_levels*number_of_levels;

  /* Add to list */
  subsystem_list[num_subsystems] = (*new_op);
  num_subsystems++;
  return;
}

/*
 * create_op creates a basic set of operators, namely the creation, annihilation, and
 * number operator.
 * Inputs:
 *        int number_of_levels: number of levels for this basic set
 * Outputs:
 *       operator *new_op: lowering op (op), raising op (op->dag), and number op (op->n)
 */

void create_vec(int number_of_levels,vec_op *new_vec) {
  operator temp = NULL;
  int i;
  _check_initialized_op();

  (*new_vec) = malloc(number_of_levels*(sizeof(struct operator*)));
  for (i=0;i<number_of_levels;i++){
    temp              = malloc(sizeof(struct operator));
    temp->initial_pop = (double)0.0;
    temp->n_before    = total_levels;
    temp->my_levels   = number_of_levels;
    temp->my_op_type  = VEC;
    /* This is a VEC operator; set its position */
    temp->position    = i;
    (*new_vec)[i]     = temp;
  }

  /*
   * Store the top of the array in vec[0], so we can access it later,
   * through subsystem_list.
   */
  (*new_vec)[0]->vec_op_list = (*new_vec);

  /* Increase total_levels */
  total_levels = total_levels*number_of_levels;
  /*
   * We store just the first VEC in the subsystem list, since it has
   * enough information to define all others
   */
  subsystem_list[num_subsystems] = (*new_vec)[0];
  num_subsystems++;
  return;

}

/*
 * add_to_ham_time_dep adds a(t)*op to the time dependent hamiltonian list
 * Inputs:
 *        double (*time_dep_func)(double): time dependent function to multiply op
 *        int num_ops:  number of operators that will be passed in
 *        operator op1, op2,...,op_{num_ops}: operators to be added to the matrix
 * Outputs:
 *        none
 */
void add_to_ham_time_dep(double (*time_dep_func)(double),int num_ops,...){
  PetscInt    i;
  operator    op;
  va_list     ap;
  _check_initialized_A();

  /*
   * Create the new PETSc matrix.
   * These matrices are incredibly sparse (1 to 2 per row)
   */

  _time_dep_list[_num_time_dep].time_dep_func = time_dep_func;
  _time_dep_list[_num_time_dep].num_ops       = num_ops;
  _time_dep_list[_num_time_dep].ops = malloc(num_ops*sizeof(operator));

  //Add the expanded op to the matrix
  va_start(ap,num_ops);
  for (i=0;i<num_ops;i++){
    op = va_arg(ap,operator);
    _time_dep_list[_num_time_dep].ops[i] = op;
  }
  _num_time_dep = _num_time_dep + 1;
  return;
}


/*
 * add_to_ham_p adds a*op1*op2*...*opn to the hamiltonian
 * Inputs:
 *        PetscScalar a:    scalar to multiply op(s)
 *        PetscIng    num_ops:    number of ops in the list (can be vecs)
 *        operator op1...: operators to multiply together and add
 * Outputs:
 *        none
 */
void add_to_ham_p(PetscScalar a,PetscInt num_ops,...){
  va_list     ap;
  PetscInt i,j,j_ig,j_gi,this_j_ig,this_j_gi,Istart,Iend;
  PetscScalar    val_ig,val_gi,tmp_val;
  PetscScalar add_to_mat;
  operator    this_op1,this_op2;

  PetscLogEventBegin(add_to_ham_event,0,0,0,0);
  _check_initialized_A();

  MatGetOwnershipRange(full_A,&Istart,&Iend);

  if (PetscAbsComplex(a)!=0) { //Don't add zero numbers to the hamiltonian
    for (i=Istart;i<Iend;i++){
      this_j_ig = i;
      this_j_gi = i;
      val_ig = 1.0;
      val_gi = 1.0;
      //Loop through operators
      va_start(ap,num_ops);
      for (j=0;j<num_ops;j++){
        this_op1 = va_arg(ap,operator);

        if(this_op1->my_op_type==VEC){
          /*
           * Since this is a VEC operator, the next operator must also
           * be a VEC operator; it is assumed they always come in pairs.
           */
          this_op2 = va_arg(ap,operator);
          if (this_op2->my_op_type!=VEC){
            if (nid==0){
              printf("ERROR! VEC operators must come in pairs in _add_to_PETSc_kron_parallel\n");
              exit(0);
            }
          }
          //Increment j
          j=j+1;

          //-1 means that it was 0 on a past operator multiplication, so we skip it if it is -1
          if (this_j_ig!=-1){
            //Get i cross G
            _get_val_j_from_global_i_vec_vec(this_j_ig,this_op1,this_op2,&j_ig,&tmp_val,-1);
            this_j_ig = j_ig;
            val_ig = tmp_val * val_ig;
          }

          if (this_j_gi!=-1){
            //Get G* cross I
            _get_val_j_from_global_i_vec_vec(this_j_gi,this_op1,this_op2,&j_gi,&tmp_val,1);
            this_j_gi = j_gi;
            val_gi = tmp_val * val_gi;
          }

        } else {
          //Normal operator
          if (this_j_ig!=-1){
            //Get i cross G
            _get_val_j_from_global_i(this_j_ig,this_op1,&j_ig,&tmp_val,-1);
            this_j_ig = j_ig;
            val_ig = tmp_val * val_ig;
          }

          if (this_j_gi!=-1){
            //Get G* cross I
            _get_val_j_from_global_i(this_j_gi,this_op1,&j_gi,&tmp_val,1);
            this_j_gi = j_gi;
            val_gi = tmp_val * val_gi;
          }
        }
      }
      va_end(ap);
      //Add -i * I cross G_1 G_2 ... G_n

      if (this_j_ig!=-1){
        add_to_mat = -a*PETSC_i*val_ig;
        MatSetValue(full_A,i,this_j_ig,add_to_mat,ADD_VALUES);
      }
      //Add i * G_1*T G_2*T ... G_n*T cross I
      if (this_j_gi!=-1){
        add_to_mat = a*PETSC_i*val_gi;
        MatSetValue(full_A,this_j_gi,i,add_to_mat,ADD_VALUES);
      }
    }
  }
  PetscLogEventEnd(add_to_ham_event,0,0,0,0);
  return;
}


/*
 * add_to_ham adds a*op to the hamiltonian
 * Inputs:
 *        PetscScalar a:    scalar to multiply op
 *        operator op: operator to add
 * Outputs:
 *        none
 */
void add_to_ham(PetscScalar a,operator op){
  PetscScalar    mat_scalar;

  PetscLogEventBegin(add_to_ham_event,0,0,0,0);

  _check_initialized_A();
  if (PetscAbsComplex(a)!=0) { //Don't add zero numbers to the hamiltonian

    /*
     * Construct the dense Hamiltonian only on the master node
     */
    if (nid==0&&_print_dense_ham) {
      mat_scalar = a;
      _add_to_dense_kron(mat_scalar,op->n_before,op->my_levels,op->my_op_type,op->position);
    }

    /*
     * Add to the Hamiltonian matrix, ham_A
     */

    mat_scalar = -a*PETSC_i;
    _add_to_PETSc_kron(ham_A,mat_scalar,op->n_before,op->my_levels,
                       op->my_op_type,op->position,1,1,0);
    /*
     * Add -i * (I cross H) to the superoperator matrix, A
     * Since this is an additional I before, we simply
     * pass total_levels as extra_before
     * We pass the -a*PETSC_i to get the sign and imaginary part correct.
     */

    mat_scalar = -a*PETSC_i;
    _add_to_PETSc_kron(full_A,mat_scalar,op->n_before,op->my_levels,
                       op->my_op_type,op->position,total_levels,1,0);

    /*
     * Add i * (H^T cross I) to the superoperator matrix, A
     * Since this is an additional I after, we simply
     * pass total_levels as extra_after.
     * We pass a*PETSC_i to get the imaginary part correct.
     */

    mat_scalar = a*PETSC_i;
    _add_to_PETSc_kron(full_A,mat_scalar,op->n_before,op->my_levels,
                       op->my_op_type,op->position,1,total_levels,1);
  }
  PetscLogEventEnd(add_to_ham_event,0,0,0,0);
  return;
}


/*
 * add_to_ham_stiff adds a*op to the stiff part of the hamiltonian
 * Inputs:
 *        PetscScalar a:    scalar to multiply op
 *        operator op: operator to add
 * Outputs:
 *        none
 */
void add_to_ham_stiff(PetscScalar a,operator op){
  PetscScalar    mat_scalar;


  _check_initialized_A();
  _stiff_solver = 1;
  /*
   * Construct the dense Hamiltonian only on the master node
   */
  if (nid==0&&_print_dense_ham) {
    mat_scalar = a;
    _add_to_dense_kron(mat_scalar,op->n_before,op->my_levels,op->my_op_type,op->position);
  }

  /*
   * Add to the Hamiltonian matrix, ham_A
   */

  mat_scalar = -a*PETSC_i;
  _add_to_PETSc_kron(ham_stiff_A,mat_scalar,op->n_before,op->my_levels,
                     op->my_op_type,op->position,1,1,0);

  /*
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * pass total_levels as extra_before
   * We pass the -a*PETSC_i to get the sign and imaginary part correct.
   */

  mat_scalar = -a*PETSC_i;
  _add_to_PETSc_kron(full_stiff_A,mat_scalar,op->n_before,op->my_levels,
                     op->my_op_type,op->position,total_levels,1,0);

  /*
   * Add i * (H^T cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * pass total_levels as extra_after.
   * We pass a*PETSC_i to get the imaginary part correct.
   */

  mat_scalar = a*PETSC_i;
  _add_to_PETSc_kron(full_stiff_A,mat_scalar,op->n_before,op->my_levels,
                     op->my_op_type,op->position,1,total_levels,1);
  return;
}



/*
 * add_to_ham_mult2 adds a*op(handle1)*op(handle2) to the hamiltonian
 * Inputs:
 *        PetscScalar a:     scalar to multiply op(handle1)
 *        operator op1: the first operator
 *        operator op2: the second operator
 * Outputs:
 *        none
 */
void add_to_ham_mult2(PetscScalar a,operator op1,operator op2){
  PetscScalar mat_scalar;
  int         multiply_vec,n_after;
  _check_initialized_A();
  multiply_vec = _check_op_type2(op1,op2);


  if (nid==0&&_print_dense_ham){
    /* Add the terms to the dense Hamiltonian */
    if (multiply_vec){
      /*
       * We are multiplying two vec ops. This will only be one value (of 1.0) in the
       * subspace, at location op1->position, op2->position.
       */
      n_after    = total_levels/(op1->my_levels*op1->n_before);
      _add_to_dense_kron_ij(a,op1->position,op2->position,op1->n_before,
                          n_after,op1->my_levels);
    } else {
      /* We are multiplying two normal ops and have to do a little more work. */
        _add_to_dense_kron_comb(a,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                              op2->n_before,op2->my_levels,op2->my_op_type,op2->position);
    }
  }

  /*
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * pass total_levels as extra_before
   * We pass the -a*PETSC_i to get the sign and imaginary part correct.
   */
  mat_scalar = -a*PETSC_i;
  if (multiply_vec){
    /*
     * We are multiplying two vec ops. This will only be one value (of 1.0) in the
     * subspace, at location op1->position, op2->position.
     */
    n_after    = total_levels/(op1->my_levels*op1->n_before);
    /* Add to the Hamiltonian matrix, -i*ham_A */
    _add_to_PETSc_kron_ij(ham_A,mat_scalar,op1->position,op2->position,op1->n_before,
                          n_after,op1->my_levels);
    /* Add to the superoperator matrix, full_A */
    _add_to_PETSc_kron_ij(full_A,mat_scalar,op1->position,op2->position,op1->n_before*total_levels,
                          n_after,op1->my_levels);
  } else {
    /* Add to the Hamiltonian matrix, -i*ham_A */
    _add_to_PETSc_kron_comb(ham_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            1,1,1,0);

    /* We are multiplying two normal ops and have to do a little more work. */
    /* Add to the superoperator matrix, full_A */
    _add_to_PETSc_kron_comb(full_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            total_levels,1,1,0);
  }

  /*
   * Add i * (H^T cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * pass total_levels as extra_after.
   * We pass a*PETSC_i to get the imaginary part correct.
   */
  mat_scalar = a*PETSC_i;
  if (multiply_vec){
    /*
     * We are multiplying two vec ops. This will only be one value (of 1.0) in the
     * subspace, at location op1->position, op2->position.
     */
    n_after    = total_levels/(op1->my_levels*op1->n_before);

    /* Add to the superoperator matrix, full_A */
    /* Flip op1 and op2 positions because we need the transpose */
    _add_to_PETSc_kron_ij(full_A,mat_scalar,op2->position,op1->position,op1->n_before,
                          n_after*total_levels,op1->my_levels);
  } else {
    /* We are multiplying two normal ops and have to do a little more work. */
    /* Add to the superoperator matrix, full_A */
    _add_to_PETSc_kron_comb(full_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            1,1,total_levels,1);
  }

  return;
}

/*
 * add_to_ham_stiff_mult2 adds a*op(handle1)*op(handle2) to the stiff hamiltonian
 * Inputs:
 *        PetscScalar a:     scalar to multiply op(handle1)
 *        operator op1: the first operator
 *        operator op2: the second operator
 * Outputs:
 *        none
 */
void add_to_ham_stiff_mult2(PetscScalar a,operator op1,operator op2){
  PetscScalar mat_scalar;
  int         multiply_vec,n_after;
  _check_initialized_A();

  _stiff_solver = 1;

  multiply_vec = _check_op_type2(op1,op2);


  if (nid==0&&_print_dense_ham){
    /* Add the terms to the dense Hamiltonian */
    if (multiply_vec){
      /*
       * We are multiplying two vec ops. This will only be one value (of 1.0) in the
       * subspace, at location op1->position, op2->position.
       */
      n_after    = total_levels/(op1->my_levels*op1->n_before);
      _add_to_dense_kron_ij(a,op1->position,op2->position,op1->n_before,
                          n_after,op1->my_levels);
    } else {
      /* We are multiplying two normal ops and have to do a little more work. */
      _add_to_dense_kron_comb(a,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                              op2->n_before,op2->my_levels,op2->my_op_type,op2->position);
    }
  }

  /*
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * pass total_levels as extra_before
   * We pass the -a*PETSC_i to get the sign and imaginary part correct.
   */

  mat_scalar = -a*PETSC_i;
  if (multiply_vec){
    /*
     * We are multiplying two vec ops. This will only be one value (of 1.0) in the
     * subspace, at location op1->position, op2->position.
     */
    n_after    = total_levels/(op1->my_levels*op1->n_before);
    /* Add to the Hamiltonian matrix, -i*ham_A */
    _add_to_PETSc_kron_ij(ham_stiff_A,mat_scalar,op1->position,op2->position,op1->n_before,
                          n_after,op1->my_levels);
    /* Add to the superoperator matrix, full_A */
    _add_to_PETSc_kron_ij(full_stiff_A,mat_scalar,op1->position,op2->position,op1->n_before*total_levels,
                          n_after,op1->my_levels);
  } else {
    /* Add to the Hamiltonian matrix, -i*ham_A */
    _add_to_PETSc_kron_comb(ham_stiff_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            1,1,1,0);

    /* We are multiplying two normal ops and have to do a little more work. */
    /* Add to the superoperator matrix, full_A */
    _add_to_PETSc_kron_comb(full_stiff_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            total_levels,1,1,0);
  }

  /*
   * Add i * (H^T cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * pass total_levels as extra_after.
   * We pass a*PETSC_i to get the imaginary part correct.
   */
  mat_scalar = a*PETSC_i;
  if (multiply_vec){
    /*
     * We are multiplying two vec ops. This will only be one value (of 1.0) in the
     * subspace, at location op1->position, op2->position.
     */
    n_after    = total_levels/(op1->my_levels*op1->n_before);

    /* Add to the superoperator matrix, full_A */
    /* We switch op1 and op2 positions to get the transpose */
    _add_to_PETSc_kron_ij(full_stiff_A,mat_scalar,op2->position,op1->position,op1->n_before,
                          n_after*total_levels,op1->my_levels);
  } else {
    /* We are multiplying two normal ops and have to do a little more work. */
    /* Add to the superoperator matrix, full_A */
    _add_to_PETSc_kron_comb(full_stiff_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            1,1,total_levels,1);
  }

  return;
}


/*
 * add_to_ham_mult3 adds a*op1*op2*op3 to the hamiltonian
 * currently assumes either (op1,op2) or (op2,op3) is a pair
 * of vector operators
 * Inputs:
 *        double a:     scalar to multiply op(handle1)
 *        operator op1: the first operator
 *        operator op2: the second operator
 *        operator op3: the second operator
 * Outputs:
 *        none
 */

void add_to_ham_mult3(PetscScalar a,operator op1,operator op2,operator op3){
  PetscScalar mat_scalar;
  int         first_pair;
  _check_initialized_A();
  first_pair = _check_op_type3(op1,op2,op3);


  /* Add to the dense hamiltonian */
  if (nid==0&&_print_dense_ham){
    if (first_pair) {
      /* The first pair is the vec pair and op3 is the normal op*/
      _add_to_dense_kron_comb_vec(a,op3->n_before,op3->my_levels,
                                  op3->my_op_type,op1->n_before,op1->my_levels,
                                  op1->position,op2->position);
    } else {
      /* The last pair is the vec pair and op1 is the normal op*/
      _add_to_dense_kron_comb_vec(a,op1->n_before,op1->my_levels,
                                  op1->my_op_type,op2->n_before,op2->my_levels,
                                  op2->position,op3->position);
    }
  }

  /*
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * pass total_levels as extra_before
   * We pass the -a*PETSC_i to get the sign and imaginary part correct.
   */
  mat_scalar  = -a*PETSC_i;
  if (first_pair) {
    /* The first pair is the vec pair and op3 is the normal op*/
    _add_to_PETSc_kron_comb_vec(full_A,mat_scalar,op3->n_before,op3->my_levels,
                                op3->my_op_type,op1->n_before,op1->my_levels,
                                op1->position,op2->position,total_levels,1,1,0);

  } else {
    /* The last pair is the vec pair and op1 is the normal op*/
    _add_to_PETSc_kron_comb_vec(full_A,mat_scalar,op1->n_before,op1->my_levels,
                                op1->my_op_type,op2->n_before,op2->my_levels,
                                op2->position,op3->position,total_levels,1,1,0);

  }
  /*
   * Add i * (H cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * pass total_levels as extra_after.
   * We pass a*PETSC_i to get the imaginary part correct.
   */
  mat_scalar  = a*PETSC_i;
  if (first_pair) {
    /* The first pair is the vec pair and op3 is the normal op*/
    _add_to_PETSc_kron_comb_vec(full_A,mat_scalar,op3->n_before,op3->my_levels,
                                op3->my_op_type,op1->n_before,op1->my_levels,
                                op1->position,op2->position,1,1,total_levels,1);
  } else {
    /* The last pair is the vec pair and op1 is the normal op*/
    _add_to_PETSc_kron_comb_vec(full_A,mat_scalar,op1->n_before,op1->my_levels,
                                op1->my_op_type,op2->n_before,op2->my_levels,
                                op2->position,op3->position,1,1,total_levels,1);
  }

  return;
}


/*
 * add_lin adds a Lindblad L(C) term to the system of equations, where
 * L(C)p = C p C^t - 1/2 (C^t C p + p C^t C)
 * Or, in superoperator space (t = conjugate transpose, T = transpose, * = conjugate)
 * Lp    = C* cross C - 1/2(C^T C* cross I + I cross C^t C) p
 * And C = op1 op2 ... opn
 * Inputs:
 *        PetscScalar a:    scalar to multiply L term (note: Full term, not sqrt())
 *        PetscInt    num_ops: number of operators to combine
 *        operator op1 ...: ops to make L(C) of
 * Outputs:
 *        none
 */

void add_lin_p(PetscScalar a,PetscInt num_ops,...){
  va_list     ap;
  PetscInt i,j,j_ig,j_gi,j_gg,this_j_ig,this_j_gi,Istart,Iend,this_j_gg;
  PetscScalar    val_ig,val_gi,val_gg,tmp_val;
  PetscScalar add_to_mat;
  operator    this_op1,this_op2;

  PetscLogEventBegin(add_lin_event,0,0,0,0);
  _check_initialized_A();
  _lindblad_terms = 1;
  MatGetOwnershipRange(full_A,&Istart,&Iend);
  if (PetscAbsComplex(a)!=0){
    for (i=Istart;i<Iend;i++){
      this_j_ig = i;
      this_j_gi = i;
      this_j_gg = i;
      val_ig = 1.0;
      val_gi = 1.0;
      val_gg = 1.0;
      //Loop through operators
      va_start(ap,num_ops);
      for (j=0;j<num_ops;j++){
        this_op1 = va_arg(ap,operator);

        if(this_op1->my_op_type==VEC){
          /*
           * Since this is a VEC operator, the next operator must also
           * be a VEC operator; it is assumed they always come in pairs.
           */
          this_op2 = va_arg(ap,operator);
          if (this_op2->my_op_type!=VEC){
            if (nid==0){
              printf("ERROR! VEC operators must come in pairs in _add_to_PETSc_kron_parallel\n");
              exit(0);
            }
          }
          //Increment j
          j=j+1;

          //-1 for this_j* means that it was 0 on a past operator multiplication, so we skip it if it is -1
          if (this_j_ig!=-1){
            //Get I cross G
            _get_val_j_from_global_i_vec_vec(this_j_ig,this_op1,this_op2,&j_ig,&tmp_val,-1);
            this_j_ig = j_ig;
            val_ig = tmp_val * val_ig;
          }

          if (this_j_gi!=-1){
            //Get G* cross I
            _get_val_j_from_global_i_vec_vec(this_j_gi,this_op1,this_op2,&j_gi,&tmp_val,1);
            this_j_gi = j_gi;
            val_gi = tmp_val * val_gi;
          }

          if (this_j_gg!=-1){
            //Get G* cross I
            _get_val_j_from_global_i_vec_vec(this_j_gg,this_op1,this_op2,&j_gg,&tmp_val,0);
            this_j_gg = j_gg;
            val_gg = tmp_val * val_gg;
          }

        } else {
          //Normal operator
          if (this_j_ig!=-1){
            //Get I cross G
            _get_val_j_from_global_i(this_j_ig,this_op1,&j_ig,&tmp_val,-1);
            this_j_ig = j_ig;
            val_ig = tmp_val * val_ig;
          }

          if (this_j_gi!=-1){
            //Get G* cross I
            _get_val_j_from_global_i(this_j_gi,this_op1,&j_gi,&tmp_val,1);
            this_j_gi = j_gi;
            val_gi = tmp_val * val_gi;
          }

          if (this_j_gg!=-1){
            //Get G* cross I
            _get_val_j_from_global_i(this_j_gg,this_op1,&j_gg,&tmp_val,0);
            this_j_gg = j_gg;
            val_gg = tmp_val * val_gg;
          }

        }
      }
      va_end(ap);
      /*
       * From above, we only have I cross G = I cross G1 G2 ... Gn
       * But, we really need is
       * I cross (G1 G2 ... Gn)^t G1 G2 ... Gn
       *
       * First, get I cross G^t G by taking:
       * (G^t G)_{ij} = sum_k G_^t_{ik}G_{kj}
       * but, only one value per row:
       *              = G^t_{ik} G_{kj}
       *              = G_ki* G_kj
       * but, again, only one value per row, so i=j
       *              = G_ki* G_ki
       * Generally, have G_ik; that is fine, we just
       * end up calculating G_kk instead of G_ii - so,
       * maybe we don't own it, but PETSc will figure it out
       */

      /*
       * Add (I cross G^t G)
       */
      if (this_j_ig!=-1){
        add_to_mat = -0.5*a*PetscConjComplex(val_ig)*val_ig;
        MatSetValue(full_A,this_j_ig,this_j_ig,add_to_mat,ADD_VALUES);
      }

      /*
       * Add ((G^t G)* cross I)
       */
      if (this_j_gi!=-1){
        //The second conjugate is redundant here?
        add_to_mat = -0.5*a*PetscConjComplex(val_gi)*val_gi;
        MatSetValue(full_A,this_j_gi,this_j_gi,add_to_mat,ADD_VALUES);
      }
      /*
       * Add (G* cross G) to the superoperator matrix, A
       */
      if (this_j_gg!=-1){
        //The second conjugate is redundant here?
        add_to_mat = a*val_gg;
        MatSetValue(full_A,i,this_j_gg,add_to_mat,ADD_VALUES);
      }
    }
  }
  PetscLogEventEnd(add_lin_event,0,0,0,0);
  return;
}


/*
 * add_lin adds a Lindblad L(C) term to the system of equations, where
 * L(C)p = C p C^t - 1/2 (C^t C p + p C^t C)
 * Or, in superoperator space (t = conjugate transpose, T = transpose, * = conjugate)
 * Lp    = C* cross C - 1/2(C^T C* cross I + I cross C^t C) p
 *
 * Inputs:
 *        PetscScalar a:    scalar to multiply L term (note: Full term, not sqrt())
 *        operator op: op to make L(C) of
 * Outputs:
 *        none
 */

void add_lin(PetscScalar a,operator op){
  PetscScalar    mat_scalar;

  PetscLogEventBegin(add_lin_event,0,0,0,0);
  _check_initialized_A();
  _lindblad_terms = 1;

  if (PetscAbsComplex(a)!=0){

    /*
     * Add (I cross C^t C) to the superoperator matrix, A
     * Which is (I_total cross I_before cross C^t C cross I_after)
     * Since this is an additional I_total before, we simply
     * set extra_before to total_levels
     */
    mat_scalar = -0.5*a;
    _add_to_PETSc_kron_lin(full_A,mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                           op->position,total_levels,1,0);
    /*
     * Add (C^T C* cross I) to the superoperator matrix, A
     * Which is (I_before cross C^T C* cross I_after cross I_total)
     * Since this is an additional I_total after, we simply
     * set extra_after to total_levels
     */
    _add_to_PETSc_kron_lin(full_A,mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                           op->position,1,total_levels,1);

    /*
     * Add (C'* cross C') to the superoperator matrix, A, where C' is the full space
     * representation of C. Let I_b = I_before and I_a = I_after
     * This simplifies to (I_b cross C cross I_a cross I_b cross C cross I_a)
     * or (I_b cross C* cross I_ab cross C cross I_a)
     * This is just like add_to_ham_comb, with n_between = n_after*n_before
     */
    mat_scalar = a;
    _add_to_PETSc_kron_lin_comb(full_A,mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                                op->position);
  }
  PetscLogEventEnd(add_lin_event,0,0,0,0);
  return;
}

/*
 * add_lin_mult2 adds a Lindblad term to the L.
 *
 * For two normal ops, C1 and C2
 * L(C1*C2)p = C1*C2 p (C1*C2)^t - 1/2 ((C1*C2)^t (C1*C2) p + p (C1*C2)^t (C1*C2))
 * Or, in superoperator space, where C = C1*C2
 * Lp    = C cross C - 1/2(C^t C cross I + I cross C^t C) p
 *
 * For two vecs, we add L(C=|op1><op2|) term to the system of equations, where
 * L(|1><2|)p = |1><2| p |2><1| - 1/2 (|2><2| p + p |2><2|)
 * Or, in superoperator space
 * Lp    = |1><2| cross |1><2| - 1/2(|2><2| cross I + I cross |2><2|) p
 *
 * where C is the outer product of two VECs
 * Inputs:
 *        PetscScalar a:     scalar to multiply L term (note: Full term, not sqrt())
 *        operator op1: VEC 1
 *        operator op2: VEC 2
 * Outputs:
 *        none
 */

void add_lin_mult2(PetscScalar a,operator op1,operator op2){
  PetscScalar mat_scalar;
  int         k3,i1,j1,i2,j2,i_comb,j_comb,comb_levels;
  int         multiply_vec,n_after;

  _check_initialized_A();
  _lindblad_terms = 1;
  multiply_vec =  _check_op_type2(op1,op2);

  if (multiply_vec){
    /*
     * Add (I cross C^t C)  = (I cross |2><2| ) to the superoperator matrix, A
     * Which is (I_total cross I_before cross C^t C cross I_after)
     * Since this is an additional I_total before, we simply
     * set extra_before to total_levels
     *
     * Since this is an outer product of VEC ops, its i,j is just op2->position,op2->position,
     * and its val is 1.0
     */
    n_after    = total_levels/(op1->my_levels*op1->n_before);
    mat_scalar = -0.5*a;

    _add_to_PETSc_kron_ij(full_A,mat_scalar,op2->position,op2->position,op2->n_before*total_levels,
                          n_after,op2->my_levels);
    /*
     * Add (C^t C cross I) = (|2><2| cross I) to the superoperator matrix, A
     * Which is (I_before cross C^t C cross I_after cross I_total)
     * Since this is an additional I_total after, we simply
     * set extra_after to total_levels
     */
    _add_to_PETSc_kron_ij(full_A,mat_scalar,op2->position,op2->position,op2->n_before,
                          n_after*total_levels,op2->my_levels);

    /*
     * Add (C' cross C') to the superoperator matrix, A, where C' is the full space
     * representation of C. Let I_b = I_before and I_a = I_after
     * This simplifies to (I_b cross C cross I_a cross I_b cross C cross I_a)
     * or (I_b cross C cross I_ab cross C cross I_a)
     * This is just like add_to_ham_comb, with n_between = n_after*n_before
     */
    comb_levels = op1->my_levels*op1->my_levels*op1->n_before*n_after;
    mat_scalar = a;
    for (k3=0;k3<op2->n_before*n_after;k3++){
      /*
       * Since this is an outer product of VEC ops, there is only
       * one entry in C (and it is 1.0); we need not loop over anything.
       */
      i1   = op1->position;
      j1   = op2->position;
      i2   = i1 + k3*op1->my_levels;
      j2   = j1 + k3*op1->my_levels;

      /*
       * Using the standard Kronecker product formula for
       * A and I cross B, we calculate
       * the i,j pair for handle1 cross I cross handle2.
       * Through we do not use it here, we note that the new
       * matrix is also diagonal.
       * We need my_levels*n_before*n_after because we are taking
       * C cross (Ia cross Ib cross C), so the the size of the second operator
       * is my_levels*n_before*n_after
       */
      i_comb = op1->my_levels*op1->n_before*n_after*i1 + i2;
      j_comb = op1->my_levels*op1->n_before*n_after*j1 + j2;


      _add_to_PETSc_kron_ij(full_A,mat_scalar,i_comb,j_comb,op1->n_before,
                            n_after,comb_levels);

    }
  } else {

    /* Multiply two normal ops */
    _add_to_PETSc_kron_lin2(full_A,a,op1,op2);

  }

  return;
}

/*
 * add_lin_mat adds a Lindblad L(C) term to the system of equations, where
 * L(C)p = C p C^t - 1/2 (C^t C p + p C^t C)
 * Or, in superoperator space (t = conjugate transpose, T = transpose, * = conjugate)
 * Lp    = C* cross C - 1/2(C^T C* cross I + I cross C^t C) p
 * For this routine, C is expressed explicitly as a previously constructed matrix
 * (rather than a compressed operator)
 * Inputs:
 *        PetscScalar a:    scalar to multiply L term (note: Full term, not sqrt())
 *        Mat add_to_lin:   mat to make L(C) of
 * Outputs:
 *        none
 */

void add_lin_mat(PetscScalar a,Mat add_to_lin){
  PetscInt       i,j,Istart,Iend,ncols,i_add,j_add,i2,j2;
  const PetscInt    *cols2;
  const PetscScalar *vals2;
  PetscScalar    vals[total_levels],val_to_add;
  PetscInt       cols[total_levels],ncols2;
  PetscScalar    mat_scalar;
  PetscReal      fill=1.0;
  Mat work_mat1,work_mat2;

  _check_initialized_A();
  _lindblad_terms = 1;

  /* Construct C^t C */
  MatHermitianTranspose(add_to_lin,MAT_INITIAL_MATRIX,&work_mat2);
  MatMatMult(work_mat2,add_to_lin,MAT_INITIAL_MATRIX,fill,&work_mat1);
  MatDestroy(&work_mat2);
  /*
   * Add (I_total cross C^t C) to the superoperator matrix, A
   */
  mat_scalar = -0.5*a;

  _add_to_PETSc_kron_lin_mat(full_A,mat_scalar,work_mat1,1,0);
  /*
   * Add (C^T C* cross I) to the superoperator matrix, A
   */
  _add_to_PETSc_kron_lin_mat(full_A,mat_scalar,work_mat1,0,1);

  /*
   * Add (C* cross C) to the superoperator matrix, A
   * using the standard tensor product between two
   * arbitrary matrices, exploiting no special structure
   * WARNING: This won't work in parallel as written
   */
  MatGetOwnershipRange(add_to_lin,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    /* Get the row */
    MatGetRow(add_to_lin,i,&ncols2,&cols2,&vals2);
    /* Copy info into temporary array */
    ncols = ncols2;
    for (j=0;j<ncols2;j++){
      cols[j] = cols2[j];
      vals[j] = vals2[j];
    }
    MatRestoreRow(add_to_lin,i,&ncols2,&cols2,&vals2);
    for (j=0;j<ncols;j++){
      for (i2=Istart;i2<Iend;i2++){
        /* Get the row */
        MatGetRow(add_to_lin,i2,&ncols2,&cols2,&vals2);
        for (j2=0;j2<ncols2;j2++){
          i_add = total_levels * i + i2;
          j_add = total_levels * cols[j] + cols2[j2];
          val_to_add = a*PetscConjComplex(vals[j])*vals2[j2];
          MatSetValue(full_A,i_add,j_add,val_to_add,ADD_VALUES);
        }
        MatRestoreRow(add_to_lin,i2,&ncols2,&cols2,&vals2);
      }
    }

  }

  MatDestroy(&work_mat1);

  return;
}

/*
 * combine_ops_to_mat takes in a list of operators, multiplies them
 * and stores it in a matrix. This should work with both operators from
 * the same space and operators from different spaces. The matrix returned
 * is of size total_levels by totel_levels; that is, it is in the 'operator'
 * space. If a subspace is missing, it is assumed to be the identity.
 * Because of this, this routine can be used to expand a single operator
 * into its full matrix form. If no operators are passed, this returns the
 * identity.
 *
 * Note that the matrix is allocated here, but must be freed outside
 * of this routine.
 *
 * Inputs:
 *        int number_of_ops: the number of operators to multiply
 *        operator op1, op2, ...: the operators to multiply
 * Outputs:
 *        Mat *matrix_out: The matrix where the result is stored.
 *
 */
void combine_ops_to_mat(Mat *matrix_out,int number_of_ops,...){
  va_list ap;
  operator *op;
  PetscScalar val,op_val;
  PetscInt Istart,Iend;
  PetscInt i,j,this_i,this_j,dim;

  va_start(ap,number_of_ops);
  op = malloc(number_of_ops*sizeof(struct operator));
  /* Loop through passed in ops and store in list */
  for (i=0;i<number_of_ops;i++){
    op[i] = va_arg(ap,operator);
  }
  va_end(ap);

  dim = total_levels;

  // Should this inherit its stucture from full_A?
  MatCreate(PETSC_COMM_WORLD,matrix_out);
  MatSetType(*matrix_out,MATMPIAIJ);
  MatSetSizes(*matrix_out,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
  MatSetFromOptions(*matrix_out);

  MatMPIAIJSetPreallocation(*matrix_out,5,NULL,5,NULL);

  /*
   * Calculate ABC using the following observation:
   *     Each operator (ABCD...) are very sparse - having less than
   *          1 value per row. This allows us to efficiently do the
   *          multiplication of ABCD... by just calculating the value
   *          for one of the indices (i); if there is no matching j,
   *          the value is 0.
   */



  MatGetOwnershipRange(*matrix_out,&Istart,&Iend);

  for (i=Istart;i<Iend;i++){
    this_i = i; // The leading index which we check
    op_val = 1.0;
    for (j=0;j<number_of_ops;j++){
      _get_val_j_from_global_i(this_i,op[j],&this_j,&val,-1); // Get the corresponding j and val
      if (this_j<0) {
        /*
         * Negative j says there is no nonzero value for a given this_i
         * As such, we can immediately break the loop for i
         */
        op_val = 0.0;
        break;
      } else {
        this_i = this_j;
        op_val = op_val*val;
      }
    }
    if (PetscAbsComplex(op_val)!=0) {
      MatSetValue(*matrix_out,i,this_i,op_val,ADD_VALUES);
    }
  }

  MatAssemblyBegin(*matrix_out,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*matrix_out,MAT_FINAL_ASSEMBLY);
  free(op);
  return;
}


/*
 * _check_initialized_op checks if petsc was initialized and sets up variables
 * for op creation. It also errors if there are too many subsystems or
 * if add_to_ham or add_lin was called.
 */

void _check_initialized_op(){
  /* Check to make sure petsc was initialize */
  if (!petsc_initialized){
    if (nid==0){
      printf("ERROR! You need to call QuaC_initialize before creating\n");
      printf("       any operators!\n");
      exit(0);
    }
  }

  /* Set up counters on first call */
  if (!op_initialized){
    op_finalized   = 0;
    _lindblad_terms = 0;
    _stiff_solver   = 0;
    total_levels   = 1;
    op_initialized = 1;
    num_subsystems = 0;
  }

  if (num_subsystems+1>MAX_SUB&&nid==0){
    if (nid==0){
      printf("ERROR! Too many systems for this MAX_SUB\n");
      exit(0);
    }
  }

  if (op_finalized){
    if (nid==0){
      printf("ERROR! You cannot add more operators after\n");
      printf("       calling add_to_ham or add_lin!\n");
      exit(0);
    }
  }

}

/*
 * _check_op_type2 checks to make sure the two ops can be
 * multiplied in a meaningful way. |s> a^\dagger doesn't make sense,
 * for instance
 * Inputs:
 *       operator op1
 *       operator op2
 * Return:
 *       0 if normal op * normal op
 *       1 if vec op * vec op
 */

int _check_op_type2(operator op1,operator op2){
  int return_value;
  /* Check if we are trying to multiply a vec and nonvec operator - should not happen */
  if (op1->my_op_type==VEC&&op2->my_op_type!=VEC){
    if (nid==0){
      printf("ERROR! Multiplying a VEC_OP and a regular OP does not make sense!\n");
      exit(0);
    }
  }
  /* Check if we are trying to multiply a vec and nonvec operator - should not happen */
  if (op2->my_op_type==VEC&&op1->my_op_type!=VEC){
    if (nid==0){
      printf("ERROR! Multiplying a VEC_OP and a regular OP does not make sense!\n");
      exit(0);
    }
  }

  /* Return 1 if we are multiplying two vec ops */
  if (op1->my_op_type==VEC&&op2->my_op_type==VEC){
    /* Check to make sure the two VEC are within the same subspace */
    if (op1->n_before!=op2->n_before) {
      if (nid==0){
        printf("ERROR! Multiplying two VEC_OPs from different subspaces does not make sense!\n");
        exit(0);
      }
    }
    return_value = 1;
  }

  /* Return 0 if we are multiplying two normal ops */
  if (op1->my_op_type!=VEC&&op2->my_op_type!=VEC){
    return_value = 0;
  }

  return return_value;
}


/*
 * _check_op_type3 checks to make sure the three ops can be
 * multiplied in a meaningful way. |s> a^\dagger a doesn't make sense,
 * for instance
 * Inputs:
 *       operator op1
 *       operator op2
 *       operator op3
 * Return:
 *       1 if the first pair is the vec pair
 *       0 if the second pair is the vec pair
 */
int _check_op_type3(operator op1,operator op2,operator op3){
  int return_value;

  /* Check to make sure the VEC op location makes sense */
  if (op1->my_op_type==VEC&&op2->my_op_type==VEC&&op3->my_op_type==VEC){
    if (nid==0){
      printf("ERROR! Multiplying three VEC_OPs does not make sense!\n");
      exit(0);
    }
  }

  if (op1->my_op_type!=VEC&&op2->my_op_type!=VEC&&op3->my_op_type==VEC){
    if (nid==0){
      printf("ERROR! Multiplying one VEC_OP and two normal ops does not make sense!\n");
      exit(0);
    }
  }

  if (op1->my_op_type==VEC&&op2->my_op_type!=VEC&&op3->my_op_type!=VEC){
    if (nid==0){
      printf("ERROR! Multiplying one VEC_OP and two normal ops does not make sense!\n");
      exit(0);
    }
  }

  if (op1->my_op_type!=VEC&&op2->my_op_type==VEC&&op3->my_op_type!=VEC){
    if (nid==0){
      printf("ERROR! Multiplying one VEC_OP and two normal ops does not make sense!\n");
      exit(0);
    }
  }

  if (op1->my_op_type==VEC&&op2->my_op_type!=VEC&&op3->my_op_type==VEC){
    if (nid==0){
      printf("ERROR! Multiplying VEC*OP*VEC does not make sense!\n");
      exit(0);
    }
  }

  if (op1->my_op_type!=VEC&&op2->my_op_type!=VEC&&op3->my_op_type!=VEC){
    if (nid==0){
      printf("ERROR! Multiplying OP*OP*OP currently not supported.\n");
      exit(0);
    }
  }

  /* Check to make sure the two VEC are in the same subsystems */
  if (op1->my_op_type==VEC&&op2->my_op_type==VEC){
    if (op1->n_before!=op2->n_before){
      if (nid==0){
        printf("ERROR! Multiplying two VEC from different subspaces does not make sense.\n");
        exit(0);
      }
    }
    return_value = 1;
  }

  /* Check to make sure the two VEC subsystems are the same */
  if (op2->my_op_type==VEC&&op3->my_op_type==VEC){
    if (op2->n_before!=op3->n_before){
      if (nid==0){
        printf("ERROR! Multiplying two VEC from different subspaces does not make sense.\n");
        exit(0);
      }
    }
    return_value = 0;
  }

  return return_value;
}

/*
 * _check_initialized_A checks to make sure petsc was initialized,
 * some ops were created, and, on first call, sets up the
 * data structures for the matrices.
 */

void _check_initialized_A(){
  int            i;
  long           dim;
  PetscInt       *d_nz,*o_nz,local;

  /* Check to make sure petsc was initialize */
  if (!petsc_initialized){
    if (nid==0){
      printf("ERROR! You need to call QuaC_initialize before creating\n");
      printf("       any operators!");
      exit(0);
    }
  }
  /* Check to make sure some operators were created */
  if (!op_initialized){
    if (nid==0){
      printf("ERROR! You need to create operators before you add anything to\n");
      printf("       the Hamiltonian or Lindblad!\n");
      exit(0);
    }
  }

  if (!op_finalized){
    op_finalized = 1;
    /* Allocate space for (dense) Hamiltonian matrix in operator space
     * (for printing and debugging purposes)
     */
    if (nid==0) {
      printf("Operators created. Total Hilbert space size: %ld\n",total_levels);
      if (_print_dense_ham){
        _hamiltonian = malloc(total_levels*sizeof(PetscScalar*));
        for (i=0;i<total_levels;i++){
          _hamiltonian[i] = malloc(total_levels*sizeof(PetscScalar));
        }
      }
    }

    dim = total_levels*total_levels;
    /* Setup petsc matrix */

    MatCreate(PETSC_COMM_WORLD,&full_A);
    MatSetType(full_A,MATMPIAIJ);
    MatSetSizes(full_A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
    MatSetFromOptions(full_A);


    if (nid==0){
      /*
       * Only the first row has extra nonzeros, from the stabilization.
       * We want to allocate extra memory for that row, but not for any others.
       * Since we put total_levels extra elements (spread evenly), we
       * add a fraction to the diagonal part and the remaining to the
       * off diagonal part. We assume that core 0 owns roughly dim/np
       * rows.
       */

      PetscMalloc1((dim/np)*5,&d_nz); /* malloc array of nnz diagonal elements*/
      PetscMalloc1((dim/np)*5,&o_nz); /* malloc array of nnz off diagonal elements*/

      /*
       * If the system is small enough, we can just allocate a lot of
       * memory for it. Fixes a bug from PETSc when you try to preallocate bigger
       * the row size
       */
      if (total_levels<MAX_NNZ_PER_ROW) {
        d_nz[0] = total_levels*total_levels/np;
        o_nz[0] = total_levels*total_levels/np;
        for (i=1;i<(dim/np)*5;i++){
          d_nz[i] = total_levels*total_levels/np;
          o_nz[i] = total_levels*total_levels/np;
        }
      } else {
        d_nz[0] = MAX_NNZ_PER_ROW + ceil(ceil(dim/np)/total_levels)+5;
        o_nz[0] = MAX_NNZ_PER_ROW + (total_levels - floor(ceil(dim/np)/total_levels))+5;
        for (i=1;i<(dim/np)*5;i++){
          d_nz[i] = MAX_NNZ_PER_ROW;
          o_nz[i] = MAX_NNZ_PER_ROW;
        }

      }

      /* MatMPIAIJSetPreallocation(full_A,0,d_nz,0,o_nz); */
      /* MatSeqAIJSetPreallocation(full_A,0,d_nz); */
      local = 100;//*MAX_NNZ_PER_ROW/np;

      MatMPIAIJSetPreallocation(full_A,local,NULL,(np-1)*MAX_NNZ_PER_ROW/np,NULL);
    } else {
        local = 100;//*MAX_NNZ_PER_ROW/np;

      if (MAX_NNZ_PER_ROW>total_levels*total_levels) {
        MatMPIAIJSetPreallocation(full_A,total_levels,NULL,total_levels,NULL);
      } else {
        //MatMPIAIJSetPreallocation(full_A,MAX_NNZ_PER_ROW,NULL,MAX_NNZ_PER_ROW,NULL);
        MatMPIAIJSetPreallocation(full_A,local,NULL,(np-1)*MAX_NNZ_PER_ROW/np,NULL);
      }
    }

    /* if (nid==0){ */
    /*   ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim, */
    /*                       ,NULL,,NULL,&full_A);CHKERRQ(ierr); */
    /* } else { */
    /*   ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim, */
    /*                       10,NULL,10,NULL,&full_A);CHKERRQ(ierr); */
    /* } */

    MatSetUp(full_A); // This might not be necessary?

    /* MatCreate(PETSC_COMM_WORLD,&full_stiff_A); */
    /* MatSetType(full_stiff_A,MATMPIAIJ); */
    /* MatSetSizes(full_stiff_A,PETSC_DECIDE,PETSC_DECIDE,dim,dim); */
    /* MatSetFromOptions(full_stiff_A); */


    /* if (nid==0){ */
    /*   /\* */
    /*    * Only the first row has extra nonzeros, from the stabilization. */
    /*    * We want to allocate extra memory for that row, but not for any others. */
    /*    * Since we put total_levels extra elements (spread evenly), we */
    /*    * add a fraction to the diagonal part and the remaining to the */
    /*    * off diagonal part. We assume that core 0 owns roughly dim/np */
    /*    * rows. */
    /*    *\/ */
    /*   /\* */
    /*    * If the system is small enough, we can just allocate a lot of */
    /*    * memory for it. Fixes a bug from PETSc when you try to preallocate bigger */
    /*    * the row size */
    /*    *\/ */
    /*   if (total_levels<MAX_NNZ_PER_ROW) { */
    /*     d_nz[0] = total_levels; */
    /*     o_nz[0] = total_levels; */
    /*     for (i=1;i<(dim/np)*5;i++){ */
    /*       d_nz[i] = total_levels; */
    /*       o_nz[i] = total_levels; */
    /*     } */
    /*   } else { */
    /*     d_nz[0] = MAX_NNZ_PER_ROW + ceil(ceil(dim/np)/total_levels)+5; */
    /*     o_nz[0] = MAX_NNZ_PER_ROW + (total_levels - floor(ceil(dim/np)/total_levels))+5; */
    /*     for (i=1;i<(dim/np)*5;i++){ */
    /*       d_nz[i] = MAX_NNZ_PER_ROW; */
    /*       o_nz[i] = MAX_NNZ_PER_ROW; */
    /*     } */
    /*   } */

    /*   /\* MatMPIAIJSetPreallocation(full_stiff_A,0,d_nz,0,o_nz); *\/ */
    /*   /\* MatSeqAIJSetPreallocation(full_stiff_A,0,d_nz); *\/ */
    /*   MatMPIAIJSetPreallocation(full_stiff_A,2*MAX_NNZ_PER_ROW/np,NULL,(np-1)*MAX_NNZ_PER_ROW/np,NULL); */
    /*   PetscFree(d_nz); */
    /*   PetscFree(o_nz); */

    /* } else { */
    /*   if (MAX_NNZ_PER_ROW>total_levels*total_levels) { */
    /*     MatMPIAIJSetPreallocation(full_stiff_A,total_levels,NULL,total_levels,NULL); */
    /*   } else { */
    /*     //MatMPIAIJSetPreallocation(full_stiff_A,MAX_NNZ_PER_ROW,NULL,MAX_NNZ_PER_ROW,NULL); */
    /*     MatMPIAIJSetPreallocation(full_stiff_A,2*MAX_NNZ_PER_ROW/np,NULL,(np-1)*MAX_NNZ_PER_ROW/np,NULL); */
    /*   } */
    /* } */

    /* /\* if (nid==0){ *\/ */
    /* /\*   ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim, *\/ */
    /* /\*                       ,NULL,,NULL,&full_stiff_A);CHKERRQ(ierr); *\/ */
    /* /\* } else { *\/ */
    /* /\*   ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim, *\/ */
    /* /\*                       10,NULL,10,NULL,&full_stiff_A);CHKERRQ(ierr); *\/ */
    /* /\* } *\/ */

    /* MatSetUp(full_stiff_A); // This might not be necessary? */


    /* Setup ham_A matrix */
    MatCreate(PETSC_COMM_WORLD,&ham_A);
    MatSetType(ham_A,MATMPIAIJ);
    MatSetSizes(ham_A,PETSC_DECIDE,PETSC_DECIDE,total_levels,total_levels);
    MatSetFromOptions(ham_A);
    if (MAX_NNZ_PER_ROW>total_levels/2) {
      if (np==1){
        MatMPIAIJSetPreallocation(ham_A,total_levels,NULL,0,NULL);
      } else {
        MatMPIAIJSetPreallocation(ham_A,total_levels/2,NULL,total_levels/2,NULL);
      }
    } else {
      MatMPIAIJSetPreallocation(ham_A,MAX_NNZ_PER_ROW,NULL,MAX_NNZ_PER_ROW,NULL);
    }
    MatSetUp(ham_A); // This might not be necessary?

    /* /\* Setup ham_stiff_A matrix *\/ */
    /* MatCreate(PETSC_COMM_WORLD,&ham_stiff_A); */
    /* MatSetType(ham_stiff_A,MATMPIAIJ); */
    /* MatSetSizes(ham_stiff_A,PETSC_DECIDE,PETSC_DECIDE,total_levels,total_levels); */
    /* MatSetFromOptions(ham_stiff_A); */
    /* if (MAX_NNZ_PER_ROW>total_levels/2) { */
    /*   MatMPIAIJSetPreallocation(ham_stiff_A,total_levels/2,NULL,total_levels/2,NULL); */
    /* } else { */
    /*   MatMPIAIJSetPreallocation(ham_stiff_A,MAX_NNZ_PER_ROW,NULL,MAX_NNZ_PER_ROW,NULL); */
    /* } */
    /* MatSetUp(ham_stiff_A); // This might not be necessary? */

  }

  return;
}

/*
 * set_initial_pop_op sets the initial population for a single operator
 * Inputs:
 *       operator op1
 *       int initial_pop
 * Return:
 *       none
 */
void set_initial_pop(operator op1,double initial_pop){

  if (initial_pop>=op1->my_levels&&op1->my_op_type!=VEC){
    if (nid==0){
      printf("ERROR! The initial population cannot be greater than the number of levels!\n");
      exit(0);
    }
  }

  op1->initial_pop = (double)initial_pop;

  return;
}
