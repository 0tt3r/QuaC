#include "kron_p.h" //Includes petscmat.h and operators_p.h
#include "quac_p.h" 
#include "operators.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* TODO? : 
 * - put wrappers into quac.h
 * - variable number of arguments to add_to_ham and add_to_lin
 * - add_to_ham_mult4 for coupling between two vec subsystems
 * - add PetscLog for getting setup time
 * - check if input DM is a valid DM (trace, hermitian, etc)
 */

#define MAX_NNZ_PER_ROW 50

static int              op_initialized = 0;
/* Declare private, library variables. Externed in operators_p.h */
int op_finalized;
Mat full_A;
PetscInt total_levels;
int num_subsystems;
operator subsystem_list[MAX_SUB];
int _print_dense_ham = 0;
int _num_time_dep = 0;
time_dep_struct _time_dep_list[MAX_SUB];
double **_hamiltonian;

/*
 * print_dense_ham tells the program to print the dense hamiltonian when it is constructed.
 */
void print_dense_ham(){
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
  operator temp = NULL;

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

  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = RAISE;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;
  (*new_op)->dag    = temp;

  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = NUMBER;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;

  (*new_op)->n      = temp;

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
 *        double (*time_dep_func)(doubl): time dependent function to multiply op
 *        operator op: operator to add
 * Outputs:
 *        none
 */
void add_to_ham_time_dep(double (*time_dep_func)(double),operator op){
  
  _check_initialized_A();
  
  return;
}

/*
 * add_to_ham adds a*op to the hamiltonian
 * Inputs:
 *        double a:    scalar to multiply op
 *        operator op: operator to add
 * Outputs:
 *        none
 */
void add_to_ham(double a,operator op){
  PetscScalar    mat_scalar;

  
  _check_initialized_A();

  /*
   * Construct the dense Hamiltonian only on the master node
   */
  if (nid==0&&_print_dense_ham) {
    mat_scalar = a;
    _add_to_dense_kron(mat_scalar,op->n_before,op->my_levels,op->my_op_type,op->position);
  }

  /*
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * pass total_levels as extra_before
   * We pass the -a*PETSC_i to get the sign and imaginary part correct.
   */

  mat_scalar = -a*PETSC_i;
  _add_to_PETSc_kron(full_A,mat_scalar,op->n_before,op->my_levels,
                     op->my_op_type,op->position,total_levels,1);

  /*
   * Add i * (H cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * pass total_levels as extra_after.
   * We pass a*PETSC_i to get the imaginary part correct.
   */

  mat_scalar = a*PETSC_i;
  _add_to_PETSc_kron(full_A,mat_scalar,op->n_before,op->my_levels,
                     op->my_op_type,op->position,1,total_levels);
  return;
}



/*
 * add_to_ham_mult2 adds a*op(handle1)*op(handle2) to the hamiltonian
 * Inputs:
 *        double a:     scalar to multiply op(handle1)
 *        operator op1: the first operator
 *        operator op2: the second operator
 * Outputs:
 *        none
 */
void add_to_ham_mult2(double a,operator op1,operator op2){
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
    _add_to_PETSc_kron_ij(full_A,mat_scalar,op1->position,op2->position,op1->n_before*total_levels,
                          n_after,op1->my_levels);
  } else {
    /* We are multiplying two normal ops and have to do a little more work. */
    _add_to_PETSc_kron_comb(full_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            total_levels,1,1);
  }

  /*
   * Add i * (H cross I) to the superoperator matrix, A
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
    _add_to_PETSc_kron_ij(full_A,mat_scalar,op1->position,op2->position,op1->n_before,
                          n_after*total_levels,op1->my_levels);
  } else {
    /* We are multiplying two normal ops and have to do a little more work. */
    _add_to_PETSc_kron_comb(full_A,mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                            1,1,total_levels);
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

void add_to_ham_mult3(double a,operator op1,operator op2,operator op3){
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
                                op1->position,op2->position,total_levels,1,1);

  } else {
    /* The last pair is the vec pair and op1 is the normal op*/  
    _add_to_PETSc_kron_comb_vec(full_A,mat_scalar,op1->n_before,op1->my_levels,
                                op1->my_op_type,op2->n_before,op2->my_levels,
                                op2->position,op3->position,total_levels,1,1);

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
                                op1->position,op2->position,1,1,total_levels);

  } else {
    /* The last pair is the vec pair and op1 is the normal op*/  
    _add_to_PETSc_kron_comb_vec(full_A,mat_scalar,op1->n_before,op1->my_levels,
                                op1->my_op_type,op2->n_before,op2->my_levels,
                                op2->position,op3->position,1,1,total_levels);

  }

  return;
}

/*
 * add_lin adds a Lindblad L(C) term to the system of equations, where
 * L(C)p = C p C^t - 1/2 (C^t C p + p C^t C)
 * Or, in superoperator space
 * Lp    = C cross C - 1/2(C^t C cross I + I cross C^t C) p
 *
 * Inputs:
 *        double a:    scalar to multiply L term (note: Full term, not sqrt())
 *        operator op: op to make L(C) of
 * Outputs:
 *        none
 */

void add_lin(double a,operator op){
  PetscScalar    mat_scalar;

  _check_initialized_A();

  /*
   * Add (I cross C^t C) to the superoperator matrix, A
   * Which is (I_total cross I_before cross C^t C cross I_after)
   * Since this is an additional I_total before, we simply
   * set extra_before to total_levels
   */
  mat_scalar = -0.5*a;
  _add_to_PETSc_kron_lin(full_A,mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                         op->position,total_levels,1);
  /*
   * Add (C^t C cross I) to the superoperator matrix, A
   * Which is (I_before cross C^t C cross I_after cross I_total)
   * Since this is an additional I_total after, we simply
   * set extra_after to total_levels
   */
  _add_to_PETSc_kron_lin(full_A,mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                         op->position,1,total_levels);

  /*
   * Add (C' cross C') to the superoperator matrix, A, where C' is the full space
   * representation of C. Let I_b = I_before and I_a = I_after
   * This simplifies to (I_b cross C cross I_a cross I_b cross C cross I_a)
   * or (I_b cross C cross I_ab cross C cross I_a)
   * This is just like add_to_ham_comb, with n_between = n_after*n_before
   */
  mat_scalar = a;
  _add_to_PETSc_kron_lin_comb(full_A,mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                              op->position);

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
 *        double a:     scalar to multiply L term (note: Full term, not sqrt())
 *        operator op1: VEC 1
 *        operator op2: VEC 2
 * Outputs:
 *        none
 */

void add_lin_mult2(double a,operator op1,operator op2){
  PetscScalar mat_scalar;
  int         k3,i1,j1,i2,j2,i_comb,j_comb,comb_levels;
  int         multiply_vec,n_after;   
  
  _check_initialized_A();
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
    /* Only allow a a^\dagger terms from the same subspace - check that this is true*/
    if (op1->n_before!=op2->n_before){
      if (nid==0){
        printf("ERROR! Operators must be from the same subspace to be multiplied\n");
        printf("       in a Lindblad term!\n");
        exit(0);
      }
    }
    if (op1->my_op_type==RAISE&&op2->my_op_type==LOWER){
      if (nid==0){
        printf("ERROR! Please use provided a->n type terms to add the \n");
        printf("       number operator to a Lindblad term!\n");
        exit(0);
      }
    }
    if (op1->my_op_type!=LOWER&&op2->my_op_type!=RAISE){
      if (nid==0){
        printf("ERROR! Only terms of the type a a^\\dagger are currently\n");
        printf("       supported for multiplied Lindblad terms.\n");
        exit(0);
      }
    }
    /*
     * Add (I cross C^t C) to the superoperator matrix, A
     */
    mat_scalar = -0.5*a;
    _add_to_PETSc_kron_lin2(full_A,mat_scalar,op1->n_before,op1->my_levels,total_levels,1);
    
    /*
     * Add (C^t C cross I) to the superoperator matrix, A
     */
    mat_scalar = -0.5*a;
    _add_to_PETSc_kron_lin2(full_A,mat_scalar,op1->n_before,op1->my_levels,1,total_levels);
    /*
     * Add (C' cross C') to the superoperator matrix, A, where C' is the full space
     */
    mat_scalar = a;
    _add_to_PETSc_kron_lin2_comb(full_A,mat_scalar,op1->n_before,op1->my_levels);

  }

  return;
}

/*
 * _check_initialized_op checks if petsc was initialized and sets up variables
 * for op creation. It also errors if there are too many subsystems or
 * if add_to_ham or add_to_lin was called.
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
    total_levels   = 1;
    op_initialized = 1;
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
      printf("       calling add_to_ham or add_to_lin!\n");
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
  PetscInt       *d_nz,*o_nz;

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
        _hamiltonian = malloc(total_levels*sizeof(double*));
        for (i=0;i<total_levels;i++){
          _hamiltonian[i] = malloc(total_levels*sizeof(double));
        }
      }
    }

    dim = total_levels*total_levels;
    /* Setup petsc matrix */
    MatCreate(PETSC_COMM_WORLD,&full_A);
    //    MatSetType(full_A,MATMPIAIJ);
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
        d_nz[0] = total_levels*total_levels;
        o_nz[0] = total_levels*total_levels;
        for (i=1;i<(dim/np)*5;i++){
          d_nz[i] = total_levels*total_levels;
          o_nz[i] = total_levels*total_levels;
        }
      } else {
        d_nz[0] = MAX_NNZ_PER_ROW + ceil(ceil(dim/np)/total_levels)+5;
        o_nz[0] = MAX_NNZ_PER_ROW + (total_levels - floor(ceil(dim/np)/total_levels))+5;
        for (i=1;i<(dim/np)*5;i++){
          d_nz[i] = MAX_NNZ_PER_ROW;
          o_nz[i] = MAX_NNZ_PER_ROW;
          
        }
      }
      MatMPIAIJSetPreallocation(full_A,0,d_nz,0,o_nz);
      MatSeqAIJSetPreallocation(full_A,0,d_nz);
      PetscFree(d_nz);
      PetscFree(o_nz);

    } else {
      MatMPIAIJSetPreallocation(full_A,MAX_NNZ_PER_ROW,NULL,MAX_NNZ_PER_ROW,NULL);
    }

    /* if (nid==0){ */
    /*   ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim, */
    /*                       ,NULL,,NULL,&full_A);CHKERRQ(ierr); */
    /* } else { */
    /*   ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim, */
    /*                       10,NULL,10,NULL,&full_A);CHKERRQ(ierr); */
    /* } */

    MatSetUp(full_A); // This might not be necessary?
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

