#include "kron_p.h" //Includes petscmat.h and operators_p.h
#include "quac_p.h" 
#include "operators.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* TODO? : 
 * - QuaC_finalize   (free subsystems)
 * - put wrappers into quac.h
 * - variable number of arguments to add_to_ham and add_to_lin
 */


static int              op_initialized = 0;
/* Declare private, library variables. Externed in operators_p.h */
int op_finalized;
Mat full_A;
long total_levels;
int num_subsystems;
double** _hamiltonian;
operator subsystem_list[MAX_SUB];

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

  check_initialized_op();

  /* First make the annihilation operator */
  temp             = malloc(sizeof(struct operator));
  temp->n_before   = total_levels;
  temp->my_levels  = number_of_levels;
  temp->my_op_type = LOWER;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position   = -1;
  *new_op          = temp;

  temp             = malloc(sizeof(struct operator));
  temp->n_before   = total_levels;
  temp->my_levels  = number_of_levels;
  temp->my_op_type = RAISE;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position   = -1;
  (*new_op)->dag   = temp;

  temp             = malloc(sizeof(struct operator));
  temp->n_before   = total_levels;
  temp->my_levels  = number_of_levels;
  temp->my_op_type = NUMBER;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position   = -1;

  (*new_op)->n     = temp;

  /* Increase total_levels */
  total_levels = total_levels * number_of_levels;

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
  check_initialized_op();

  (*new_vec) = malloc(number_of_levels*(sizeof(struct operator*)));
  for (i=0;i<number_of_levels;i++){
    temp             = malloc(sizeof(struct operator));
    temp->n_before   = total_levels;
    temp->my_levels  = number_of_levels;
    temp->my_op_type = VEC;
    /* This is a VEC operator; set its position */
    temp->position   = i;
    (*new_vec)[i]       = temp;
  }

  /* Increase total_levels */
  total_levels = total_levels * number_of_levels;

  subsystem_list[num_subsystems] = (*new_vec);
  num_subsystems++;
  return;

}

/*
 * add_to_ham adds a*op(handle1) to the hamiltonian
 * Inputs:
 *        double a:    scalar to multiply op(handle1)
 *        operator op: operator to add
 * Outputs:
 *        none
 */
void add_to_ham(double a,operator op){
  PetscScalar    mat_scalar;

  
  check_initialized_A();

  /*
   * Construct the dense Hamiltonian only on the master node
   * extra_before and extra_after are 1, letting the code know to
   * do the dense, operator space H.
   */
  if (nid==0) {
    mat_scalar = a;
    _add_to_PETSc_kron(mat_scalar,op->n_before,op->my_levels,op->my_op_type,op->position,1,1);
  }

  /*
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * pass total_levels as extra_before
   * We pass the -a*PETSC_i to get the sign and imaginary part correct.
   */


  mat_scalar = -a*PETSC_i;
  _add_to_PETSc_kron(mat_scalar,op->n_before,op->my_levels,
                     op->my_op_type,op->position,total_levels,1);

  /*
   * Add i * (H cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * pass total_levels as extra_after.
   * We pass a*PETSC_i to get the imaginary part correct.
   */

  mat_scalar = a*PETSC_i;
  _add_to_PETSc_kron(mat_scalar,op->n_before,op->my_levels,
                     op->my_op_type,op->position,1,total_levels);
  return;
}


/*
 * add_to_ham_comb adds a*op(handle1)*op(handle2) to the hamiltonian
 * Inputs:
 *        double a:     scalar to multiply op(handle1)
 *        operator op1: the first operator
 *        operator op2: the second operator
 * Outputs:
 *        none
 */
void add_to_ham_comb(double a,operator op1,operator op2){
  PetscScalar    mat_scalar;

  check_initialized_A();

  /*
   * Construct the dense Hamiltonian only on the master node
   * extra_before and extra_after are 1, letting the code know to
   * do the dense, operator space H.
   */
  if (nid==0) {
    mat_scalar = a;
    _add_to_PETSc_kron_comb(mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                            op2->n_before,op2->my_levels,op2->my_op_type,op2->position,1,1,1);
  }

  /*
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * pass total_levels as extra_before
   * We pass the -a*PETSC_i to get the sign and imaginary part correct.
   */

  mat_scalar = -a*PETSC_i;
  _add_to_PETSc_kron_comb(mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                          op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                          total_levels,1,1);

  /*
   * Add i * (H cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * pass total_levels as extra_after.
   * We pass a*PETSC_i to get the imaginary part correct.
   */

  mat_scalar = a*PETSC_i;
  _add_to_PETSc_kron_comb(mat_scalar,op1->n_before,op1->my_levels,op1->my_op_type,op1->position,
                          op2->n_before,op2->my_levels,op2->my_op_type,op2->position,
                          1,1,total_levels);


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
p *        int handle1: handle of operator to add
 * Outputs:
 *        none
 */

void add_lin(double a,operator op){
  PetscScalar    mat_scalar;

  check_initialized_A();

  /*
   * Add (I cross C^t C) to the superoperator matrix, A
   * Which is (I_total cross I_before cross C^t C cross I_after)
   * Since this is an additional I_total before, we simply
   * set extra_before to total_levels
   */
  mat_scalar = -0.5*a;
  _add_to_PETSc_kron_lin(mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                         op->position,total_levels,1);
  /*
   * Add (C^t C cross I) to the superoperator matrix, A
   * Which is (I_before cross C^t C cross I_after cross I_total)
   * Since this is an additional I_total after, we simply
   * set extra_after to total_levels
   */
  _add_to_PETSc_kron_lin(mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                         op->position,1,total_levels);

  /*
   * Add (C' cross C') to the superoperator matrix, A, where C' is the full space
   * representation of C. Let I_b = I_before and I_a = I_after
   * This simplifies to (I_b cross C cross I_a cross I_b cross C cross I_a)
   * or (I_b cross C cross I_ab cross C cross I_a)
   * This is just like add_to_ham_comb, with n_between = n_after*n_before
   */
  mat_scalar = a;
  _add_to_PETSc_kron_lin_comb(mat_scalar,op->n_before,op->my_levels,op->my_op_type,
                              op->position);

  return;
}

/*
 * check_initialized_op checks if petsc was initialized and sets up variables
 * for op creation. It also errors if there are too many subsystems or
 * if add_to_ham or add_to_lin was called.
 */

void check_initialized_op(){
  /* Check to make sure petsc was initialize */
  if (!petsc_initialized){ 
    if (nid==0){
      printf("ERROR! You need to call QuaC_initialize before creating\n");
      printf("       any operators!");
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
 * check_initialized_A checks to make sure petsc was initialized,
 * some ops were created, and, on first call, sets up the 
 * data structures for the matrices.
 */

void check_initialized_A(){
  int            i,j;
  long           dim;
  PetscErrorCode ierr;

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
      printf("Operators created. Total Hilbert space size: %d\n",total_levels);
      _hamiltonian = malloc(total_levels*sizeof(double*));
      for (i=0;i<total_levels;i++){
        _hamiltonian[i] = malloc(total_levels*sizeof(double));
        /* Initialize to 0 */
        for (j=0;j<total_levels;j++){
          _hamiltonian[i][j] = 0.0;
        }
      }
    }
    dim = total_levels*total_levels;
    /* Setup petsc matrix */
    //FIXME - do something better than 5*total_levels!!
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim,
                        5*total_levels,NULL,5*total_levels,NULL,&full_A);CHKERRQ(ierr);
    ierr = MatSetFromOptions(full_A);CHKERRQ(ierr);
    ierr = MatSetUp(full_A);CHKERRQ(ierr);
  }

  return;

}

/*
 * print_ham prints the dense hamiltonian
 */
void print_ham(){
  int i,j;
  FILE *fp_ham;

  fp_ham = fopen("ham","w");

  if (nid==0){
    for (i=0;i<total_levels;i++){
      for (j=0;j<total_levels;j++){
        fprintf(fp_ham,"%e ",_hamiltonian[i][j]);
      }
      fprintf(fp_ham,"\n");
    }
  }
  fclose(fp_ham);
  return;
}





/*
  NOTE: Don't allow addition of operators? Addition handled only in add_to_ham?
        Simplifies storage of operators - ALL will be defined by a single diagonal
  combine_op calculates a*op(handle1) * b*op(handle2)
  Inputs:
         double a:    scalar to multiply op(handle1)
         int handle1: handle of first operator to combine
         double b:    scalar to multiply op(handle2)
         int handle2: handle of second operator to combine
  Outputs:
         int handle3: handle of combined operator


         NOTE: PROBABLY WON'T USE! add_to_ham_comb instead
 */
/* void combine_op(double a,int handle1,double b,int handle2,int handle3){ */
/*   int i; */
/*   int diag1,diag2,levels1,levels2,new_levels,new_diag; */

/*   diag1 = operator_list[handle1].diagonal_start; */
/*   diag2 = operator_list[handle2].diagonal_start; */
  
/*   operator_list[num_ops].n_before = 1; */
/*   new_levels                       = levels1*levels2; */
/*   operator_list[num_ops].my_levels = new_levels; */
  
/*   /\* */
/*     The new diagonal offset is n_a * k_a + k_b, */
/*     where k_i = diagonal offset of i */
/*   *\/ */
/*   new_diag                              = levels1*diag1+diag2; */
/*   operator_list[num_ops].diagonal_start = new_diag; */

  
/*   if (abs(new_diag)>new_levels&&nid==0) { */
/*     printf("ERROR! new_diag is greater than new_levels!\n"); */
/*     printf("       The new operator will be 0 everywhere!\n"); */
/*     printf("       This should not happen!\n"); */
/*     exit(0); */
/*   } */
  
/*   operator_list[num_ops].matrix_diag    = malloc(new_levels-abs(new_diag)*sizeof(double)); */

/*   /\* Set the values of the new operator - first initialize to 0 *\/ */
/*   for (i=0;i<new_levels-abs(new_diag);i++){ */
/*     operator_list[num_ops].matrix_diag[i] = 0.0; */
/*   } */

  
/*   /\* Increment number of ops, since we've finished adding this *\/ */
/*   num_ops++; */
/*   return; */
/* } */
