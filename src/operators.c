
#include <petscmat.h>
#include <math.h>
#include <stdlib.h>
#define MAX_OP 100  //Consider making this not a define


struct operator {
  int n_before;
  int my_levels;
  /* 

     All of our operators are guaranteed to be a diagonal of some type
     (likely a super diagonal or sub diagonal, but a diagonal nonetheless)
     As such, we only store said diagonal, as well as the starting
     row / column.
     diagonal_start is the offset from the true diagonal
     positive means super diagonal, negative means sub diagnal

  */
  double *matrix_diag;
  int diagonal_start;
};


static struct operator operator_list[MAX_OP]; //static should keep it in this file
static int num_ops = 0;
static int total_levels = 0;
static double** hamiltonian;
static int op_finalized = 0;

/*
  create_op creates a basic set of operators, namely the creation, annihilation, and
  number operator. 
  Inputs:
         int number_of_levels: number of levels for this basic set
  Outputs:
         int op_create:  handle of creation operator
         int op_destroy: handle of annilitation operator
         int op_number:  handle of number operator

 */

void create_op(int number_of_levels,int op_create,int op_destroy,int op_number) {
  int i;


  if (num_ops+3>MAX_OP&&nid==0) {
    printf("ERROR! Too many ops for this MAX_OP\n");
    exit(0);
  }

  if (op_finalized) {
    printf("ERROR! You cannot add more operators after\n");
    printf("       calling add_to_ham!\n");
    exit(0);
  }


  /* First, make the creation operator */
  operator_list[num_ops].n_before       = total_levels;
  operator_list[num_ops].my_levels      = number_of_levels;
  operator_list[num_ops].diagonal_start = -1;
  /* Allocate the diagonal */
  /* number_of_levels-1 because subdiagonal by 1 */
  operator_list[num_ops].matrix_diag    = malloc((number_of_levels-1)*sizeof(double)); 
  /* Set the values */
  for(i=0;i<number_of_levels-1;i++){
    operator_list[num_ops].matrix[i] = sqrt(i+1);
  }

  op_create = num_ops;
  /* Increment number of ops, since we've finished adding this */
  num_ops++;

  /* Next, make the annihilation operator */
  operator_list[num_ops].n_before       = total_levels;
  operator_list[num_ops].my_levels      = number_of_levels;
  operator_list[num_ops].diagonal_start = 1;
  /* Allocate the diagonal */
  /* number_of_levels-1 because superdiagonal by 1 */
  operator_list[num_ops].matrix_diag    = malloc((number_of_levels-1)*sizeof(double)); 
  /* Set the values */
  for (i=0;i<number_of_levels-1;i++){
    operator_list[num_ops].matrix[i] = sqrt(i+1);
  }

  op_destroy = num_ops;
  /* Increment number of ops, since we've finished adding this */
  num_ops++;
 
  /* Finally, make the number operator */
  operator_list[num_ops].n_before       = total_levels;
  operator_list[num_ops].my_levels      = number_of_levels;
  operator_list[num_ops].diagonal_start = 0;
  /* Allocate the diagonal */
  operator_list[num_ops].matrix_diag    = malloc(number_of_levels*sizeof(double));
  /* Set the values */
  for (i=0;i<number_of_levels;i++){
    operator_list[num_ops].matrix[i] = i;
  }
 
  op_number = num_ops;
  /* Increment number of ops, since we've finished adding this */
  num_ops++;



  /* Increase total_levels */
  if (total_levels==0){
    total_levels = number_of_levels;
  } else {
    total_levels = total_levels * number_of_levels;
  }

  return;
}

/*
 * add_to_ham adds a*op(handle1) to the hamiltonian
 * Inputs:
 *        double a:    scalar to multiply op(handle1)
 *        int handle1: handle of first operator to combine
 *        int handle2: handle of second operator to combine
 * Outputs:
 *        none
 */
void add_to_ham(double a,int handle1){
  int            k1,k2,i,j,n_before,n_after,my_levels,diag_start;
  int            i_ham,j_ham,i_op,j_op;
  PetscScalar    add_to_mat;
  PetscErrorCode ierr;

  if (!op_finalized){
    op_finalized = 1;
    if (nid==0) {
      printf("Operators created. Total Hilbert space size: %d\n",total_levels);
      /* Allocate space for (dense) Hamiltonian matrix in operator space
       * (for printing and debugging purposes)
       */
      hamiltonian = malloc(total_levels*sizeof(double*));
      for (i=0;i<total_levels;i++){
        hamiltonian[i] = malloc(total_levels*sizeof(double));
        /* Initialize to 0 */
        for (j=0;j<total_levels;j++){
          hamiltonian[i][j] = 0.0;
        }
      }
    }
  }


  my_levels  = operator_list[handle1].my_levels;
  diag_start = operator_list[handle1].diagonal_start;
  n_before   = operator_list[handle1].n_before;
  n_after    = total_levels/(n_before*my_levels);

  /* Construct the dense Hamiltonian only on the master node */
  if(nid==0) {
    for (k1=0;k1<n_before;k1++){
      for (k2=0;k2<n_after;k2++){
        for (i=0;i<my_levels-abs(diag_start);i++){
          /* 
           * Since we store our matrix as a diagonal, we need to
           * calculate the actual i,j location for our operator.
           *
           * If it is a superdiagonal (diag_start > 0),
           * our data index (i) describes the i_op value
           * 
           * If is is a subdiagonal (diag_start < 0), 
           * it is reversed.
           */

          if (diag_start>=0) {
            i_op  = i;
            j_op  = i+diag_start;
          } else {
            i_op = i-diag_start;
            j_op = i;
          }
           
          /*
           * Now we need to calculate the apropriate location of this
           * within the Hamiltonian matrix. We need to expand the operator
           * from its small Hilbert space to the total Hilbert space.
           * This expansion depends on the order in which the operators
           * were added. For example, if we added 3 operators:
           * A, B, and C (with sizes n_a, n_b, n_c, respectively), we would
           * have (where the ' denotes in the full space and I_(n) means
           * the identity matrix of size n):
           *
           * A' = A cross I_(n_b) cross I_(n_c)
           * B' = I_(n_a) cross B cross I_(n_c)
           * C' = I_(n_a) cross I_(n_b) cross C
           * 
           * For an arbitrary number of operators, we only care about
           * the Hilbert space size before and the Hilbert space size
           * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b)
           *
           * The calculation of i_ham and j_ham exploit the structure of 
           * the tensor products - they are general for kronecker products
           * of identity matrices with some matrix A
           */
          
          i_ham = i_op*n_after+k2+k1*my_levels*n_after;
          j_ham = j_op*n_after+k2+k1*my_levels*n_after;
          hamiltonian[i_ham][j_ham] = hamiltonian[i_ham][j_ham]
            +a*operator_list[handle1].matrix_diag[i];
        }
      }
    }
  }


  /* 
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * increase n_before by total_levels
   *
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */
  for (k1=0;k1<n_before*total_levels;k1++){
    for (k2=0;k2<n_after;k2++){
      for (i=0;i<my_levels-abs(diag_start);i++){
        /* 
         * Since we store our matrix as a diagonal, we need to
         * calculate the actual i,j location for our operator.
         */

        if (diag_start>=0) {
          i_op  = i;
          j_op  = i+diag_start;
        } else {
          i_op = i-diag_start;
          j_op = i;
        }
           
        /*
         * Now we need to calculate the apropriate location of this
         * within the superoperator Hamiltonian matrix. 
         * See more detailed notes above
         */
        i_ham = i_op*n_after+k2+k1*my_levels*n_after;
        j_ham = j_op*n_after+k2+k1*my_levels*n_after;
        add_to_mat = 0.0 - a*operator_list[handle1].matrix_diag[i]*PETSC_i;
        ierr = MatSetValue(A,i_ham,j_ham,mat_tmp,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }


  /* 
   * Add i * (H cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * multiplt n_after by total_levels
   *
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   * Note: Switched k1 and k2 loops to aid in parallelization
   */
  for (k2=0;k2<n_after*total_levels;k2++){
    for (k1=0;k1<n_before;k1++){
      for (i=0;i<my_levels-abs(diag_start);i++){
        /* 
         * Since we store our matrix as a diagonal, we need to
         * calculate the actual i,j location for our operator.
         */

        if (diag_start>=0) {
          i_op  = i;
          j_op  = i+diag_start;
        } else {
          i_op = i-diag_start;
          j_op = i;
        }
           
        /*
         * Now we need to calculate the apropriate location of this
         * within the superoperator Hamiltonian matrix. 
         * See more detailed notes above
         */
        i_ham = i_op*n_after+k2+k1*my_levels*n_after;
        j_ham = j_op*n_after+k2+k1*my_levels*n_after;
        add_to_mat = 0.0 + a*operator_list[handle1].matrix_diag[i]*PETSC_i;
        ierr = MatSetValue(A,i_ham,j_ham,mat_tmp,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  return;
}
/*
  add_to_ham_comb adds a*op(handle1)*op(handle2) to the hamiltonian
  Inputs:
         double a:    scalar to multiply op(handle1)
         int handle1: handle of first operator to combine
         int handle2: handle of second operator to combine
  Outputs:
         none
 */
void add_to_ham_comb(double a,int handle1,int handle2){
  int i;






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
