#include "operators_p.h"
#include "operators.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_OP 100  //Consider making this not a define
/* TODO? : Make Kron product (with I) as a function? 
 * MatSetValue would be within this.
 */

struct operator {
  int n_before;
  int my_levels;
  /* 
   * All of our operators are guaranteed to be a diagonal of some type
   * (likely a super diagonal or sub diagonal, but a diagonal nonetheless)
   * As such, we only store said diagonal, as well as the starting
   * row / column.
   * diagonal_start is the offset from the true diagonal
   * positive means super diagonal, negative means sub diagnal
   */
  double *matrix_diag;
  int diagonal_start;
};

static struct operator operator_list[MAX_OP]; //static should keep it in this file
static int             num_ops;
static double**        hamiltonian;
static int             petsc_initialized = 0;
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

void create_op(int number_of_levels,int *op_create,int *op_destroy,int *op_number) {
  int i;
  PetscErrorCode ierr;

  /* Set up petsc and get nid on first call */
  if (!petsc_initialized){
#if !defined(PETSC_USE_COMPLEX)
    SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
    ierr              = MPI_Comm_rank(PETSC_COMM_WORLD,&nid);CHKERRQ(ierr);
    op_finalized      = 0;
    total_levels      = 1;
    num_subsystems    = 0;
    num_ops           = 0;
    petsc_initialized = 1;
  }
    
  if (num_ops+3>MAX_OP&&nid==0){
    printf("ERROR! Too many ops for this MAX_OP\n");
    exit(0);
  }

  if (op_finalized){
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
    operator_list[num_ops].matrix_diag[i] = sqrt(i+1);
  }

  *op_create = num_ops;
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
    operator_list[num_ops].matrix_diag[i] = sqrt(i+1);
  }

  *op_destroy = num_ops;
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
    operator_list[num_ops].matrix_diag[i] = i;
  }
 
  *op_number = num_ops;
  /* Increment number of ops, since we've finished adding this */
  num_ops++;

  /* Increase total_levels */
  if (total_levels==0){
    total_levels = number_of_levels;
  } else {
    total_levels = total_levels * number_of_levels;
  }

  num_subsystems++;
  return;
}

/*
 * add_to_ham adds a*op(handle1) to the hamiltonian
 * Inputs:
 *        double a:    scalar to multiply op(handle1)
 *        int handle1: handle of first operator to combine
 * Outputs:
 *        none
 */
void add_to_ham(double a,int handle1){
  int            k1,k2,i,j,my_levels,diag_start;
  long           i_ham,j_ham,i_op,j_op,n_before,n_after;
  PetscScalar    add_to_mat;
  PetscErrorCode ierr;
  
  check_initialized_A();

  my_levels  = operator_list[handle1].my_levels;
  diag_start = operator_list[handle1].diagonal_start;
  n_before   = operator_list[handle1].n_before;
  n_after    = total_levels/(n_before*my_levels);

  /* Construct the dense Hamiltonian only on the master node */
  if (nid==0) {
    for (k1=0;k1<n_after;k1++){
      for (k2=0;k2<n_before;k2++){
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
           * For an arbitrary operator, we only care about
           * the Hilbert space size before and the Hilbert space size
           * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b)
           *
           * The calculation of i_ham and j_ham exploit the structure of 
           * the tensor products - they are general for kronecker products
           * of identity matrices with some matrix A
           */
          
          i_ham = i_op*n_after+k1+k2*my_levels*n_after;
          j_ham = j_op*n_after+k1+k2*my_levels*n_after;
          hamiltonian[i_ham][j_ham] = hamiltonian[i_ham][j_ham]
            +a*operator_list[handle1].matrix_diag[i];
        }
      }
    }
  }


  /* 
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * multiply n_before by total_levels
   *
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */
  for (k1=0;k1<n_after;k1++){
    for (k2=0;k2<n_before*total_levels;k2++){
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
        i_ham = i_op*n_after+k1+k2*my_levels*n_after;
        j_ham = j_op*n_after+k1+k2*my_levels*n_after;
        add_to_mat = 0.0 - a*operator_list[handle1].matrix_diag[i]*PETSC_i;
        ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }


  /* 
   * Add i * (H cross I) to the superoperator matrix, A
   * Since this is an additional I after, we simply
   * multiply n_after by total_levels
   *
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   * Note: Switched k1 and k2 loops to aid in parallelization
   */
  for (k2=0;k2<n_before;k2++){
    for (k1=0;k1<n_after*total_levels;k1++){
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
        i_ham = i_op*n_after*total_levels+k1+k2*my_levels*n_after*total_levels;
        j_ham = j_op*n_after*total_levels+k1+k2*my_levels*n_after*total_levels;
        add_to_mat = 0.0 + a*operator_list[handle1].matrix_diag[i]*PETSC_i;
        ierr = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }

  return;
}


/*
 * add_to_ham_comb adds a*op(handle1)*op(handle2) to the hamiltonian
 * Inputs:
 *        double a:    scalar to multiply op(handle1)
 *        int handle1: handle of first operator to combine
 *        int handle2: handle of second operator to combine
 * Outputs:
 *        none
 */
void add_to_ham_comb(double a,int handle1,int handle2){
  int            k1,k2,k3,i,j,levels1,levels2,diag1,diag2,l_handle1,l_handle2;
  int            i1,i2,j1,j2;
  long           i_ham,j_ham,i_comb,j_comb,n_before,n_after,n_between;
  long           my_levels,n_before1,n_before2;
  PetscScalar    add_to_mat;
  PetscErrorCode ierr;

  check_initialized_A();

  n_before1   = operator_list[handle1].n_before;
  n_before2   = operator_list[handle2].n_before;

  /* 
   * We want n_before2 to be the larger of the two,
   * because the kroneckor product only cares about
   * what order the operators were added. 
   * I.E a' * b' = b' * a', where a' is the full space
   * representation of a.
   */

  if (n_before2>n_before1){
    levels1   = operator_list[handle1].my_levels;
    diag1     = operator_list[handle1].diagonal_start;
    levels2   = operator_list[handle2].my_levels;
    diag2     = operator_list[handle2].diagonal_start;
    /* Local handles, since we may need to reverse them */
    l_handle1 = handle1;
    l_handle2 = handle2;
  } else {
    n_before2 = operator_list[handle1].n_before;
    levels2   = operator_list[handle1].my_levels;
    diag2     = operator_list[handle1].diagonal_start;
    n_before1 = operator_list[handle2].n_before;
    levels1   = operator_list[handle2].my_levels;
    diag1     = operator_list[handle2].diagonal_start;
    /* Local handles, since we may need to reverse them */
    l_handle1 = handle2;
    l_handle2 = handle1;
  }

  /* 
   * We need to calculate n_between, since, in general,
   * A=op(1) and B=op(2) may not be next to each other (in kroneckor terms)
   * We may have:
   * A = a cross I_c cross I_b
   * B = I_a cross I_c cross b
   * So, A*B = a cross I_c cross b, where I_c is the Hilbert space size
   * of all operators between.
   * n_between is the hilbert space size between the operators. 
   * We take the larger n_before (say n2), then divide out all operators 
   * before the other operator (say n1), and divide out the other operator's
   * hilbert space (l1), giving n_between = n2/(n1*l1)
   *
   */
  n_between = n_before2/(n_before1*levels1);

  /* 
   * n_before and n_after refer to before and after a cross I_c cross b
   * and my_levels is the size of a cross I_c cross b
   */
  n_before   = n_before1;
  my_levels  = levels1*levels2*n_between;
  n_after    = total_levels/(my_levels*n_before);

  /* Construct the dense Hamiltonian only on the master node */
  if (nid==0) {
    for (k1=0;k1<n_after;k1++){
      for (k2=0;k2<n_before;k2++){
        for (i=0;i<levels1-abs(diag1);i++){
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

          if (diag1>=0) {
            i1  = i;
            j1  = i+diag1;
          } else {
            i1 = i-diag1;
            j1 = i;
          }
          /* 
           * Since we are taking a cross I cross b, we do 
           * I_n_between cross b below 
           */
          for (k3=0;k3<n_between;k3++){
            for (j=0;j<levels2-abs(diag2);j++){

              /* Get the i,j for operator 2 */
              if (diag2>=0) {
                i2  = j;
                j2  = j+diag2;
              } else {
                i2 = j-diag2;
                j2 = j;
              }
              /* Update i2,j2 with the kroneckor product value */
              i2 = i2 + k3*levels2;
              j2 = j2 + k3*levels2;
              /* 
               * Using the standard Kronecker product formula for 
               * A and I cross B, we calculate
               * the i,j pair for handle1 cross I cross handle2.
               * Through we do not use it here, we note that the new
               * matrix is also diagonal, with offset = levels2*n_between*diag1 + diag2;
               * We need levels2*n_between because we are taking
               * a cross (I cross b), so the the size of the second operator
               * is n_between*levels2
               */
              i_comb = levels2*n_between*i1 + i2;
              j_comb = levels2*n_between*j1 + j2;
              /*
               * Now we need to calculate the apropriate location of this
               * within the Hamiltonian matrix. We need to expand the operator
               * from its small Hilbert space to the total Hilbert space.xp
               * This expansion depends on the order in which the operators
               *
               * For an arbitrary operator, we only care about
               * the Hilbert space size before and the Hilbert space size
               * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b)
               *
               * The calculation of i_ham and j_ham exploit the structure of 
               * the tensor products - they are general for kronecker products
               * of identity matrices with some matrix A
               */
              
              i_ham = i_comb*n_after+k1+k2*my_levels*n_after;
              j_ham = j_comb*n_after+k1+k2*my_levels*n_after;
              hamiltonian[i_ham][j_ham] = hamiltonian[i_ham][j_ham]
                +a*operator_list[l_handle1].matrix_diag[i]
                *operator_list[l_handle2].matrix_diag[j];
            }
          }
        }
      }
    }
  }


  /* 
   * Add -i * (I cross H) to the superoperator matrix, A
   * Since this is an additional I before, we simply
   * multiply n_before by total_levels
   *
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */

  for (k2=0;k2<n_before*total_levels;k2++){
    for (k1=0;k1<n_after;k1++){
      for (i=0;i<levels1-abs(diag1);i++){
        /* 
         * Since we store our matrix as a diagonal, we need to
         * calculate the actual i,j location for our first operator.
         */

        if (diag1>=0) {
          i1  = i;
          j1  = i+diag1;
        } else {
          i1 = i-diag1;
          j1 = i;
        }
        /* 
         * Since we are taking a cross I cross b, we do 
         * I_n_between cross b below 
         */
        for (k3=0;k3<n_between;k3++){
          for (j=0;j<levels2-abs(diag2);j++){

            /* Get the i,j for operator 2 */
            if (diag2>=0) {
              i2  = j;
              j2  = j+diag2;
            } else {
              i2 = j-diag2;
              j2 = j;
            }
            /* Update i2,j2 with the I cross b value */
            i2 = i2 + k3*levels2;
            j2 = j2 + k3*levels2;
            /* 
             * Using the standard Kronecker product formula for 
             * A and I cross B, we calculate
             * the i,j pair for handle1 cross I cross handle2.
             * Through we do not use it here, we note that the new
             * matrix is also diagonal, with offset = levels2*n_between*diag1 + diag2;
             * We need levels2*n_between because we are taking
             * a cross (I cross b), so the the size of the second operator
             * is n_between*levels2
             */
            i_comb = levels2*n_between*i1 + i2;
            j_comb = levels2*n_between*j1 + j2;
            /*
             * Now we need to calculate the apropriate location of this
             * within the Hamiltonian matrix. We need to expand the operator
             * from its small Hilbert space to the total Hilbert space.
             * This expansion depends on the order in which the operators
             *
             * For an arbitrary operator, we only care about
             * the Hilbert space size before and the Hilbert space size
             * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b)
             *
             * The calculation of i_ham and j_ham exploit the structure of 
             * the tensor products - they are general for kronecker products
             * of identity matrices with some matrix A
             */
            i_ham = i_comb*n_after+k1+k2*my_levels*n_after;
            j_ham = j_comb*n_after+k1+k2*my_levels*n_after;

            add_to_mat = 0.0 - a*operator_list[l_handle1].matrix_diag[i]
              *operator_list[l_handle2].matrix_diag[j]*PETSC_i;
            ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
          }
        }
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

  for (k1=0;k1<n_after*total_levels;k1++){
    for (k2=0;k2<n_before;k2++){
      for (i=0;i<levels1-abs(diag1);i++){
        /* 
         * Since we store our matrix as a diagonal, we need to
         * calculate the actual i,j location for our first operator.
         */

        if (diag1>=0) {
          i1  = i;
          j1  = i+diag1;
        } else {
          i1 = i-diag1;
          j1 = i;
        }
        /* 
         * Since we are taking a cross I cross b, we do 
         * I_n_between cross b below 
         */
        for (k3=0;k3<n_between;k3++){
          for (j=0;j<levels2-abs(diag2);j++){

            /* Get the i,j for operator 2 */
            if (diag2>=0) {
              i2  = j;
              j2  = j+diag2;
            } else {
              i2 = j-diag2;
              j2 = j;
            }
            /* Update i2,j2 with the I cross b value */
            i2 = i2 + k3*levels2;
            j2 = j2 + k3*levels2;
            /* 
             * Using the standard Kronecker product formula for 
             * A and I cross B, we calculate
             * the i,j pair for handle1 cross I cross handle2.
             * Through we do not use it here, we note that the new
             * matrix is also diagonal, with offset = levels2*n_between*diag1 + diag2;
             * We need levels2*n_between because we are taking
             * a cross (I cross b), so the the size of the second operator
             * is n_between*levels2
             */
            i_comb = levels2*n_between*i1 + i2;
            j_comb = levels2*n_between*j1 + j2;
            /*
             * Now we need to calculate the apropriate location of this
             * within the Hamiltonian matrix. We need to expand the operator
             * from its small Hilbert space to the total Hilbert space.
             * This expansion depends on the order in which the operators
             *
             * For an arbitrary operator, we only care about
             * the Hilbert space size before and the Hilbert space size
             * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b)
             *
             * The calculation of i_ham and j_ham exploit the structure of 
             * the tensor products - they are general for kronecker products
             * of identity matrices with some matrix A
             */
            i_ham = i_comb*n_after*total_levels+k1+k2*my_levels*n_after*total_levels;
            j_ham = j_comb*n_after*total_levels+k1+k2*my_levels*n_after*total_levels;

            add_to_mat = 0.0 + a*operator_list[l_handle1].matrix_diag[i]
              *operator_list[l_handle2].matrix_diag[j]*PETSC_i;
            ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
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
p *        int handle1: handle of operator to add
 * Outputs:
 *        none
 */

void add_lin(double a,int handle){
  int            k1,k2,k3,i,j,my_levels,diag_start,i1,j1,i2,j2,comb_levels;
  long           i_ham,j_ham,j_op,n_before,n_after,i_comb,j_comb;
  PetscScalar    add_to_mat;
  PetscErrorCode ierr;

  check_initialized_A();

  my_levels  = operator_list[handle].my_levels;
  diag_start = operator_list[handle].diagonal_start;
  n_before   = operator_list[handle].n_before;
  n_after    = total_levels/(n_before*my_levels);


  /* 
   * Add (I cross C^t C) to the superoperator matrix, A
   * Which is (I_total cross I_before cross C^t C cross I_after)
   * Since this is an additional I_total before, we simply
   * multiply n_before by total_levels
   *
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */
  for (k2=0;k2<n_before*total_levels;k2++){
    for (k1=0;k1<n_after;k1++){
      for (i=0;i<my_levels-abs(diag_start);i++){
        /* 
         * For this term, we have to calculate Ct C.
         * We know, a priori, that all operators are (sub or super) diagonal.
         * Any matrix such as this will be (true) diagonal after doing Ct C.
         * Ct C is simple to calculate with these diagonal matrices:
         * for all elements of C[k], where C is the stored diagonal,
         * location in Ct C = j,j of the
         * original element C[k], and the value is C[k]*C[k]
         * Since we store our matrix as a diagonal, we need to
         * calculate the actual i,j location for our operator.
         */

        if (diag_start>=0) {
          j_op  = i+diag_start;
        } else {
          j_op = i;
        }

        /*
         * Now we need to calculate the apropriate location of this
         * within the superoperator Hamiltonian matrix. 
         * See more detailed notes above
         * note j_op, j_op because Ct C
         */
        i_ham = j_op*n_after+k1+k2*my_levels*n_after;
        j_ham = j_op*n_after+k1+k2*my_levels*n_after;
        add_to_mat = -0.5*a*operator_list[handle].matrix_diag[i]
          *operator_list[handle].matrix_diag[i] + 0.0*PETSC_i;
        ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  

  /* 
   * Add (C^t C cross I) to the superoperator matrix, A
   * Which is (I_before cross C^t C cross I_after cross I_total)
   * Since this is an additional I_total after, we simply
   * multiply n_after by total_levels
   *
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */

  for (k1=0;k1<n_after*total_levels;k1++){
    for (k2=0;k2<n_before;k2++){
      for (i=0;i<my_levels-abs(diag_start);i++){
        /* 
         * For this term, we have to calculate Ct C.
         * We know, a priori, that all operators are (sub or super) diagonal.
         * Any matrix such as this will be (true) diagonal after doing Ct C.
         * Ct C is simple to calculate with these diagonal matrices:
         * for all elements of C[k], where C is the stored diagonal,
         * location in Ct C = j,j of the
         * original element C[k], and the value is C[k]*C[k]
         * Since we store our matrix as a diagonal, we need to
         * calculate the actual i,j location for our operator.
         */

        if (diag_start>=0) {
          j_op  = i+diag_start;
        } else {
          j_op = i;
        }

        /*
         * Now we need to calculate the apropriate location of this
         * within the superoperator Hamiltonian matrix. 
         * See more detailed notes above
         * note j_op, j_op because Ct C
         */
        i_ham = j_op*n_after*total_levels+k1+k2*my_levels*n_after*total_levels;
        j_ham = j_op*n_after*total_levels+k1+k2*my_levels*n_after*total_levels;
        add_to_mat = -0.5*a*operator_list[handle].matrix_diag[i]
          *operator_list[handle].matrix_diag[i] + 0.0*PETSC_i;
        ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }


  /* 
   * Add (C' cross C') to the superoperator matrix, A, where C' is the full space
   * representation of C. Let I_b = I_before and I_a = I_after
   * This simplifies to (I_b cross C cross I_a cross I_b cross C cross I_a)
   * or (I_b cross C cross I_ab cross C cross I_a)
   * This is just like add_to_ham_comb, with n_between = n_after*n_before
   *
   * We want to do this in parallel, so we chunk it up between cores
   * We move the k3 loop to the top level to assist in parallelization
   * TODO: CHUNK THIS UP
   */

  comb_levels = my_levels*my_levels*n_before*n_after;

  for (k3=0;k3<n_before*n_after;k3++){
    for (k1=0;k1<n_after;k1++){
      for (k2=0;k2<n_before;k2++){
        for (i=0;i<my_levels-abs(diag_start);i++){
        /* 
         * Since we store our matrix as a diagonal, we need to
         * calculate the actual i,j location for our first operator.
         */

          if (diag_start>=0) {
            i1  = i;
            j1  = i+diag_start;
          } else {
            i1 = i-diag_start;
            j1 = i;
          }

          /* 
           * Since we are taking c cross I cross c, we do 
           * I_ab cross c below - the k3 loop is moved to the 
           * top
           */
          for (j=0;j<my_levels-abs(diag_start);j++){

            /* Get the i,j for operator 2 */
            if (diag_start>=0) {
              i2  = j;
              j2  = j+diag_start;
            } else {
              i2 = j-diag_start;
              j2 = j;
            }
            /* Update i2,j2 with the I cross b value */
            i2 = i2 + k3*my_levels;
            j2 = j2 + k3*my_levels;
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
            i_comb = my_levels*n_before*n_after*i1 + i2;
            j_comb = my_levels*n_before*n_after*j1 + j2;

            /*
             * Now we need to calculate the apropriate location of this
             * within the Hamiltonian matrix. We need to expand the operator
             * from its small Hilbert space to the total Hilbert space.
             * This expansion depends on the order in which the operators
             *
             * For an arbitrary operator, we only care about
             * the Hilbert space size before and the Hilbert space size
             * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b)
             *
             * The calculation of i_ham and j_ham exploit the structure of 
             * the tensor products - they are general for kronecker products
             * of identity matrices with some matrix A
             */
            i_ham = i_comb*n_after+k1+k2*comb_levels*n_after;
            j_ham = j_comb*n_after+k1+k2*comb_levels*n_after;

            add_to_mat = a*operator_list[handle].matrix_diag[i]
              *operator_list[handle].matrix_diag[j] + PETSC_i*0;
            ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }


  return;
}


void check_initialized_A(){
  int            i,j;
  long           dim;
  PetscErrorCode ierr;
  if (!petsc_initialized){
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
      hamiltonian = malloc(total_levels*sizeof(double*));
      for (i=0;i<total_levels;i++){
        hamiltonian[i] = malloc(total_levels*sizeof(double));
        /* Initialize to 0 */
        for (j=0;j<total_levels;j++){
          hamiltonian[i][j] = 0.0;
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
        fprintf(fp_ham,"%e ",hamiltonian[i][j]);
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
