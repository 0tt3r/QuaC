#include "operators.h"
#include "kron_p.h" //Includes operators_p.h
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*                                              
 * _get_loop_limit is a simple function that returns the
 * appropriate loop limit for a given op_type
 * Inputs:
 *      op_type my_op_type: operator type
 *      int my_levels:      number of levels for operator
 * Outputs:
 *      none
 * Return value:
 *      calculated loop_limit
 */

long _get_loop_limit(op_type my_op_type,int my_levels){
  int loop_limit;
  /* 
   * Raising and lowering operators both have 
   * one less length in the loop (loop_limit)
   */
  loop_limit = 1;
  if (my_op_type==NUMBER){
    /* Number operator needs to loop through the full my_levels*/
    loop_limit = 0;
  } else if (my_op_type==VEC){
    /* 
     * Vec operators have only one value in their subspace,
     * so the loop size is 1 and loop_limit is my_levels-1
     */
    loop_limit = my_levels-1;
  }

  return loop_limit;
}

/*                                              
 * _get_val_in_subspace is a simple function that returns the
 * i_op,j_op pair and val for a given i;
 * Inputs:
 *      int i:              current index in the loop over the subspace
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 * Outputs:
 *      int *i_op:          row value in subspace
 *      int *j_op:          column value in subspace
 * Return value:
 *      double val:         value at i_op,j_op
 */

double _get_val_in_subspace(long i,op_type my_op_type,int position,long *i_op,long *j_op){
  double val;
  /* 
   * Since we store our operators as a type and number of levels
   * calculate the actual i,j location for our operator,
   * within its subspace, as well as its values.
   *
   * If it is a lowering operator, it is super diagonal.
   * If it is a number operator, it is diagonal.
   * If it is a raising operator, it is sub diagonal.
   * If it is a vector operation, it is only one location in the matrix.
   */
  
  if (my_op_type==LOWER) {
    /* Lowering operator */
    *i_op = i;
    *j_op = i+1;
    val   = sqrt((double)i+1.0);
  } else if (my_op_type==NUMBER){
    /* Number operator */
    *i_op = i;
    *j_op = i;
    val   = (double)i;
  } else if (my_op_type==RAISE){
    /* Raising operator */
    *i_op = i+1;
    *j_op = i;
    val  = sqrt((double)i+1);
  } else {
    /* Vec operator */
    /* 
     * Since we assume 1 vec operator means |e><e|,
     * the only i,j pair is on the diagonal, at it's position
     * And the value is 1
     */
    *i_op = position;
    *j_op = position;
    val   = 1.0;
  }
  
  return val;
}

/*                                              
 * _add_to_dense_ham expands an operator from its subspace to the
 * full Hilbert space and then adds that to the hamiltonian
 *
 * Inputs:
 *      double a:           scalar to multiply operator
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 * Outputs:
 *      none, but adds to dense hamiltonian
 */

void _add_to_dense_ham(double a,int n_before,int my_levels,op_type my_op_type,int position){
  long loop_limit,k1,k2,i,i_op,j_op,i_ham,j_ham,n_after;
  double val;
  loop_limit = _get_loop_limit(my_op_type,my_levels);
  n_after    = total_levels/(n_before*my_levels);
 
  for (k1=0;k1<n_after;k1++){
    for (k2=0;k2<n_before;k2++){
      for (i=0;i<my_levels-loop_limit;i++){
        /* 
         * Since we store our operators as a type and number of levels
         * calculate the actual i,j location for our operator,
         * within its subspace, as well as its values.
         */
        
        val = _get_val_in_subspace(i,my_op_type,position,&i_op,&j_op);
        /*
         * Now we need to calculate the apropriate location of this
         * within the full Hamiltonian matrix. We need to expand the operator
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
        _hamiltonian[i_ham][j_ham] = _hamiltonian[i_ham][j_ham]
          +a*val;
      }
    }
  }
  return;
}


/*                                              
 * _add_to_dense_ham_comb combines and expands a*op1*op2 in to the 
 * full Hilbert space and adds to the hamiltonian
 *
 * Inputs:
 *      double a:           scalar to multiply operators
 *      int n_before1:      Hilbert space size before op1
 *      int levels1:        number of levels for op1
 *      op_type op_type1:   op1's type
 *      int position1:      vec op1's position variable
 *      int n_before2:      Hilbert space size before op2
 *      int levels2:        number of levels for op2
 *      op_type op_type2:   op1's type
 *      int position2:      vec op2's position variable
 * Outputs:
 *      none, but adds to dense hamiltonian
 */

void _add_to_dense_ham_comb(double a,int n_before1,int levels1,op_type op_type1,int position1,
                            int n_before2,int levels2,op_type op_type2,int position2){
  long loop_limit1,loop_limit2,k1,k2,k3,i,j,i1,j1,i2,j2,i_ham,j_ham;
  long n_before,n_after,n_between,my_levels,tmp_switch,i_comb,j_comb;
  double val1,val2;
  op_type tmp_op_switch;

  loop_limit1 = _get_loop_limit(op_type1,levels1);
  loop_limit2 = _get_loop_limit(op_type2,levels2);

  /* 
   * We want n_before2 to be the larger of the two,
   * because the kroneckor product only cares about
   * what order the operators were added. 
   * I.E a' * b' = b' * a', where a' is the full space
   * representation of a.
   * If that is not true, flip them
   */

  if (n_before2<n_before1){
    tmp_switch    = levels1;
    levels1       = levels2;
    levels2       = tmp_switch;

    tmp_switch    = n_before1;
    n_before1     = n_before2;
    n_before2     = tmp_switch;

    tmp_switch    = loop_limit1;
    loop_limit1   = loop_limit2;
    loop_limit2   = tmp_switch;

    tmp_op_switch = op_type1;
    op_type1      = op_type2;
    op_type2      = tmp_op_switch;
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

  for (k1=0;k1<n_after;k1++){
    for (k2=0;k2<n_before;k2++){
      for (i=0;i<levels1-loop_limit1;i++){
        /* 
         * Since we store our operators as a type and number of levels
         * calculate the actual i,j location for our operator,
         * within its subspace, as well as its values.
         */
        val1 = _get_val_in_subspace(i,op_type1,position1,&i1,&j1);
        /* 
         * Since we are taking a cross I cross b, we do 
         * I_n_between cross b below 
         */
        for (k3=0;k3<n_between;k3++){
          for (j=0;j<levels2-loop_limit2;j++){

            /* Get the i,j and val for operator 2 */
            val2 = _get_val_in_subspace(j,op_type2,position2,&i2,&j2);

            /* Update i2,j2 with the kroneckor product for I_between */
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
            _hamiltonian[i_ham][j_ham] = _hamiltonian[i_ham][j_ham]
              +a*val1*val2;
          }
        }
      }
    }
  }

  return;
}


/*                                              
 * _add_to_PETSc_kron expands an operator given a Hilbert space size
 * before and affter and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 *      int extra_before:   extra Hilbert space size before
 *      int extra_after:    extra Hilbert space size after
 * Outputs:
 *      none, but adds to dense hamiltonian
 */

void _add_to_PETSc_kron(PetscScalar a,int n_before,int my_levels,
                        op_type my_op_type,int position,
                        int extra_before,int extra_after){
  long loop_limit,k1,k2,i,i_op,j_op,i_ham,j_ham,n_after;
  PetscReal    val;
  PetscScalar add_to_mat;
  PetscErrorCode ierr;
  loop_limit = _get_loop_limit(my_op_type,my_levels);

  n_after    = total_levels/(my_levels*n_before);
  /*
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */
  for (k1=0;k1<n_after*extra_after;k1++){
    for (k2=0;k2<n_before*extra_before;k2++){
      for (i=0;i<my_levels-loop_limit;i++){
        /* 
         * Since we store our operators as a type and number of levels
         * calculate the actual i,j location for our operator,
         * within its subspace, as well as its values.
         */
        
        val = _get_val_in_subspace(i,my_op_type,position,&i_op,&j_op);
        /*
         * Now we need to calculate the apropriate location of this
         * within the full Hamiltonian matrix. We need to expand the operator
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
        i_ham = i_op*n_after*extra_after+k1+k2*my_levels*n_after*extra_after;
        j_ham = j_op*n_after*extra_after+k1+k2*my_levels*n_after*extra_after;

        /* 
         * If there is no extra hilbert space size before or after,
         * then we assume this is for the dense hamiltonian, and we
         * store it in the H matrix.
         */
        if (extra_before==1&&extra_after==1){
          _hamiltonian[i_ham][j_ham] = _hamiltonian[i_ham][j_ham]+
            (double)PetscRealPart(a*val);
        } else {
          add_to_mat = a*val; //Will this do the complex multiplication right?!
          ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  return;
}

/*                                              
 * _add_to_PETSc_kron_comb expands a*op1*op2 given a Hilbert space size
 * before and affter and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 * Outputs:
 *      none, but adds to dense hamiltonian
 */

void _add_to_PETSc_kron_comb(PetscScalar a,int n_before1,int levels1,op_type op_type1,int position1,
                             int n_before2,int levels2,op_type op_type2,int position2,
                             int extra_before,int extra_between,int extra_after){
  long loop_limit1,loop_limit2,k1,k2,k3,i,j,i1,j1,i2,j2,i_ham,j_ham;
  long n_before,n_after,n_between,my_levels,tmp_switch,i_comb,j_comb;
  double val1,val2;
  PetscScalar add_to_mat;
  PetscErrorCode ierr;
  op_type tmp_op_switch;

  loop_limit1 = _get_loop_limit(op_type1,levels1);
  loop_limit2 = _get_loop_limit(op_type2,levels2);

  /* 
   * We want n_before2 to be the larger of the two,
   * because the kroneckor product only cares about
   * what order the operators were added. 
   * I.E a' * b' = b' * a', where a' is the full space
   * representation of a.
   * If that is not true, flip them
   */

  if (n_before2<n_before1){
    tmp_switch    = levels1;
    levels1       = levels2;
    levels2       = tmp_switch;

    tmp_switch    = n_before1;
    n_before1     = n_before2;
    n_before2     = tmp_switch;

    tmp_switch    = loop_limit1;
    loop_limit1   = loop_limit2;
    loop_limit2   = tmp_switch;

    tmp_switch    = position1;
    position1     = position2;
    position2     = position1;

    tmp_op_switch = op_type1;
    op_type1      = op_type2;
    op_type2      = tmp_op_switch;
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

  for (k1=0;k1<n_after*extra_after;k1++){
    for (k2=0;k2<n_before*extra_before;k2++){
      for (i=0;i<levels1-loop_limit1;i++){
        /* 
         * Since we store our operators as a type and number of levels
         * calculate the actual i,j location for our operator,
         * within its subspace, as well as its values.
         */
        val1 = _get_val_in_subspace(i,op_type1,position1,&i1,&j1);
        /* 
         * Since we are taking a cross I cross b, we do 
         * I_n_between cross b below 
         */
        for (k3=0;k3<n_between*extra_between;k3++){
          for (j=0;j<levels2-loop_limit2;j++){

            /* Get the i,j and val for operator 2 */
            val2 = _get_val_in_subspace(j,op_type2,position2,&i2,&j2);

            /* Update i2,j2 with the kroneckor product for I_between */
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
              
            i_ham = i_comb*n_after*extra_after+k1+k2*my_levels*n_after*extra_after;
            j_ham = j_comb*n_after*extra_after+k1+k2*my_levels*n_after*extra_after;

            /* 
             * If there is no extra hilbert space size before or after,
             * then we assume this is for the dense hamiltonian, and we
             * store it in the H matrix.
             */
            if (extra_before==1&&extra_after==1&&extra_between==1){
              _hamiltonian[i_ham][j_ham] = _hamiltonian[i_ham][j_ham]+
                (double)PetscRealPart(a*val1*val2);
            } else {
              add_to_mat = a*val1*val2; //Will this do the complex multiplication right?!
              ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }
      }
    }
  }

  return;
}


/*                                              
 * _add_to_PETSc_kron_lin expands an op^dag op given a Hilbert space size
 * before and after and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 *      int extra_before:   extra Hilbert space size before
 *      int extra_after:    extra Hilbert space size after
 * Outputs:
 *      none, but adds to PETSc matrix 
 */

void _add_to_PETSc_kron_lin(PetscScalar a,int n_before,int my_levels,
                        op_type my_op_type,int position,
                        int extra_before,int extra_after){
  long loop_limit,k1,k2,i,i_op,j_op,i_ham,j_ham,n_after;
  PetscReal    val;
  PetscErrorCode ierr;
  PetscScalar add_to_mat;
  loop_limit = _get_loop_limit(my_op_type,my_levels);

  n_after    = total_levels/(my_levels*n_before);
  /*
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */
  for (k1=0;k1<n_after*extra_after;k1++){
    for (k2=0;k2<n_before*extra_before;k2++){
      for (i=0;i<my_levels-loop_limit;i++){
       
        /* 
         * For this term, we have to calculate Ct C.
         * We know, a priori, that all operators are (sub or super) diagonal.
         * Any matrix such as this will be (true) diagonal after doing Ct C.
         * Ct C is simple to calculate with these diagonal matrices:
         * for all elements of C[k], where C is the stored diagonal,
         * location in Ct C = j,j of the
         * original element C[k], and the value is C[k]*C[k]
         */
        val  = _get_val_in_subspace(i,my_op_type,position,&i_op,&j_op);
        i_op = j_op;
        val  = val*val;
        /*
         * Now we need to calculate the apropriate location of this
         * within the full Hamiltonian matrix. We need to expand the operator
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
          
        i_ham = i_op*n_after*extra_after+k1+k2*my_levels*n_after*extra_after;
        j_ham = j_op*n_after*extra_after+k1+k2*my_levels*n_after*extra_after;
            
        add_to_mat = a*val; //Will this do the complex multiplication right?!
        ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  return;
}


/*                                              
 * _add_to_PETSc_kron_lin_comb adds C' cross C' to the full_A
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 * Outputs:
 *      none, but adds to PETSc matrix 
 */

void _add_to_PETSc_kron_lin_comb(PetscScalar a,int n_before,int my_levels,op_type my_op_type,
                                 int position){
  long loop_limit,k1,k2,k3,i,j,i1,j1,i2,j2,i_ham,j_ham,i_comb,j_comb;
  long n_after,comb_levels;
  double val1,val2;
  PetscScalar add_to_mat;
  PetscErrorCode ierr;

  n_after     = total_levels/(n_before*my_levels);
  comb_levels = my_levels*my_levels*n_before*n_after;

  loop_limit = _get_loop_limit(my_op_type,my_levels);

  for (k1=0;k1<n_after;k1++){
    for (k2=0;k2<n_before;k2++){
      for (k3=0;k3<n_before*n_after;k3++){
        for (i=0;i<my_levels-loop_limit;i++){
        /* 
         * Since we store our operators as a type and number of levels
         * calculate the actual i,j location for our operator,
         * within its subspace, as well as its values.
         */
          val1 = _get_val_in_subspace(i,my_op_type,position,&i1,&j1);
          /* 
           * Since we are taking c cross I cross c, we do 
           * I_ab cross c below - the k3 loop is moved to the 
           * top
           */
          for (j=0;j<my_levels-loop_limit;j++){

            val2 = _get_val_in_subspace(j,my_op_type,position,&i2,&j2);
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

            add_to_mat = a*val1*val2 + PETSC_i*0;
            ierr       = MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  return;
}
