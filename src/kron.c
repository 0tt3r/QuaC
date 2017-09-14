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
 *      long i:             current index in the loop over the subspace
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 * Outputs:
 *      long *i_op:         row value in subspace
 *      long *j_op:         column value in subspace
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
 * _get_val_j_from_global_i returns the val and global j for a given global i.
 * If there is no nonzero value for a given i, it returns a negative j
 * Inputs:
 *      long i:             global i
 *      operator:           operator to get
 * Outputs:
 *      long *j:             global j for nonzer of given i; or negative if none
 *      double *val:        value of op for global i,j
  */

void _get_val_j_from_global_i(PetscInt i,operator this_op,PetscInt *j,double *val){
  int i_sub,n_after,tmp_int,k1,k2;
  /*
   * We store our operators as a type and number of levels;
   * we use the stored information to calculate the global j location
   * and nonzero value for a give global i
   *
   * If it is a lowering operator, it is super diagonal.
   * If it is a number operator, it is diagonal.
   * If it is a raising operator, it is sub diagonal.
   */
  n_after = total_levels/(this_op->my_levels*this_op->n_before);
  i_sub = i/n_after%this_op->my_levels; //Use integer arithmetic to get floor function

  if (this_op->my_op_type==LOWER) {
    /*
     * Lowering operator
     * From i, use the generating function from the kronecker product:
     *    i = i_sub * n_af + k1 + k2*n_l*n_af
     *    j = (i_sub+1) * n_af + k1 + k2*n_l*n_af
     * We work out k1 and k2 from i to get j.
     */
    if (i_sub>=(this_op->my_levels-1)){
      //There is no nonzero value for given global i; return -1 as flag
      *j = -1;
      *val = 0.0;
    } else {
      tmp_int = i - i_sub * n_after;
      k2      = tmp_int/(this_op->my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(this_op->my_levels*n_after);
      *j = (i_sub + 1) * n_after + k1 + k2*this_op->my_levels*n_after;
      *val   = sqrt((double)i_sub+1.0);
    }
  } else if (this_op->my_op_type==NUMBER){
    /* Number operator */
    if (i_sub!=0){
      *j = i; //Diagonal, even in global space
      *val   = (double)i_sub;
    } else {
      //There is no nonzero value for given global i; return -1 as flag
      *j = -1;
      *val = 0.0;
    }
  } else if (this_op->my_op_type==RAISE){
    /*
     * Raising operator
     *
     * From i, use the generating function from the kronecker product:
     *    i = (i_sub+1) * n_af + k1 + k2*n_me*n_af
     *    j = i_sub * n_af + k1 + k2*n_me*n_af
     * We work out k1 and k2 from i to get j.
     */
    i_sub = i_sub - 1;
    if(i_sub<0){
      //There is no nonzero value for given global i; return -1 as flag
      *j = -1;
      *val = 0.0;
    } else {
      tmp_int = i - (i_sub+1) * n_after;
      k2      = tmp_int/(this_op->my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(this_op->my_levels*n_after);
      *j = i_sub * n_after + k1 + k2*this_op->my_levels*n_after;
      *val   = sqrt((double)i_sub+1.0);
    }
  } else {
    /* Vec operator */
    /*
     * Since we assume 1 vec operator means |e><e|,
     * the only i,j pair is on the diagonal, at it's position
     * And the value is 1
     */
      if (nid==0){
        printf("ERROR! Vec Operators not currently supported for _get_val_j_from_global_i\n");
        printf("       (maybe from get_expectation_value)");
        exit(0);
      }
    /* *i_op = position; */
    /* *j_op = position; */
    /* *val   = 1.0; */
  }

  return;
}

/*
 * _add_to_PETSc_kron_ij is the main driver of the kronecker
 * products. It takes an i,j pair from some subspace which
 * then needs to be expanded to the larger space, defined by
 * the Kronecker product with I_before and with I_after.
 *
 * This is where the parallelization of the matrix generation
 * could happen.
 *
 * Inputs:
 *       Mat matrix:             matrix to add to
 *       PetscScalar add_to_mat: value to add
 *       int i_op:               i of the subspace
 *       int j_op:               j of the subspace
 *       int n_before:           size of I_before
 *       int n_after:            size of I_after
 *       int my_levels:          size of subspace
 * Outputs:
 *       none, but adds to PETSc matrix A
 *
 */


void _add_to_PETSc_kron_ij(Mat matrix,PetscScalar add_to_mat,int i_op,int j_op,
                           int n_before,int n_after,int my_levels){
  long k1,k2,i_ham,j_ham;
  //  int  my_start_af,my_end_af,my_start_bef,my_end_bef;
  PetscInt Istart,Iend;

  MatGetOwnershipRange(matrix,&Istart,&Iend); //FIXME: Make these library global?

  for (k1=0;k1<n_after;k1++){ /* n_after loop */
    for (k2=0;k2<n_before;k2++){ /* n_before loop */
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
      if (i_ham>=Istart&&i_ham<Iend) MatSetValue(matrix,i_ham,j_ham,add_to_mat,ADD_VALUES);
    }
  }

  //FIXME: Put this code in separate routine?
  /* /\*  */
  /*  * We want to parallelize on the largest of n_after or n_before, */
  /*  * because n_after and n_before will be 1 in some cases, so we */
  /*  * would have no parallelization of that loop. As such, we */
  /*  * check here which is bigger and split based on that. */
  /*  *\/ */
  /* if (n_after>n_before){ */
  /*   /\* Parallelize the n_after loop *\/ */
  /*   my_start_af      = (n_after/np)*nid; */
  /*   if (n_after%np>nid){ */
  /*     my_start_af   += nid; */
  /*     my_end_af      = my_start_af+(n_after/np)+1; */
  /*   } else { */
  /*     my_start_af   += n_after % np; */
  /*     my_end_af      = my_start_af+(n_after/np); */
  /*   } */

  /*   for (k1=my_start_af;k1<my_end_af;k1++){ /\* n_after loop *\/ */
  /*     for (k2=0;k2<n_before;k2++){ /\* n_before loop *\/ */
  /*       /\* */
  /*        * Now we need to calculate the apropriate location of this */
  /*        * within the full Hamiltonian matrix. We need to expand the operator */
  /*        * from its small Hilbert space to the total Hilbert space. */
  /*        * This expansion depends on the order in which the operators */
  /*        * were added. For example, if we added 3 operators: */
  /*        * A, B, and C (with sizes n_a, n_b, n_c, respectively), we would */
  /*        * have (where the ' denotes in the full space and I_(n) means */
  /*        * the identity matrix of size n): */
  /*        * */
  /*        * A' = A cross I_(n_b) cross I_(n_c) */
  /*        * B' = I_(n_a) cross B cross I_(n_c) */
  /*        * C' = I_(n_a) cross I_(n_b) cross C */
  /*        *  */
  /*        * For an arbitrary operator, we only care about */
  /*        * the Hilbert space size before and the Hilbert space size */
  /*        * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b) */
  /*        * */
  /*        * The calculation of i_ham and j_ham exploit the structure of  */
  /*        * the tensor products - they are general for kronecker products */
  /*        * of identity matrices with some matrix A */
  /*        *\/ */
  /*       i_ham = i_op*n_after+k1+k2*my_levels*n_after; */
  /*       j_ham = j_op*n_after+k1+k2*my_levels*n_after; */
  /*       MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES); */
  /*     } */
  /*   } */

  /* } else { */
  /*   /\* Parallelize the n_before loop *\/ */
  /*   my_start_bef     = (n_before/np)*nid; */
  /*   if (n_before%np>nid){ */
  /*     my_start_bef  += nid; */
  /*     my_end_bef     = my_start_bef+(n_before/np)+1; */
  /*   } else { */
  /*     my_start_bef  += n_before % np; */
  /*     my_end_bef     = my_start_bef+(n_before/np); */
  /*   } */

  /*   for (k1=0;k1<n_after;k1++){ /\* n_after loop *\/ */
  /*     for (k2=my_start_bef;k2<my_end_bef;k2++){ /\* n_before loop *\/ */
  /*       /\* */
  /*        * Now we need to calculate the apropriate location of this */
  /*        * within the full Hamiltonian matrix. We need to expand the operator */
  /*        * from its small Hilbert space to the total Hilbert space. */
  /*        * This expansion depends on the order in which the operators */
  /*        * were added. For example, if we added 3 operators: */
  /*        * A, B, and C (with sizes n_a, n_b, n_c, respectively), we would */
  /*        * have (where the ' denotes in the full space and I_(n) means */
  /*        * the identity matrix of size n): */
  /*        * */
  /*        * A' = A cross I_(n_b) cross I_(n_c) */
  /*        * B' = I_(n_a) cross B cross I_(n_c) */
  /*        * C' = I_(n_a) cross I_(n_b) cross C */
  /*        *  */
  /*        * For an arbitrary operator, we only care about */
  /*        * the Hilbert space size before and the Hilbert space size */
  /*        * after the target operator (since I_(n_a) cross I_(n_b) = I_(n_a*n_b) */
  /*        * */
  /*        * The calculation of i_ham and j_ham exploit the structure of  */
  /*        * the tensor products - they are general for kronecker products */
  /*        * of identity matrices with some matrix A */
  /*        *\/ */
  /*       i_ham = i_op*n_after+k1+k2*my_levels*n_after; */
  /*       j_ham = j_op*n_after+k1+k2*my_levels*n_after; */
  /*       MatSetValue(full_A,i_ham,j_ham,add_to_mat,ADD_VALUES); */
  /*     } */
  /*   } */
  /* } */
  return;
}


/*
 * _add_PETSc_DM_kron_ij is the main driver of the kronecker
 * products. It takes an i,j pair from some subspace which
 * then needs to be expanded to the larger space, defined by
 * the Kronecker product with I_before and with I_after. This
 * routine specifically adds to the initial density matrix, rho.
 *
 * This is where the parallelization of the matrix generation
 * could happen.
 *
 * Inputs:
 *       PetscScalar add_to_mat: value to add
 *       Mat subspace_dm:        subspace density matrix
 *       Mat rho_mat:            initial density matrix
 *       int i_op:               i of the subspace
 *       int j_op:               j of the subspace
 *       int n_before:           size of I_before
 *       int n_after:            size of I_after
 *       int my_levels:          size of subspace
 * Outputs:
 *       none, but adds to PETSc matrix
 *
 */

void _add_PETSc_DM_kron_ij(PetscScalar add_to_rho,Mat subspace_dm,Mat rho_mat,int i_op,int j_op,
                            int n_before,int n_after,int my_levels){
  long k1,k2,i_dm,j_dm;

  for (k1=0;k1<n_after;k1++){ /* n_after loop */
    for (k2=0;k2<n_before;k2++){ /* n_before loop */
      /*
       * Now we need to calculate the apropriate location of this
       * within the full DM vector. We need to expand the operator
       * from its small Hilbert space to the total Hilbert space.
       * This expansion depends on the order in which the operators
       * were added. For example, if we multiplied 3 operators:
       * A, B, and C (with sizes n_a, n_b, n_c, respectively), we would
       * have (where the ' denotes in the full space and I_(n) means
       * the identity matrix of size n):
       * DM = A' B' C' = A cross B cross C
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
      i_dm = i_op*n_after+k1+k2*my_levels*n_after;
      j_dm = j_op*n_after+k1+k2*my_levels*n_after;
      MatSetValue(subspace_dm,i_dm,j_dm,add_to_rho,ADD_VALUES);
    }
  }
  return;
}

/*
 * _mult_PETSc_init_DM takes in a (fully expanded) subspace's
 * density matrix and does rho = rho*sub_DM. Since each DM is from a
 * separate Hilbert space, this is valid.
 * Inputs:
 *      Mat subspace_dm - the (fully expanded) subspace's DM
 *      Mat rho_mat     - the initial DM
 */
void _mult_PETSc_init_DM(Mat subspace_dm,Mat rho_mat,double trace){
  Mat tmp_mat;

  /* Assemble matrix */

  MatAssemblyBegin(subspace_dm,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(subspace_dm,MAT_FINAL_ASSEMBLY);

  /*
   * Check to make sure trace is 1; if not, normalize and print a warning.
   */
  if (trace!=(double)1.0){
    printf("WARNING! The trace over the subsystem is not 1.0!\n");
    printf("         The initial populations were normalized.\n");
    MatScale(subspace_dm,1./trace);
  }

  /*
   * Do rho = rho*subspace_dm - this is correct because the initial DMs
   * are all from different subspaces
   */
  MatMatMult(rho_mat,subspace_dm,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tmp_mat);
  MatCopy(tmp_mat,rho_mat,SAME_NONZERO_PATTERN);
  MatDestroy(&tmp_mat);

  return;
}

/*
 * _add_to_dense_kron_ij is the main driver of the kronecker
 * products. It takes an i,j pair from some subspace which
 * then needs to be expanded to the larger space, defined by
 * the Kronecker product with I_before and with I_after
 * Inputs:
 *       PetscScalar a:               value to add
 *       int i_op:               i of the subspace
 *       int j_op:               j of the subspace
 *       int n_before:           size of I_before
 *       int n_after:            size of I_after
 *       int my_levels:          size of subspace
 * Outputs:
 *       none, but adds to dense matrix Ham
 *
 */

void _add_to_dense_kron_ij(PetscScalar a,int i_op,int j_op,
                           int n_before,int n_after,int my_levels){
  long k1,k2,i_ham,j_ham;

  for (k1=0;k1<n_after;k1++){
    for (k2=0;k2<n_before;k2++){
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
      _hamiltonian[i_ham][j_ham] = _hamiltonian[i_ham][j_ham] + a;
     }
  }

}

/*
 * _add_to_PETSc_kron expands an operator given a Hilbert space size
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
 *      none, but adds to PETSc matrix full_A
 */

void _add_to_PETSc_kron(Mat matrix, PetscScalar a,int n_before,int my_levels,
                        op_type my_op_type,int position,
                        int extra_before,int extra_after){
  long loop_limit,i,i_op,j_op,n_after;
  PetscReal    val;
  PetscScalar add_to_mat;
  loop_limit = _get_loop_limit(my_op_type,my_levels);

  n_after    = total_levels/(my_levels*n_before);
  /*
   * We want to do this in parallel, so we chunk it up between cores
   * TODO: CHUNK THIS UP
   */

  for (i=0;i<my_levels-loop_limit;i++){
    /*
     * Since we store our operators as a type and number of levels
     * calculate the actual i,j location for our operator,
     * within its subspace, as well as its values.
     */
    val = _get_val_in_subspace(i,my_op_type,position,&i_op,&j_op);
    add_to_mat = a*val;
    _add_to_PETSc_kron_ij(matrix,add_to_mat,i_op,j_op,n_before*extra_before,n_after*extra_after,my_levels);
   }
  return;
}

/*
 * _add_to_PETSc_kron_comb expands a*op1*op2 given a Hilbert space size
 * before and after and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a      scalar to multiply operator (can be complex)
 *      int n_before1:     Hilbert space size before op1
 *      int levels1:       levels of op1
 *      op_type op_type1:  operator type of op1
 *      int position1:     vec op1's position variable
 *      int n_before2:     Hilbert space size before op2
 *      int levels2:       levels of op2
 *      op_type op_type2:  operator type of op2
 *      int position2:     vec op2's position variable
 *      int extra_before:  extra Hilbert space size before
 *      int extra_between: extra Hilbert space size between
 *      int extra_after:   extra Hilbert space size after
 *      int transpose:     whether or not to take the transpose
 * Outputs:
 *      none, but adds to full_A
 */

void _add_to_PETSc_kron_comb(Mat matrix,PetscScalar a,int n_before1,int levels1,op_type op_type1,int position1,
                             int n_before2,int levels2,op_type op_type2,int position2,
                             int extra_before,int extra_between,int extra_after,
                             int transpose){
  long loop_limit1,loop_limit2,k3,i,j,i1,j1,i2,j2;
  long n_before,n_after,n_between,my_levels,tmp_switch,i_comb,j_comb;
  double val1,val2;
  PetscScalar add_to_mat;
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
        if (transpose) {
          /*
           * We can transpose the combined operators because
           * the transpose of Kronecker products is the
           * kronecker product of the transposes
           */
          j_comb = levels2*n_between*i1 + i2;
          i_comb = levels2*n_between*j1 + j2;
        } else {
          i_comb = levels2*n_between*i1 + i2;
          j_comb = levels2*n_between*j1 + j2;
        }
        add_to_mat = a*val1*val2;
        _add_to_PETSc_kron_ij(matrix,add_to_mat,i_comb,j_comb,n_before*extra_before,
                              n_after*extra_after,my_levels);
      }
    }
  }

  return;
}

/*
 * _add_to_PETSc_kron_comb_vec expands a*vec*vec*op or a*op*vec*vec
 * given a Hilbert space size before and after and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before_op:    Hilbert space size before op
 *      int levels_op:      number of levels for op
 *      op_type op_type_op: operator type of op
 *      int n_before_vec:   Hilbert space size before vec
 *      int levels_vec:     number of levels for vec
 *      int i_vec:          vec*vec row index
 *      int j_vec:          vec*vec column index
 *      int extra_before:  extra Hilbert space size before
 *      int extra_between: extra Hilbert space size between
 *      int extra_after:   extra Hilbert space size after
 *      int transpose:     whether or not to take the transpose
 * Outputs:
 *      none, but adds to full_A
 */

void _add_to_PETSc_kron_comb_vec(Mat matrix,PetscScalar a,int n_before_op,int levels_op,op_type op_type_op,
                                 int n_before_vec,int levels_vec,int i_vec,int j_vec,
                                 int extra_before,int extra_between,int extra_after,
                                 int transpose){
  long loop_limit_op,k3,i,j,i1,j1,i2,j2;
  long n_before,n_after,n_between,my_levels,i_comb,j_comb;
  double val1,val2;
  PetscScalar add_to_mat;

  loop_limit_op = _get_loop_limit(op_type_op,levels_op);

  /*
   * We want n_before2 to be the larger of the two,
   * because the kroneckor product only cares about
   * what order the operators were added.
   * I.E a' * b' = b' * a', where a' is the full space
   * representation of a.
   * If that is not true, flip them
   */

  if (n_before_vec < n_before_op){
    /* n_before_vec => n_before1, n_before_op => n_before2 */

    /* The normal op is farther in Hilbert space */

    n_between = n_before_op/(n_before_vec*levels_vec);

    /*
     * n_before and n_after refer to before and after a cross I_c cross b
     * and my_levels is the size of a cross I_c cross b
     */
    n_before   = n_before_vec;
    my_levels  = levels_op*levels_vec*n_between;
    n_after    = total_levels/(my_levels*n_before);

    /* The vec pair is i1, and we know it is only one value in one spot in its subspace */
    val1 = 1.0;
    i1   = i_vec;
    j1   = j_vec;
    /*
     * Since we are taking a cross I cross b, we do
     * I_n_between cross b below
     */
    for (k3=0;k3<n_between*extra_between;k3++){
      for (j=0;j<levels_op-loop_limit_op;j++){
        /* Get the i,j and val for operator 2 */
        val2 = _get_val_in_subspace(j,op_type_op,-1,&i2,&j2);

        /* Update i2,j2 with the kroneckor product for I_between */
        i2 = i2 + k3*levels_op;
        j2 = j2 + k3*levels_op;
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
        if (transpose) {
          /*
           * We can transpose the combined operators because
           * the transpose of Kronecker products is the
           * kronecker product of the transposes
           */
          j_comb = levels_op*n_between*i1 + i2;
          i_comb = levels_op*n_between*j1 + j2;
        } else {
          i_comb = levels_op*n_between*i1 + i2;
          j_comb = levels_op*n_between*j1 + j2;
        }
        add_to_mat = a*val1*val2;

        _add_to_PETSc_kron_ij(matrix,add_to_mat,i_comb,j_comb,n_before*extra_before,
                              n_after*extra_after,my_levels);
      }
    }
  } else {
    /* n_before_vec => n_before2, n_before_op => n_before1 */

    /* The vec op pair is farther in Hilbert space */

    n_between = n_before_vec/(n_before_op*levels_op);

    /*
     * n_before and n_after refer to before and after a cross I_c cross b
     * and my_levels is the size of a cross I_c cross b
     */
    n_before   = n_before_op;
    my_levels  = levels_vec*levels_op*n_between;
    n_after    = total_levels/(my_levels*n_before);

    for (i=0;i<levels_op-loop_limit_op;i++){
      /*
       * Since we store our operators as a type and number of levels
       * calculate the actual i,j location for our operator,
       * within its subspace, as well as its values.
       */
      val1 = _get_val_in_subspace(i,op_type_op,-1,&i1,&j1);
      /*
       * Since we are taking a cross I cross b, we do
       * I_n_between cross b below
       */
      for (k3=0;k3<n_between*extra_between;k3++){

        /* The vec pair is op2, and we know it is only one value in one spot in its subspace */
        val2 = 1.0;
        i2   = i_vec;
        j2   = j_vec;

        /* Update i2,j2 with the kroneckor product for I_between */
        i2 = i2 + k3*levels_vec;
        j2 = j2 + k3*levels_vec;
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
        if (transpose) {
          /*
           * We can transpose the combined operators because
           * the transpose of Kronecker products is the
           * kronecker product of the transposes
           */
          j_comb = levels_vec*n_between*i1 + i2;
          i_comb = levels_vec*n_between*j1 + j2;
        } else {
          i_comb = levels_vec*n_between*i1 + i2;
          j_comb = levels_vec*n_between*j1 + j2;
        }
        add_to_mat = a*val1*val2;

        _add_to_PETSc_kron_ij(matrix,add_to_mat,i_comb,j_comb,n_before*extra_before,
                              n_after*extra_after,my_levels);
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

void _add_to_PETSc_kron_lin(Mat matrix,PetscScalar a,int n_before,int my_levels,
                        op_type my_op_type,int position,
                        int extra_before,int extra_after){
  long loop_limit,i,i_op,j_op,n_after;
  PetscReal    val;
  PetscScalar add_to_mat;
  loop_limit = _get_loop_limit(my_op_type,my_levels);

  n_after    = total_levels/(my_levels*n_before);

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
    add_to_mat = a*val;
    _add_to_PETSc_kron_ij(matrix,add_to_mat,i_op,j_op,n_before*extra_before,
                          n_after*extra_after,my_levels);
  }

  return;
}


/*
 * WARNING: A LITTLE BIT OF A HACK
 * _add_to_PETSc_kron_lin2 assumes op = a a^\dag and
 * expands an op^dag op given a Hilbert space size
 * before and after and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      int extra_before:   extra Hilbert space size before
 *      int extra_after:    extra Hilbert space size after
 * Outputs:
 *      none, but adds to PETSc matrix
 */

void _add_to_PETSc_kron_lin2(Mat matrix,PetscScalar a,int n_before,int my_levels,
                             int extra_before,int extra_after){
  long i,i_op,j_op,n_after;
  PetscReal    val;
  PetscScalar add_to_mat;


  n_after    = total_levels/(my_levels*n_before);

  for (i=1;i<my_levels;i++){
    /*
     * For this term, we have to calculate Ct C.
     * We are assuming that C = aa^\dagger, so we
     * exploit that structure directly
     */
    i_op = i-1;
    j_op = i-1;
    val  = (double)i*(double)i;
    add_to_mat = a*val;
    _add_to_PETSc_kron_ij(matrix,add_to_mat,i_op,j_op,n_before*extra_before,
                          n_after*extra_after,my_levels);
  }

  return;
}


/*
 * add_to_PETSc_kron_lin_comb adds C' cross C' to matrix
 *
 * Inputs:
 *      Mat matrix          matrix to add to
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 * Outputs:
 *      none, but adds to PETSc matrix
 */

void _add_to_PETSc_kron_lin_comb(Mat matrix, PetscScalar a,int n_before,int my_levels,op_type my_op_type,
                                 int position){
  long loop_limit,k3,i,j,i1,j1,i2,j2,i_comb,j_comb;
  long n_after,comb_levels;
  double val1,val2;
  PetscScalar add_to_mat;


  n_after     = total_levels/(n_before*my_levels);
  comb_levels = my_levels*my_levels*n_before*n_after;

  loop_limit = _get_loop_limit(my_op_type,my_levels);

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

        add_to_mat = a*val1*val2 + PETSC_i*0;
        _add_to_PETSc_kron_ij(matrix,add_to_mat,i_comb,j_comb,n_before,
                              n_after,comb_levels);
      }
    }
  }

  return;
}

/* WARNING: A BIT OF A HACK
 * add_to_PETSc_kron_lin2_comb adds C' cross C' to the full_A, assuming
 * C' = a a\dag form
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 * Outputs:
 *      none, but adds to PETSc matrix
 */

void _add_to_PETSc_kron_lin2_comb(Mat matrix,PetscScalar a,int n_before,int my_levels){
  long k3,i,j,i1,j1,i2,j2,i_comb,j_comb;
  long n_after,comb_levels;
  double val1,val2;
  PetscScalar add_to_mat;


  n_after     = total_levels/(n_before*my_levels);
  comb_levels = my_levels*my_levels*n_before*n_after;

  for (k3=0;k3<n_before*n_after;k3++){
    for (i=1;i<my_levels;i++){
      /*
       * We are assuming that C = aa^\dagger, so we
       * exploit that structure directly
       */
      i1 = i-1;
      j1 = i-1;
      val1  = (double)i*(double)i;

      /*
       * Since we are taking c cross I cross c, we do
       * I_ab cross c below - the k3 loop is moved to the
       * top
       */
      for (j=1;j<my_levels;j++){
        /*
         * We are assuming that C = aa^\dagger, so we
         * exploit that structure directly
         */
        i2 = j-1;
        j2 = j-1;
        val2  = (double)i*(double)i;

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

        add_to_mat = a*val1*val2 + PETSC_i*0;
        _add_to_PETSc_kron_ij(matrix,add_to_mat,i_comb,j_comb,n_before,
                              n_after,comb_levels);
      }
    }
  }

  return;
}


/*
 * _add_to_dense_kron expands an operator given a Hilbert space size
 * before and after and adds that to the dense Hamiltonian
 *
 * Inputs:
 *      PetscScalar a:           scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 * Outputs:
 *      none, but adds to dense hamiltonian
 */

void _add_to_dense_kron(PetscScalar a,int n_before,int my_levels,
                        op_type my_op_type,int position){
  long loop_limit,i,i_op,j_op,n_after;
  PetscReal    val;
  double add_to_mat;
  loop_limit = _get_loop_limit(my_op_type,my_levels);

  n_after    = total_levels/(my_levels*n_before);

  for (i=0;i<my_levels-loop_limit;i++){
    /*
     * Since we store our operators as a type and number of levels
     * calculate the actual i,j location for our operator,
     * within its subspace, as well as its values.
     */
    val = _get_val_in_subspace(i,my_op_type,position,&i_op,&j_op);
    add_to_mat = a*val;
    _add_to_dense_kron_ij(add_to_mat,i_op,j_op,n_before,n_after,my_levels);
   }
  return;
}


/*
 * _add_to_dense_kron_comb expands a*op1*op2 given a Hilbert space size
 * before and after and adds that to the dense Ham matrix
 *
 * Inputs:
 *      double a:          scalar to multiply operator
 *      int n_before1:     Hilbert space size before op1
 *      int levels1:       levels of op1
 *      op_type op_type1:  operator type of op1
 *      int position1:     vec op1's position variable
 *      int n_before2:     Hilbert space size before op2
 *      int levels2:       levels of op2
 *      op_type op_type2:  operator type of op2
 *      int position2:     vec op2's position variable
 * Outputs:
 *      none, but adds to dense Hamiltonian
 */

void _add_to_dense_kron_comb(PetscScalar a,int n_before1,int levels1,op_type op_type1,int position1,
                             int n_before2,int levels2,op_type op_type2,int position2){
  long loop_limit1,loop_limit2,k3,i,j,i1,j1,i2,j2;
  long n_before,n_after,n_between,my_levels,tmp_switch,i_comb,j_comb;
  double val1,val2;
  PetscScalar add_to_mat;
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
        add_to_mat = a*val1*val2;
        _add_to_dense_kron_ij(add_to_mat,i_comb,j_comb,n_before,
                              n_after,my_levels);
      }
    }
  }

  return;
}



/*
 * _add_to_dense_kron_comb_vec expands a*vec*vec*op or a*op*vec*vec
 * given a Hilbert space size before and after and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a            scalar to multiply operator
 *      int n_before_op:    Hilbert space size before op
 *      int levels_op:      number of levels for op
 *      op_type op_type_op: operator type of op
 *      int n_before_vec:   Hilbert space size before vec
 *      int levels_vec:     number of levels for vec
 *      int i_vec:          vec*vec row index
 *      int j_vec:          vec*vec column index
 * Outputs:
 *      none, but adds to dense hamiltonian
 */

void _add_to_dense_kron_comb_vec(PetscScalar a,int n_before_op,int levels_op,op_type op_type_op,
                                 int n_before_vec,int levels_vec,int i_vec,int j_vec){
  long loop_limit_op,k3,i,j,i1,j1,i2,j2;
  long n_before,n_after,n_between,my_levels,i_comb,j_comb;
  double val1,val2;
  PetscScalar add_to_mat;

  loop_limit_op = _get_loop_limit(op_type_op,levels_op);

  /*
   * We want n_before2 to be the larger of the two,
   * because the kroneckor product only cares about
   * what order the operators were added.
   * I.E a' * b' = b' * a', where a' is the full space
   * representation of a.
   * If that is not true, flip them
   */

  if (n_before_vec < n_before_op){
    /* n_before_vec => n_before1, n_before_op => n_before2 */

    /* The normal op is farther in Hilbert space */

    n_between = n_before_op/(n_before_vec*levels_vec);

    /*
     * n_before and n_after refer to before and after a cross I_c cross b
     * and my_levels is the size of a cross I_c cross b
     */
    n_before   = n_before_vec;
    my_levels  = levels_op*levels_vec*n_between;
    n_after    = total_levels/(my_levels*n_before);

    /* The vec pair is i1, and we know it is only one value in one spot in its subspace */
    val1 = 1.0;
    i1   = i_vec;
    j1   = j_vec;
    /*
     * Since we are taking a cross I cross b, we do
     * I_n_between cross b below
     */
    for (k3=0;k3<n_between;k3++){
      for (j=0;j<levels_op-loop_limit_op;j++){
        /* Get the i,j and val for operator 2 */
        val2 = _get_val_in_subspace(j,op_type_op,-1,&i2,&j2);

        /* Update i2,j2 with the kroneckor product for I_between */
        i2 = i2 + k3*levels_op;
        j2 = j2 + k3*levels_op;
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
        i_comb = levels_op*n_between*i1 + i2;
        j_comb = levels_op*n_between*j1 + j2;
        add_to_mat = a*val1*val2;

        _add_to_dense_kron_ij(add_to_mat,i_comb,j_comb,n_before,
                              n_after,my_levels);
      }
    }
  } else {
    /* n_before_vec => n_before2, n_before_op => n_before1 */

    /* The vec op pair is farther in Hilbert space */

    n_between = n_before_vec/(n_before_op*levels_op);

    /*
     * n_before and n_after refer to before and after a cross I_c cross b
     * and my_levels is the size of a cross I_c cross b
     */
    n_before   = n_before_op;
    my_levels  = levels_vec*levels_op*n_between;
    n_after    = total_levels/(my_levels*n_before);

    for (i=0;i<levels_op-loop_limit_op;i++){
      /*
       * Since we store our operators as a type and number of levels
       * calculate the actual i,j location for our operator,
       * within its subspace, as well as its values.
       */
      val1 = _get_val_in_subspace(i,op_type_op,-1,&i1,&j1);
      /*
       * Since we are taking a cross I cross b, we
       * I_n_between cross b below
       */
      for (k3=0;k3<n_between;k3++){

        /* The vec pair is op2, and we know it is only one value in one spot in its subspace */
        val2 = 1.0;
        i2   = i_vec;
        j2   = j_vec;

        /* Update i2,j2 with the kroneckor product for I_between */
        i2 = i2 + k3*levels_vec;
        j2 = j2 + k3*levels_vec;
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
        i_comb = levels_vec*n_between*i1 + i2;
        j_comb = levels_vec*n_between*j1 + j2;
        add_to_mat = a*val1*val2;

        _add_to_dense_kron_ij(add_to_mat,i_comb,j_comb,n_before,
                              n_after,my_levels);
      }
    }
  }

  return;
}
