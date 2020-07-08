#include "kron_p.h" //Includes petscmat.h and operators_p.h
#include "quac_p.h"
#include "error_correction.h"
#include "operators.h"
#include "quantum_gates.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

Mat *_DQEC_mats;
int _discrete_ec = 0;

void build_recovery_lin(Mat *recovery_mat,operator error,char commutation_string[],int n_stabilizers,...){

  va_list ap;
  PetscScalar plus_or_minus_1=1.0,scale_val;
  PetscInt i;
  PetscReal fill;
  Mat temp_op_mat, work_mat1, work_mat2, this_stab;
  /*
   * We are calculating the recovery operator, which is defined as:
   *     R = E * (1 +/- M_1)/2 * (1 +/- M_2)/2 * ...
   * where E is the error and M_i is the stabilizer. The +/- is chosen
   * based on whether E commutes or anti-commutes with the stabilizer
   */

  /* Construct our error matrix */
  combine_ops_to_mat(&temp_op_mat,total_levels,1,error);

  /*
   * Copy the error matrix into our recovery matrix
   * This works as an initialization
   */
  MatConvert(temp_op_mat,MATSAME,MAT_INITIAL_MATRIX,recovery_mat);
  MatConvert(temp_op_mat,MATSAME,MAT_INITIAL_MATRIX,&work_mat1);
  va_start(ap,n_stabilizers);

  fill = 1.0;
  /* Loop through stabilizers */
  for (i=0;i<n_stabilizers;i++){
    this_stab = va_arg(ap,Mat);
    /* Look up commutation pattern from commutation_string */
    if (commutation_string[i]=='1') {
      plus_or_minus_1 = 1.0;
    } else if (commutation_string[i]=='0') {
      plus_or_minus_1 = -1.0;
    } else {
      if (nid==0){
        printf("ERROR! commutation_string had a bad character! It can \n");
        printf("       only have 0 or 1!\n");
        exit(0);
      }
    }
    /* Copy our stabilizer */
    MatCopy(this_stab,work_mat1,DIFFERENT_NONZERO_PATTERN);;

    /* Calculate M +/- I */
    MatShift(work_mat1,plus_or_minus_1);

    /* Calculate +/- 0.5 * (M +/- I) */
    scale_val = 0.5 * plus_or_minus_1;
    MatScale(work_mat1,scale_val);

    /* Calculate C * +/- 0.5 * (M +/- I) */
    MatMatMult(*recovery_mat,work_mat1,MAT_INITIAL_MATRIX,fill,&work_mat2);

    /* Copy the result back into recovery_mat */
    MatCopy(work_mat2,*recovery_mat,DIFFERENT_NONZERO_PATTERN);

    /* Free work_mat2 */
    MatDestroy(&work_mat2);
  }
  va_end(ap);
  MatDestroy(&work_mat1);

  return;
}

/*
 * create_stabilizer stores the set of operators which make up a stabilizer
 * Inputs:
 *        int n_ops:  the number of operators in the stabilizer
 *        operators op1, op2,...: the operators that make up the stabilizer
 * Outputs:
 *       stabilizer *stab: a new stabilizer object, which just stores the list of operators
 */

void create_stabilizer(stabilizer *stab,int n_ops,...){
  va_list ap;
  PetscInt i;

  va_start(ap,n_ops);
  (*stab).n_ops = n_ops;
  (*stab).ops = malloc(n_ops*sizeof(struct operator));
  /* Loop through passed in ops and store in list */
  for (i=0;i<n_ops;i++){
    (*stab).ops[i] = va_arg(ap,operator);
  }
  va_end(ap);
  return;
}


/*
 * destroy_stabilizer frees the memory from a stabilizer.
 * Inputs:
 *       stabilizer *stab - pointer to stabilizer to be freed
 */
void destroy_stabilizer(stabilizer *stab){
  free((*stab).ops);
}
/*
 * add_lin_recovery adds a Lindblad L(C) term to the system of equations, where
 * L(C)p = C p C^t - 1/2 (C^t C p + p C^t C)
 * Or, in superoperator space (t = conjugate transpose, T = transpose, * = conjugate)
 * Lp    = C* cross C - 1/2(C^T C* cross I + I cross C^t C) p
 * For this routine, C is a recovery operator, constructed of an error,
 * commutation relations, stabilizers.
 * Inputs:
 *        PetscScalar a:    scalar to multiply L term (note: Full term, not sqrt())
 *        Mat add_to_lin:   mat to make L(C) of
 * Outputs:
 *        none
 */

void add_lin_recovery(PetscScalar a,PetscInt same_rate,operator error,char commutation_string[],int n_stabilizers,...){
  va_list ap;
  PetscScalar mat_scalar,add_to_mat,op_val;
  PetscInt   i,Istart,Iend,this_i,i_stab,j_stab,k_stab,l_stab;
  PetscInt i1,i2,j1,j2,num_nonzero1,num_nonzero2,i_comb,j_comb;
  /*
   * The following arrays are used in C* C calculationsg
   * Maybe this is memory inefficient, but it takes
   * far less memory than the DM, so it should be fine
   */
  PetscScalar this_row1[total_levels],this_row2[total_levels];
  PetscInt row_nonzeros1[total_levels],row_nonzeros2[total_levels];
  stabilizer     *stabs;

  /*
   * We are calculating the recovery operator, which is defined as:
   *     R = E * (1 +/- M_1)/2 * (1 +/- M_2)/2 * ...
   * where E is the error and M_i is the stabilizer. The +/- is chosen
   * based on whether E commutes or anti-commutes with the stabilizer
   *
   * We will directly add the superoperator expanded values into the
   * full_A matrix, never explicitly building R, but rather,
   * building I cross R^t R + (R^t R)* cross I + R* cross R
   */
  PetscLogEventBegin(add_lin_recovery_event,0,0,0,0);
  _check_initialized_A();
  _lindblad_terms = 1;

  if (PetscAbsComplex(a)!=0) {
    MatGetOwnershipRange(full_A,&Istart,&Iend);

    va_start(ap,n_stabilizers);
    stabs = malloc(n_stabilizers*sizeof(struct stabilizer));
    /* Loop through passed in ops and store in list */
    for (i=0;i<n_stabilizers;i++){
      stabs[i] = va_arg(ap,stabilizer);
    }
    va_end(ap);

    /*
     * Construct R^t R. Due to interesting relations among the pauli operators
     * (sig_i * sig_i) = I and sig_i = sig_i^t, as well as the fact that
     * pauli operators from different subspaces commute and the stabilizers
     * themselve commute, R^t R has a rather simple form:
     *
     * R^t R = \prod_i (I + M_i)/2
     *
     * This is almost the same as R, but without the error!
     * This product is very similar to the elementary symmetric polynomials, and there
     * are formula for calculating the values (which we don't make use of at this point,
     * given we plan to have few stabilizers at this time).
     * An example, with 3 stabilizers:
     *
     * R^t R = 1/2^3 (I + M_1 + M_2 + M_3 + M_1*M_2 + M_1*M_3 + M_2*M_3 + M_1*M_2*M_3)
     *
     * Generally, the form is:
     * R^t R = 1/2^n (I + \sum_i M_i + \sum_i<j M_i*M_j + \sum_i<j<k M_i*M_j*M_k + ...)
     *
     * Since all M_i are constructed of tensor products (like I \cross Z \cross Z), and each of
     * the submatrices are very sparse, we can use tricks (like in combine_ops_to_mat) to
     * efficiently generate each of the members of the sum.
     *
     * Due to the form of commutation strings, if all of the error rates for each of
     * the recover operators are the same, you only need to add the U* cross U terms
     */

    mat_scalar = -1/pow(2,n_stabilizers) * 0.5 * a; //Store the common multiplier for all terms, -0.5*a*1/4^n
    /* First, do I cross C^t C and (C^t C)* cross I */
    /*
     * We break it up into different numbers of stabilizers, for the different
     * numbers of terms (i.e., M_i or M_i*M_j, or M_i*M_j*M_k, etc)
     * There should be a general way to do this that supports any number,
     * but we code only to a maximum of 4 for now.
     */
    /* I terms - a * (I cross I + I cross I) = 2*a*I in the total space*/
    MatGetOwnershipRange(full_A,&Istart,&Iend);
    for (i=Istart;i<Iend;i++){
      add_to_mat = 2*mat_scalar;
      MatSetValue(full_A,i,i,add_to_mat,ADD_VALUES);
    }

    if (same_rate==0) {
      /* Single M_i terms */
      for (i=Istart;i<Iend;i++){
        for (i_stab=0;i_stab<2;i_stab++){//n_stabilizers;i_stab++){
          /* First loop through i_stab's ops */
          /* Reset this_i and op_val to identity */
          this_i = i; // The leading index which we check
          op_val = 1.0;
          // Add I cross U
          _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);
          add_to_mat = op_val * mat_scalar;
          /* printf("add_to_mat: %f i: %d this_i: %d\n",add_to_mat,i,this_i); */
          MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);

          // Add  U* cross I
          /* Reset this_i and op_val to identity */
          this_i = i; // The leading index which we check
          op_val = 1.0;
          _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],1);
          add_to_mat = op_val * mat_scalar;
          MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);
        }
      }

      if (n_stabilizers > 1) {
        /* M_i*M_j terms */
        for (i=Istart;i<Iend;i++){
          for (i_stab=0;i_stab<n_stabilizers;i_stab++){
            for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
              /* First loop through i_stab's ops */
              /* Reset this_i and op_val to identity */
              // Add I cross U
              this_i = i; // The leading index which we check
              op_val = 1.0;
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);
              if (op_val!= 0.0) {
                _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],-1);
              }
              add_to_mat = op_val * mat_scalar;
              MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);

              // Add U* cross I
              this_i = i; // The leading index which we check
              op_val = 1.0;
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],1);
              if (op_val!= 0.0) {
                _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],1);
              }
              add_to_mat = op_val * mat_scalar;
              MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);
            }
          }
        }
      }
      if (n_stabilizers>2) {
        /* M_i*M_j*M_k terms */
        for (i=Istart;i<Iend;i++){
          for (i_stab=0;i_stab<n_stabilizers;i_stab++){
            for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
              for (k_stab=j_stab+1;k_stab<n_stabilizers;k_stab++){
                // Add I cross U
                /* Reset this_i and op_val to identity */
                this_i = i; // The leading index which we check
                op_val = 1.0;
                _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);
                if (op_val!= 0.0) {
                  _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],-1);
                }
                if (op_val!= 0.0) {
                  _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab],-1);
                }
                add_to_mat = op_val * mat_scalar;
                MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);

                // Add I cross U
                /* Reset this_i and op_val to identity */
                this_i = i; // The leading index which we check
                op_val = 1.0;
                _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],1);
                if (op_val!= 0.0) {
                  _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],1);
                }
                if (op_val!= 0.0) {
                  _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab],1);
                }
                add_to_mat = op_val * mat_scalar;
                MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);
              }
            }
          }
        }
      }

      if (n_stabilizers>3) {
        /* M_i*M_j*M_k*M_l terms */
        for (i=Istart;i<Iend;i++){
          for (i_stab=0;i_stab<n_stabilizers;i_stab++){
            for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
              for (k_stab=j_stab+1;k_stab<n_stabilizers;k_stab++){
                for (l_stab=k_stab+1;l_stab<n_stabilizers;l_stab++){
                  // Add I cross U
                  /* Reset this_i and op_val to identity */
                  this_i = i; // The leading index which we check
                  op_val = 1.0;
                  _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);
                  if (op_val!= 0.0) {
                    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],-1);
                  }
                  if (op_val!= 0.0) {
                    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab],-1);
                  }
                  if (op_val!= 0.0) {
                    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[l_stab],commutation_string[l_stab],-1);
                  }

                  add_to_mat = op_val * mat_scalar;
                  MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);

                  // Add U* cross I
                  /* Reset this_i and op_val to identity */
                  this_i = i; // The leading index which we check
                  op_val = 1.0;
                  _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],1);
                  if (op_val!= 0.0) {
                    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],1);
                  }
                  if (op_val!= 0.0) {
                    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab],1);
                  }
                  if (op_val!= 0.0) {
                    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[l_stab],commutation_string[l_stab],1);
                  }

                  add_to_mat = op_val * mat_scalar;
                  MatSetValue(full_A,i,this_i,add_to_mat,ADD_VALUES);

                }
              }
            }
          }
        }
      }
    }
    if (n_stabilizers>4) {
      if (nid==0){
        printf("ERROR! A maximum of 4 stabilizers is supported at this time!\n");
        exit(0);
      }
    }

    /*
     * Add (C* cross C) to the superoperator matrix,
     * We expand R* and R, get their respective i,j
     * and use _add_to_PETSc_kron_ij to add the value
     * to the full_A
     *
     *     R = E * (1 +/- M_1)/2 * (1 +/- M_2)/2 * ...
     * or
     *     R = E * \prod_i (1 +/- M_i)/2
     *
     * Similar to the calculation of R^t R above,
     * we expand the product as:
     *
     *     R = E * (I + M_1 + M_2 + M_3 + M_1 * M_2 + ...)
     *
     * a la elementary symmetric polynomials.
     * We take a given i, calculate the value for each
     * of the individual terms, then sum it all up.
     */
    // Store common prefactors. 2*n_stab because C* cross C
    mat_scalar = 1/pow(2,2*n_stabilizers) * a;
    // FIXME Consider distributing this loop in some smart fashion
    /* for (i1=0;i1<total_levels;i1++){ */
    /*   /\* Get the nonzeros for row i1 of C *\/ */
    /*   _get_row_nonzeros(this_row1,row_nonzeros1,&num_nonzero1,i1,error,commutation_string,n_stabilizers,stabs); */
    /*   for (i2=0;i2<total_levels;i2++){ */
    /*     /\* Get the nonzeros for row i2 of C *\/ */
    /*     /\* FIXME: Consider skipping the i1=i2 spot *\/ */
    /*     _get_row_nonzeros(this_row2,row_nonzeros2,&num_nonzero2,i2,error,commutation_string,n_stabilizers,stabs); */
    /*     /\* */
    /*      * Use the general formula for the kronecker product between */
    /*      * two matrices to find the full value */
    /*      *\/ */
    /*     for (j1=0;j1<num_nonzero1;j1++){ */
    /*       for (j2=0;j2<num_nonzero2;j2++){ */
    /*         /\* Get the combind indices *\/ */
    /*         i_comb = total_levels*i1 + i2; */
    /*         j_comb = total_levels*row_nonzeros1[j1] + row_nonzeros2[j2]; */
    /*         add_to_mat = mat_scalar * */
    /*           PetscConjComplex(this_row1[row_nonzeros1[j1]])* */
    /*           this_row2[row_nonzeros2[j2]]; */
    /*         if (i_comb>=Istart&&i_comb<Iend) MatSetValue(full_A,i_comb,j_comb,add_to_mat,ADD_VALUES); */
    /*       } */
    /*     } */

    /*   } */
    /* } */

    for (i=Istart;i<Iend;i++){
      /* Calculate i1, i2 */
      i1 = i/total_levels;
      i2 = i%total_levels;
      _get_row_nonzeros(this_row1,row_nonzeros1,&num_nonzero1,i1,error,commutation_string,n_stabilizers,stabs);
      _get_row_nonzeros(this_row2,row_nonzeros2,&num_nonzero2,i2,error,commutation_string,n_stabilizers,stabs);
      for (j1=0;j1<num_nonzero1;j1++){
        for (j2=0;j2<num_nonzero2;j2++){
          /* Get the combind indices */
          i_comb = total_levels*i1 + i2;
          j_comb = total_levels*row_nonzeros1[j1] + row_nonzeros2[j2];
          add_to_mat = mat_scalar *
            PetscConjComplex(this_row1[row_nonzeros1[j1]])*
            this_row2[row_nonzeros2[j2]];
          if (PetscAbsComplex(add_to_mat)>1e-5){
            MatSetValue(full_A,i_comb,j_comb,add_to_mat,ADD_VALUES);
          }
        }
      }
    }
  }
  PetscLogEventEnd(add_lin_recovery_event,0,0,0,0);
  return;
}

/* Get the j and val from a stabilizer - essentially multiply the ops for a given row
* tensor_control - switch on which superoperator to compute
*                  -1: I cross G
*                   0: G* cross G
*                   1: G* cross I
*/
void _get_this_i_and_val_from_stab(PetscInt *this_i, PetscScalar *op_val,stabilizer stab,
                                   char commutation_char,PetscInt tensor_control){
  PetscInt j,this_j;
  PetscScalar val;
  PetscReal plus_or_minus_1=1.0;

  if (commutation_char=='1') {
    plus_or_minus_1 = 1.0;
  } else if (commutation_char=='0') {
    plus_or_minus_1 = -1.0;
  } else {
    if (nid==0){
      printf("ERROR! commutation_string had a bad character! It can \n");
      printf("       only have 0 or 1!\n");
      exit(0);
    }
  }


  for (j=0;j<stab.n_ops;j++){
    _get_val_j_from_global_i(total_levels,*this_i,stab.ops[j],&this_j,&val,tensor_control); // Get the corresponding j and val
    if (this_j<0) {
      /*
       * Negative j says there is no nonzero value for a given this_i
       * As such, we can immediately break the loop for i
       */
      *op_val = 0.0;
      break;
    } else {
      *this_i = this_j;
      *op_val = *op_val*val;
    }
  }

  if (tensor_control!=0){
    /*
     * We don't need to apply plus_or_minus_1 if tensor
     * control is 0 because we would apply it twice (U* and U),
     * always leading to +1
     */
    *op_val = *op_val*plus_or_minus_1; //Include commutation information
  }


}

/* Gets the nonzeros in a row for a given i, error operator, and stabilizers */
void _get_row_nonzeros(PetscScalar this_row[],PetscInt row_nonzeros[],PetscInt *num_nonzero,PetscInt i,operator error,char commutation_string[],int n_stabilizers,stabilizer stabs[]){
  PetscScalar error_val,op_val;
  PetscInt j,this_i,error_i,i_stab,j_stab,k_stab,l_stab,found;
  /* Get the error terms nonzero for this i */
  _get_val_j_from_global_i(total_levels,i,error,&error_i,&error_val,-1); // Get the corresponding j and val
  this_i = error_i; // The leading index which we check
  op_val = error_val;

  /*
   * Reset num_nonzero. We need not reset the arrays because
   * we will only access the indices described in the list,
   * and those values will be correspondingly updated
   */

  *num_nonzero = 0;
  /* E * I term - since this is the first term, we know we haven't added anything yet*/
  this_row[this_i] = op_val;
  row_nonzeros[*num_nonzero] = this_i;
  *num_nonzero = *num_nonzero + 1;

  /*
   * Single E * M_i terms. The error term is included from the
   * fact tht this_i and op_val are reset to the error
   * operator's values for each new stabilizer.
   */
  for (i_stab=0;i_stab<n_stabilizers;i_stab++){
    /* Reset this_i and op_val to error values */
    this_i = error_i; // The leading index which we check
    op_val = error_val;
    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);
    if (op_val!=0.0) {
      /*
       * Check if we already have a value in this spot.
       * This is a linear search; it should be fine, since
       * these matrices are incredibly sparse. But, consider
       * changing to faster search.
       */
      found = 0;
      for (j=0;j<*num_nonzero;j++){
        if (this_i==row_nonzeros[j]) {
          // Found it!
          found = 1;
          break;
        }
      }
      /* Add to this_row1 */
      if (found==1){
        /* There was a value here, so add to it */
        this_row[this_i] = this_row[this_i] + op_val;
      } else {
        /* This is a new value, so just set */
        this_row[this_i] = op_val;
        row_nonzeros[*num_nonzero] = this_i;
        *num_nonzero = *num_nonzero + 1;
      }
    }
  }

  if (n_stabilizers>1) {
    /* M_i*M_j terms */
    for (i_stab=0;i_stab<n_stabilizers;i_stab++){
      for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
        /* Reset this_i and op_val to error values */
        this_i = error_i; // The leading index which we check
        op_val = error_val;

        _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);

        if (op_val!= 0.0) {
          _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],-1);
        }

        /*
         * Now we have an element of the matrix form of E*M_i*M_j,
         * add it to the row_nonzero value
         */
        if (op_val!=0.0) {
          /*
           * Check if we already have a value in this spot.
           * This is a linear search; it should be fine, since
           * these matrices are incredibly sparse. But, consider
           * changing to faster search.
           */
          found = 0;
          for (j=0;j<*num_nonzero;j++){
            if (this_i==row_nonzeros[j]) {
              // Found it!
              found = 1;
              break;
            }
          }
          /* Add to this_row1 */
          if (found==1){
            /* There was a value here, so add to it */
            this_row[this_i] = this_row[this_i] + op_val;
          } else {
            /* This is a new value, so just set */
            this_row[this_i] = op_val;
            row_nonzeros[*num_nonzero] = this_i;
            *num_nonzero = *num_nonzero + 1;
          }
        }

      }
    }
  }

  if (n_stabilizers>2) {
    /* M_i*M_j*M_k terms */
    for (i_stab=0;i_stab<n_stabilizers;i_stab++){
      for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
        for (k_stab=j_stab+1;k_stab<n_stabilizers;k_stab++){
          /* Reset this_i and op_val to error values */
          this_i = error_i; // The leading index which we check
          op_val = error_val;

          _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);

          if (op_val!= 0.0) {
            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],-1);
          }

          if (op_val!= 0.0) {
            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab],-1);
          }

          /*
           * Now we have an element of the matrix form of E*M_i*M_j,
           * add it to the row_nonzero value
           */
          if (op_val!=0.0) {
            /*
             * Check if we already have a value in this spot.
             * This is a linear search; it should be fine, since
             * these matrices are incredibly sparse. But, consider
             * changing to faster search.
             */
            found = 0;
            for (j=0;j<*num_nonzero;j++){
              if (this_i==row_nonzeros[j]) {
                // Found it!
                found = 1;
                break;
              }
            }
            /* Add to this_row1 */
            if (found==1){
              /* There was a value here, so add to it */
              this_row[this_i] = this_row[this_i] + op_val;
            } else {
              /* This is a new value, so just set */
              this_row[this_i] = op_val;
              row_nonzeros[*num_nonzero] = this_i;
              *num_nonzero = *num_nonzero + 1;
            }
          }
        }
      }
    }
  }

  if (n_stabilizers>3) {
    /* M_i*M_j*M_k*M_l terms */
    for (i_stab=0;i_stab<n_stabilizers;i_stab++){
      for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
        for (k_stab=j_stab+1;k_stab<n_stabilizers;k_stab++){
          for (l_stab=k_stab+1;l_stab<n_stabilizers;l_stab++){
            /* Reset this_i and op_val to error values */
            this_i = error_i; // The leading index which we check
            op_val = error_val;

            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab],-1);

            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab],-1);
            }

            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab],-1);
            }

            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[l_stab],commutation_string[l_stab],-1);
            }

            /*
             * Now we have an element of the matrix form of E*M_i*M_j,
             * add it to the row_nonzero value
             */
            if (op_val!=0.0) {
              /*
               * Check if we already have a value in this spot.
               * This is a linear search; it should be fine, since
               * these matrices are incredibly sparse. But, consider
               * changing to faster search.
               */
              found = 0;
              for (j=0;j<*num_nonzero;j++){
                if (this_i==row_nonzeros[j]) {
                  // Found it!
                  found = 1;
                  break;
                }
              }
              /* Add to this_row1 */
              if (found==1){
                /* There was a value here, so add to it */
                this_row[this_i] = this_row[this_i] + op_val;
              } else {
                /* This is a new value, so just set */
                this_row[this_i] = op_val;
                row_nonzeros[*num_nonzero] = this_i;
                *num_nonzero = *num_nonzero + 1;
              }
            }
          }
        }
      }
    }
  }

  return;
}


/*
 * Create an encoder. The first of the passed in systems is assumed to be the
 * qubit that is encoded/decoded to.
 */
void create_encoded_qubit(encoded_qubit *new_encoder,encoder_type my_encoder_type,...){
  PetscInt num_qubits,i,qubit;
  va_list ap;

  if(my_encoder_type==NONE){
    num_qubits = 1;
    (*new_encoder).num_qubits = num_qubits;
    (*new_encoder).my_encoder_type = NONE;
    (*new_encoder).qubits = malloc(num_qubits*sizeof(PetscInt));
    va_start(ap,num_qubits);
    for (i=0;i<num_qubits;i++){
      qubit = va_arg(ap,int);
      (*new_encoder).qubits[i] = qubit;
    }
    create_circuit(&((*new_encoder).encoder_circuit),1);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,EYE,(*new_encoder).qubits[0]);

    create_circuit(&((*new_encoder).decoder_circuit),1);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,EYE,(*new_encoder).qubits[0]);

  } else if(my_encoder_type==BIT){
    num_qubits = 3;
    (*new_encoder).num_qubits = num_qubits;
    (*new_encoder).my_encoder_type = BIT;
    (*new_encoder).qubits = malloc(num_qubits*sizeof(PetscInt));
    va_start(ap,num_qubits);
    for (i=0;i<num_qubits;i++){
      qubit = va_arg(ap,int);
      (*new_encoder).qubits[i] = qubit;
    }
    create_circuit(&((*new_encoder).encoder_circuit),2);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[2]);

    create_circuit(&((*new_encoder).decoder_circuit),2);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[1]);

  } else if(my_encoder_type==PHASE){

    num_qubits = 3;
    (*new_encoder).num_qubits = num_qubits;
    (*new_encoder).my_encoder_type = PHASE;
    (*new_encoder).qubits = malloc(num_qubits*sizeof(PetscInt));
    va_start(ap,num_qubits);
    for (i=0;i<num_qubits;i++){
      qubit = va_arg(ap,int);
      (*new_encoder).qubits[i] = qubit;
    }
    create_circuit(&((*new_encoder).encoder_circuit),5);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[2]);

    create_circuit(&((*new_encoder).decoder_circuit),5);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CNOT,(*new_encoder).qubits[0],(*new_encoder).qubits[1]);

  } else if(my_encoder_type==FIVE){
    num_qubits = 5;
    (*new_encoder).num_qubits = num_qubits;
    (*new_encoder).my_encoder_type = FIVE;
    (*new_encoder).qubits = malloc(num_qubits*sizeof(PetscInt));
    va_start(ap,num_qubits);
    for (i=0;i<num_qubits;i++){
      qubit = va_arg(ap,int);
      (*new_encoder).qubits[i] = qubit;
    }

    create_circuit(&((*new_encoder).encoder_circuit),21);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CmZ,(*new_encoder).qubits[0],(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[0],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[4],(*new_encoder).qubits[3]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[4],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CXZ,(*new_encoder).qubits[4],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,SIGMAZ,(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[3]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[3],(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[3],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CNOT,(*new_encoder).qubits[3],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[2],(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[2],(*new_encoder).qubits[3]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CNOT,(*new_encoder).qubits[2],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[1],(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CZ,(*new_encoder).qubits[1],(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,CXZ,(*new_encoder).qubits[1],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).encoder_circuit),1.0,SIGMAZ,(*new_encoder).qubits[1]);


    create_circuit(&((*new_encoder).decoder_circuit),21);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,SIGMAZ,(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZX,(*new_encoder).qubits[1],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[1],(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[1],(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CNOT,(*new_encoder).qubits[2],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[2],(*new_encoder).qubits[3]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[2],(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CNOT,(*new_encoder).qubits[3],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[3],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[3],(*new_encoder).qubits[2]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[3]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,SIGMAZ,(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZX,(*new_encoder).qubits[4],(*new_encoder).qubits[0]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[4],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[4],(*new_encoder).qubits[3]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,HADAMARD,(*new_encoder).qubits[4]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CZ,(*new_encoder).qubits[0],(*new_encoder).qubits[1]);
    add_gate_to_circuit(&((*new_encoder).decoder_circuit),1.0,CmZ,(*new_encoder).qubits[0],(*new_encoder).qubits[4]);

  } else {
    if (nid==0){
      printf("ERROR! Encoding type not understood!\n");
      exit(0);
    }
  }

  return;
}

//The ... are the encoders, assumes we decode to the first of the list in the encoder
void add_encoded_gate_to_circuit(circuit *circ,PetscReal time,gate_type my_gate_type,...){
  int num_qubits=0,qubit,i;
  va_list ap;
  encoded_qubit *encoders;
  PetscReal theta;

  if (_gate_array_initialized==0){
    //Initialize the array of gate function pointers
    _initialize_gate_function_array();
    _gate_array_initialized = 1;
  }

  _check_gate_type(my_gate_type,&num_qubits);

  if ((*circ).num_gates==(*circ).gate_list_size){
    if (nid==0){
      printf("ERROR! Gate list not large enough (encoded)!\n");
      exit(1);
    }
  }

  PetscMalloc1(num_qubits,&encoders);

  if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ) {
    va_start(ap,num_qubits+1); //FIXME This +1 is probably useless?
  } else {
    va_start(ap,num_qubits);
  }

  //First, get the encoders
  for (i=0;i<num_qubits;i++){
    encoders[i] = va_arg(ap,encoded_qubit);
  }

  // Now, add the decoding circuit(s) to the output circuit
  for (i=0;i<num_qubits;i++){
    // Only add a decoder if we have an encoded qubit
    if (encoders[i].my_encoder_type!=NONE){
      add_circuit_to_circuit(circ,encoders[i].decoder_circuit,time);
    }
  }

  // FIXME: Call add_gate_to_circuit here
  // Store arguments for the logical operation in list
  (*circ).gate_list[(*circ).num_gates].qubit_numbers = malloc(num_qubits*sizeof(int));
  (*circ).gate_list[(*circ).num_gates].time = time;
  (*circ).gate_list[(*circ).num_gates].my_gate_type = my_gate_type;
  (*circ).gate_list[(*circ).num_gates]._get_val_j_from_global_i = _get_val_j_functions_gates[my_gate_type+_min_gate_enum];

  // Loop through and store qubits
  for (i=0;i<num_qubits;i++){
    qubit = encoders[i].qubits[0]; //assumes we decode to the first of the list
    (*circ).gate_list[(*circ).num_gates].qubit_numbers[i] = qubit;
  }

  if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ) {
    //Get the theta parameter from the last argument passed in
    theta = va_arg(ap,PetscReal);
    (*circ).gate_list[(*circ).num_gates].theta = theta;
  } else {
    //Set theta to 0
    (*circ).gate_list[(*circ).num_gates].theta = 0;
  }

  (*circ).num_gates = (*circ).num_gates + 1;

  // Now, reencode our qubits
  for (i=0;i<num_qubits;i++){
    // Only add a decoder if we have an encoded qubit
    if (encoders[i].my_encoder_type!=NONE){
      add_circuit_to_circuit(circ,encoders[i].encoder_circuit,time);
    }
  }

  return;
}

//The ... are the encoders, assumes we encode from the first of the list in the encoder
void encode_state(Vec rho,PetscInt num_logical_qubits,...){
  PetscInt i,j;
  va_list ap;
  encoded_qubit this_qubit;
  va_start(ap,num_logical_qubits);

  //Loop through the qubit, multiplying rho by the encoding circuit
  for (i=0;i<num_logical_qubits;i++){
    this_qubit = va_arg(ap,encoded_qubit);
    for (j=0;j<this_qubit.encoder_circuit.num_gates;j++) {
      _apply_gate(this_qubit.encoder_circuit.gate_list[j],rho);
    }
  }

  return;
}

//The ... are the encoders, assumes we encode from the first of the list in the encoder
void decode_state(Vec rho,PetscInt num_logical_qubits,...){
  PetscInt i,j;
  va_list ap;
  encoded_qubit this_qubit;

  va_start(ap,num_logical_qubits);

  //Loop through the qubit, multiplying rho by the encoding circuit
  for (i=0;i<num_logical_qubits;i++){
    this_qubit = va_arg(ap,encoded_qubit);
    for (j=0;j<this_qubit.decoder_circuit.num_gates;j++) {
      _apply_gate(this_qubit.decoder_circuit.gate_list[j],rho);
    }
  }

  return;
}


void add_continuous_error_correction(encoded_qubit this_qubit,PetscReal correction_rate){
  stabilizer     S1,S2,S3,S4;
  operator       qubit0,qubit1,qubit2,qubit3,qubit4;

  if (this_qubit.my_encoder_type == NONE){
    //No encoding, no error correction needed
  } else if (this_qubit.my_encoder_type == BIT){
    qubit0 = subsystem_list[this_qubit.qubits[0]];
    qubit1 = subsystem_list[this_qubit.qubits[1]];
    qubit2 = subsystem_list[this_qubit.qubits[2]];

    create_stabilizer(&S1,2,qubit0->sig_z,qubit1->sig_z);
    create_stabilizer(&S2,2,qubit1->sig_z,qubit2->sig_z);

    add_lin_recovery(correction_rate,1,qubit0->eye,(char *)"11",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit0->sig_x,(char *)"01",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit1->sig_x,(char *)"00",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit2->sig_x,(char *)"10",2,S1,S2);

    destroy_stabilizer(&S1);
    destroy_stabilizer(&S2);

  } else if (this_qubit.my_encoder_type == PHASE){
    qubit0 = subsystem_list[this_qubit.qubits[0]];
    qubit1 = subsystem_list[this_qubit.qubits[1]];
    qubit2 = subsystem_list[this_qubit.qubits[2]];

    create_stabilizer(&S1,2,qubit0->sig_x,qubit1->sig_x);
    create_stabilizer(&S2,2,qubit1->sig_x,qubit2->sig_x);

    add_lin_recovery(correction_rate,1,qubit0->eye,(char *)"11",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit0->sig_z,(char *)"01",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit1->sig_z,(char *)"00",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit2->sig_z,(char *)"10",2,S1,S2);

    destroy_stabilizer(&S1);
    destroy_stabilizer(&S2);

  } else if (this_qubit.my_encoder_type == FIVE) {
    qubit0 = subsystem_list[this_qubit.qubits[0]];
    qubit1 = subsystem_list[this_qubit.qubits[1]];
    qubit2 = subsystem_list[this_qubit.qubits[2]];
    qubit3 = subsystem_list[this_qubit.qubits[3]];
    qubit4 = subsystem_list[this_qubit.qubits[4]];

    create_stabilizer(&S1,4,qubit0->sig_x,qubit1->sig_z,qubit2->sig_z,qubit3->sig_x);
    create_stabilizer(&S2,4,qubit1->sig_x,qubit2->sig_z,qubit3->sig_z,qubit4->sig_x);
    create_stabilizer(&S3,4,qubit2->sig_x,qubit3->sig_z,qubit4->sig_z,qubit0->sig_x);
    create_stabilizer(&S4,4,qubit3->sig_x,qubit4->sig_z,qubit0->sig_z,qubit1->sig_x);

    add_lin_recovery(correction_rate,1,qubit0->eye,(char *)"1111",4,S1,S2,S3,S4);

    //Qubit 0 errors
    add_lin_recovery(correction_rate,1,qubit0->sig_x,(char *)"1110",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit0->sig_y,(char *)"0100",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit0->sig_z,(char *)"0101",4,S1,S2,S3,S4);

    //Qubit 1 errors
    add_lin_recovery(correction_rate,1,qubit1->sig_x,(char *)"0111",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit1->sig_y,(char *)"0010",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit1->sig_z,(char *)"1010",4,S1,S2,S3,S4);

    //Qubit 2 errors
    add_lin_recovery(correction_rate,1,qubit2->sig_x,(char *)"0011",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit2->sig_y,(char *)"1101",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit2->sig_z,(char *)"0001",4,S1,S2,S3,S4);

    //Qubit 3 errors
    add_lin_recovery(correction_rate,1,qubit3->sig_x,(char *)"1001",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit3->sig_y,(char *)"0110",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit3->sig_z,(char *)"0000",4,S1,S2,S3,S4);

    //Qubit 4 errors
    add_lin_recovery(correction_rate,1,qubit4->sig_x,(char *)"1100",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit4->sig_y,(char *)"1011",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit4->sig_z,(char *)"1000",4,S1,S2,S3,S4);

    destroy_stabilizer(&S1);
    destroy_stabilizer(&S2);
    destroy_stabilizer(&S3);
    destroy_stabilizer(&S4);

  } else {
    if (nid==0){
      printf("ERROR! Encoder type not understood!\n");
      exit(1);
    }
  }
  return;
}

void add_discrete_error_correction(encoded_qubit this_qubit,PetscReal correction_rate){
  stabilizer     S1,S2,S3,S4;
  operator       qubit0,qubit1,qubit2,qubit3,qubit4;
  if (this_qubit.my_encoder_type == NONE){
    //No encoding, no error correction needed
  } else if (this_qubit.my_encoder_type == BIT){
    qubit0 = subsystem_list[this_qubit.qubits[0]];
    qubit1 = subsystem_list[this_qubit.qubits[1]];
    qubit2 = subsystem_list[this_qubit.qubits[2]];

    create_stabilizer(&S1,2,qubit0->sig_z,qubit1->sig_z);
    create_stabilizer(&S2,2,qubit1->sig_z,qubit2->sig_z);

    add_lin_recovery(correction_rate,1,qubit0->eye,(char *)"11",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit0->sig_x,(char *)"01",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit1->sig_x,(char *)"00",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit2->sig_x,(char *)"10",2,S1,S2);

    destroy_stabilizer(&S1);
    destroy_stabilizer(&S2);
  } else if (this_qubit.my_encoder_type == PHASE){
    qubit0 = subsystem_list[this_qubit.qubits[0]];
    qubit1 = subsystem_list[this_qubit.qubits[1]];
    qubit2 = subsystem_list[this_qubit.qubits[2]];

    create_stabilizer(&S1,2,qubit0->sig_x,qubit1->sig_x);
    create_stabilizer(&S2,2,qubit1->sig_x,qubit2->sig_x);
    add_lin_recovery(correction_rate,1,qubit0->eye,(char *)"11",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit0->sig_z,(char *)"01",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit1->sig_z,(char *)"00",2,S1,S2);
    add_lin_recovery(correction_rate,1,qubit2->sig_z,(char *)"10",2,S1,S2);

    destroy_stabilizer(&S1);
    destroy_stabilizer(&S2);

  } else if (this_qubit.my_encoder_type == FIVE) {
    qubit0 = subsystem_list[this_qubit.qubits[0]];
    qubit1 = subsystem_list[this_qubit.qubits[1]];
    qubit2 = subsystem_list[this_qubit.qubits[2]];
    qubit3 = subsystem_list[this_qubit.qubits[3]];
    qubit4 = subsystem_list[this_qubit.qubits[4]];

    create_stabilizer(&S1,4,qubit0->sig_x,qubit1->sig_z,qubit2->sig_z,qubit3->sig_x);
    create_stabilizer(&S2,4,qubit1->sig_x,qubit2->sig_z,qubit3->sig_z,qubit4->sig_x);
    create_stabilizer(&S3,4,qubit2->sig_x,qubit3->sig_z,qubit4->sig_z,qubit0->sig_x);
    create_stabilizer(&S4,4,qubit3->sig_x,qubit4->sig_z,qubit0->sig_z,qubit1->sig_x);

    add_lin_recovery(correction_rate,1,qubit0->eye,(char *)"1111",4,S1,S2,S3,S4);

    //Qubit 0 errors
    add_lin_recovery(correction_rate,1,qubit0->sig_x,(char *)"1110",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit0->sig_y,(char *)"0100",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit0->sig_z,(char *)"0101",4,S1,S2,S3,S4);

    //Qubit 1 errors
    add_lin_recovery(correction_rate,1,qubit1->sig_x,(char *)"0111",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit1->sig_y,(char *)"0010",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit1->sig_z,(char *)"1010",4,S1,S2,S3,S4);

    //Qubit 2 errors
    add_lin_recovery(correction_rate,1,qubit2->sig_x,(char *)"0011",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit2->sig_y,(char *)"1101",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit2->sig_z,(char *)"0001",4,S1,S2,S3,S4);

    //Qubit 3 errors
    add_lin_recovery(correction_rate,1,qubit3->sig_x,(char *)"1001",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit3->sig_y,(char *)"0110",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit3->sig_z,(char *)"0000",4,S1,S2,S3,S4);

    //Qubit 4 errors
    add_lin_recovery(correction_rate,1,qubit4->sig_x,(char *)"1100",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit4->sig_y,(char *)"1011",4,S1,S2,S3,S4);
    add_lin_recovery(correction_rate,1,qubit4->sig_z,(char *)"1000",4,S1,S2,S3,S4);

    destroy_stabilizer(&S1);
    destroy_stabilizer(&S2);
    destroy_stabilizer(&S3);
    destroy_stabilizer(&S4);

  } else {
    if (nid==0){
      printf("ERROR! Encoder type not understood!\n");
      exit(1);
    }
  }
  return;
}


/* EventFunction is one step in Petsc to apply some action at a specific time.
 * This function checks to see if an event has happened.
 */
PetscErrorCode _DQEC_EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx) {
  /* Check if the time has passed a gate */

  /* /\* We signal that we passed the time by returning a negative number *\/ */
  /* for (i=0;i<_num_DQEC;i++){ */
  /*   fvalue[i] = _correction_time[i] - t; */
  /*   if (fvalue[i]<0){ */
  /*     _correction_time[i] = _correction_time[i] + _correction_dt[i]; */
  /*   } */
  /* } */

  return(0);
}

/* PostEventFunction is the other step in Petsc. If an event has happend, petsc will call this function
 * to apply that event.
*/
PetscErrorCode _DQEC_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forward,void* ctx) {
  PetscInt i,i_ev;
  Vec tmp_answer;

  VecDuplicate(U,&tmp_answer);

  if (nevents) {
    //Loop through events
    for (i_ev=0;i_ev<nevents;i_ev++){
      MatMult(_DQEC_mats[i],U,tmp_answer);
      VecCopy(tmp_answer,U);
    }
    VecDestroy(&tmp_answer);
  }

  TSSetSolution(ts,U);
  return(0);
}

//Take an old circuit and encode it
void encode_circuit(circuit old_circ,circuit *encoded_circ,PetscInt num_encoders,...){
  PetscInt i,j;
  int num_qubits=0;
  PetscReal time,theta;
  va_list ap;
  gate_type my_gate_type;
  encoded_qubit encoders[50]; //50 is more systems than we will be able to do
  int qubit_numbers[50];

  va_start(ap,num_encoders);
  for (i=0;i<num_encoders;i++){
    encoders[i] = va_arg(ap,encoded_qubit);
  }

  for (i=0;i<old_circ.num_gates;i++){
    time = old_circ.gate_list[i].time;
    my_gate_type = old_circ.gate_list[i].my_gate_type;
    _check_gate_type(my_gate_type,&num_qubits);
    theta = old_circ.gate_list[i].theta;
    for (j=0;j<num_qubits;j++){
      qubit_numbers[j] = old_circ.gate_list[i].qubit_numbers[j];
    }
    if (num_qubits==1){
      // Get the encoder for that qubit
      add_encoded_gate_to_circuit(encoded_circ,time,my_gate_type,encoders[qubit_numbers[0]],theta);
    } else {
      add_encoded_gate_to_circuit(encoded_circ,time,my_gate_type,
                                  encoders[qubit_numbers[0]],encoders[qubit_numbers[1]]);
    }
  }
  return;
}

