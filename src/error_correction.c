#include "kron_p.h" //Includes petscmat.h and operators_p.h
#include "quac_p.h"
#include "error_correction.h"
#include "operators.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>


void build_recovery_lin(Mat *recovery_mat,operator error,char commutation_string[],int n_stabilizers,...){

  va_list ap;
  PetscScalar plus_or_minus_1,scale_val;
  PetscInt i,dim;
  PetscReal fill;
  Mat temp_op_mat, work_mat1, work_mat2, this_stab;
  /*
   * We are calculating the recovery operator, which is defined as:
   *     R = E * (1 +/- M_1)/2 * (1 +/- M_2)/2 * ...
   * where E is the error and M_i is the stabilizer. The +/- is chosen
   * based on whether E commutes or anti-commutes with the stabilizer
   */

  /* Construct our error matrix */
  combine_ops_to_mat(&temp_op_mat,1,error);

  dim = total_levels;

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
  free(*stab->ops);
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

void add_lin_recovery(PetscScalar a,operator error,char commutation_string[],int n_stabilizers,...){
  va_list ap;
  PetscReal   plus_or_minus_1;
  PetscScalar mat_scalar,add_to_mat,op_val,val,error_val,error_i;
  PetscInt   i,j,Istart,Iend,this_i,this_j,i_stab,j_stab,k_stab,l_stab;
  PetscInt i1,i2,j1,j2,num_nonzero1,num_nonzero2,i_comb,j_comb;
  /*
   * The following arrays are used in C* C calculationsg
   * Maybe this is memory inefficient, but it takes
   * far less memory than the DM, so it should be fine
   */
  PetscScalar this_row1[total_levels],this_row2[total_levels];
  PetscInt row_nonzeros1[total_levels],row_nonzeros2[total_levels];
  stabilizer     *stabs;

  MatGetOwnershipRange(full_A,&Istart,&Iend);
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

  _check_initialized_A();
  _lindblad_terms = 1;


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
   */

  mat_scalar = -1/pow(2,n_stabilizers) * 0.5 * a; //Store the common multiplier for all terms, -0.5*a*1/4^n

  /* First, do I cross C^t C and (C^t C)* cross I */

  /*
   * We break it up into different numbers of stabilizers, for the different
   * numbers of terms (i.e., M_i or M_i*M_j, or M_i*M_j*M_k, etc)
   * There should be a general way to do this that supports any number,
   * but we code only to a maxmimum of 4 for now.
   */

  /* I terms - a * (I cross I + I cross I) = 2*a*I in the total space*/
  MatGetOwnershipRange(full_A,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    add_to_mat = 2*mat_scalar;
    MatSetValue(full_A,i,i,add_to_mat,ADD_VALUES);
  }

  /* Single M_i terms */
  // FIXME Consider distributing this loop in some smart fashion
  for (i=0;i<total_levels;i++){
    for (i_stab=0;i_stab<n_stabilizers;i_stab++){
      /* First loop through i_stab's ops */
      /* Reset this_i and op_val to identity */
      this_i = i; // The leading index which we check
      op_val = 1.0;
      _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);
      add_to_mat = op_val * mat_scalar; // Include common prefactor terms

      /*
       * Now we have the matrix form of M_i_stab, but we need still need to
       * expand it to add it to the superoperator space.
       */
      /* I cross C^t C part */
      _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,total_levels,1,total_levels);

      /* (C^t C)* cross I part */
      add_to_mat = PetscConjComplex(add_to_mat);
      _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,1,total_levels,total_levels);
    }
  }

  if (n_stabilizers > 1) {
    /* M_i*M_j terms */
    for (i=0;i<total_levels;i++){
      for (i_stab=0;i_stab<n_stabilizers;i_stab++){
        for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
          /* Reset this_i and op_val to identity */
          this_i = i; // The leading index which we check
          op_val = 1.0;
          _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);
          if (op_val!= 0.0) {
            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab]);
          }
          add_to_mat = op_val * mat_scalar; // Include common prefactor terms
          /*
           * Now we have an element of the matrix form of M_i*M_j but we need still need to
           * expand it to add it to the superoperator space.
           */
          /* I cross C^t C part */
          _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,total_levels,1,total_levels);

          /* (C^t C)* cross I part */
          add_to_mat = PetscConjComplex(add_to_mat);
          _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,1,total_levels,total_levels);
        }
      }
    }
  }
  if (n_stabilizers>2) {
    /* M_i*M_j*M_k terms */
    for (i=0;i<total_levels;i++){
      for (i_stab=0;i_stab<n_stabilizers;i_stab++){
        for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
          for (k_stab=j_stab+1;k_stab<n_stabilizers;k_stab++){
            /* Reset this_i and op_val to identity */
            this_i = i; // The leading index which we check
            op_val = 1.0;
            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);
            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab]);
            }
            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab]);
            }

            add_to_mat = op_val * mat_scalar; // Include common prefactor terms
            /*
             * Now we have an element of the matrix form of M_i*M_j but we need still need to
             * expand it to add it to the superoperator space.
             */
            /* I cross C^t C part */
            _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,total_levels,1,total_levels);

            /* (C^t C)* cross I part */
            add_to_mat = PetscConjComplex(add_to_mat);
            _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,1,total_levels,total_levels);
          }
        }
      }
    }
  }

  if (n_stabilizers>3) {
    /* M_i*M_j*M_k*M_l terms */
    for (i=0;i<total_levels;i++){
      for (i_stab=0;i_stab<n_stabilizers;i_stab++){
        for (j_stab=i_stab+1;j_stab<n_stabilizers;j_stab++){
          for (k_stab=j_stab+1;k_stab<n_stabilizers;k_stab++){
            for (l_stab=k_stab+1;l_stab<n_stabilizers;l_stab++){
              /* Reset this_i and op_val to identity */
              this_i = i; // The leading index which we check
              op_val = 1.0;
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);
              if (op_val!= 0.0) {
                _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab]);
              }
              if (op_val!= 0.0) {
                _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab]);
              }
              if (op_val!= 0.0) {
                _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[l_stab],commutation_string[l_stab]);
              }

              add_to_mat = op_val * mat_scalar; // Include common prefactor terms
              /*
               * Now we have an element of the matrix form of M_i*M_j but we need still need to
               * expand it to add it to the superoperator space.
               */
              /* I cross C^t C part */
              _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,total_levels,1,total_levels);

              /* (C^t C)* cross I part */
              add_to_mat = PetscConjComplex(add_to_mat);
              _add_to_PETSc_kron_ij(full_A,add_to_mat,i,this_i,1,total_levels,total_levels);
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
  for (i1=0;i1<total_levels;i1++){
    /* Get the nonzeros for row i1 of C */
    _get_row_nonzeros(this_row1,row_nonzeros1,&num_nonzero1,i1,error,commutation_string,n_stabilizers,stabs);
    for (i2=0;i2<total_levels;i2++){
      /* Get the nonzeros for row i2 of C */
      /* FIXME: Consider skipping the i1=i2 spot */
      _get_row_nonzeros(this_row2,row_nonzeros2,&num_nonzero2,i2,error,commutation_string,n_stabilizers,stabs);
      /*
       * Use the general formula for the kronecker product between
       * two matrices to find the full value
       */
      for (j1=0;j1<num_nonzero1;j1++){
        for (j2=0;j2<num_nonzero2;j2++){
          /* Get the combind indices */
          i_comb = total_levels*i1 + i2;
          j_comb = total_levels*row_nonzeros1[j1] + row_nonzeros2[j2];
          add_to_mat = mat_scalar *
            PetscConjComplex(this_row1[row_nonzeros1[j1]])*
            this_row2[row_nonzeros2[j2]];
          if (i_comb>=Istart&&i_comb<Iend) MatSetValue(full_A,i_comb,j_comb,add_to_mat,ADD_VALUES);
        }
      }

    }
  }

  return;
}

/* Get the j and val from a stabilizer - essentially multiply the ops for a given row */
void _get_this_i_and_val_from_stab(PetscInt *this_i, PetscScalar *op_val,stabilizer stab,char commutation_char){
  PetscInt j,this_j;
  PetscScalar val;
  PetscReal plus_or_minus_1;

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
    _get_val_j_from_global_i(*this_i,stab.ops[j],&this_j,&val); // Get the corresponding j and val
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
  *op_val = *op_val*plus_or_minus_1; //Include commutation information

}

/* Gets the nonzeros in a row for a given i, error operator, and stabilizers */
void _get_row_nonzeros(PetscScalar this_row[],PetscInt row_nonzeros[],PetscInt *num_nonzero,PetscInt i,operator error,char commutation_string[],int n_stabilizers,stabilizer stabs[]){
  PetscScalar error_val,op_val,val;
  PetscReal   plus_or_minus_1;
  PetscInt j,this_i,this_j,error_i,i_stab,j_stab,k_stab,l_stab,found;
  /* Get the error terms nonzero for this i */
  _get_val_j_from_global_i(i,error,&error_i,&error_val); // Get the corresponding j and val
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
    _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);
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

        _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);

        if (op_val!= 0.0) {
          _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab]);
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

          _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);

          if (op_val!= 0.0) {
            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab]);
          }

          if (op_val!= 0.0) {
            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab]);
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

            _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[i_stab],commutation_string[i_stab]);

            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[j_stab],commutation_string[j_stab]);
            }

            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[k_stab],commutation_string[k_stab]);
            }

            if (op_val!= 0.0) {
              _get_this_i_and_val_from_stab(&this_i,&op_val,stabs[l_stab],commutation_string[l_stab]);
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
