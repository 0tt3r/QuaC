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
  if (my_op_type==NUMBER||my_op_type==IDENTITY){
    /* Number and identity operators needs to loop through the full my_levels*/
    loop_limit = 0;
  } else if (my_op_type==VEC){
    /*
     * Vec operators have only one value in their subspace,
     * so the loop size is 1 and loop_limit is my_levels-1
     */
    loop_limit = my_levels-1;
  } else if (my_op_type==SIGMA_X||my_op_type==SIGMA_Y||my_op_type==SIGMA_Z){
    if (my_levels!=2) {
      if (nid==0){
        printf("ERROR! Pauli Operators are only defined for qubits\n");
        exit(0);
      }
    }
    /* Qubit Pauli Operators also need to loop through all my_levels */
    loop_limit = 0;
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

PetscScalar _get_val_in_subspace(long i,op_type my_op_type,int position,long *i_op,long *j_op){
  PetscScalar val=1.0;
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
  } else if (my_op_type==IDENTITY){
    /* Number operator */
    *i_op = i;
    *j_op = i;
    val   = 1.0;
  } else if (my_op_type==RAISE){
    /* Raising operator */
    *i_op = i+1;
    *j_op = i;
    val  = sqrt((double)i+1);
  } else if (my_op_type==SIGMA_X){
    if (i==0) {
      *i_op = 0;
      *j_op = 1;
      val = 1.0;
    } else if (i==1) {
      *i_op = 1;
      *j_op = 0;
      val = 1.0;
    } else {
      if (nid==0){
        printf("ERROR! Pauli Operators are only defined for qubits\n");
        exit(0);
      }
    }
    val = 1.0;
  } else if (my_op_type==SIGMA_Y){
    if (i==0) {
      *i_op = 0;
      *j_op = 1;
      val = -PETSC_i;
    } else if (i==1) {
      *i_op = 1;
      *j_op = 0;
      val = PETSC_i;
    } else {
      if (nid==0){
        printf("ERROR! Pauli Operators are only defined for qubits\n");
        exit(0);
      }
    }

  } else if (my_op_type==SIGMA_Z){
    if (i==0) {
      *i_op = 0;
      *j_op = 0;
      val = 1.0;
    } else if (i==1) {
      *i_op = 1;
      *j_op = 1;
      val = -1.0;
    } else {
      if (nid==0){
        printf("ERROR! Pauli Operators are only defined for qubits\n");
        exit(0);
      }
    }
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
 * Given a list of operators, row index j, and tensor_control variable, unroil
 * the operators to get the column index and value. -1 means the value is 0.
 */
void _get_val_j_ops(PetscScalar *val_out, PetscInt *j_out,PetscInt tot_levels,PetscInt num_ops,
                    operator *ops,tensor_control_enum tensor_control){
  PetscInt this_j,j,j2;
  PetscScalar val,tmp_val;
  operator this_op1,this_op2;
  this_j = *j_out;
  val = 1.0;

  for (j=0;j<num_ops;j++){
    this_op1 = ops[j];
    if(this_op1->my_op_type==VEC){
      /*
       * Since this is a VEC operator, the next operator must also
       * be a VEC operator; it is assumed they always come in pairs.
       */
      this_op2 = ops[j+1];
      if (this_op2->my_op_type!=VEC){
        if (nid==0){
          printf("ERROR! VEC operators must come in pairs in _get_val_j_ops\n");
          exit(0);
        }
      }
      //Increment j
      j=j+1;

      //-1 means that it was 0 on a past operator multiplication, so we skip it if it is -1
      if (this_j!=-1){
        //Get val
        _get_val_j_from_global_i_vec_vec(tot_levels,this_j,this_op1,this_op2,&j2,&tmp_val,tensor_control);
        this_j = j2;
        val = tmp_val * val;
      }
    } else {
      //Normal operator
      if (this_j!=-1){
        _get_val_j_from_global_i(tot_levels,this_j,this_op1,&j2,&tmp_val,tensor_control);
        this_j = j2;
        val = tmp_val * val;
      }

    }
  }

  *val_out = val;
  *j_out   = this_j;
  return;
}

/*
 * _get_val_j_from_global_i returns the val and global j for a given global i.
 * If there is no nonzero value for a given i, it returns a negative j
 * Inputs:
 *      PetscInt tot_levels total hilbert space size
 *      long i:             global i
 *      operator:           operator to get
 *      tensor_control - switch on which superoperator to compute
 *                          -1: I cross G or just G (the difference is controlled by the passed in i's, but
 *                                           the internal logic is exactly the same)
 *                           0: G* cross G
 *                           1: G* cross I
 * Outputs:
 *      long *j:            global j for nonzero of given i; or negative if none
 *      double *val:        value of op for global i,j
  */

void _get_val_j_from_global_i(PetscInt tot_levels,PetscInt i,operator this_op,PetscInt *j,PetscScalar *val,tensor_control_enum tensor_control){
  PetscInt i_sub,n_after,tmp_int,k1,k2,extra_after,j_i1,j_i2,i1,i2;
  PetscScalar val_i1,val_i2;

  /*
   * We store our operators as a type and number of levels;
   * we use the stored information to calculate the global j location
   * and nonzero value for a give global i
   *
   * If it is a lowering operator, it is super diagonal.
   * If it is a number operator, it is diagonal.
   * If it is a raising operator, it is sub diagonal.
   */

  if (tensor_control!=TENSOR_GG) {
    if (tensor_control==TENSOR_GI) {
      extra_after = tot_levels;
    } else {
      extra_after = 1;
    }

    n_after = tot_levels/(this_op->my_levels*this_op->n_before)*extra_after;
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
    } else if (this_op->my_op_type==SIGMA_X){
      /*
       * SIGMA_X
       * if (i==1)
       *    i = 1 * n_af + k1 + k2*n_me*n_af
       *    j = 0 * n_af + k1 + k2*n_me*n_af
       * if (i==0)
       *    i = 0 * n_af + k1 + k2*n_l*n_af
       *    j = 1 * n_af + k1 + k2*n_l*n_af
       * We work out k1 and k2 from i to get j.
       */
      if (i_sub==0) {
        tmp_int = i - 0 * n_after;
        k2      = tmp_int/(this_op->my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(this_op->my_levels*n_after);
        *j = (0 + 1) * n_after + k1 + k2*this_op->my_levels*n_after;
        *val   = 1.0;
      } else if (i_sub==1) {
        tmp_int = i - (0+1) * n_after;
        k2      = tmp_int/(this_op->my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(this_op->my_levels*n_after);
        *j = 0 * n_after + k1 + k2*this_op->my_levels*n_after;
        *val   = 1.0;
      } else {
        if (nid==0){
          printf("ERROR! Pauli Operators are only defined for qubits\n");
          exit(0);
        }
      }
    } else if (this_op->my_op_type==SIGMA_Y){
      /*
       * SIGMA_Y
       * if (i==1)
       *    i = 1 * n_af + k1 + k2*n_me*n_af
       *    j = 0 * n_af + k1 + k2*n_me*n_af
       * if (i==0)
       *    i = 0 * n_af + k1 + k2*n_l*n_af
       *    j = 1 * n_af + k1 + k2*n_l*n_af
       * We work out k1 and k2 from i to get j.
       */
      if (i_sub==0) {
        tmp_int = i - 0 * n_after;
        k2      = tmp_int/(this_op->my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(this_op->my_levels*n_after);
        *j = (0 + 1) * n_after + k1 + k2*this_op->my_levels*n_after;
        *val   = -PETSC_i;
      } else if (i_sub==1) {
        tmp_int = i - (0+1) * n_after;
        k2      = tmp_int/(this_op->my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(this_op->my_levels*n_after);
        *j = 0 * n_after + k1 + k2*this_op->my_levels*n_after;
        *val   = PETSC_i;
      } else {
        if (nid==0){ 
          printf("ERROR! Pauli Operators are only defined for qubits\n");
          exit(0);
        }
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
    } else if (this_op->my_op_type==IDENTITY){
      /* Identity operator */
      /* diagonal, even in global space */
      *j = i;
      *val = 1.0;
    } else if (this_op->my_op_type==SIGMA_Z){
      /*
       * SIGMA_Z
       * diagonal, even in global space
       * if (i==0) val = 1.0
       * if (i==1) val = -1.0
       * We work out k1 and k2 from i to get j.
       */
      if (i_sub==0) {
        *j = i;
        *val   = 1.0;
      } else if (i_sub==1) {
        *j = i;
        *val   = -1.0;
      } else {
        if (nid==0){
          printf("ERROR! Pauli Operators are only defined for qubits\n");
          exit(0);
        }
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
        printf("       (maybe from get_expectation_value)\n");
        exit(0);
      }
      /* *i_op = position; */
      /* *j_op = position; */
      /* *val   = 1.0; */
    }

    if (tensor_control==TENSOR_GI){
      //Take complex conjugate of answer
      *val = PetscConjComplex(*val);
    }
  } else {
    /*
     * U* cross U
     * To calculate this, we first take our i_global, convert
     * it to i1 (for U*) and i2 (for U) within their own
     * part of the Hilbert space. pWe then treat i1 and i2 as
     * global i's for the matrices U* and U themselves, which
     * gives us j's for those matrices. We then expand the j's
     * to get the full space representation, using the normal
     * tensor product.
     */

    /* Calculate i1, i2 */
    i1 = i/tot_levels;
    i2 = i%tot_levels;

    /* Now, get js for U* (i1) by calling this function */
    _get_val_j_from_global_i(tot_levels,i1,this_op,&j_i1,&val_i1,TENSOR_IG);

    /* Now, get js for U (i2) by calling this function */
    _get_val_j_from_global_i(tot_levels,i2,this_op,&j_i2,&val_i2,TENSOR_IG);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    if (j_i1==-1||j_i2==-1){
      *j = -1;
      *val = 0;
    } else {
      *j = tot_levels * j_i1 + j_i2;
      *val = PetscConjComplex(val_i1)*val_i2;
    }

  }
  return;
}

void _get_val_j_from_global_i_vec_vec(PetscInt tot_levels,PetscInt i,operator this_op1,operator this_op2,PetscInt *j,PetscScalar *val,tensor_control_enum tensor_control){
  PetscInt i_sub,n_after,tmp_int,k1,k2,extra_after,j_i1,j_i2,i1,i2;
  PetscScalar val_i1,val_i2;

  /*
   * We store our vec operators as location only.
   * we use the stored information to calculate the global j location
   * and nonzero value for a given global i
   */

  //Check for correct operator types
  if (this_op1->my_op_type!=VEC||this_op2->my_op_type!=VEC) {
    if (nid==0){
      printf("ERROR! Only Vec Operators are allowed in _get_val_j_from_global_i_vec_vec\n");
      exit(0);
    }
  }

  //Check that both operators are from the super hilbert space
  if (this_op1->n_before!=this_op2->n_before) {
    if (nid==0){
      printf("ERROR! Vec Operators must be from the same Hilbert subspace in _get_val_j_from_global_i_vec_vec\n");
      exit(0);
    }
  }


  if (tensor_control!=TENSOR_GG) {
    if (tensor_control==TENSOR_GI) {
      extra_after = tot_levels;
    } else {
      extra_after = 1;
    }

    /*
     * Because this is a vec vec, there is only one location in the subspace;
     * namely, since it is |vec1><vec2|, i_s is the position of vec1 and
     * j_s is the position of vec2.
     *
     * If the global lines up with this, we return the global j. If
     * not, we return -1.
     */

    n_after = tot_levels/(this_op1->my_levels*this_op1->n_before)*extra_after;
    i_sub = i/n_after%this_op1->my_levels; //Use integer arithmetic to get floor function
    if (i_sub==this_op1->position){
      tmp_int = i - i_sub * n_after;
      k2      = tmp_int/(this_op1->my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(this_op1->my_levels*n_after);
      *j = this_op2->position * n_after + k1 + k2*this_op1->my_levels*n_after;
      *val = 1.0;
    } else {
      //There is no nonzero value for given global i; return -1 as flag
      *j = -1;
      *val = 0.0;
    }
    if (tensor_control==TENSOR_GI){
      //Take complex conjugate of answer
      *val = PetscConjComplex(*val);
    }
  } else {
    /*
     * U* cross U
     * To calculate this, we first take our i_global, convert
     * it to i1 (for U*) and i2 (for U) within their own
     * part of the Hilbert space. pWe then treat i1 and i2 as
     * global i's for the matrices U* and U themselves, which
     * gives us j's for those matrices. We then expand the j's
     * to get the full space representation, using the normal
     * tensor product.
     */

    /* Calculate i1, i2 */
    i1 = i/tot_levels;
    i2 = i%tot_levels;

    /* Now, get js for U* (i1) by calling this function */
    _get_val_j_from_global_i_vec_vec(tot_levels,i1,this_op1,this_op2,&j_i1,&val_i1,TENSOR_IG);

    /* Now, get js for U (i2) by calling this function */
    _get_val_j_from_global_i_vec_vec(tot_levels,i2,this_op1,this_op2,&j_i2,&val_i2,TENSOR_IG);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    if (j_i1==-1||j_i2==-1){
      *j = -1;
      *val = 0;
    } else {
      *j = tot_levels * j_i1 + j_i2;
      *val = PetscConjComplex(val_i1)*val_i2;
    }
  }
  return;
}


void _add_ops_to_mat_ham_only(PetscScalar a,Mat A,PetscInt tot_levels,PetscInt num_ops,operator *ops){
  PetscInt i,j_ig,this_j_ig,Istart,Iend;
  PetscScalar    val_ig;
  PetscScalar add_to_mat;
  MatGetOwnershipRange(A,&Istart,&Iend);

  for (i=Istart;i<Iend;i++){
    this_j_ig = i;
    _get_val_j_ops(&val_ig,&this_j_ig,tot_levels,num_ops,ops,TENSOR_IG);

    //Add G_1 G_2 ... G_n
    if (this_j_ig!=-1){
      add_to_mat = a*val_ig;
      //MatSetValues(A,1,&i,1,&this_j_ig,&add_to_mat,ADD_VALUES);
      MatSetValue(A,i,this_j_ig,add_to_mat,ADD_VALUES);
    }
  }

  return;
}

/*
 * Add the non-hermitian part of the Lindblad to the Hamiltonian for the MCWF solver.
 * For some Lindblad $L$, this terms is:
 *         -i/2 a  L^\dag L
 */
void _add_ops_to_mat_lin_mcwf(PetscScalar a,Mat A,PetscInt tot_levels,PetscInt num_ops,operator *ops){
  PetscInt i,j_ig,this_j_ig,Istart,Iend;
  operator *tmp_ops;
  PetscScalar    val_ig;
  PetscScalar add_to_mat;
  MatGetOwnershipRange(A,&Istart,&Iend);

  /*
   * We double the number of ops and add each ops complex conjugate to the list
   */
  tmp_ops = malloc(2*num_ops*sizeof(operator));
  for(i=0;i<num_ops;i++){
    tmp_ops[num_ops-i-1] = ops[i]->dag; //take reverse order of ops because (ab)^\dag = b^\dag a^\dag
    tmp_ops[num_ops+i] = ops[i];
  }

  for (i=Istart;i<Iend;i++){
    this_j_ig = i;

    _get_val_j_ops(&val_ig,&this_j_ig,tot_levels,2*num_ops,tmp_ops,TENSOR_IG);

    //Add G_1 G_2 ... G_n
    if (this_j_ig!=-1){
      add_to_mat = -PETSC_i*a/2*val_ig;
      MatSetValue(A,i,this_j_ig,add_to_mat,ADD_VALUES);
    }
  }

  free(tmp_ops);
  return;
}

void _add_ops_to_mat_ham(PetscScalar a,Mat A,PetscInt tot_levels,PetscInt num_ops,operator *ops){
  PetscInt i,j_ig,j_gi,this_j_ig,this_j_gi,Istart,Iend;
  PetscScalar    val_ig,val_gi;
  PetscScalar add_to_mat;

  MatGetOwnershipRange(A,&Istart,&Iend);

  for (i=Istart;i<Iend;i++){
    this_j_ig = i;
    this_j_gi = i;

    _get_val_j_ops(&val_ig,&this_j_ig,tot_levels,num_ops,ops,TENSOR_IG);
    _get_val_j_ops(&val_gi,&this_j_gi,tot_levels,num_ops,ops,TENSOR_GI);
    //Add -i * I cross G_1 G_2 ... G_n
    if (this_j_ig!=-1){
      add_to_mat = -a*PETSC_i*val_ig;
      MatSetValue(A,i,this_j_ig,add_to_mat,ADD_VALUES);
    }

    //Add i * G_1*T G_2*T ... G_n*T cross I
    if (this_j_gi!=-1){
      add_to_mat = a*PETSC_i*PetscConjComplex(val_gi);
      MatSetValue(A,this_j_gi,i,add_to_mat,ADD_VALUES);
    }
  }

  return;
}

void _count_ops_in_mat(PetscInt *d_nnz,PetscInt *o_nnz,PetscInt tot_levels,PetscInt Istart,PetscInt Iend,
                       Mat A,mat_term_type my_term_type,PetscBool dm_equations,PetscBool mcwf_solver,
                       PetscInt num_ops,operator *ops){
  if (dm_equations==PETSC_TRUE){
    //Use lindblad terms
    if (my_term_type==LINDBLAD||my_term_type==TD_LINDBLAD){
      if(mcwf_solver==PETSC_TRUE){
        //count only non-Hermitian part
        _count_ops_in_mat_lin_mcwf(d_nnz,o_nnz,tot_levels,Istart,Iend,A,num_ops,ops);
      } else {
        _count_ops_in_mat_lin(d_nnz,o_nnz,tot_levels,Istart,Iend,A,num_ops,ops);
      }
    } else if (my_term_type==HAM||my_term_type==TD_HAM){
      if(mcwf_solver==PETSC_TRUE){
        //count only the smaller vector space size
        _count_ops_in_mat_ham_only(d_nnz,o_nnz,tot_levels,Istart,Iend,A,num_ops,ops);
      } else {
        _count_ops_in_mat_ham(d_nnz,o_nnz,tot_levels,Istart,Iend,A,num_ops,ops);
      }
    } else {
      if (nid==0){
        printf("ERROR! Wrong mat_term_type in _count_ops_in_mat - dm!\n");
        exit(0);
      }
    }
  } else {
    if (my_term_type==HAM||my_term_type==TD_HAM){
      _count_ops_in_mat_ham_only(d_nnz,o_nnz,tot_levels,Istart,Iend,A,num_ops,ops);
    } else {
      if (nid==0){
        printf("ERROR! Wrong mat_term_type in _count_ops_in_mat - ham!\n");
        exit(0);
      }
    }
  }
  return;
}

void _count_ops_in_mat_ham(PetscInt *d_nnz,PetscInt *o_nnz,PetscInt tot_levels,
                           PetscInt Istart,PetscInt Iend,Mat A,PetscInt num_ops,operator *ops){
  PetscInt i,this_j_ig,this_j_gi,n;
  PetscScalar    val_ig,val_gi;

  MatGetSize(A,&n,NULL);

  for (i=Istart;i<Iend;i++){
    this_j_ig = i;
    this_j_gi = i;

    _get_val_j_ops(&val_ig,&this_j_ig,tot_levels,num_ops,ops,TENSOR_IG);
    _get_val_j_ops(&val_gi,&this_j_gi,tot_levels,num_ops,ops,TENSOR_GI);

    //Count -i * I cross G_1 G_2 ... G_n
    if (this_j_ig!=-1){
      if(this_j_ig==i){
        //Diagonal element, already accounted for
      } else if(this_j_ig>=Istart && this_j_ig < Iend){
        //Diagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (d_nnz[i - Istart] < Iend-Istart){
          d_nnz[i - Istart] += 1;
        }
      } else {
        //Offdiagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (o_nnz[i - Istart] < n - (Iend-Istart)){
          o_nnz[i - Istart] += 1;
        }
      }
    }

    //Count i * G_1*T G_2*T ... G_n*T cross I
    if (this_j_gi!=-1){
      if(this_j_gi==i){
        //Diagonal element, already accounted for
      } else if(this_j_gi>=Istart && this_j_gi < Iend){
        //Diagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (d_nnz[this_j_gi - Istart] < Iend-Istart){
          d_nnz[this_j_gi - Istart] += 1;
        }
      } else {
        //Offdiagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (o_nnz[this_j_gi - Istart] < n - (Iend-Istart)){
          o_nnz[this_j_gi - Istart] += 1;
        }
      }
    }
  }

  return;
}

void _count_ops_in_mat_ham_only(PetscInt *d_nnz,PetscInt *o_nnz,PetscInt tot_levels,
                                PetscInt Istart,PetscInt Iend,Mat A,PetscInt num_ops,operator *ops){
  PetscInt i,this_j_ig,n;
  PetscScalar    val_ig;

  MatGetSize(A,&n,NULL);
  for (i=Istart;i<Iend;i++){
    this_j_ig = i;

    _get_val_j_ops(&val_ig,&this_j_ig,tot_levels,num_ops,ops,TENSOR_IG);

    //Count -i * G_1 G_2 ... G_n
    if (this_j_ig!=-1){
      if(this_j_ig==i){
        //Diagonal element, already accounted for
      } else if(this_j_ig>=Istart && this_j_ig < Iend){
        //Diagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (d_nnz[i - Istart] < Iend-Istart){
          d_nnz[i - Istart] += 1;
        }
      } else {
        //Offdiagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (o_nnz[i - Istart] < n - (Iend-Istart)){
          o_nnz[i - Istart] += 1;
        }
      }
    }

  }

  return;
}

void _count_ops_in_mat_lin_mcwf(PetscInt *d_nnz,PetscInt *o_nnz,PetscInt tot_levels,
                                PetscInt Istart,PetscInt Iend,Mat A,PetscInt num_ops,operator *ops){
  PetscInt i,this_j_ig,n;
  PetscScalar    val_ig;
  operator *tmp_ops;

  /*
   * We double the number of ops and add each ops complex conjugate to the list
   */
  tmp_ops = malloc(2*num_ops*sizeof(operator));
  for(i=0;i<num_ops;i++){
    tmp_ops[2*i] = ops[i];
    tmp_ops[2*i+1] = ops[i]->dag;
  }

  MatGetSize(A,&n,NULL);
  for (i=Istart;i<Iend;i++){
    this_j_ig = i;

    _get_val_j_ops(&val_ig,&this_j_ig,tot_levels,2*num_ops,tmp_ops,TENSOR_IG);

    //Count -i * G_1 G_2 ... G_n
    if (this_j_ig!=-1){
      if(this_j_ig==i){
        //Diagonal element, already accounted for
      } else if(this_j_ig>=Istart && this_j_ig < Iend){
        //Diagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (d_nnz[i - Istart] < Iend-Istart){
          d_nnz[i - Istart] += 1;
        }
      } else {
        //Offdiagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (o_nnz[i - Istart] < n - (Iend-Istart)){
          o_nnz[i - Istart] += 1;
        }
      }
    }

  }

  free(tmp_ops);
  return;
}

void _count_ops_in_mat_lin(PetscInt *d_nnz,PetscInt *o_nnz,PetscInt tot_levels,
                           PetscInt Istart,PetscInt Iend,Mat A,PetscInt num_ops,operator *ops){
  PetscInt i,this_j_ig,this_j_gi,this_j_gg,n;
  PetscScalar    val_ig,val_gi,val_gg;

  MatGetSize(A,&n,NULL);
  for (i=Istart;i<Iend;i++){

    /*
     * I cross G^t G and G^t G cross I are diagonal terms.
     * We already count the diagonal, we need not add it here
     */

    this_j_gg = i;

    _get_val_j_ops(&val_gg,&this_j_gg,tot_levels,num_ops,ops,TENSOR_GG);

    /*
     * Count (G* cross G)
     */
    if (this_j_gg!=-1){
      if(this_j_gg==i){
        //Diagonal element, already accounted for
      } else if(this_j_gg>=Istart && this_j_gg < Iend){
        //Diagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (d_nnz[i - Istart] < Iend-Istart){
          d_nnz[i - Istart] += 1;
        }
      } else {
        //Offdiagonal block
        //Check so that we do not double count and end up with allocation larger than the row
        if (o_nnz[i - Istart] < n - (Iend-Istart)){
          o_nnz[i - Istart] += 1;
        }
      }
    }
  }

  return;
}

void _add_ops_to_mat(PetscScalar a,Mat A,mat_term_type my_term_type,PetscInt tot_levels,PetscBool dm_equations,
                     PetscBool mcwf_solver,PetscInt num_ops,operator *ops){

  if (dm_equations==PETSC_TRUE){
    //Use lindblad terms
    if(mcwf_solver==PETSC_TRUE){
      //Use mcwf_solver, which only adds nonhermitian part of the Lindblad and only the vector sized Ham
      if (my_term_type==LINDBLAD||my_term_type==TD_LINDBLAD){
        _add_ops_to_mat_lin_mcwf(a,A,tot_levels,num_ops,ops);
      } else if (my_term_type==HAM||my_term_type==TD_HAM){
        _add_ops_to_mat_ham_only(a,A,tot_levels,num_ops,ops);
      } else {
        if (nid==0){
          printf("ERROR! Wrong mat_term_type in construct_matrix - dm mcwf!\n");
          exit(0);
        }
      }
    } else {
      if (my_term_type==LINDBLAD||my_term_type==TD_LINDBLAD){
        _add_ops_to_mat_lin(a,A,tot_levels,num_ops,ops);
      } else if (my_term_type==HAM||my_term_type==TD_HAM){
        _add_ops_to_mat_ham(a,A,tot_levels,num_ops,ops);
      } else {
        if (nid==0){
          printf("ERROR! Wrong mat_term_type in construct_matrix - dm!\n");
          exit(0);
        }
      }
    }
  } else {
    if (my_term_type==HAM){
      _add_ops_to_mat_ham_only(a,A,tot_levels,num_ops,ops);
    } else {
      if (nid==0){
        printf("ERROR! Wrong mat_term_type in construct_matrix - ham!\n");
        exit(0);
      }
    }
  }
  return;
}

void _add_ops_to_mat_lin(PetscScalar a,Mat A,PetscInt tot_levels,PetscInt num_ops,operator *ops){
  PetscInt i,j,j_ig,j_gi,j_gg,this_j_ig,this_j_gi,Istart,Iend,this_j_gg;
  PetscScalar    val_ig,val_gi,val_gg,tmp_val;
  PetscScalar add_to_mat;

  MatGetOwnershipRange(A,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    this_j_ig = i;
    this_j_gi = i;
    this_j_gg = i;

    _get_val_j_ops(&val_ig,&this_j_ig,tot_levels,num_ops,ops,TENSOR_IG);
    _get_val_j_ops(&val_gi,&this_j_gi,tot_levels,num_ops,ops,TENSOR_GI);
    _get_val_j_ops(&val_gg,&this_j_gg,tot_levels,num_ops,ops,TENSOR_GG);
    /*
     * Add (I cross G^t G)
     */
    if (this_j_ig!=-1){
      add_to_mat = -0.5*a*PetscConjComplex(val_ig)*val_ig;
      MatSetValue(A,this_j_ig,this_j_ig,add_to_mat,ADD_VALUES);
    }

    /*
     * Add ((G^t G)* cross I)
     */
    if (this_j_gi!=-1){
      //The second conjugate is redundant here?
      add_to_mat = -0.5*a*PetscConjComplex(PetscConjComplex(val_gi)*val_gi);
      MatSetValue(A,this_j_gi,this_j_gi,add_to_mat,ADD_VALUES);
    }
    /*
     * Add (G* cross G) to the superoperator matrix, A
     */
    if (this_j_gg!=-1){
      //The second conjugate is redundant here?
      add_to_mat = a*val_gg;
      MatSetValue(A,i,this_j_gg,add_to_mat,ADD_VALUES);
    }
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
                        int extra_before,int extra_after,int transpose){
  long loop_limit,i,i_op,j_op,n_after;
  PetscScalar    val;
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
    if (transpose){
      _add_to_PETSc_kron_ij(matrix,add_to_mat,j_op,i_op,n_before*extra_before,n_after*extra_after,my_levels);
    } else {
      _add_to_PETSc_kron_ij(matrix,add_to_mat,i_op,j_op,n_before*extra_before,n_after*extra_after,my_levels);
    }
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
  PetscScalar val1,val2;
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

  if (n_before2==n_before1){
    n_between = 1;
  } else {
    n_between = n_before2/(n_before1*levels1);
  }

  /*
   * n_before and n_after refer to before and after a cross I_c cross b
   * and my_levels is the size of a cross I_c cross b
   */
  n_before   = n_before1;
  my_levels  = levels1*levels2*n_between;
  n_after    = total_levels/(my_levels*n_before);

  if (n_before2==n_before1){
    n_after = 1;
  } else {
    n_after    = total_levels/(my_levels*n_before);
  }


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
  PetscScalar val1,val2;
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
 *      int transpose:      whether or not to take the transpose
 * Outputs:
 *      none, but adds to PETSc matrix
 */

void _add_to_PETSc_kron_lin(Mat matrix,PetscScalar a,int n_before,int my_levels,
                        op_type my_op_type,int position,
                            int extra_before,int extra_after,int transpose){
  long loop_limit,i,i_op,j_op,n_after;
  PetscScalar    val;
  PetscScalar add_to_mat;
  loop_limit = _get_loop_limit(my_op_type,my_levels);

  n_after    = total_levels/(my_levels*n_before);

  for (i=0;i<my_levels-loop_limit;i++){
    /*
     * For this term, we have to calculate Ct C or (Ct C)^T = C^T C* = (C^t C)* .
     * We know, a priori, that all operators are (sub or super) diagonal.
     * Any matrix such as this will be (true) diagonal after doing either operation.
     * Ct C and (Ct C)* are simple to calculate with these diagonal matrices:
     * for all elements of C[k], where C is the stored diagonal,
     * location in Ct C = j,j of the
     * original element C[k], and the value is C[k]*C[k].
     *
     */
    val  = _get_val_in_subspace(i,my_op_type,position,&i_op,&j_op);
    i_op = j_op;
    if (transpose){
      val  = PetscConjComplex(val*PetscConjComplex(val));
    } else{
      val  = val*PetscConjComplex(val);
    }
    add_to_mat = a*val;
    _add_to_PETSc_kron_ij(matrix,add_to_mat,i_op,j_op,n_before*extra_before,
                            n_after*extra_after,my_levels);
  }

  return;
}

/*
 * _add_to_PETSc_kron_lin_mat adds a formed matrix in the operator space
 * to the full_A, expanding either before or after with the identity matrix.
 *
 * Inputs:
 *      Mat matrix:         matrix to add to
 *      PetscScalar a       scalar to multiply matrix (can be complex)
 *      Mat matrix_to_add:  matrix to be added
 *      int before:         whether to exand before (otherwise, expand after)
 *      int transpose:      whether or not to take the conjugate transpose
 * Outputs:
 *      none, but adds to PETSc matrix
n */

void _add_to_PETSc_kron_lin_mat(Mat matrix,PetscScalar a, Mat matrix_to_add,
                                int before, int transpose){
  PetscScalar add_to_mat;
  PetscInt    i,j,Istart,Iend,ncols,extra_before,extra_after;
  const PetscInt    *cols;
  const PetscScalar *vals;

  MatGetOwnershipRange(matrix_to_add,&Istart,&Iend);

  if (before) {
    extra_before = total_levels;
    extra_after  = 1;
  } else {
    extra_after  = total_levels;
    extra_before = 1;
  }

  /*
   * For this term, we have to calculate Ct C or (Ct C)^T = C^T C* = (C^t C)* .
   */
  for (i=Istart;i<Iend;i++){
    /* Get the row */
    MatGetRow(matrix_to_add,i,&ncols,&cols,&vals);
    for (j=0;j<ncols;j++){
      if (transpose) {
        add_to_mat = a*PetscConjComplex(vals[j]);
      } else {
        add_to_mat = a*vals[j];
      }
      /*
       * We add the i,j element (which is stored in vals[j]) to
       * the full matrix, expanding appropriately
       */
      _add_to_PETSc_kron_ij(matrix,add_to_mat,i,cols[j],extra_before,
                            extra_after,total_levels);
    }
    MatRestoreRow(matrix_to_add,i,&ncols,&cols,&vals);
  }

  return;
}


/*
 * _add_to_PETSc_kron_lin2
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

void _add_to_PETSc_kron_lin2(Mat matrix,PetscScalar a,operator op1,operator op2){
  long i,i_op,j_op,n_after;
  PetscScalar add_to_mat,val,op_val;
  PetscInt Istart,Iend,this_i,this_j;

  //Check operator type for op2
  if ((op1->my_op_type!=RAISE && op2->my_op_type!=LOWER)&&(op2->my_op_type!=RAISE && op1->my_op_type!=LOWER)){
    if (nid==0){
      printf("ERROR! kron_lin2 only supports raising and lowering operators for now.\n");
      exit(0);
    }
  }



  MatGetOwnershipRange(matrix,&Istart,&Iend);

  for (i=Istart;i<Iend;i++){

    /* First get
     * I cross C^t C = I cross b^t a^t a b
     * since C = a b, and a = op1, b = op2
     */
    this_i = i; // The leading index which we check
    op_val = -0.5*a;

    // b^t
    _get_val_j_from_global_i(total_levels,this_i,op2->dag,&this_j,&val,-1); // Get the corresponding j and val for op1
    op_val = val*op_val;
    this_i = this_j;
    // a^t
    _get_val_j_from_global_i(total_levels,this_i,op1->dag,&this_j,&val,-1); // Get the corresponding j and val for op2
    op_val = val*op_val;
    this_i = this_j;
    // a
    _get_val_j_from_global_i(total_levels,this_i,op1,&this_j,&val,-1); // Get the corresponding j and val for op2
    op_val = val*op_val;
    this_i = this_j;
    // b
    _get_val_j_from_global_i(total_levels,this_i,op2,&this_j,&val,-1); // Get the corresponding j and val for op2
    op_val = val*op_val;
    this_i = this_j;

    if (PetscAbsComplex(op_val)!=0) {
      //Add to matrix if appropriate
      MatSetValue(matrix,i,this_i,op_val,ADD_VALUES);
    }

    /* Now get
     * (C^t C)* cross I = (b^t a^t a b)* cross I
     * since C = a b, and a = op1, b = op2
     */
    this_i = i; // The leading index which we check
    op_val = -0.5*a;

    // b^t
    _get_val_j_from_global_i(total_levels,this_i,op2->dag,&this_j,&val,1); // Get the corresponding j and val for op1
    op_val = val*op_val;
    this_i = this_j;
    // a^t
    _get_val_j_from_global_i(total_levels,this_i,op1->dag,&this_j,&val,1); // Get the corresponding j and val for op2
    op_val = val*op_val;
    this_i = this_j;
    // a
    _get_val_j_from_global_i(total_levels,this_i,op1,&this_j,&val,1); // Get the corresponding j and val for op2
    op_val = val*op_val;
    this_i = this_j;
    // b
    _get_val_j_from_global_i(total_levels,this_i,op2,&this_j,&val,1); // Get the corresponding j and val for op2
    op_val = val*op_val;
    op_val = PetscConjComplex(op_val);
    this_i = this_j;
    if (PetscAbsComplex(op_val)!=0) {
      //Add to matrix if appropriate
      MatSetValue(matrix,i,this_i,op_val,ADD_VALUES);
    }

    /* Now get
     * C* cross C = a* b* cross a b
     * since C = a b, and a = op1, b = op2
     */
    this_i = i; // The leading index which we check
    op_val = a;//Is this correct? FIXME

    // a* cross a
    _get_val_j_from_global_i(total_levels,this_i,op1,&this_j,&val,0); // Get the corresponding j and val for op1
    op_val = val*op_val;
    this_i = this_j;
    // b* cross b
    _get_val_j_from_global_i(total_levels,this_i,op2,&this_j,&val,0); // Get the corresponding j and val for op2
    op_val = val*op_val;
    this_i = this_j;

    if (PetscAbsComplex(op_val)!=0) {
      //Add to matrix if appropriate
      MatSetValue(matrix,i,this_i,op_val,ADD_VALUES);
    }


  }


  return;
}


/*
 * add_to_PETSc_kron_lin_comb adds C'* cross C' to matrix
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
  PetscScalar val1,val2;
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
       * Make sure to take complex conjugate here
       */
      val1 = PetscConjComplex(_get_val_in_subspace(i,my_op_type,position,&i1,&j1));
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
  PetscScalar    val;
  PetscScalar add_to_mat;
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
  PetscScalar val1,val2;
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
  i2 = 0; j2 = 0;

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
