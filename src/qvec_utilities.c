#include "qvec_utilities.h"
#include "operators_p.h"
#include "operators.h"
#include "kron_p.h"
#include <stdlib.h>
#include <stdio.h>
#include <petscblaslapack.h>
#include <string.h>
/*
 * FIXME:
 *    Add create_{dm,wf}_qvec to create any sized dm, even if it isn't tied to a system
 *    Add print_qvec_sparse
 *    Add print_qvec_file
 */




void qvec_mat_mult(Mat circ_mat,qvec state){
  Vec tmp;
  VecDuplicate(state->data,&tmp);
  MatMult(circ_mat,state->data,tmp);
  VecCopy(tmp,state->data);
  VecDestroy(&tmp);
  return;
}
/*
 * Print the qvec densely
 * Not recommended for large matrices.
 * NOTE: Should be called from all cores!
 */
void print_qvec_file(qvec state,char filename[]){

  if(state->my_type==DENSITY_MATRIX){
    print_dm_qvec_file(state,filename);
  } else {
    print_wf_qvec_file(state,filename);
  }

}

void print_dm_qvec_file(qvec dm,char filename[]){
  PetscScalar val;
  PetscInt i,j,h_dim;
  FILE           *fp;

  fp = fopen(filename,"w");
  if (!fp) {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Cannot open file in print_dm_qvec_file= %s \n",filename);
    exit(1);
  }

  h_dim = sqrt(dm->n);
  for (i=0;i<h_dim;i++){
    for (j=0;j<h_dim;j++){
      get_dm_element_qvec(dm,i,j,&val);
      PetscFPrintf(PETSC_COMM_WORLD,fp,"%4.3e + %4.3ei ",PetscRealPart(val),
                  PetscImaginaryPart(val));
    }
    PetscFPrintf(PETSC_COMM_WORLD,fp,"\n");
  }
  PetscFPrintf(PETSC_COMM_WORLD,fp,"\n");

  if (fp) {
    fclose(fp);
    fp = NULL;
  }

  return;
}

/*
 * Print the dense wf
 */
void print_wf_qvec_file(qvec state,char filename[]){
  PetscScalar val;
  PetscInt i;
  FILE           *fp;

  fp = fopen(filename,"w");
  if (!fp) {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Cannot open file in print_dm_qvec_file= %s \n",filename);
    exit(1);
  }

  for(i=0;i<state->n;i++){
    get_wf_element_qvec(state,i,&val);
    PetscFPrintf(PETSC_COMM_WORLD,fp,"%40.30e  %40.30e\n",PetscRealPart(val),
                PetscImaginaryPart(val));
  }

  if (fp) {
    fclose(fp);
    fp = NULL;
  }
  return;
}


/*
 * Print the qvec densely
 * Not recommended for large matrices.
 * NOTE: Should be called from all cores!
 */
void print_qvec(qvec state){

  if(state->my_type==DENSITY_MATRIX){
    print_dm_qvec(state);
  } else {
    print_wf_qvec(state);
  }

}

void print_dm_qvec(qvec dm){
  PetscScalar val;
  PetscInt i,j,h_dim;

  h_dim = sqrt(dm->n);
  for (i=0;i<h_dim;i++){
    for (j=0;j<h_dim;j++){
      get_dm_element_qvec(dm,i,j,&val);
      PetscPrintf(PETSC_COMM_WORLD,"%4.3e + %4.3ei ",PetscRealPart(val),
                  PetscImaginaryPart(val));
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n");
  }
  PetscPrintf(PETSC_COMM_WORLD,"\n");
  return;
}

/*
 * void get_dm_element_qvec gets a specific i,j element from the
 * input density matrix - global version; will grab from other
 * cores.
 * NOTE: This should be called by all processors, or the code will hang!
 *
 * Inputs:
 *        qvec dm   - density matrix from which to get element
 *        PetscInt row - i location of requested element
 *        PetscInt col - j location of requested element
 * Outpus:
 *        PetscScalar val - requested density matrix element
 *
 */
void get_dm_element_qvec(qvec dm,PetscInt row,PetscInt col,PetscScalar *val){
  PetscInt location[1];
  PetscScalar val_array[1];

  location[0] = sqrt(dm->n)*col + row;

  if (location[0]>=dm->Istart&&location[0]<dm->Iend) {
    VecGetValues(dm->data,1,location,val_array);
  } else{
    val_array[0] = 0.0 + 0.0*PETSC_i;
  }
  MPI_Allreduce(MPI_IN_PLACE,val_array,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

  *val = val_array[0];
  return;
}

/*
 * void get_dm_element_qvec_local gets a specific i,j element from the
 * input density matrix - local version; will only grab from self
 *
 * Inputs:
 *        qvec dm   - density matrix from which to get element
 *        PetscInt row - i location of requested element
 *        PetscInt col - j location of requested element
 * Outpus:
 *        PetscScalar val - requested density matrix element
 *
 */
void get_dm_element_qvec_local(qvec dm,PetscInt row,PetscInt col,PetscScalar *val){
  PetscInt location[1],dm_size,my_start,my_end;
  PetscScalar val_array[1];

  location[0] = sqrt(dm->n)*col + row;

  if (location[0]>=dm->Istart&&location[0]<dm->Iend) {
    VecGetValues(dm->data,1,location,val_array);
  } else{
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Can only get elements local to a core in !\n");
    PetscPrintf(PETSC_COMM_WORLD,"       get_dm_element_qvec_local.\n");
    exit(0);
  }

  *val = val_array[0];
  return;
}


/*
 * Print the dense wf
 */
void print_wf_qvec(qvec state){
  PetscScalar val;
  PetscInt i;

  for(i=0;i<state->n;i++){
    get_wf_element_qvec(state,i,&val);
    PetscPrintf(PETSC_COMM_WORLD,"%e + %ei\n",PetscRealPart(val),
                PetscImaginaryPart(val));
  }
  return;
}

void get_wf_element_qvec(qvec state,PetscInt i,PetscScalar *val){
  PetscInt location[1];
  PetscScalar val_array[1];

  location[0] = i;
  if (location[0]>=state->Istart&&location[0]<state->Iend) {
    VecGetValues(state->data,1,location,val_array);
  } else{
    val_array[0] = 0.0 + 0.0*PETSC_i;
  }
  MPI_Allreduce(MPI_IN_PLACE,val_array,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

  *val = val_array[0];
  return;
}

/*
 * create a qvec object as a dm or wf based on
 * the operators added thus far
 */
void create_qvec_sys(qsystem sys,qvec *new_qvec){
  //Automatically decide whether to make a DM or WF
  if (sys->dm_equations){
    create_dm_sys(sys,new_qvec);
  } else {
    create_wf_sys(sys,new_qvec);
  }

  return;
}

/*
 * create a dm that is correctly sized for the qsystem
 * and set to solve DM equations? FIXME
 */
void create_dm_sys(qsystem sys,qvec *new_dm){
  qvec temp = NULL;
  PetscInt n,Istart,Iend;

  /* Check to make sure some operators were created */
  if (sys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to create operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a density matrix!\n");
    exit(0);
  }


  //Even if we had no lindblad terms, maybe we start in a density matrix
  sys->dm_equations = 1;
  sys->dim = sys->total_levels*sys->total_levels;
  _setup_distribution(sys);
  sys->hspace_frozen = 1;

  temp = malloc(sizeof(struct qvec));

  PetscPrintf(PETSC_COMM_WORLD,"Creating density matrix vector for Liouvillian solver.\n");
  _create_vec(&(temp->data),sys->dim,sys->my_num);

  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = DENSITY_MATRIX;
  temp->n = n;
  temp->Istart = Istart;
  temp->Iend = Iend;

  *new_dm = temp;
  return;
}

/*
 * create a wf that is correctly sized for the qsystem
 */
void create_wf_sys(qsystem sys,qvec *new_wf){
  qvec temp = NULL;
  PetscInt n,Istart,Iend;

  /* Check to make sure some operators were created */
  if (sys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to create operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a wavefunction!\n");
    exit(0);
  }


  if (sys->dm_equations==1){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR!\n");
    PetscPrintf(PETSC_COMM_WORLD,"Must use density matrices if Lindblad terms are used.\n");
    exit(0);
  }

  PetscPrintf(PETSC_COMM_WORLD,"Creating wavefunction vector for Schrodinger solver.\n");
  sys->dim = sys->total_levels;

  _setup_distribution(sys);
  sys->hspace_frozen = 1;

  temp = malloc(sizeof(struct qvec));

  _create_vec(&(temp->data),sys->dim,sys->my_num);

  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = WAVEFUNCTION;
  temp->n = n;
  temp->Istart = Istart;
  temp->Iend = Iend;

  *new_wf = temp;

  return;
}



/*
 * Add alpha at the specified fock_state for operator op in qvec state
 * FIXME: Bad function name?
 */
void add_to_qvec_fock_op(PetscScalar alpha,qvec state,PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  PetscInt *fock_states,i,loc;


  ops = malloc(num_ops*sizeof(struct operator));
  fock_states = malloc(num_ops*sizeof(PetscInt));

  va_start(ap,num_ops);
  for (i=0;i<num_ops;i++){
    ops[i] = va_arg(ap,operator);
    fock_states[i] = va_arg(ap,PetscInt);
  }
  va_end(ap);

  get_qvec_loc_fock_op_list(state,&loc,num_ops,ops,fock_states);
  add_to_qvec_loc(state,alpha,loc);

  free(ops);
  free(fock_states);

  return;
}

/*
 * Add alpha at the specified fock_state for operator op in qvec state
 * FIXME: Bad function name?
 */
void add_to_qvec_fock_op_list(PetscScalar alpha,qvec state,PetscInt num_ops,operator *ops,PetscInt *fock_states){
  va_list  ap;
  PetscInt loc;

  get_qvec_loc_fock_op_list(state,&loc,num_ops,ops,fock_states);

  add_to_qvec_loc(state,alpha,loc);

  return;
}

/*
 * Add alpha to the element at location loc in the qvec
 * Does not discriminate between WF and DM -- loc needs to be correctly vectorized if DM!
 */

void add_to_qvec_loc(qvec state,PetscScalar alpha,PetscInt loc){

  if (loc>state->n||loc<0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Adding to a qvec location that is out of range!\n");
    exit(0);
  }

  if (loc>=state->Istart&&loc<state->Iend){
    VecSetValue(state->data,loc,alpha,ADD_VALUES);
  }

  return;
}


/*
 * Add alpha to the element at location loc in the qvec
 */

void add_to_qvec(qvec state,PetscScalar alpha,...){
  va_list  ap;
  PetscInt loc,loc2,num_locs;

  if (state->my_type==DENSITY_MATRIX){
    /*
     * Since this is a density matrix, we have to 'vectorize' the locations
     * and then call add_to_qvec_loc
     */
    num_locs = 2;
    va_start(ap,num_locs);
    loc = va_arg(ap,PetscInt);
    loc2 = va_arg(ap,PetscInt);
    ///  location = sqrt(dm_size)*col + row;
    loc = sqrt(state->n)*loc + loc2;

    add_to_qvec_loc(state,alpha,loc);
    va_end(ap);

  } else if (state->my_type==WAVEFUNCTION){
    /*
     * Since this is a wavefunction, the input loc is the true location
     * so we can just call add_to_qvec_loc
     */
    num_locs = 1;
    va_start(ap,num_locs);
    loc = va_arg(ap,PetscInt);
    add_to_qvec_loc(state,alpha,loc);
    va_end(ap);
  }

  return;
}

/*
 * Get the location in the qvec (either dm or wf) for the given fock state of operator op in system sys
 * FIXME Does not work for vec_op
 */
void get_qvec_loc_fock_op(qvec state,PetscInt *loc,PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  PetscInt *fock_states,i;


  ops = malloc(num_ops*sizeof(struct operator));
  fock_states = malloc(num_ops*sizeof(PetscInt));

  va_start(ap,num_ops);
  for (i=0;i<num_ops;i++){
    ops[i] = va_arg(ap,operator);
    fock_states[i] = va_arg(ap,PetscInt);
  }
  va_end(ap);

  get_qvec_loc_fock_op_list(state,loc,num_ops,ops,fock_states);

  free(ops);
  free(fock_states);

  return;
}


void get_qvec_loc_fock_op_list(qvec state,PetscInt *loc,PetscInt num_ops,operator *ops,PetscInt *fock_states){
  PetscInt i;

  /*
   * Here, we are getting the location in the Vec of the state
   * ...0..n1..00 n2 00..n30...
   * That is, just for the operators being in their n, and the rest all being 0.
   *
   * Each operator can have its state calculated independently and they can all be
   * summed up
   *
   * loc = sum_i n_i * n_i_b
   *
   * where n_i is the fock state number and n_i_b is the number of levels before
   */
  *loc = 0;
  for(i=0;i<num_ops;i++){
    //Check that the Fock state makes sense
    if (fock_states[i]>=ops[i]->my_levels||fock_states[i]<0){
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! The Fock State must be less than the number of levels and greater than 0!\n");
      exit(0);
    }
    *loc = *loc + fock_states[i] * ops[i]->n_before;
  }

  if (state->my_type==DENSITY_MATRIX){
    //Density matrix; we take it as the diagonal
    *loc = *loc * sqrt(state->n) + *loc;
  } else if (state->my_type==WAVEFUNCTION){
    //WF, just take the direct state
    *loc = *loc;
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvec type not understood.");
    exit(0);
  }
  return;
}



/*
 * void assemble_dm puts all the cached values in the right place and
 * allows for the dm to be used
 *
 * Inputs:
 *         qvec state - state to assemble
 * Outpus:
 *         None, but assembles the state
 *
 */
void assemble_qvec(qvec state){
  VecAssemblyBegin(state->data);
  VecAssemblyEnd(state->data);
}

/*
 * void destroy_qvec frees the memory from a previously created qvec object
 *
 * Inputs:
 *         qvec state - memory to free
 * Outpus:
 *         None, but frees the memory from state
 *
 */

void destroy_qvec(qvec *state){
  VecDestroy(&((*state)->data));
  free(*state);
}

/*
 * private routine which actually allocates the memory.
 */
void _create_vec(Vec *dm,PetscInt dim,PetscInt local_size){
  /* Create the dm, partition with PETSc */
  VecCreate(PETSC_COMM_WORLD,dm);
  VecSetType(*dm,VECMPI);
  VecSetSizes(*dm,local_size,dim);
  /* Set all elements to 0 */
  VecSet(*dm,0.0);
  return;
}


/*
 * void get_expectation_value_qvec calculates the expectation value of the multiplication
 * of a list of operators.
 * The expectation value is defined as:
 *              <ABC...> = Tr(ABC...*rho)
 * where A,B,C,... are operators and rho is the density matrix.
 * This function only accepts operators to be multiplied. To find the
 * expectation value of a sum, call this function once for each
 * element of the sum. i.e.
 *              <A+B+C> = Tr((A+B+C)*rho)
 *                      = Tr(A*rho) + Tr(B*rho) * Tr(C*rho)
 *                      = <A> + <B> + <C>
 *
 * Inputs:
 *         qvec state        - state to get ev of
 *         PetscInt number_of_ops - number of operators in the list
 *          ...              - list of operators
 * Outputs:
 *         PetscScalar *trace_val - the expectation value of the multiplied operators
 *
 * An example calling this function:
 *      get_expectation_value_qvec(dm,&expect,4,ph[0]->dag,ph[1]->dag,ph[0],ph[1]);
 *
 */
void get_expectation_value_qvec(qvec state,PetscScalar *trace_val,PetscInt num_ops,...){
  va_list ap;
  PetscInt i;
  operator *ops;

  ops = malloc(num_ops*sizeof(struct operator));

  va_start(ap,num_ops);
  for (i=0;i<num_ops;i++){
    ops[i] = va_arg(ap,operator);
  }
  va_end(ap);

  get_expectation_value_qvec_list(state,trace_val,num_ops,ops);

  return;
}

/*
 * void get_expectation_value_qvec_list calculates the expectation value of the multiplication
 * of a list of operators.
 * The expectation value is defined as:
 *              <ABC...> = Tr(ABC...*rho)
 * where A,B,C,... are operators and rho is the density matrix.
 * This function only accepts operators to be multiplied. To find the
 * expectation value of a sum, call this function once for each
 * element of the sum. i.e.
 *              <A+B+C> = Tr((A+B+C)*rho)
 *                      = Tr(A*rho) + Tr(B*rho) * Tr(C*rho)
 *                      = <A> + <B> + <C>
 *
 * Inputs:
 *         qvec state        - state to get ev of
 *         PetscInt number_of_ops - number of operators in the list
 *         operator ops           - list of operators
 * Outputs:
 *         PetscScalar *trace_val - the expectation value of the multiplied operators
 *
 * An example calling this function:
 *      get_expectation_value_qvec_list(dm,&expect,4,op_list);
 *
 */
void get_expectation_value_qvec_list(qvec state,PetscScalar *trace_val,PetscInt num_ops,operator *ops){

  if(state->my_type==WAVEFUNCTION){
    _get_expectation_value_wf(state,trace_val,num_ops,ops);
  } else if(state->my_type==DENSITY_MATRIX){
    _get_expectation_value_dm(state,trace_val,num_ops,ops);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvec type not understood.");
    exit(0);
  }

  return;
}

void _get_expectation_value_dm(qvec rho,PetscScalar *trace_val,PetscInt num_ops,operator *ops){
  PetscInt i,j,this_i,this_j,my_j_start,my_j_end,my_start,my_end,dim,dm_size,j_op_tmp;
  PetscInt this_loc;
  PetscScalar dm_element,val,op_val;

  /*
   * Calculate Tr(ABC...*rho) using the following observations:
   *     Tr(A*rho) = sum_i (A*rho)_ii = sum_i sum_k A_ik rho_ki
   *          i.e., we do not need a j loop.
   *     Each operator (ABCD...) is very sparse - having less than
   *          1 value per row on average. This allows us to efficiently do the
   *          multiplication of ABCD... by just calculating the value
   *          for one of the indices (i); if there is no matching j,
   *          the value is 0.
   *
   */
  *trace_val = 0.0 + 0.0*PETSC_i;
  my_start = rho->Istart;
  my_end = rho->Iend;

  /*
   * Find the range of j values stored on a core.
   * Some columns will be shared by more than 1 core;
   * In that case the core who has the first element
   * of the row calculates that value, potentially
   * communicating to get values it does not have
   */
  my_j_start = my_start/total_levels; // Rely on integer division to get 'floor'
  my_j_end  = my_end/total_levels;

  for (i=my_j_start;i<my_j_end;i++){
    this_i = i; // The leading index which we check
    op_val = 1.0;
    for (j=0;j<num_ops;j++){
      if(ops[j]->my_op_type==VEC){
        PetscPrintf(PETSC_COMM_WORLD,"ERROR! VEC operators not yet supported!\n");
        exit(0);
        /*
         * Since this is a VEC operator, the next operator must also
         * be a VEC operator; it is assumed they always come in pairs.
         */
        if (ops[j+1]->my_op_type!=VEC){
          PetscPrintf(PETSC_COMM_WORLD,"ERROR! VEC operators must come in pairs in get_expectation_value\n");
        }
        _get_val_j_from_global_i_vec_vec(this_i,ops[j],ops[j+1],&this_j,&val,-1);
        //Increment j
        j=j+1;
      } else {
        //Standard operator
        _get_val_j_from_global_i(this_i,ops[j],&this_j,&val,-1); // Get the corresponding j and val
      }
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
    /*
     * Check that this i is on this core;
     * most of the time, it will be, but sometimes
     * columns are split up by core.
     */
    this_loc = total_levels*i + this_i;
    if (this_loc>=my_start&&this_loc<my_end) {
      get_dm_element_qvec_local(rho,this_i,i,&dm_element);
      /*
       * Take complex conjugate of dm_element (since we relied on the fact
       * that rho was hermitian to get better data locality)
       */
      *trace_val = *trace_val + op_val*(dm_element);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,trace_val,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

  return;
}

void _get_expectation_value_wf(qvec psi,PetscScalar *trace_val,PetscInt num_ops,operator *ops){
  PetscInt Istart,Iend,location[1],i,j,this_j,this_i;
  PetscScalar val_array[1],val,op_val;
  Vec op_psi;

  /*
   * FIXME: Check for consistency in operator sizes and wf size
   */


  *trace_val = 0.0;
  VecGetOwnershipRange(psi->data,&Istart,&Iend);
  VecDuplicate(psi->data,&op_psi);
  //Calculate A * B * Psi
  for (i=0;i<total_levels;i++){
    this_i = i; // The leading index which we check
    op_val = 1.0;
    for (j=0;j<num_ops;j++){
      _get_val_j_from_global_i(this_i,ops[j],&this_j,&val,-1); // Get the corresponding j and val
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
    //Now we have op1*op2*op3...
    if (this_i>=Istart&&this_i<Iend&&op_val!=0.0){
      //this val belongs to me, do the (local) multiplication
      location[0] = this_i;
      VecGetValues(psi->data,1,location,val_array);
      op_val = op_val * val_array[0];
      //Add the value to the op_psi
      VecSetValue(op_psi,i,op_val,ADD_VALUES);
    }
  }
  // Now, calculate the inner product between psi^H * OP_psi
  VecAssemblyBegin(op_psi);
  VecAssemblyEnd(op_psi);
  VecDot(op_psi,psi->data,trace_val);
  VecDestroy(&op_psi);

  return;
}
