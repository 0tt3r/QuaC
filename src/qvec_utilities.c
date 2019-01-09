#include "qvec_utilities.h"
#include "operators_p.h"
#include "operators.h"
#include <stdlib.h>
#include <stdio.h>
#include <petscblaslapack.h>
#include <string.h>
/*
 * FIXME:
 *    Add create_{dm,wf}_qvec to create any sized dm, even if it isn't tied to a system
 */



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

void print_dm_qvec(qvec state){
  PetscScalar val;
  PetscInt i,j,h_dim;
  h_dim = sqrt(state->n);
  for (i=0;i<h_dim;i++){
    for (j=0;j<h_dim;j++){
      //get_dm_element(rho,i,j,&val); FIXME Write this
      PetscPrintf(PETSC_COMM_WORLD,"%4.3e + %4.3ei ",PetscRealPart(val),
                  PetscImaginaryPart(val));
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n");
  }
  PetscPrintf(PETSC_COMM_WORLD,"\n");
  return;
}

void print_wf_qvec(qvec state){
  PetscScalar val;
  PetscInt i;

  for(i=0;i<state->n;i++){
    get_wf_element_qvec(state,i,&val);
    PetscPrintf(PETSC_COMM_WORLD,"%4.3e + %4.3ei\n",PetscRealPart(val),
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
  add_to_qvec_loc(alpha,loc,state);

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

  add_to_qvec_loc(alpha,loc,state);

  return;
}

/*
 * Add alpha to the element at location loc in the qvec
 */

void add_to_qvec_loc(PetscScalar alpha,PetscInt loc,qvec state){

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
    *loc = *loc * state->n + *loc;
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
    _get_expectation_value_wf(state->data,trace_val,num_ops,ops);
  }

  return;
}


void _get_expectation_value_wf(Vec psi,PetscScalar *trace_val,PetscInt num_ops,operator *ops){
  PetscInt Istart,Iend,location[1],i,j,this_j,this_i;
  PetscScalar val_array[1],val,op_val;
  Vec op_psi;
  *trace_val = 0.0;
  VecGetOwnershipRange(psi,&Istart,&Iend);
  VecDuplicate(psi,&op_psi);
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
      VecGetValues(psi,1,location,val_array);
      op_val = op_val * val_array[0];
      //Add the value to the op_psi
      VecSetValue(op_psi,i,op_val,ADD_VALUES);
    }
  }
  // Now, calculate the inner product between psi^H * OP_psi
  VecAssemblyBegin(op_psi);
  VecAssemblyEnd(op_psi);
  VecDot(op_psi,psi,trace_val);
  VecDestroy(&op_psi);
}
