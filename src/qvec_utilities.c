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
 *    Add print_qvec_sparse
 *    Add print_qvec_file
 */

void check_qvec_consistent(qvec state1,qvec state2){
  PetscInt error=0;
  if(state1->my_type!=state2->my_type){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvecs have different types!\n");
    error=1;
  }
  if(state1->n!=state2->n){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvecs have different sizes!\n");
    error=1;
  }
  if(error==1){
    exit(1);
  }
  return;
}

void copy_qvec(qvec source,qvec destination){

  //check that the qvecs are consistent
  if(source->my_type==WAVEFUNCTION&&destination->my_type==DENSITY_MATRIX){
    if((source->n*source->n)!=destination->n){
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvecs have different sizes!\n");
      exit(1);
    }
    copy_qvec_wf_to_dm(source,destination);
  } else if(source->my_type==destination->my_type){
    if(source->n!=destination->n){
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvecs have different sizes!\n");
      exit(1);
    }
    VecCopy(source->data,destination->data);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Coping a DM to a WF does not make sense!\n");
    exit(1);
  }

  return;
}

// Done in serial for now, inefficient
void copy_qvec_wf_to_dm(qvec source,qvec destination){
  VecScatter        ctx_wf;
  Vec wf_local;
  PetscScalar       *wf_a,this_val;
  PetscInt          n,i,j,this_row;
  if(source->n*source->n!=destination->n){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! wf and dm do not have consistent sizes!\n");
    exit(1);
  }

  //Collect all data for the wf on one core
  VecScatterCreateToZero(source->data,&ctx_wf,&wf_local);

  VecScatterBegin(ctx_wf,source->data,wf_local,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(ctx_wf,source->data,wf_local,INSERT_VALUES,SCATTER_FORWARD);

  /* Rank 0 now has a local copy of the wf, it will do the calculation */
  if (nid==0){
    VecGetArray(wf_local,&wf_a);
    VecGetSize(wf_local,&n);
    //Now calculate the outer product
    for(i=0;i<n;i++){
      for(j=0;j<n;j++){
        this_val = wf_a[i]*PetscConjComplex(wf_a[j]);
        this_row = j*n+i;
        VecSetValue(destination->data,this_row,this_val,INSERT_VALUES);
      }
    }
  }
  VecAssemblyBegin(destination->data);
  VecAssemblyEnd(destination->data);
  VecDestroy(&wf_local);
  return;
}



//Read in a qvec from a qutip generated file with function file_data_store
//Not recommended for large vectors
void read_qvec_wf_binary(qvec *newvec,const char filename[]){
  qvec temp = NULL;
  PetscViewer viewer;
  PetscInt n,Istart,Iend;

  temp = malloc(sizeof(struct qvec));
  VecCreate(PETSC_COMM_WORLD,&(temp->data));
  VecSetType(temp->data,VECMPI);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);
  VecLoad(temp->data,viewer);
  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = WAVEFUNCTION;
  temp->n = n;
  temp->Istart = Istart;
  temp->Iend = Iend;

  *newvec = temp;
  return;
}

//Read in a qvec from a qutip generated file with function file_data_store
//Not recommended for large vectors
void read_qvec_dm_binary(qvec *newvec,const char filename[]){
  qvec temp = NULL;
  PetscViewer viewer;
  PetscInt n,Istart,Iend;

  temp = malloc(sizeof(struct qvec));
  VecCreate(PETSC_COMM_WORLD,&(temp->data));
  VecSetType(temp->data,VECMPI);
  PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);
  VecLoad(temp->data,viewer);
  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = DENSITY_MATRIX;
  temp->n = n;
  temp->Istart = Istart;
  temp->Iend = Iend;

  *newvec = temp;
  return;
}

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
      PetscFPrintf(PETSC_COMM_WORLD,fp,"%40.30e  %40.30e ",PetscRealPart(val),
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

void get_wf_element_qvec_local(qvec state,PetscInt i,PetscScalar *val){
  PetscInt location[1];
  PetscScalar val_array[1];

  location[0] = i;
  if (location[0]>=state->Istart&&location[0]<state->Iend) {
    VecGetValues(state->data,1,location,val_array);
  } else{
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Can only get elements local to a core in !\n");
    PetscPrintf(PETSC_COMM_WORLD,"       get_wf_element_qvec_local.\n");
    exit(0);
  }

  *val = val_array[0];
  return;
}

/*
 * create a qvec object with arbitrary dimensions
 */
void create_arb_qvec(qvec *new_qvec,PetscInt nstates,qvec_type my_type){
  qvec temp = NULL;
  PetscInt n,Istart,Iend,iVar;
  PetscReal fVar;

  temp = malloc(sizeof(struct qvec));

  _create_vec(&(temp->data),nstates,-1);

  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = my_type;
  temp->n = n;
  if(my_type==WAVEFUNCTION){
    temp->total_levels = n;
  } else if(my_type==DENSITY_MATRIX){
    //Perhaps check that n in a true square?
    fVar = sqrt((double)(n));
    iVar = fVar;
    if(iVar==fVar){
      temp->total_levels = iVar;
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! n for DENSITY_MATRIX must be perfect square!\n");
      exit(9);
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Type not understood in create_arb_qvec.\n");
    exit(9);
  }
  temp->Istart = Istart;
  temp->Iend = Iend;

  *new_qvec = temp;

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
  temp->total_levels = sqrt(n);
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
  temp->total_levels = n;
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
  if(local_size<0){
    //Negative numbers mean we want PETSc to distribute our
    VecSetSizes(*dm,PETSC_DECIDE,dim);
  }else{
    VecSetSizes(*dm,local_size,dim);
  }
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



void get_hilbert_schmidt_dist_qvec(qvec q1,qvec q2,PetscReal *hs_dist) {
  Vec temp;
  PetscScalar alpha=-1.0;

  //Check sizes


  if (q1->my_type==DENSITY_MATRIX && q2->my_type==DENSITY_MATRIX){
    if (q1->n==q2->n){
      VecDuplicate(q1->data,&temp);
      VecCopy(q1->data,temp);
      VecAXPY(temp,alpha,q2->data);
      VecNorm(temp,NORM_2,hs_dist);
      *hs_dist = *hs_dist * (*hs_dist);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! DMs are not the same size in hs_dist\n");
      exit(9);
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! HS is only for DM-DM comparisons\n");
    exit(9);
  }
  return;
}

void get_bitstring_probs(qvec rho,PetscInt *nloc,PetscReal **probs){
  //FIXME Dangerous to return nloc, exposes parallelism to user

  if(rho->my_type==DENSITY_MATRIX){
    _get_bitstring_probs_dm(rho,nloc,probs);
  } else if(rho->my_type==WAVEFUNCTION){
    _get_bitstring_probs_wf(rho,nloc,probs);
  }

  return;
}

void get_linear_xeb_fidelity_probs(PetscReal *probs_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscInt nloc_exp,PetscInt h_dim,PetscReal *lin_xeb_fid){
  PetscReal tmp_lin_xeb_fid[1];
  PetscInt i;

  if(nloc_ref!=nloc_exp){
    printf("ERROR! Ref and Exp are not the same size in get_linear_xeb_fidelity_probs!\n");
    exit(9);
  }
  /*
   * linear xeb fid estimator:
   * F_lxeb = <D p(q) - 1>, but we have the distribution of the bitstrings q, so
   * F_lxeb = D \sum_i p_exp(q_i) p_true(q_i) - 1
   * where D is total hilbert space size
   */
  tmp_lin_xeb_fid[0] = 0;
  for(i=0;i<nloc_ref;i++){
    tmp_lin_xeb_fid[0] = tmp_lin_xeb_fid[0] + probs_exp[i]*probs_ref[i];
  }
  //Collect all cores
  MPI_Allreduce(MPI_IN_PLACE,tmp_lin_xeb_fid,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  *lin_xeb_fid = h_dim * tmp_lin_xeb_fid[0] - 1;

  return;
}

void get_linear_xeb_fidelity(qvec ref,qvec exp,PetscReal *lin_xeb_fid){
  PetscReal *probs_ref,*probs_exp;
  PetscInt nloc_ref,nloc_exp;
  /*
   * Our 'experiment' is stored in a dm and we can extract the bitstring
   * probabilities by just taking the diagonal part
   * Similarly, for our reference
   */
  if(ref->total_levels!=exp->total_levels){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Ref and Exp are not the same size in get_linear_xeb_fidelity!\n");
    exit(9);
  }
  get_bitstring_probs(ref,&nloc_ref,&probs_ref);
  get_bitstring_probs(exp,&nloc_exp,&probs_exp);

  get_linear_xeb_fidelity_probs(probs_ref,nloc_ref,probs_exp,nloc_exp,ref->total_levels,lin_xeb_fid);

  free(probs_exp);
  free(probs_ref);
  return;
}

void get_log_xeb_fidelity_probs(PetscReal *probs_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscInt nloc_exp,PetscInt h_dim,PetscReal *log_xeb_fid){
  PetscReal gamma=0.57721566490153286060651209008240243104215933593992;
  PetscReal tmp_log_xeb_fid[1];
  PetscInt i;

  if(nloc_ref!=nloc_exp){
    //FIXME Dangerous print here
    printf("ERROR! Ref and Exp are not the same size in get_log_xeb_fidelity_probs!\n");
    exit(9);
  }

  /*
   * log xeb fid estimator:
   * F_lxeb = <log(D p(q)) + gamma>, but we have the distribution of the bitstrings q, so
   * F_lxeb =  \sum_i p_exp (q_i) log(D p_true(q_i)) + gamma
   * where D is total hilbert space size and gamma is the Euler Mascheroni Constant
   */

  tmp_log_xeb_fid[0] = 0;
  for(i=0;i<nloc_ref;i++){
    if(probs_ref[i]==0.0){
      tmp_log_xeb_fid[0] = tmp_log_xeb_fid[0] + probs_exp[i]*PetscLogReal(3e-33);
    } else{
      tmp_log_xeb_fid[0] = tmp_log_xeb_fid[0] + probs_exp[i]*PetscLogReal(probs_ref[i]);
    }
  }
  //Collect all cores
  MPI_Allreduce(MPI_IN_PLACE,tmp_log_xeb_fid,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  *log_xeb_fid = PetscLogReal(h_dim) + tmp_log_xeb_fid[0] + gamma;

  return;
}

void get_log_xeb_fidelity(qvec ref,qvec exp,PetscReal *log_xeb_fid){
  PetscReal *probs_ref,*probs_exp;
  PetscInt nloc_ref,nloc_exp;

  /*
   * Our 'experiment' is stored in a dm and we can extract the bitstring
   * probabilities by just taking the diagonal part
   * Similarly, for our reference
   */
  if(ref->total_levels!=exp->total_levels){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Ref and Exp are not the same size in get_linear_xeb_fidelity!\n");
    exit(9);
  }

  get_bitstring_probs(ref,&nloc_ref,&probs_ref);
  get_bitstring_probs(exp,&nloc_exp,&probs_exp);

  get_log_xeb_fidelity_probs(probs_ref,nloc_ref,probs_exp,nloc_exp,ref->total_levels,log_xeb_fid);

  free(probs_exp);
  free(probs_ref);

  return;
}

void get_hog_score_probs(PetscReal *probs_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscInt nloc_exp,PetscInt h_dim,PetscReal *hog_score){
  PetscReal tmp_hog_score[1];
  PetscInt i;

  if(nloc_ref!=nloc_exp){
    //FIXME Dangerous print here
    printf("ERROR! Ref and Exp are not the same size in get_hog_score_probs!\n");
    exit(9);
  }

  /*
   * HOG Score description
   */

  tmp_hog_score[0] = 0;
  for(i=0;i<nloc_ref;i++){
    if(probs_ref[i]>=log(2.0)/h_dim){
      tmp_hog_score[0] = tmp_hog_score[0] + probs_exp[i];
    }
  }
  //Collect all cores
  MPI_Allreduce(MPI_IN_PLACE,tmp_hog_score,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  *hog_score = tmp_hog_score[0];

  return;
}

void get_hog_score(qvec ref,qvec exp,PetscReal *hog_score){
  PetscReal *probs_ref,*probs_exp;
  PetscInt nloc_ref,nloc_exp;
  /*
   * Our 'experiment' is stored in a dm and we can extract the bitstring
   * probabilities by just taking the diagonal part
   * Similarly, for our reference
   */

  if(ref->total_levels!=exp->total_levels){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Ref and Exp are not the same size in get_linear_xeb_fidelity!\n");
    exit(9);
  }
  get_bitstring_probs(ref,&nloc_ref,&probs_ref);
  get_bitstring_probs(exp,&nloc_exp,&probs_exp);

  get_hog_score_probs(probs_ref,nloc_ref,probs_exp,nloc_exp,ref->total_levels,hog_score);

  free(probs_exp);
  free(probs_ref);

  return;
}


void get_hog_score_fidelity_probs(PetscReal *probs_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscInt nloc_exp,PetscInt h_dim,PetscReal *hog_score_fid){
  PetscReal hog_score;

  if(nloc_ref!=nloc_exp){
    //FIXME Dangerous print here
    printf("ERROR! Ref and Exp are not the same size in get_hog_score_probs!\n");
    exit(9);
  }

  get_hog_score_probs(probs_ref,nloc_ref,probs_exp,nloc_exp,h_dim,&hog_score);
  *hog_score_fid = (2*hog_score - 1)/log(2.0);

  return;
}

void get_hog_score_fidelity(qvec ref,qvec exp,PetscReal *hog_score_fid){
  PetscReal hog_score;

  /*
   * Our 'experiment' is stored in a dm and we can extract the bitstring
   * probabilities by just taking the diagonal part
   * Similarly, for our reference
   */
  if(ref->total_levels!=exp->total_levels){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Ref and Exp are not the same size in get_hog_score_fidelity!\n");
    exit(9);
  }

  get_hog_score(ref,exp,&hog_score);
  *hog_score_fid = (2*hog_score - 1)/log(2.0);
  return;
}


void _get_bitstring_probs_dm(qvec q1,PetscInt *num_loc,PetscReal **probs){
  PetscScalar tmp_scalar=0;
  PetscInt i=0,loc=0,num_loc_t=0;

  num_loc_t = 0;
  //Loops over 2^n, but only touches local elements. Perhaps rewrite so it only loops over local parts
  //Count number of local diagonal elements
  for(i=0;i<q1->total_levels;i++){
    loc = q1->total_levels*i+i;
    if(loc>=q1->Istart && loc<q1->Iend){
      num_loc_t = num_loc_t+1;
    }
  }
  *num_loc = num_loc_t-1; //Don't forget to subtract one?
  //Allocate array
  (*probs) = malloc(num_loc_t*sizeof(PetscReal));
  num_loc_t=0;
  for(i=0;i<q1->total_levels;i++){
    loc = q1->total_levels*i+i;
    if(loc>=q1->Istart && loc<q1->Iend){
      //Get local diagonal parts for both q1 and q2
      get_dm_element_qvec_local(q1,i,i,&tmp_scalar);
      (*probs)[num_loc_t] = PetscRealPart(tmp_scalar);
      num_loc_t = num_loc_t+1;
    }
  }

  return;
}


void _get_bitstring_probs_wf(qvec q1,PetscInt *num_loc,PetscReal **probs){
  PetscScalar tmp_scalar=0;
  PetscInt i=0,loc=0,j=0;

  //Allocate array
  *num_loc = q1->Iend-q1->Istart;
  (*probs) = malloc((*num_loc)*sizeof(PetscReal));
  for(i=0;i<q1->total_levels;i++){
    loc = i;
    if(loc>=q1->Istart && loc<q1->Iend){
      //Get local diagonal parts for both q1 and q2
      get_wf_element_qvec_local(q1,loc,&tmp_scalar);
      (*probs)[j] = pow(PetscRealPart(PetscAbsComplex(tmp_scalar)),2);
      j++;
    }
  }

  return;
}


/*
 * void get_superfidelity calculates the superfidelity between two matrices,
 * where the superfidelity for density matrices is defined as:
 *         F = Tr(rho sigma) + sqrt(1-Tr(rho*rho))*sqrt(1-Tr(sigma*sigma))
 * where rho, sigma are the density matrices to calculate the
 * See https://arxiv.org/pdf/0805.2037.pdf
 * Inputs:
 *         qvec q1  - one quantum system
 *         qvec q2 - the other quantum system
 * Outpus:
 *         PetscReal *superfidelity - the fidelity between the two dms
 *
 */
void get_superfidelity_qvec(qvec q1,qvec q2,PetscReal *superfidelity) {
  qvec tmp_dm;
  if (q1->my_type==DENSITY_MATRIX && q2->my_type==DENSITY_MATRIX){
    _get_superfidelity_dm_dm(q1->data,q2->data,superfidelity);
  } else if(q1->my_type==WAVEFUNCTION && q2->my_type==WAVEFUNCTION){
    PetscPrintf(PETSC_COMM_WORLD,"Calculating superfidelity between two wavefunctions is not recommended. Use get_fidelity_qvec!\n");
    exit(9);
  } else if(q1->my_type==WAVEFUNCTION && q2->my_type==DENSITY_MATRIX){
    //Copy wf into DM
    //Inefficient because copy_qvec_wf_to_dm is inefficient
    create_arb_qvec(&tmp_dm,q2->n,DENSITY_MATRIX);
    copy_qvec_wf_to_dm(q1,tmp_dm);
    _get_superfidelity_dm_dm(q2->data,tmp_dm->data,superfidelity);
    destroy_qvec(&tmp_dm);
  } else if(q2->my_type==WAVEFUNCTION && q1->my_type==DENSITY_MATRIX){
    //Copy wf into DM
    //Inefficient because copy_qvec_wf_to_dm is inefficient
    create_arb_qvec(&tmp_dm,q1->n,DENSITY_MATRIX);
    copy_qvec_wf_to_dm(q2,tmp_dm);
    _get_superfidelity_dm_dm(q1->data,tmp_dm->data,superfidelity);
    destroy_qvec(&tmp_dm);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Types not understand in get_superfidelity_qvec!\n");
    exit(9);
  }
  return;
}

void  _get_superfidelity_dm_dm(Vec dm1,Vec dm2,PetscReal *superfidelity){
  PetscScalar val1,val2,val3;

  //Get Tr(rho sigma) = <<rho | sigma >>
  VecDot(dm1,dm2,&val1);
  //Get Tr(rho rho) = <<rho | rho >>
  VecDot(dm1,dm1,&val2);
  //Get Tr(sigma sigma) = <<sigma | sigma >>
  VecDot(dm2,dm2,&val3);

  *superfidelity = PetscSqrtReal(PetscRealPart(val1 + PetscSqrtComplex(1-val2)*PetscSqrtComplex(1-val3)));
  return;
}


/*
 * void get_fidelity calculates the fidelity between two matrices,
 * where the fidelity for density matrices is defined as:
 *         F = Tr(sqrt(sqrt(rho) sigma sqrt(rho)))
 * where rho, sigma are the density matrices to calculate the
 * fidelity between and fidelity for wave functions is defined as
 *         F = |<psi_1 | psi_2>|^2
 * Inputs:
 *         qvec  q1  - one quantum system
 *         qvec q2 - the other quantum system
 * Outpus:
 *         PetscReal *fidelity - the fidelity between the two dms
 *
 */
void get_fidelity_qvec(qvec q1,qvec q2,PetscReal *fidelity) {

  if (q1->my_type==DENSITY_MATRIX && q2->my_type==DENSITY_MATRIX){
    _get_fidelity_dm_dm(q1->data,q2->data,fidelity);
  } else if(q1->my_type==WAVEFUNCTION && q2->my_type==WAVEFUNCTION){
    _get_fidelity_wf_wf(q1->data,q2->data,fidelity);
  } else if(q1->my_type==WAVEFUNCTION && q2->my_type==DENSITY_MATRIX){
    _get_fidelity_dm_wf(q2->data,q1->data,fidelity);
  } else if(q2->my_type==WAVEFUNCTION && q1->my_type==DENSITY_MATRIX){
    _get_fidelity_dm_wf(q1->data,q2->data,fidelity);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Types not understand in get_fidelity_qvec!\n");
    exit(9);
  }
  return;
}

void _get_fidelity_wf_wf(Vec wf1, Vec wf2,PetscReal *fidelity){
  PetscScalar val;
  //F = |<psi_1 | psi_2>|^2
  VecDot(wf1,wf2,&val);
  *fidelity = pow(PetscAbsComplex(val),2);
  return;
}


/*
 * void get_fidelity calculates the fidelity between two matrices,
 * where the fidelity is defined as:
 *         F = <psi | rho | psi >
 * where rho, psi are the density matrix, wavefunction to calculate the
 * fidelity between
 *
 * Inputs:
 *         Vec dm   - one density matrix with which to find the fidelity
 *         Vec wf - the wavefunction
 * Outpus:
 *         double *fidelity - the fidelity between the two dms
 *
 * NOTE: This is probably inefficient at scale
 */
void _get_fidelity_dm_wf(Vec dm,Vec wf,PetscReal *fidelity) {
  PetscInt          i,j,dm_size,wf_size,local_dm_size;
  PetscScalar       val;
  Mat               dm_mat;
  Vec               result_vec;
  /* Variables needed for LAPACK */


  VecGetSize(dm,&dm_size);
  VecGetSize(wf,&wf_size);

  if (sqrt(dm_size)!=wf_size){
    if (nid==0){
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! The input density matrix and wavefunction are not compatible!\n");
      PetscPrintf(PETSC_COMM_WORLD,"       Fidelity cannot be calculated.\n");
      exit(0);
    }
  }


  //Make a new temporary vector
  VecDuplicate(wf,&result_vec);

  /*
   * We want to work with the density matrix as a matrix directly,
   */
  MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,wf_size,wf_size,NULL,&dm_mat);

  for (i=0;i<wf_size;i++){
    for (j=0;j<wf_size;j++){
      get_dm_element(dm,i,j,&val);
      MatSetValue(dm_mat,i,j,val,INSERT_VALUES);
    }
  }
  MatAssemblyBegin(dm_mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(dm_mat,MAT_FINAL_ASSEMBLY);

  /*
   * calculate rho | psi>
   */
  MatMult(dm_mat,wf,result_vec);

  /*
   * calculate <psi | rho | psi >
   */
  VecDot(wf,result_vec,&val);
  *fidelity = val;
  MatDestroy(&dm_mat);
  VecDestroy(&result_vec);
  return;
}


/*
 * void get_fidelity calculates the fidelity between two matrices,
 * where the fidelity is defined as:
 *         F = Tr(sqrt(sqrt(rho) sigma sqrt(rho)))
 * where rho, sigma are the density matrices to calculate the
 * fidelity between
 *
 * Inputs:
 *         Vec dm   - one density matrix with which to find the fidelity
 *         Vec dm_r - the other density matrix
 * Outpus:
 *         double *fidelity - the fidelity between the two dms
 *
 * NOTE: This is probably inefficient at scale
 */
void _get_fidelity_dm_dm(Vec dm,Vec dm_r,PetscReal *fidelity) {
  VecScatter        ctx_dm,ctx_dm_r;
  PetscInt          i,dm_size,dm_r_size,levels;
  PetscScalar       *dm_a,*dm_r_a;
  Mat               dm_mat,dm_mat_r,result_mat;
  Vec               dm_local,dm_r_local;
  /* Variables needed for LAPACK */
  PetscScalar  *work,*eigs,sdummy;
  PetscReal    *rwork;
  PetscBLASInt idummy,lwork,lierr,nb;

  VecGetSize(dm,&dm_size);
  VecGetSize(dm_r,&dm_r_size);

  if (dm_size!=dm_r_size){
    if (nid==0){
      printf("ERROR! The input density matrices are not the same size!\n");
      printf("       Fidelity cannot be calculated.\n");
      exit(0);
    }
  }

  /* Collect both DM's onto master core */
  VecScatterCreateToZero(dm,&ctx_dm,&dm_local);
  VecScatterCreateToZero(dm_r,&ctx_dm_r,&dm_r_local);

  VecScatterBegin(ctx_dm,dm,dm_local,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterBegin(ctx_dm_r,dm_r,dm_r_local,INSERT_VALUES,SCATTER_FORWARD);

  VecScatterEnd(ctx_dm,dm,dm_local,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(ctx_dm_r,dm_r,dm_r_local,INSERT_VALUES,SCATTER_FORWARD);

  /* Rank 0 now has a local copy of the matrices, so it does the calculations */
  if (nid==0){
    levels = sqrt(dm_size);
    /*
     * We want to work with the density matrices as matrices directly,
     * so that we can get eigenvalues, etc.
     */
    VecGetArray(dm_local,&dm_a);
    MatCreateSeqDense(PETSC_COMM_SELF,levels,levels,dm_a,&dm_mat);

    VecGetArray(dm_r_local,&dm_r_a);
    MatCreateSeqDense(PETSC_COMM_SELF,levels,levels,dm_r_a,&dm_mat_r);

    /* Get the sqrt of the matrix */
    /* printf("before sqrt1\n"); */
    /* MatView(dm_mat,PETSC_VIEWER_STDOUT_SELF); */

    sqrt_mat(dm_mat);
    /* MatView(dm_mat,PETSC_VIEWER_STDOUT_SELF);     */
    /* calculate sqrt(dm_mat)*dm_mat_r */
    MatMatMult(dm_mat,dm_mat_r,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&result_mat);
    /*
     * calculate (sqrt(dm_mat)*dm_mat_r)*sqrt(dm_mat)
     * we reuse dm_mat_r as our result to save memory, since we are done with
     * that data
     */
    MatMatMult(result_mat,dm_mat,MAT_REUSE_MATRIX,PETSC_DEFAULT,&dm_mat_r);


    /* Get eigenvalues of result_mat */

    idummy = levels;
    lwork  = 5*levels;
    PetscMalloc1(5*levels,&work);
    PetscMalloc1(2*levels,&rwork);
    PetscMalloc1(levels,&eigs);
    PetscBLASIntCast(levels,&nb);

    /* Call LAPACK through PETSc to ensure portability */
    LAPACKgeev_("N","N",&nb,dm_r_a,&nb,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,rwork,&lierr);
    *fidelity = 0;
    for (i=0;i<levels;i++){
      /*
       * Only positive values because sometimes we get small, negative eigenvalues
       * Also, we take the real part, because of small, imaginary parts
       */
      if (PetscRealPart(eigs[i])>0){
        *fidelity = *fidelity + sqrt(PetscRealPart(eigs[i]));
      }
    }
    VecRestoreArray(dm_local,&dm_a);
    VecRestoreArray(dm_r_local,&dm_r_a);
    /* Clean up memory */
    MatDestroy(&dm_mat);
    MatDestroy(&dm_mat_r);
    MatDestroy(&result_mat);
    PetscFree(work);
    PetscFree(rwork);
    PetscFree(eigs);
  }

  /* Broadcast the value to all cores */
  MPI_Bcast(fidelity,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  VecDestroy(&dm_local);
  VecDestroy(&dm_r_local);

  VecScatterDestroy(&ctx_dm);
  VecScatterDestroy(&ctx_dm_r);


  return;
}

void load_sparse_mat_qvec(char filename[],Mat *write_mat,qvec rho){
  PetscViewer    fd;

  PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&fd);
  MatCreate(PETSC_COMM_WORLD,write_mat);
  MatLoad(*write_mat,fd);
  PetscViewerDestroy(&fd);
  return;
}
