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

void check_qvec_equal(qvec state1,qvec state2,PetscBool *flag){
  check_qvec_consistent(state1,state2);
  VecEqual(state1->data,state2->data,flag);
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
  VecScatter        scatter_ctx;
  Vec wf_local;
  PetscScalar       *wf_a,this_val;
  PetscInt          n,i,j,this_row;
  if(source->n*source->n!=destination->n){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! wf and dm do not have consistent sizes!\n");
    exit(1);
  }

  //Collect all data for the wf on one core
  VecScatterCreateToZero(source->data,&scatter_ctx,&wf_local);

  VecScatterBegin(scatter_ctx,source->data,wf_local,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(scatter_ctx,source->data,wf_local,INSERT_VALUES,SCATTER_FORWARD);

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
  VecScatterDestroy(&scatter_ctx);
  return;
}



//Read in a qvec from a PetscBinary format
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

  temp->total_levels = n;
  //Assume one big hspace for now
  temp->n_ops = 1;
  temp->hspace_dims = malloc(sizeof(PetscInt));
  temp->hspace_dims[0] = temp->total_levels;

  *newvec = temp;

  PetscViewerDestroy(&viewer);
  return;
}

//Read in a qvec from a from a PetscBinary format
//Not recommended for large vectors
void read_qvec_dm_binary(qvec *newvec,const char filename[]){
  qvec temp = NULL;
  PetscViewer viewer;
  PetscInt n,Istart,Iend,iVar;
  PetscReal fVar;

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

  fVar = sqrt((double)(n));
  iVar = fVar;
  if(iVar==fVar){
    temp->total_levels = iVar;
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! n for DENSITY_MATRIX must be perfect square!\n");
    exit(9);
  }

  temp->n_ops = 1;
  temp->hspace_dims = malloc(2*sizeof(PetscInt));
  temp->hspace_dims[0] = temp->total_levels;
  temp->hspace_dims[1] = temp->total_levels;

  *newvec = temp;

  PetscViewerDestroy(&viewer);
  return;
}

//A*state - be careful with use, might not be intuitive behavior for DM
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
    if(state->my_type==WF_ENSEMBLE && state->ens_spawned==PETSC_TRUE){
      print_wf_ens_i_qvec(state);
    } else {
      print_wf_qvec(state);
    }
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
  PetscInt loc;

  loc = sqrt(dm->n)*col + row;

  _get_qvec_element_local(dm,loc,val);

  return;
}

void _get_qvec_element_local(qvec state,PetscInt loc,PetscScalar *val){
  PetscInt location[1],dm_size,my_start,my_end;
  PetscScalar val_array[1];

  location[0] = loc;

  if (location[0]>=state->Istart&&location[0]<state->Iend) {
    VecGetValues(state->data,1,location,val_array);
  } else{
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Can only get elements local to a core in !\n");
    PetscPrintf(PETSC_COMM_WORLD,"       _get_qvec_element_local.\n");
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

void get_wf_element_qvec(qvec state,PetscInt loc,PetscScalar *val){

  _get_qvec_element_local(state,loc,val);

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


void get_wf_element_ens_i_qvec_local(qvec state,PetscInt ens_idx,PetscInt i,PetscScalar *val){
  PetscInt location[1];
  PetscScalar val_array[1];

  location[0] = i;
  if (location[0]>=state->Istart&&location[0]<state->Iend) {
    VecGetValues(state->ens_datas[ens_idx],1,location,val_array);
  } else{
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Can only get elements local to a core in !\n");
    PetscPrintf(PETSC_COMM_WORLD,"       get_wf_element_qvec_local.\n");
    exit(0);
  }

  *val = val_array[0];
  return;
}



/*
 * Print the dense wf_ens[ens_i]
 */
void print_wf_ens_i_qvec(qvec state){
  PetscScalar val;
  PetscInt i;

  for(i=0;i<state->n;i++){
    get_wf_element_ens_i_qvec_local(state,state->ens_i,i,&val);
    PetscPrintf(PETSC_COMM_WORLD,"%e + %ei\n",PetscRealPart(val),
                PetscImaginaryPart(val));
  }
  return;
}

/*
 * create a qvec object with arbitrary dimension
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
    //Assume one big hspace for now
    temp->n_ops = 1;
    temp->hspace_dims = malloc(sizeof(PetscInt));
    temp->hspace_dims[0] = temp->total_levels;

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
    temp->n_ops = 1;
    temp->hspace_dims = malloc(2*sizeof(PetscInt));
    temp->hspace_dims[0] = temp->total_levels;
    temp->hspace_dims[1] = temp->total_levels;
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Type not understood in create_arb_qvec.\n");
    exit(9);
  }
  temp->Istart = Istart;
  temp->Iend = Iend;
  temp->ndims_hspace = 1;
  temp->ens_spawned=PETSC_FALSE;
  *new_qvec = temp;

  return;

}

/*
 * create a qvec object with arbitrary dimensions
 */
void create_arb_qvec_dims(qvec *new_qvec,PetscInt ndims,PetscInt *dims,qvec_type my_type){
  qvec temp = NULL;
  PetscInt n,Istart,Iend,nstates,i;

  //Calculate the total nstates from the dims array
  nstates=1;
  for(i=0;i<ndims;i++){
    nstates = nstates * dims[i];
  }

  temp = malloc(sizeof(struct qvec));

  if(my_type==WAVEFUNCTION){

    _create_vec(&(temp->data),nstates,-1);
    VecGetSize(temp->data,&n);
    VecGetOwnershipRange(temp->data,&Istart,&Iend);
    temp->total_levels = n;
    temp->n_ops = ndims;
    temp->ndims_hspace = ndims;
    temp->hspace_dims = malloc(ndims*sizeof(PetscInt));

    for(i=0;i<ndims;i++){
      temp->hspace_dims[i] = dims[i];
    }

  } else if(my_type==DENSITY_MATRIX){

    _create_vec(&(temp->data),nstates,-1);
    VecGetSize(temp->data,&n);
    VecGetOwnershipRange(temp->data,&Istart,&Iend);
    temp->total_levels = pow(nstates,0.5);
    temp->n_ops = ndims/2;
    temp->ndims_hspace = ndims;
    temp->hspace_dims = malloc(ndims*sizeof(PetscInt));
    for(i=0;i<ndims;i++){
      temp->hspace_dims[i] = dims[i];
    }

  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Type not understood in create_arb_qvec_dims.\n");
    exit(9);
  }

  temp->my_type = my_type;
  temp->n = n;
  temp->Istart = Istart;
  temp->Iend = Iend;


  *new_qvec = temp;

  return;

}


/*
 * change a qvec object's with arbitrary dimensions
 */
void change_qvec_dims(qvec state,PetscInt ndims,PetscInt *dims){
  PetscInt i, nstates;

  //Calculate the total nstates from the dims array
  nstates=1;
  for(i=0;i<ndims;i++){
    nstates = nstates * dims[i];
  }


  //Check for consistency
  if(nstates!=state->n){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvec dims not consistent in change_qvec_dims!\n");
    exit(9);
  }


  if(state->my_type==WAVEFUNCTION){

    state->n_ops = ndims;
    state->ndims_hspace = ndims;
    free(state->hspace_dims);
    state->hspace_dims = malloc(ndims*sizeof(PetscInt));

    for(i=0;i<ndims;i++){
      state->hspace_dims[i] = dims[i];
    }

  } else if(state->my_type==DENSITY_MATRIX){

    state->n_ops = ndims/2;
    state->ndims_hspace = ndims;
    free(state->hspace_dims);
    state->hspace_dims = malloc(ndims*sizeof(PetscInt));
    for(i=0;i<ndims;i++){
      state->hspace_dims[i] = dims[i];
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Type not understood in change_qvec_dims.\n");
    exit(9);
  }


  return;
}


/*
 * create a qvec object as a dm or wf based on
 * the operators added thus far
 */
void create_qvec_sys(qsystem qsys,qvec *new_qvec){
  //Automatically decide whether to make a DM or WF
  if (qsys->dm_equations==PETSC_TRUE && qsys->mcwf_solver==PETSC_FALSE){
    create_dm_sys(qsys,new_qvec);
  } else if(qsys->dm_equations==PETSC_TRUE && qsys->mcwf_solver==PETSC_TRUE){
    create_wf_ensemble_sys(qsys,new_qvec);
  } else {
    create_wf_sys(qsys,new_qvec);
  }

  return;
}

/*
 * create a dm that is correctly sized for the qsystem
 * and set to solve DM equations? FIXME
 */
void create_dm_sys(qsystem qsys,qvec *new_dm){
  qvec temp = NULL;
  PetscInt n,Istart,Iend,i;

  /* Check to make sure some operators were created */
  if (qsys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to create operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a density matrix!\n");
    exit(0);
  }


  //Even if we had no lindblad terms, maybe we start in a density matrix
  qsys->dm_equations = 1;
  qsys->dim = qsys->total_levels*qsys->total_levels;
  if(qsys->my_num<0){//Only setup the distribution once. Could be created otherwise
    _setup_distribution(qsys->nid,qsys->np,qsys->dim,&(qsys->my_num),&(qsys->Istart),&(qsys->Iend));
  }
  //total_levels = qsys->total_levels;
  qsys->hspace_frozen = 1;

  temp = malloc(sizeof(struct qvec));

  PetscPrintf(PETSC_COMM_WORLD,"Creating density matrix vector for Liouvillian solver.\n");
  _create_vec(&(temp->data),qsys->dim,qsys->my_num);

  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = DENSITY_MATRIX;
  temp->n = n;
  temp->total_levels = sqrt(n);
  temp->Istart = Istart;
  temp->Iend = Iend;
  temp->n_ops = qsys->num_subsystems;
  //The density matrix is N_tot by N_tot, so we treat it as having double the number of dimensinons
  temp->ndims_hspace = qsys->num_subsystems*2;
  temp->hspace_dims = malloc(2*qsys->num_subsystems*sizeof(PetscInt));
  temp->n_ensemble = -1;
  temp->ens_spawned = PETSC_FALSE;
  temp->ens_i = -1;

  //For a density matrix with 3 subsystems with L_i levels, we store the :
  //[L_1 L_2 L_3 L_1 L_2 L_3]
  for(i=0;i<qsys->num_subsystems;i++){
    temp->hspace_dims[i] = qsys->subsystem_list[i]->my_levels;
    temp->hspace_dims[i+qsys->num_subsystems] = qsys->subsystem_list[i]->my_levels;
  }
  *new_dm = temp;
  return;
}

/*
 * create a wf that is correctly sized for the qsystem
 */
void create_wf_sys(qsystem qsys,qvec *new_wf){
  qvec temp = NULL;
  PetscInt n,Istart,Iend,i;

  /* Check to make sure some operators were created */
  if (qsys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to crveate operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a wavefunction!\n");
    exit(0);
  }


  if (qsys->dm_equations==PETSC_TRUE && qsys->mcwf_solver==PETSC_FALSE){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR!\n");
    PetscPrintf(PETSC_COMM_WORLD,"Must use density matrices if Lindblad terms are used without");
    PetscPrintf(PETSC_COMM_WORLD,"using mcwf_solver.\n");
    exit(0);
  }

  PetscPrintf(PETSC_COMM_WORLD,"Creating wavefunction vector for Schrodinger solver.\n");
  qsys->dim = qsys->total_levels;

  if(qsys->my_num<0){//Only setup the distribution once. Could be created otherwise
    _setup_distribution(qsys->nid,qsys->np,qsys->dim,&(qsys->my_num),&(qsys->Istart),&(qsys->Iend));
  }
  //total_levels = qsys->total_levels;
  qsys->hspace_frozen = 1;

  temp = malloc(sizeof(struct qvec));

  _create_vec(&(temp->data),qsys->dim,qsys->my_num);

  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = WAVEFUNCTION;
  temp->n = n;
  temp->total_levels = n;
  temp->Istart = Istart;
  temp->Iend = Iend;
  temp->n_ops = qsys->num_subsystems;
  temp->ndims_hspace = qsys->num_subsystems;
  temp->hspace_dims = malloc(qsys->num_subsystems*sizeof(PetscInt));
  temp->n_ensemble = -1;
  temp->ens_spawned = PETSC_FALSE;
  temp->ens_i = -1;

  for(i=0;i<qsys->num_subsystems;i++){
    temp->hspace_dims[i] = qsys->subsystem_list[i]->my_levels;
  }

  *new_wf = temp;

  return;
}


/*
 * create a wf that is correctly sized for the qsystem
 */
void create_wf_ensemble_sys(qsystem qsys,qvec *new_wf){
  qvec temp = NULL;
  PetscInt n,Istart,Iend,i;

  /* Check to make sure some operators were created */
  if (qsys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to create operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a wavefunction!\n");
    exit(0);
  }


  if (qsys->dm_equations==PETSC_TRUE && qsys->mcwf_solver==PETSC_FALSE){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR!\n");
    PetscPrintf(PETSC_COMM_WORLD,"Must use density matrices if Lindblad terms are used without");
    PetscPrintf(PETSC_COMM_WORLD,"using mcwf_solver.\n");
    exit(0);
  }

  PetscPrintf(PETSC_COMM_WORLD,"Creating wavefunction vector ensemble for Monte Carlo wavefunction solver.\n");
  qsys->dim = qsys->total_levels;

  if(qsys->my_num<0){//Only setup the distribution once. Could be created otherwise
    _setup_distribution(qsys->nid,qsys->np,qsys->dim,&(qsys->my_num),&(qsys->Istart),&(qsys->Iend));
  }
  //total_levels = qsys->total_levels;
  qsys->hspace_frozen = 1;

  temp = malloc(sizeof(struct qvec));

  //temp->data will define the initial condition. The ens_datas will those spawned from it
  _create_vec(&(temp->data),qsys->dim,qsys->my_num);
  temp->ens_datas = malloc(qsys->num_local_trajs*sizeof(Vec));
  for(i=0;i<qsys->num_local_trajs;i++){
    _create_vec(&(temp->ens_datas[i]),qsys->dim,qsys->my_num);
  }

  VecGetSize(temp->data,&n);
  VecGetOwnershipRange(temp->data,&Istart,&Iend);

  temp->my_type = WF_ENSEMBLE;
  temp->n = n;
  temp->total_levels = n;
  temp->Istart = Istart;
  temp->Iend = Iend;
  temp->n_ops = qsys->num_subsystems;
  temp->ndims_hspace = qsys->num_subsystems;
  temp->hspace_dims = malloc(qsys->num_subsystems*sizeof(PetscInt));
  temp->n_ensemble = qsys->num_local_trajs;
  temp->ens_spawned = PETSC_FALSE;
  temp->ens_i = -1;

  for(i=0;i<qsys->num_subsystems;i++){
    temp->hspace_dims[i] = qsys->subsystem_list[i]->my_levels;
  }

  *new_wf = temp;

  return;
}



/*
 * Add alpha at the specified fock_state for operator op in qvec state
 * FIXME: Tensor ordering is backwards and the fock_op series is wrong
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
 * Add alpha to the elempent at location loc in the qvec
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
 * Add alpha to the elempent at location loc in the qvec wf_ens i_ens
 * Does not discriminate between WF and DM -- loc needs to be correctly vectorized if DM!
 */

void add_to_wf_ens_loc(qvec state,PetscInt i_ens,PetscScalar alpha,PetscInt loc){

  if (loc>state->n||loc<0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Adding to a qvec location that is out of range!\n");
    exit(0);
  }
  state->ens_spawned=PETSC_TRUE;
  if (loc>=state->Istart&&loc<state->Iend){
    VecSetValue(state->ens_datas[i_ens],loc,alpha,ADD_VALUES);
  }

  return;
}


/*
 * Add alpha to the element at location loc in the qvec
 * FIXME: Put some safety check on DM with only i (no j) passed in?
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
  } else if (state->my_type==WF_ENSEMBLE){
    if(state->ens_spawned==PETSC_FALSE){
      /*
       * Since this is a wavefunction, the input loc is the true location
       * so we can just call add_to_qvec_loc
       */
      num_locs = 1;
      va_start(ap,num_locs);
      loc = va_arg(ap,PetscInt);
      add_to_qvec_loc(state,alpha,loc);
      va_end(ap);
    } else {
      PetscPrintf(PETSC_COMM_WORLD,"Adding to spawned wf_ensemble not yet implemented\n");
    }
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
  } else if (state->my_type==WAVEFUNCTION||state->my_type==WF_ENSEMBLE){
    //WF, just take the direct state
    //WF_Ensemble, the state is the same either way
    *loc = *loc;
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! qvec type not understood.");
    exit(0);
  }
  return;
}



/*
 * void assemble_qvec puts all the cached values in the right place and
 * allows for the qvec to be used
 *
 * Inputs:
 *         qvec state - state to assemble
 * Outpus:
 *         None, but assembles the state
 *
 */
void assemble_qvec(qvec state){
  PetscInt i;
  if(state->my_type==WF_ENSEMBLE&&state->ens_spawned==PETSC_TRUE){
    /*
     * Split begin and end into two loops for possible
     * reduction in communication time?
     */
    for(i=0;i<state->n_ensemble;i++){
      VecAssemblyBegin(state->ens_datas[i]);
    }
    for(i=0;i<state->n_ensemble;i++){
      VecAssemblyEnd(state->ens_datas[i]);
    }
  } else {
    VecAssemblyBegin(state->data);
    VecAssemblyEnd(state->data);
  }
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
  PetscInt i;
  free(((*state)->hspace_dims));
  if((*state)->my_type==WF_ENSEMBLE){
    for(i=0;i<(*state)->n_ensemble;i++){
      VecDestroy(&((*state)->ens_datas[i]));
    }
    free((*state)->ens_datas);
  }
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

  free(ops);
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
  } else if(state->my_type==WF_ENSEMBLE){
    if(state->ens_spawned==PETSC_FALSE){
      //Haven't actually made an ensemble, so revert to just WF
      _get_expectation_value_wf(state,trace_val,num_ops,ops);
    } else{
      _get_expectation_value_wf_ens(state,trace_val,num_ops,ops);
    }
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
  my_j_start = my_start/rho->total_levels; // Rely on integer division to get 'floor'
  my_j_end  = my_end/rho->total_levels;

  for (i=my_j_start;i<my_j_end;i++){
    this_i = i; // The leading index which we check
    op_val = 1.0;
    for (j=0;j<num_ops;j++){
      if(ops[j]->my_op_type==VEC){
        /* PetscPrintf(PETSC_COMM_WORLD,"ERROR! VEC operators not yet supported!\n"); */
        /* exit(0); */
        /*
         * Since this is a VEC operator, the next operator must also
         * be a VEC operator; it is assumed they always come in pairs.
         */
        if (ops[j+1]->my_op_type!=VEC){
          PetscPrintf(PETSC_COMM_WORLD,"ERROR! VEC operators must come in pairs in get_expectation_value\n");
        }
        _get_val_j_from_global_i_vec_vec(rho->total_levels,this_i,ops[j],ops[j+1],&this_j,&val,-1);
        //Increment j
        j=j+1;
      } else {
        //Standard operator
        _get_val_j_from_global_i(rho->total_levels,this_i,ops[j],&this_j,&val,-1); // Get the corresponding j and val
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

  //This should be put in a routine called "apply op to psi"?
  //Calculate A * B * Psi
  for (i=0;i<psi->total_levels;i++){
    this_i = i; // The leading index which we check
    op_val = 1.0;
    for (j=0;j<num_ops;j++){
      _get_val_j_from_global_i(psi->total_levels,this_i,ops[j],&this_j,&val,-1); // Get the corresponding j and val
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


void _get_expectation_value_wf_ens(qvec psi,PetscScalar *trace_val,PetscInt num_ops,operator *ops){
  PetscInt Istart,Iend,location[1],i,j,this_j,this_i,i_ens;
  PetscScalar val_array[1],val,op_val,tmp_trace_val,norm;
  Vec op_psi;

  /*
   * FIXME: Check for consistency in operator sizes and wf size
   * Consider merging with _get_expectation_value_wf in some way
   */

  *trace_val = 0.0;
  VecGetOwnershipRange(psi->data,&Istart,&Iend);
  VecDuplicate(psi->data,&op_psi);
  for (i_ens=0;i_ens<psi->n_ensemble;i_ens++){
    //Calculate A * B * Psi
    for (i=0;i<psi->total_levels;i++){
      this_i = i; // The leading index which we check
      op_val = 1.0;
      for (j=0;j<num_ops;j++){
        _get_val_j_from_global_i(psi->total_levels,this_i,ops[j],&this_j,&val,-1); // Get the corresponding j and val
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
        VecGetValues(psi->ens_datas[i_ens],1,location,val_array);
        op_val = op_val * val_array[0];
        //Add the value to the op_psi

        VecSetValue(op_psi,i,op_val,ADD_VALUES);
      }
    }
    // Now, calculate the inner product between psi^H * OP_psi
    VecAssemblyBegin(op_psi);
    VecAssemblyEnd(op_psi);
    VecDot(op_psi,psi->ens_datas[i_ens],&tmp_trace_val);
    VecDot(psi->ens_datas[i_ens],psi->ens_datas[i_ens],&norm);
    *trace_val = *trace_val + tmp_trace_val/norm;
    VecSet(op_psi,0.0);//Reset vector
  }

  *trace_val = *trace_val/psi->n_ensemble;

  VecDestroy(&op_psi);

  return;
}

void get_qvec_local_idxs(qvec state,PetscInt global_idx,PetscInt *local_idxs){
  PetscInt i,this_global_idx;

  this_global_idx = global_idx;

  for(i=0;i<state->ndims_hspace;i++){
    local_idxs[i] = this_global_idx % state->hspace_dims[i];
    this_global_idx = this_global_idx/state->hspace_dims[i];
  }

  return;
}


void _ptrace_over_list_qvec_dm(qvec full_dm,PetscInt n_ops,PetscInt *op_loc_list,PetscInt n_ops_keep, PetscInt *ops_keep_list,qvec *new_dm){
  PetscInt i,j1;
  PetscInt *local_idxs,keep_val,new_idx,this_n_bef;
  PetscScalar this_val;

  local_idxs = malloc(full_dm->ndims_hspace*sizeof(PetscInt));

  //Loop through local elements of the full_dm
  for(i=full_dm->Istart;i<full_dm->Iend;i++){
    //Get the array of i1,i2,...,j1,j2... values for this global i
    get_qvec_local_idxs(full_dm,i,local_idxs);
    //Assume we will keep this value
    keep_val = 1;
    //Loop over the operators to ptrace away
    for(j1=0;j1<n_ops;j1++){
      //Check if the two indices for this op do not match
      //If they do not match for any of the ops, we will not keep
      //This does n^2 work instead of n
      if(local_idxs[op_loc_list[j1]]!=local_idxs[op_loc_list[j1]+full_dm->n_ops]){
          keep_val = 0;
      }
    }
    if(keep_val==1){
      //All the trace_over ops idx matched, so we keep this one
      new_idx = 0;
      this_n_bef = 1;
      // new_idx = \sum_i_k n_bef * local_idx[i_k]
      for(j1=0;j1<n_ops_keep;j1++){
        new_idx = new_idx + this_n_bef*local_idxs[ops_keep_list[j1]]; // i position
        new_idx = new_idx + this_n_bef*(*new_dm)->total_levels*local_idxs[ops_keep_list[j1]+full_dm->n_ops]; // j position - it is always total_levels later
        this_n_bef = this_n_bef * (*new_dm)->hspace_dims[j1];
      }
      _get_qvec_element_local(full_dm,i,&this_val);
      add_to_qvec_loc((*new_dm),this_val,new_idx);
    }
  }

    free(local_idxs);

  return;
}


void _ptrace_over_list_qvec_wf(qvec full_wf,PetscInt n_ops,PetscInt *op_loc_list,PetscInt n_ops_keep, PetscInt *ops_keep_list,qvec *new_dm){
  PetscInt *local_idxs,*local_idxsj,keep_val,new_idx,this_n_bef,i,j,j1;
  PetscScalar this_val,this_valj;
  VecScatter        scatter_ctx;
  qvec wf_local;

  local_idxs = malloc(full_wf->ndims_hspace*sizeof(PetscInt));
  local_idxsj = malloc(full_wf->ndims_hspace*sizeof(PetscInt));
  //Gather WF to one core - inefficient! Serial
  //Other option is to write complicated MPI_GET, since PETSc only allows local VecGetValues?

  //Collect all data for the wf on one core - not necessary for single core examples
  //Can be sped up by not doing this on one core
  wf_local = malloc(sizeof(struct qvec));
  VecScatterCreateToZero(full_wf->data,&scatter_ctx,&(wf_local->data));
  VecScatterBegin(scatter_ctx,full_wf->data,wf_local->data,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(scatter_ctx,full_wf->data,wf_local->data,INSERT_VALUES,SCATTER_FORWARD);

  wf_local->Istart = 0;
  wf_local->Iend   = full_wf->total_levels;
  wf_local->total_levels = full_wf->total_levels;
  wf_local->ndims_hspace = full_wf->ndims_hspace;
  wf_local->hspace_dims = malloc(full_wf->ndims_hspace*sizeof(PetscInt));
  for(i=0;i<full_wf->ndims_hspace;i++){
    wf_local->hspace_dims[i] = full_wf->hspace_dims[i];
  }


  //Loop through local elements of the full_wf
  for(i=0;i<wf_local->total_levels;i++){
    for(j=0;j<wf_local->total_levels;j++){
      //Get the array of i1,i2,...,j1,j2... values for this global i
      get_qvec_local_idxs(wf_local,i,local_idxs);
      get_qvec_local_idxs(wf_local,j,local_idxsj);
      //Assume we will keep this value
      keep_val = 1;
      //Loop over the operators to ptrace away
      for(j1=0;j1<n_ops;j1++){
        //Check if the two indices for this op do not match
        //If they do not match for any of the ops, we will not keep
        if(local_idxs[op_loc_list[j1]]!=local_idxsj[op_loc_list[j1]]){
          keep_val = 0;
        }
      }
      if(keep_val==1){
        //All the trace_over ops idx matched, so we keep this one
        new_idx = 0;
        this_n_bef = 1;
        // new_idx = \sum_i_k n_bef * local_idx[i_k]
        for(j1=0;j1<n_ops_keep;j1++){
          new_idx = new_idx + this_n_bef*local_idxs[ops_keep_list[j1]]; // i position
          new_idx = new_idx + this_n_bef*(*new_dm)->total_levels*local_idxsj[ops_keep_list[j1]]; // j position
          this_n_bef = this_n_bef * (*new_dm)->hspace_dims[j1];
        }
        _get_qvec_element_local(wf_local,i,&this_val);
        _get_qvec_element_local(wf_local,j,&this_valj);
        add_to_qvec_loc((*new_dm),this_val*PetscConjComplex(this_valj),new_idx);
      }
    }
  }

  VecScatterDestroy(&scatter_ctx);
  free(local_idxs);
  free(local_idxsj);
  destroy_qvec(&wf_local);

  return;
}

void ptrace_over_list_qvec(qvec full_state,PetscInt n_ops,PetscInt *op_loc_list,qvec *new_dm){
  PetscInt i,j,ndims_new,*hspace_dims_new,i_h,found=0;
  PetscInt n_ops_keep,*ops_keep_list;


  if(n_ops>full_state->n_ops){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Trying to ptrace more ops than are in the state!\n");
    exit(9);
  }

  for(i=0;i<n_ops;i++){
    if(op_loc_list[i]>(full_state->n_ops-1)){
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! op_loc_list has an index greater than nops in state!\n");
      exit(9);
    }
  }

  //2 * ndims_hspace because the wf is implicitly turned into a DM
  //2 * nops because we are removing 2 of the ndims per n_op
  if(full_state->my_type==WAVEFUNCTION){
    ndims_new = 2*full_state->ndims_hspace - 2*n_ops; //WF specific
  } else{
    ndims_new = full_state->ndims_hspace - 2*n_ops; //WF specific
  }

  n_ops_keep = full_state->n_ops - n_ops;
  ops_keep_list = malloc(n_ops_keep*sizeof(PetscInt));
  hspace_dims_new = malloc(ndims_new*sizeof(PetscInt));
  i_h = 0;

  for(i=0;i<full_state->n_ops;i++){
    found=0;
    //Find if i is to be kept or traced over
    //There is no guarantee that op_list is ordered, so we search through it
    //It will be size <50 almost always, so it is OK to do this extra work
    for(j=0;j<n_ops;j++){
      if(i==op_loc_list[j]){
        found = 1;
      }
    }

    //Save the hspace_dims of the subsytems we want to keep
    if(found==0){
      ops_keep_list[i_h] = i;
      hspace_dims_new[i_h] = full_state->hspace_dims[i];
      hspace_dims_new[i_h+n_ops_keep] = full_state->hspace_dims[i];
      i_h = i_h+1;
    }
  }

  //Create new DM with correct sizes
  //Always a DM when partial tracing
  create_arb_qvec_dims(new_dm,ndims_new,hspace_dims_new,DENSITY_MATRIX);

  if(full_state->my_type==WAVEFUNCTION){
    _ptrace_over_list_qvec_wf(full_state,n_ops,op_loc_list,n_ops_keep,ops_keep_list,new_dm);
  } else {
    _ptrace_over_list_qvec_dm(full_state,n_ops,op_loc_list,n_ops_keep,ops_keep_list,new_dm);
  }

  assemble_qvec(*new_dm);
  free(hspace_dims_new);
  free(ops_keep_list);

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

void get_bitstring_probs(qvec state,PetscInt *nloc,PetscReal **probs,PetscReal **vars){
  PetscInt i;
  //FIXME Dangerous to return nloc, exposes parallelism to user

  if(state->my_type==DENSITY_MATRIX){
    _get_bitstring_probs_dm(state,nloc,probs);
    (*vars) = malloc((*nloc)*sizeof(PetscReal));
    //0 variance because we aren't sampling over anything
    for(i=0;i<(*nloc);i++){
      (*vars)[i] = 0.0;
    }

  } else if(state->my_type==WAVEFUNCTION){
    _get_bitstring_probs_wf(state,nloc,probs);
    (*vars) = malloc((*nloc)*sizeof(PetscReal));
    //0 variance because we aren't sampling over anything
    for(i=0;i<(*nloc);i++){
      (*vars)[i] = 0.0;
    }

  } else if(state->my_type==WF_ENSEMBLE){
    if(state->ens_spawned==PETSC_FALSE){
      //Haven't actually made an ensemble, so revert to just WF
      _get_bitstring_probs_wf(state,nloc,probs);
      (*vars) = malloc((*nloc)*sizeof(PetscReal));
      //0 variance because we aren't sampling over anything
      for(i=0;i<(*nloc);i++){
        (*vars)[i] = 0.0;
      }

    } else {
      _get_bitstring_probs_wf_ens(state,nloc,probs,vars);
    }
  }

  return;
}

void get_linear_xeb_fidelity_probs(PetscReal *probs_ref,PetscReal *vars_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscReal *vars_exp,
                                   PetscInt nloc_exp,PetscInt h_dim,PetscReal *lin_xeb_fid,PetscReal *lin_xeb_fid_var){
  PetscReal tmp_lin_xeb_fid[1],tmp_lin_xeb_fid_var[1];
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
  tmp_lin_xeb_fid_var[0] = 0;
  for(i=0;i<nloc_ref;i++){
    tmp_lin_xeb_fid[0] = tmp_lin_xeb_fid[0] + probs_exp[i]*probs_ref[i];
    tmp_lin_xeb_fid_var[0] = tmp_lin_xeb_fid_var[0] + vars_exp[i]*probs_ref[i]*probs_ref[i];
  }

  //Collect all cores
  MPI_Allreduce(MPI_IN_PLACE,tmp_lin_xeb_fid,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,tmp_lin_xeb_fid_var,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  *lin_xeb_fid = h_dim * tmp_lin_xeb_fid[0] - 1;
  *lin_xeb_fid_var = h_dim * h_dim * tmp_lin_xeb_fid_var[0];
  return;
}

void get_linear_xeb_fidelity(qvec ref,qvec exp,PetscReal *lin_xeb_fid,PetscReal *lin_xeb_fid_var){
  PetscReal *probs_ref,*probs_exp,*vars_ref,*vars_exp;
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
  get_bitstring_probs(ref,&nloc_ref,&probs_ref,&vars_ref);
  get_bitstring_probs(exp,&nloc_exp,&probs_exp,&vars_exp);

  get_linear_xeb_fidelity_probs(probs_ref,vars_ref,nloc_ref,probs_exp,vars_exp,nloc_exp,ref->total_levels,lin_xeb_fid,lin_xeb_fid_var);

  free(probs_exp);
  free(probs_ref);
  free(vars_exp);
  free(vars_ref);
  return;
}

void get_log_xeb_fidelity_probs(PetscReal *probs_ref,PetscReal *vars_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscReal *vars_exp,PetscInt nloc_exp,
                                PetscInt h_dim,PetscReal *log_xeb_fid,PetscReal *log_xeb_fid_var){
  PetscReal gamma=0.57721566490153286060651209008240243104215933593992;
  PetscReal tmp_log_xeb_fid[1],tmp_log_xeb_fid_var[1];
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
  tmp_log_xeb_fid_var[0] = 0;
  for(i=0;i<nloc_ref;i++){
    if(probs_ref[i]==0.0){
      //Protection from log(0)
      tmp_log_xeb_fid_var[0] = tmp_log_xeb_fid_var[0] + vars_exp[i]*PetscLogReal(3e-33)*PetscLogReal(3e-33);
    } else{
      tmp_log_xeb_fid[0] = tmp_log_xeb_fid[0] + probs_exp[i]*PetscLogReal(probs_ref[i]);
      tmp_log_xeb_fid_var[0] = tmp_log_xeb_fid_var[0] + vars_exp[i]*PetscLogReal(probs_ref[i])*PetscLogReal(probs_ref[i]);
    }
  }
  //Collect all cores
  MPI_Allreduce(MPI_IN_PLACE,tmp_log_xeb_fid,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,tmp_log_xeb_fid_var,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  *log_xeb_fid = PetscLogReal(h_dim) + tmp_log_xeb_fid[0] + gamma;
  *log_xeb_fid_var = tmp_log_xeb_fid_var[0];

  return;
}

void get_log_xeb_fidelity(qvec ref,qvec exp,PetscReal *log_xeb_fid,PetscReal *log_xeb_fid_var){
  PetscReal *probs_ref,*probs_exp,*vars_ref,*vars_exp;
  PetscInt nloc_ref,nloc_exp;

  /*
   * Our 'experiment' is stored in a dm and we can extract the bitstring
   * probabilities by just taking the diagonal part
   * Similarly, for our reference
   */
  if(ref->total_levels!=exp->total_levels){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Ref and Exp are not the same size in get_log_xeb_fidelity!\n");
    exit(9);
  }

  get_bitstring_probs(ref,&nloc_ref,&probs_ref,&vars_ref);
  get_bitstring_probs(exp,&nloc_exp,&probs_exp,&vars_exp);

  get_log_xeb_fidelity_probs(probs_ref,vars_ref,nloc_ref,probs_exp,vars_exp,nloc_exp,ref->total_levels,log_xeb_fid,log_xeb_fid_var);

  free(probs_exp);
  free(probs_ref);
  free(vars_exp);
  free(vars_ref);

  return;
}

void get_hog_score_probs(PetscReal *probs_ref,PetscReal *vars_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscReal *vars_exp,PetscInt nloc_exp,PetscInt h_dim,PetscReal *hog_score,PetscReal *hog_var){
  PetscReal tmp_hog_score[1],tmp_hog_var[1];
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
  tmp_hog_var[0] = 0;
  for(i=0;i<nloc_ref;i++){
    if(probs_ref[i]>=log(2.0)/h_dim){
      tmp_hog_score[0] = tmp_hog_score[0] + probs_exp[i];
      tmp_hog_var[0] = tmp_hog_var[0] + vars_exp[i];
    }
  }
  //Collect all cores
  MPI_Allreduce(MPI_IN_PLACE,tmp_hog_score,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,tmp_hog_var,1,MPIU_REAL,MPI_SUM,PETSC_COMM_WORLD);
  *hog_score = tmp_hog_score[0];
  *hog_var = tmp_hog_var[0];

  return;
}

void get_hog_score(qvec ref,qvec exp,PetscReal *hog_score,PetscReal *hog_var){
  PetscReal *probs_ref,*probs_exp,*vars_ref,*vars_exp;
  PetscInt nloc_ref,nloc_exp;
  /*
   * Our 'experiment' is stored in a dm and we can extract the bitstring
   * probabilities by just taking the diagonal part
   * Similarly, for our reference
   */

  if(ref->total_levels!=exp->total_levels){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Ref and Exp are not the same size in get_hog_score!\n");
    exit(9);
  }
  get_bitstring_probs(ref,&nloc_ref,&probs_ref,&vars_ref);
  get_bitstring_probs(exp,&nloc_exp,&probs_exp,&vars_exp);

  get_hog_score_probs(probs_ref,vars_ref,nloc_ref,probs_exp,vars_exp,nloc_exp,ref->total_levels,hog_score,hog_var);

  free(probs_exp);
  free(probs_ref);
  free(vars_exp);
  free(vars_ref);

  return;
}


void get_hog_score_fidelity_probs(PetscReal *probs_ref,PetscReal *vars_ref,PetscInt nloc_ref,PetscReal *probs_exp,PetscReal *vars_exp,PetscInt nloc_exp,PetscInt h_dim,PetscReal *hog_score_fid,PetscReal *hog_fid_var){
  PetscReal hog_score,hog_var;

  if(nloc_ref!=nloc_exp){
    //FIXME Dangerous print here
    printf("ERROR! Ref and Exp are not the same size in get_hog_score_probs!\n");
    exit(9);
  }

  get_hog_score_probs(probs_ref,vars_ref,nloc_ref,probs_exp,vars_exp,nloc_exp,h_dim,&hog_score,&hog_var);
  *hog_score_fid = (2*hog_score - 1)/log(2.0);
  *hog_fid_var   = (2/log(2.0))*(2/log(2.0)) * hog_var;

  return;
}

void get_hog_score_fidelity(qvec ref,qvec exp,PetscReal *hog_score_fid,PetscReal *hog_fid_var){
  PetscReal hog_score,hog_var;

  /*
   * Our 'experiment' is stored in a dm and we can extract the bitstring
   * probabilities by just taking the diagonal part
   * Similarly, for our reference
   */
  if(ref->total_levels!=exp->total_levels){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Ref and Exp are not the same size in get_hog_score_fidelity!\n");
    exit(9);
  }

  get_hog_score(ref,exp,&hog_score,&hog_var);
  *hog_score_fid = (2*hog_score - 1)/log(2.0);
  //Var(aX + b) = a^2 *Var(X)
  *hog_fid_var   = (2/log(2.0))*(2/log(2.0)) * hog_var;

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
  *num_loc = num_loc_t; //Don't forget to subtract one?
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
  PetscInt i=0;

  //Allocate array
  *num_loc = q1->Iend-q1->Istart;
  (*probs) = malloc((*num_loc)*sizeof(PetscReal));
  for(i=q1->Istart;i<q1->Iend;i++){
    //Get local element
    get_wf_element_qvec_local(q1,i,&tmp_scalar);
    (*probs)[i] = pow(PetscRealPart(PetscAbsComplex(tmp_scalar)),2);
  }

  return;
}

void _get_bitstring_probs_wf_ens(qvec q1,PetscInt *num_loc,PetscReal **probs,PetscReal **vars){
  PetscScalar tmp_scalar=0,norm;
  PetscReal tmp_real;
  PetscInt i=0,i_ens=0;

  //Allocate array
  *num_loc = q1->Iend-q1->Istart;
  (*probs) = malloc((*num_loc)*sizeof(PetscReal));
  (*vars) = malloc((*num_loc)*sizeof(PetscReal));
  for(i=0;i<*num_loc;i++){
    (*probs)[i] = 0.0;
    (*vars)[i]  = 0.0;
  }
  for(i_ens=0;i_ens<q1->n_ensemble;i_ens++){
    //Get norm
    VecDot(q1->ens_datas[i_ens],q1->ens_datas[i_ens],&norm);
    for(i=q1->Istart;i<q1->Iend;i++){
      //Get local element
      get_wf_element_ens_i_qvec_local(q1,i_ens,i,&tmp_scalar);
      (*probs)[i] += pow(PetscRealPart(PetscAbsComplex(tmp_scalar)),2)/norm;
    }
  }

  for(i=q1->Istart;i<q1->Iend;i++){
    (*probs)[i] = (*probs)[i]/q1->n_ensemble;
  }

  //Now calculate variances
  for(i_ens=0;i_ens<q1->n_ensemble;i_ens++){
    //Get norm
    VecDot(q1->ens_datas[i_ens],q1->ens_datas[i_ens],&norm);
    for(i=q1->Istart;i<q1->Iend;i++){
      //Get local element
      get_wf_element_ens_i_qvec_local(q1,i_ens,i,&tmp_scalar);
      tmp_real = pow(PetscRealPart(PetscAbsComplex(tmp_scalar)),2)/norm; //X_i
      (*vars)[i] += (tmp_real - (*probs)[i])*(tmp_real - (*probs)[i]); //(X_i - X)**2
    }
  }

  for(i=q1->Istart;i<q1->Iend;i++){
    (*vars)[i] = (*vars)[i]/(q1->n_ensemble-1); //\sum_i (X_i - X)**2/ (n-1)

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
    PetscPrintf(PETSC_COMM_WORLD,"Calculating superfidelity between two wavefunctions is not recommended. Use get_superfidelity_qvec!\n");
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
void get_fidelity_qvec(qvec q1,qvec q2,PetscReal *fidelity,PetscReal *fid_var) {
  PetscInt i;
  PetscScalar norm;
  PetscReal fid_tmp;
  *fid_var = 0;
  if (q1->my_type==DENSITY_MATRIX && q2->my_type==DENSITY_MATRIX){
    _get_fidelity_dm_dm(q1->data,q2->data,fidelity);
  } else if(q1->my_type==WAVEFUNCTION && q2->my_type==WAVEFUNCTION){
    _get_fidelity_wf_wf(q1->data,q2->data,fidelity);
  } else if(q1->my_type==WAVEFUNCTION && q2->my_type==DENSITY_MATRIX){
    _get_fidelity_dm_wf(q2->data,q1->data,fidelity);
  } else if(q2->my_type==WAVEFUNCTION && q1->my_type==DENSITY_MATRIX){
    _get_fidelity_dm_wf(q1->data,q2->data,fidelity);
  } else if(q1->my_type==WF_ENSEMBLE && q2->my_type==WAVEFUNCTION){
    if(q1->ens_spawned==PETSC_FALSE){
      _get_fidelity_wf_wf(q1->data,q2->data,fidelity);
    } else {
      *fidelity = 0.0;
      for(i=0;i<q1->n_ensemble;i++){
        VecDot(q1->ens_datas[i],q1->ens_datas[i],&norm);
        _get_fidelity_wf_wf(q1->ens_datas[i],q2->data,&fid_tmp);
        *fidelity = *fidelity + fid_tmp/norm;
      }

      *fidelity = *fidelity/q1->n_ensemble;

      for(i=0;i<q1->n_ensemble;i++){
        VecDot(q1->ens_datas[i],q1->ens_datas[i],&norm);
        _get_fidelity_wf_wf(q1->ens_datas[i],q2->data,&fid_tmp);
        *fid_var += (fid_tmp - *fidelity)*(fid_tmp - *fidelity);
      }

      *fid_var = *fid_var/(q1->n_ensemble-1);
    }
  } else if(q2->my_type==WF_ENSEMBLE && q1->my_type==WAVEFUNCTION){
    if(q2->ens_spawned==PETSC_FALSE){
      _get_fidelity_wf_wf(q2->data,q1->data,fidelity);
    } else {
      *fidelity = 0.0;
      for(i=0;i<q2->n_ensemble;i++){
        VecDot(q2->ens_datas[i],q2->ens_datas[i],&norm);
        _get_fidelity_wf_wf(q2->ens_datas[i],q1->data,&fid_tmp);
        *fidelity = *fidelity + fid_tmp/norm;
      }
      *fidelity = *fidelity/q2->n_ensemble;
      for(i=0;i<q2->n_ensemble;i++){
        VecDot(q2->ens_datas[i],q2->ens_datas[i],&norm);
        _get_fidelity_wf_wf(q2->ens_datas[i],q1->data,&fid_tmp);
        *fid_var += (fid_tmp - *fidelity)*(fid_tmp - *fidelity);
      }
      *fid_var = *fid_var/(q2->n_ensemble-1);
    }

    //Not the correct way to do it - need to explicitly construct DM from ensemble, maybe?
  /* } else if(q1->my_type==WF_ENSEMBLE && q2->my_type==DENSITY_MATRIX){ */
  /*   if(q1->ens_spawned==PETSC_FALSE){ */
  /*     _get_fidelity_dm_wf(q2->data,q1->data,fidelity); */
  /*   } else { */
  /*     *fidelity = 0.0; */
  /*     for(i=0;i<q1->n_ensemble;i++){ */
  /*       VecDot(q1->ens_datas[i],q1->ens_datas[i],&norm); */
  /*       _get_fidelity_dm_wf(q2->data,q1->ens_datas[i],&fid_tmp); */
  /*       *fidelity = *fidelity + fid_tmp/norm; */
  /*     } */
  /*     *fidelity = *fidelity/q1->n_ensemble; */
  /*   } */
  /* } else if(q2->my_type==WF_ENSEMBLE && q1->my_type==DENSITY_MATRIX){ */
  /*   if(q2->ens_spawned==PETSC_FALSE){ */
  /*     printf("here1\n"); */
  /*     _get_fidelity_dm_wf(q1->data,q2->data,fidelity); */
  /*   } else { */
  /*     printf("here2 %d\n",q2->n_ensemble); */
  /*     *fidelity = 0.0; */
  /*     for(i=0;i<q2->n_ensemble;i++){ */
  /*       VecDot(q2->ens_datas[i],q2->ens_datas[i],&norm); */
  /*       _get_fidelity_dm_wf(q1->data,q2->ens_datas[i],&fid_tmp); */
  /*       *fidelity = *fidelity + fid_tmp/norm; */
  /*     } */
  /*     *fidelity = *fidelity/q2->n_ensemble; */
  /*   } */
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
      get_dm_element(dm,i,j,&val);//FIXME: update this to get_dm_element_qvec
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
  *fidelity = *fidelity * (*fidelity); //We want fidelity, not sqrt(fidelity)
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


/*
 * void sqrt_mat takes the square root of square, hermitian matrix
 *
 * Inputs:
 *         Mat dm_mat - matrix to take square root of
 * Outpus:
 *         None, but does square root in place
 *
 */
void sqrt_mat(Mat dm_mat){
  Mat V,sqrt_D,result_mat;
  PetscInt rows,columns,i;
  PetscScalar  *array,*work,*eigs,*evec,sdummy;
  PetscReal    *rwork;
  PetscBLASInt idummy,lwork,lierr,nb;

  MatGetSize(dm_mat,&rows,&columns);

  if (rows!=columns){
    if (nid==0){
      printf("ERROR! The input matrix in sqrt_mat is not square!\n");
      exit(0);
    }
  }


  /* We need the actual array so that we can pass it to lapack */
  MatDenseGetArray(dm_mat,&array);

  /* Lots of setup for LAPACK stuff */
  idummy = rows;
  lwork  = 5*rows;
  PetscMalloc1(5*rows,&work);
  PetscMalloc1(2*rows,&rwork);
  PetscMalloc1(rows*rows,&evec);
  PetscMalloc1(rows,&eigs);
  PetscBLASIntCast(rows,&nb);

  /* Call LAPACK through PETSc to ensure portability */
  LAPACKgeev_("N","V",&nb,array,&nb,eigs,&sdummy,&idummy,evec,&nb,work,&lwork,rwork,&lierr);
  /* Create matrices to store eigenvectors / values */
  MatCreateSeqDense(PETSC_COMM_SELF,rows,rows,evec,&V);
  MatCreateSeqDense(PETSC_COMM_SELF,rows,rows,NULL,&sqrt_D);

  MatSetUp(V);
  MatSetUp(sqrt_D);

  for (i=0;i<rows;i++){
    /* Stop NaN's from roundoff error by checking that eigs be positive */
    if (PetscRealPart(eigs[i])>0){
      MatSetValue(sqrt_D,i,i,sqrt(PetscRealPart(eigs[i])),INSERT_VALUES);
    }
  }

  MatAssemblyBegin(sqrt_D,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sqrt_D,MAT_FINAL_ASSEMBLY);
  /* Calculate V*sqrt(D) */
  MatMatMult(V,sqrt_D,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&result_mat);
  /* Calculate V^\dagger */
  MatHermitianTranspose(V,MAT_INPLACE_MATRIX,&V);
  MatDenseRestoreArray(dm_mat,&array);
  /* Calculate (V*sqrt(D))*V^\dagger, store in dm_mat */
  MatMatMult(result_mat,V,MAT_REUSE_MATRIX,PETSC_DEFAULT,&dm_mat);


  MatDestroy(&V);
  MatDestroy(&sqrt_D);
  MatDestroy(&result_mat);
  PetscFree(work);
  PetscFree(rwork);
  PetscFree(evec);
  PetscFree(eigs);
  return;
}


/*
 * set_qvec_from_init_excited sets the total initial condition from the
 * initial conditions provided via the set_init_excited_op routine.
 *
 * Inputs:
 *        qsystem qsys
 *        qvec state
 */
void set_qvec_from_init_excited_op(qsystem qsys,qvec state){
  PetscInt    i,init_row_op=0,n_after;
  PetscScalar mat_tmp_val;

  /*
   * See if there are any vec operators
   */

  /*
   * We can only use this simpler initialization if all of the operators
   * are ladder operators, and the user hasn't used any special initialization routine
   */
  for (i=0;i<qsys->num_subsystems;i++){
    n_after   = qsys->total_levels/(qsys->subsystem_list[i]->my_levels*qsys->subsystem_list[i]->n_before);
    init_row_op += (qsys->subsystem_list[i]->initial_exc)*n_after;
  }

  if(qsys->dm_equations==PETSC_TRUE) {
    init_row_op = qsys->total_levels*init_row_op + init_row_op;
  } else {
    init_row_op = init_row_op;
  }
  mat_tmp_val = 1. + 0.0*PETSC_i;
  add_to_qvec_loc(state,mat_tmp_val,init_row_op);

  return;
}
