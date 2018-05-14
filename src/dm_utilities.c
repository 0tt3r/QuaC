#include "dm_utilities.h"
#include "operators_p.h"
#include <stdlib.h>
#include <stdio.h>
#include <petscblaslapack.h>
#include <string.h>

/*
 * Print the DM as a matrix.
 * Not recommended for large matrices.
 * NOTE: Should be called from all cores!
 */
void print_dm(Vec rho,int h_dim){
  PetscScalar val;
  int i,j;
  for (i=0;i<h_dim;i++){
    for (j=0;j<h_dim;j++){
      get_dm_element(rho,i,j,&val);
      PetscPrintf(PETSC_COMM_WORLD,"%e + %ei ",PetscRealPart(val),
                  PetscImaginaryPart(val));
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n");
  }
    PetscPrintf(PETSC_COMM_WORLD,"\n");
}

/*
 * Print the DM as a sparse matrix.
 * Not recommended for large matrices.
 * NOTE: Should be called from all cores!
 */
void print_dm_sparse(Vec rho,int h_dim){
  PetscScalar val;
  int i,j;
  for (i=0;i<h_dim;i++){
    for (j=0;j<h_dim;j++){
      get_dm_element(rho,i,j,&val);
      if (PetscAbsComplex(val)>1e-10){
        PetscPrintf(PETSC_COMM_WORLD,"%d %d %e %e\n",i,j,PetscRealPart(val),PetscImaginaryPart(val));
      }
    }
  }
}


/*
 * Print the DM as a sparse matrix.
 * Not recommended for large matrices.
 * NOTE: Should be called from all cores!
 */
void print_dm_sparse_to_file(Vec rho,int h_dim,char filename[]){
  PetscScalar val;
  int i,j;
  FILE *fp;

  fp = fopen(filename,"w");
  for (i=0;i<h_dim;i++){
    for (j=0;j<h_dim;j++){
      get_dm_element(rho,i,j,&val);
      if (PetscAbsComplex(val)>1e-10){
        PetscFPrintf(PETSC_COMM_WORLD,fp,"%d %d %e %e\n",i,j,PetscRealPart(val),PetscImaginaryPart(val));
      }
    }
  }
}

/*
 * Print psi
 * Not recommended for large systems
 * NOTE: Won't work in parallel well at this time
 */

void print_psi(Vec rho,int h_dim){
  PetscScalar val_array[1];
  PetscInt location[1];
  int i;

  for (i=0;i<h_dim;i++){
    location[0] = i;
    VecGetValues(rho,1,location,val_array);
    PetscPrintf(PETSC_COMM_WORLD,"%f + %f i\n",PetscRealPart(val_array[0]),
                PetscImaginaryPart(val_array[0]));
  }
  PetscPrintf(PETSC_COMM_WORLD,"\n");
}

/*
 * partial_trace_over does the partial trace over a list of operators,
 * leaving the operators not listed.
 *
 * Inputs:
 *     Vec full_dm: the full Hilbert space density matrix to trace over.
 *                  Note: MUST be the full density matrix; use other routines
 *                        if starting with an already traced DM
 *     int number_of_ops: number of ops in list to trace over
 *     <list of ops>: A list of operators which are to be traced over
 *
 * Outpus:
 *     Vec ptraced_dm: the result of the partial trace is stored here.
 *                     Note: Assumed to already be allocated via create_dm()
 */

void partial_trace_over(Vec full_dm,Vec ptraced_dm,int number_of_ops,...){
  va_list ap;
  Vec tmp_dm,tmp_full_dm;
  operator op;
  PetscInt i,j,current_total_levels,*nbef_prev,*nop_prev,dm_size,nbef,naf,previous_total_levels;

  if (number_of_ops>num_subsystems){
    if (nid==0){
      printf("ERROR! number_of_ops cannot be greater than num_subsystems\n");
      exit(0);
    }
  }

  va_start(ap,number_of_ops);

  /* Check that the full_dm is of size total_levels */

  VecGetSize(full_dm,&dm_size);

  if (dm_size!=pow(total_levels,2)){
    if (nid==0){
      printf("ERROR! You need to use the full Hilbert space sized DM in \n");
      printf("       partial_trace_over!\n");
      exit(0);
    }
  }

  PetscMalloc1(number_of_ops,&nbef_prev);
  PetscMalloc1(number_of_ops,&nop_prev);
  current_total_levels = total_levels;
  /* Initially create a copy of the full DM */
  VecDuplicate(full_dm,&tmp_full_dm);
  VecCopy(full_dm,tmp_full_dm);
  // Loop through ops that we are tracing over
  for (i=0;i<number_of_ops;i++){
    op = va_arg(ap,operator);
    previous_total_levels = current_total_levels;
    current_total_levels = current_total_levels/op->my_levels;

    /* Creat a smaller, temporary DM to store the current partial trace */
    create_dm(&tmp_dm,current_total_levels);

    nbef = op->n_before;
    naf  = total_levels/(op->my_levels*nbef);

    /* Update nbef and naf with the already removed operators */
    for (j=0;j<i;j++) {
      if (nbef_prev[j]==op->n_before){
        if (nid==0){
          printf("ERROR! Partial tracing the same operator twice does not make sense!\n");
          exit(0);
        }
      }
      /*
       * If the current operator was before a previous in the ordering of the
       * Hilbert spaces, we decrease nbef. If it was after, we decrease
       * naf
       */
      if (nbef_prev[j]<op->n_before){
        nbef = nbef/nop_prev[j];
      } else {
        naf = naf/nop_prev[j];
      }
    }

    partial_trace_over_one(tmp_full_dm,tmp_dm,nbef,op->my_levels,naf,previous_total_levels);

    /* Destroy old large copy, copy smaller DM into a new 'large' copy for the next trace */
    destroy_dm(tmp_full_dm);
    VecDuplicate(tmp_dm,&tmp_full_dm);
    VecCopy(tmp_dm,tmp_full_dm);
    destroy_dm(tmp_dm);
    /* Store this ops information in the *_prev arrays */
    nbef_prev[i] = op->n_before;
    nop_prev[i]  = op->my_levels;
  }

  /* Check that ptraced_dm is big enough */

  VecGetSize(ptraced_dm,&dm_size);

  if (dm_size<pow(current_total_levels,2)){
    if (nid==0){
      printf("ERROR! ptraced_dm is not large enough to store the traced over density matrix!\n");
      printf("       Please ensure that the Hilbert space size of the ptraced_dm is large enough\n");
      printf("       to store the Hilbert space size that you are tracing down to.\n");
      exit(0);
    }
  }

  if (dm_size>pow(current_total_levels,2)){
    if (nid==0){
      printf("Warning! ptraced_dm is larger than the traced over density matrix!\n");
      printf("         This will work, but it may not be what you meant.\n");
    }
  }


  /* Assume ptraced_dm has been created, copy ptraced information into in */
  VecCopy(tmp_full_dm,ptraced_dm);

  /* Destroy tmp_full_dm */
  destroy_dm(tmp_full_dm);


  PetscFree(nbef_prev);
  PetscFree(nop_prev);

  va_end(ap);
  return;
}



/*
 *Measure a dm by a given operator, forcing a change:
 *    dm -> U^\dag dm U
 * or, in vectored notation
 *    dm -> (U* cross U) dm
 */
void measure_dm(Vec dm,operator op){
  Mat tmp_op_mat;
  PetscInt Istart,Iend,i,j,dim;
  PetscScalar val;
  Vec tmp_dm;
  dim = total_levels*total_levels;
  MatCreate(PETSC_COMM_WORLD,&tmp_op_mat);
  MatSetType(tmp_op_mat,MATMPIAIJ);
  MatSetSizes(tmp_op_mat,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
  MatSetFromOptions(tmp_op_mat);
  MatMPIAIJSetPreallocation(tmp_op_mat,5,NULL,5,NULL);
  MatGetOwnershipRange(tmp_op_mat,&Istart,&Iend);
  VecDuplicate(dm,&tmp_dm);
  //Construct U* cross U
  for (i=Istart;i<Iend;i++){
    _get_val_j_from_global_i(i,op,&j,&val,0); // Get the corresponding j and val
    MatSetValue(tmp_op_mat,i,j,val,INSERT_VALUES);
  }
  MatAssemblyBegin(tmp_op_mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tmp_op_mat,MAT_FINAL_ASSEMBLY);


  //Do (U* cross U) * dm
  MatMult(tmp_op_mat,dm,tmp_dm);
  //Normalize resulting density matrix
  //Get trace(tmp_dm)
  //FIXME: assumes op has identity; I think all do, but vec_ops might break here?
  get_expectation_value(tmp_dm,&val,1,subsystem_list[0]->eye);
  //Divide all elements by trace val
  VecScale(tmp_dm,1/val);

  VecCopy(tmp_dm,dm);

  //Cleanup tmp objects
  MatDestroy(&tmp_op_mat);
  VecDestroy(&tmp_dm);
  return;
}

/*
 * partial_trace_keep does the partial trace, keeping only a list of operators
 * tracing out the operators not listed. Assumes systems are listed in order they
 * were created.
 *
 * Inputs:
 *     Vec full_dm: the full Hilbert space density matrix to trace over.
 *                  Note: MUST be the full density matrix; use other routines
 *                        if starting with an already traced DM
 *     int number_of_ops: number of ops in list to keep
 *     <list of ops>: A list of operators which are to be kept
 *
 * Outpus:
 *     Vec ptraced_dm: the result of the partial trace is stored here.
 *                     Note: Assumed to already be allocated via create_dm()
 */

void partial_trace_keep(Vec full_dm,Vec ptraced_dm,int number_of_ops,...){
  va_list ap;
  Vec tmp_dm,tmp_full_dm;
  operator op,*keeper_systems;
  PetscInt i,j,k,l,current_total_levels,*nbef_prev,*nop_prev,dm_size,nbef,naf,previous_total_levels;
  PetscInt num_trace;

  if (number_of_ops>num_subsystems){
    if (nid==0){
      printf("ERROR! number_of_ops cannot be greater than num_subsystems\n");
      exit(0);
    }
  }

  va_start(ap,number_of_ops);

  /* Check that the full_dm is of size total_levels */

  VecGetSize(full_dm,&dm_size);

  if (dm_size!=pow(total_levels,2)){
    if (nid==0){
      printf("ERROR! You need to use the full Hilbert space sized DM in \n");
      printf("       partial_trace_keep!\n");
      exit(0);
    }
  }

  num_trace = num_subsystems - number_of_ops;
  PetscMalloc1(num_trace,&nbef_prev);
  PetscMalloc1(num_trace,&nop_prev);
  PetscMalloc1(number_of_ops,&keeper_systems);
  current_total_levels = total_levels;
  /* Initially create a copy of the full DM */
  VecDuplicate(full_dm,&tmp_full_dm);
  VecCopy(full_dm,tmp_full_dm);
  // Loop through ops that we are tracing over
  for (i=0;i<number_of_ops;i++){
    op = va_arg(ap,operator);
    keeper_systems[i] = op;
  }
  k=0;
  l=0;
  for (i=0;i<num_subsystems;i++){
    op = subsystem_list[i];
    if (op->n_before==keeper_systems[k]->n_before){
      //this is a keeper system, so skip it!
      k = k+1;
      if (k==number_of_ops) {
        /*
         * Small hack to go back to the last one of the list and
         * keep checking that one; otherwise k will go larger
         * than the list size and give a seg fault. This is safe
         * because, once we have passed all of the ops, we
         * won't hit op->n_before ever again.
         */
        k = k-1;
      }
    } else {
      previous_total_levels = current_total_levels;
      current_total_levels = current_total_levels/op->my_levels;

      /* Creat a smaller, temporary DM to store the current partial trace */
      create_dm(&tmp_dm,current_total_levels);

      nbef = op->n_before;
      naf  = total_levels/(op->my_levels*nbef);

      /* Update nbef and naf with the already removed operators */
      for (j=0;j<l;j++) {
        if (nbef_prev[j]==op->n_before){
          if (nid==0){
            printf("ERROR! Partial tracing the same operator twice does not make sense!\n");
            exit(0);
          }
        }
        /*
         * If the current operator was before a previous in the ordering of the
         * Hilbert spaces, we decrease nbef. If it was after, we decrease
         * naf
         */
        if (nbef_prev[j]<op->n_before){
          nbef = nbef/nop_prev[j];
        } else {
          naf = naf/nop_prev[j];
        }
      }

      partial_trace_over_one(tmp_full_dm,tmp_dm,nbef,op->my_levels,naf,previous_total_levels);

      /* Destroy old large copy, copy smaller DM into a new 'large' copy for the next trace */
      destroy_dm(tmp_full_dm);
      VecDuplicate(tmp_dm,&tmp_full_dm);
      VecCopy(tmp_dm,tmp_full_dm);
      destroy_dm(tmp_dm);
      /* Store this ops information in the *_prev arrays */
      nbef_prev[l] = op->n_before;
      nop_prev[l]  = op->my_levels;
      l = l+1;
    }
  }

  /* Check that ptraced_dm is big enough */

  VecGetSize(ptraced_dm,&dm_size);

  if (dm_size<pow(current_total_levels,2)){
    if (nid==0){
      printf("ERROR! ptraced_dm is not large enough to store the traced over density matrix!\n");
      printf("       Please ensure that the Hilbert space size of the ptraced_dm is large enough\n");
      printf("       to store the Hilbert space size that you are tracing down to (keep).\n");
      exit(0);
    }
  }

  if (dm_size>pow(current_total_levels,2)){
    if (nid==0){
      printf("Warning! ptraced_dm is larger than the traced over density matrix!\n");
      printf("         This will work, but it may not be what you meant.\n");
    }
  }


  /* Assume ptraced_dm has been created, copy ptraced information into in */
  VecCopy(tmp_full_dm,ptraced_dm);

  /* Destroy tmp_full_dm */
  destroy_dm(tmp_full_dm);


  PetscFree(nbef_prev);
  PetscFree(nop_prev);
  PetscFree(keeper_systems);
  va_end(ap);
  return;
}

/*
 * void create_dm creates a new density matrix object
 * and initializes it to 0
 *
 * Inputs:
 *        Vec* new_dm   - where the new density matrix will be stored
 *        PetscInt size - size of the Hilbert space (N if the matrix is NxN)
 * Outpus:
 *        Vec* new_dm   - new, initialized DM
 *
 */
void create_dm(Vec* new_dm,PetscInt size){
  /* Create the dm, partition with PETSc */
  VecCreate(PETSC_COMM_WORLD,new_dm);
  VecSetType(*new_dm,VECMPI);
  VecSetSizes(*new_dm,PETSC_DECIDE,pow(size,2));
  /* Set all elements to 0 */
  VecSet(*new_dm,0.0);
}
/*
 * void create_dm creates a new density matrix object
 * and initializes it to 0
 *
 * Inputs:
 *        Vec* new_dm   - where the new density matrix will be stored
 *        PetscInt size - size of the Hilbert space (N if the matrix is NxN)
 * Outpus:
 *        Vec* new_dm   - new, initialized DM
 *
 */
void create_full_dm(Vec* new_dm){
  PetscInt size = total_levels;

  _check_initialized_A();

  /* Create the dm, partition with PETSc */
  /* VecCreate(PETSC_COMM_WORLD,new_dm); */
  /* VecSetType(*new_dm,VECMPI); */
  /* if (_lindblad_terms) { */
  /*   VecSetSizes(*new_dm,PETSC_DECIDE,pow(size,2)); */
  /* } else { */
  /*   VecSetSizes(*new_dm,PETSC_DECIDE,size); */
  /* } */
  if (_lindblad_terms) {
    MatCreateVecs(full_A,new_dm,NULL);
  } else {
    MatCreateVecs(ham_A,new_dm,NULL);
  }

  /* Set all elements to 0 */
  VecSet(*new_dm,0.0);
}

/*
 * set_dm_from_initial_pop sets the initial condition from the
 * initial conditions provided via the set_initial_pop routine.
 * Currently a sequential routine. May need to be updated in the future.
 *
 * Inputs:
 *      Vec x
 */
void set_dm_from_initial_pop(Vec x){
  PetscInt    i,j,init_row_op=0,n_after,i_sub,j_sub,n_before;
  PetscScalar mat_tmp_val;
  PetscInt    *index_array;
  Mat         subspace_dm,rho_mat;
  int         simple_init_pop=1;
  MatScalar   *rho_mat_array;
  PetscReal   vec_pop;

  /*
   * See if there are any vec operators
   */

  for (i=0;i<num_subsystems&&simple_init_pop==1;i++){
    if (subsystem_list[i]->my_op_type==VEC){
      simple_init_pop = 0;
    }
  }

  if (nid==0&&simple_init_pop==1){
    /*
     * We can only use this simpler initialization if all of the operators
     * are ladder operators, and the user hasn't used any special initialization routine
     */
    for (i=0;i<num_subsystems;i++){
      n_after   = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);
      init_row_op += ((int)subsystem_list[i]->initial_pop)*n_after;
      //      init_row_op += ((int)subsystem_list[i]->initial_pop)*subsystem_list[i]->n_before;
    }

    if(_lindblad_terms) {
      init_row_op = total_levels*init_row_op + init_row_op;
    } else {
      init_row_op = init_row_op;
    }
    mat_tmp_val = 1. + 0.0*PETSC_i;
    VecSetValue(x,init_row_op,mat_tmp_val,INSERT_VALUES);

  } else if (nid==0){
    /*
     * This more complicated initialization routine allows for the vec operator
     * to take distributed values (say, 1/3 1/3 1/3)
     */


    /* Create temporary PETSc matrices */
    MatCreate(PETSC_COMM_SELF,&subspace_dm);
    MatSetType(subspace_dm,MATSEQDENSE);
    MatSetSizes(subspace_dm,total_levels,total_levels,total_levels,total_levels);
    MatSetUp(subspace_dm);

    MatCreate(PETSC_COMM_SELF,&rho_mat);
    MatSetType(rho_mat,MATSEQDENSE);
    MatSetSizes(rho_mat,total_levels,total_levels,total_levels,total_levels);
    MatSetUp(rho_mat);
    /*
     * Set initial density matrix to the identity matrix, because
     * A' cross B' cross C' = I (A cross I_b cross I_c) (I_a cross B cross I_c)
     */
    for (i=0;i<total_levels;i++){
      MatSetValue(rho_mat,i,i,1.0,INSERT_VALUES);
    }
    MatAssemblyBegin(rho_mat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(rho_mat,MAT_FINAL_ASSEMBLY);
    /* Loop through subsystems */
    for (i=0;i<num_subsystems;i++){
      n_after = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);
      //      n_before = subsystem_list[i]->n_before;
      /*
       * If the subsystem is a ladder operator, the population will be just on a
       * diagonal element within the subspace.
       * LOWER is in the if because that is the op in the subsystem list for ladder operators
       */
      if (subsystem_list[i]->my_op_type==LOWER){

        /* Zero out the subspace density matrix */
        MatZeroEntries(subspace_dm);
        i_sub = (int)subsystem_list[i]->initial_pop;
        j_sub = i_sub;
        mat_tmp_val = 1. + 0.0*PETSC_i;
        _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
        //     _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,n_after,n_before,subsystem_list[i]->my_levels);
        _mult_PETSc_init_DM(subspace_dm,rho_mat,(double)1.0);

      } else if (subsystem_list[i]->my_op_type==VEC){
        /*
         * If the subsystem is a vec type operator, we loop over each
         * state and set the initial population.
         */
        n_after = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);
        //        n_before = subsystem_list[i]->n_before;
        /* Zero out the subspace density matrix */
        MatZeroEntries(subspace_dm);
        vec_pop = 0.0;
        for (j=0;j<subsystem_list[i]->my_levels;j++){
          i_sub   = subsystem_list[i]->vec_op_list[j]->position;
          j_sub   = i_sub;
          vec_pop += subsystem_list[i]->vec_op_list[j]->initial_pop;
          mat_tmp_val = subsystem_list[i]->vec_op_list[j]->initial_pop + 0.0*PETSC_i;
          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
          //          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,n_after,n_before,subsystem_list[i]->my_levels);
        }

        if (vec_pop==(double)0.0){
          printf("WARNING! No initial population set for a vector operator!\n");
          printf("         Defaulting to all population in the 0th element\n");
          mat_tmp_val = 1.0 + 0.0*PETSC_i;
          vec_pop     = 1.0;
          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,0,0,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
          //          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,0,0,n_after,n_before,subsystem_list[i]->my_levels);
        }
        /*
         * Now that the subspace_dm is fully constructed, we multiply it into the full
         * initial DM
         */
        _mult_PETSc_init_DM(subspace_dm,rho_mat,vec_pop);
      }
    }
    /* vectorize the density matrix */

    /* First, get the array where matdense is stored */
    MatDenseGetArray(rho_mat,&rho_mat_array);

    /* Create an index list for, needed for VecSetValues */
    PetscMalloc1(total_levels*total_levels,&index_array);
    for (i=0;i<total_levels;i++){
      for (j=0;j<total_levels;j++){
        index_array[i+j*total_levels] = i+j*total_levels;
      }
    }

    VecSetValues(x,total_levels*total_levels,index_array,rho_mat_array,INSERT_VALUES);
    MatDenseRestoreArray(rho_mat,&rho_mat_array);
    MatDestroy(&subspace_dm);
    MatDestroy(&rho_mat);
    PetscFree(index_array);
  }
  assemble_dm(x);
  return;
}

/*
 * add_to_dm_from_string takes a string of occupation numbers
 * and adds that state to the starting density matrix
 * (or psi, eventually)
 */

void add_to_dm_from_string(Vec rho,PetscScalar val,char string[]){
  PetscInt string_length,i,passed_levels,real_levels;

  string_length = strlen(string);
  if(string_length!=num_subsystems) {
    if (nid==0){
      printf("ERROR! String must be the same size as the number of subsystems!\n");
      printf("       (in add_to_dm_from_string)!\n");
      exit(0);
    }
  }


  for(i=0;i<num_subsystems;i++){
    real_levels   = subsystem_list[i]->my_levels;
    if(real_levels>9){
      if (nid==0){
        printf("ERROR! Number of levels must be less than 10 to use this routinie!\n");
        printf("       (in add_to_dm_from_string)!\n");
        exit(0);
      }
    }
    passed_levels = string[i] - '0';
    if(passed_levels>real_levels){
      if (nid==0){
        printf("ERROR! Number of levels in string must be the less than or equal to the true number!\n");
        printf("       (in add_to_dm_from_string)!\n");
        exit(0);
      }
    }
  }

}


/*
 * set_dm_from_initial_pop sets the initial condition from the
 * initial conditions provided via the set_initial_pop routine.
 * Currently a sequential routine. May need to be updated in the future.
 *
 * Inputs:
 *      Vec x
 */
void set_initial_dm_2qds_first_plus_pop(Vec x,Vec rho_2qds){
  PetscInt    i,j,init_row_op=0,n_after,i_sub,j_sub,n_before;
  PetscScalar mat_tmp_val;
  PetscInt    *index_array;
  Mat         subspace_dm,rho_mat;
  int         simple_init_pop=1;
  MatScalar   *rho_mat_array;
  PetscReal   vec_pop;

  /*
   * See if there are any vec operators
   */
  /* for (i=0;i<num_subsystems&&simple_init_pop==1;i++){ */
  /*   if (subsystem_list[i]->my_op_type==VEC){ */
  /*     simple_init_pop = 0; */
  /*   } */
  /* } */




    /* Create temporary PETSc matrices */
  MatCreate(PETSC_COMM_SELF,&subspace_dm);
  MatSetType(subspace_dm,MATSEQDENSE);
  MatSetSizes(subspace_dm,total_levels,total_levels,total_levels,total_levels);
  MatSetUp(subspace_dm);
  MatZeroEntries(subspace_dm);
  for (i=0;i<4;i++){
    for (j=0;j<4;j++){
      get_dm_element(rho_2qds,i,j,&mat_tmp_val);
      n_after = total_levels/(4);
      _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i,j,1.0,n_after,4);
    }
  }
  if (nid==0){
    MatCreate(PETSC_COMM_SELF,&rho_mat);
    MatSetType(rho_mat,MATSEQDENSE);
    MatSetSizes(rho_mat,total_levels,total_levels,total_levels,total_levels);
    MatSetUp(rho_mat);
    /*
     * Set initial density matrix to the identity matrix, because
     * A' cross B' cross C' = I (A cross I_b cross I_c) (I_a cross B cross I_c)
     */
    for (i=0;i<total_levels;i++){
      MatSetValue(rho_mat,i,i,1.0,INSERT_VALUES);
    }
    MatAssemblyBegin(rho_mat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(rho_mat,MAT_FINAL_ASSEMBLY);
    /* Loop through subsystems */
    /* Do the first 2 systems, which are assumed to be 2 levels */


    _mult_PETSc_init_DM(subspace_dm,rho_mat,(double) 1.0);

    for (i=2;i<num_subsystems;i++){

      n_after = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);
      //      n_before = subsystem_list[i]->n_before;
      /*
       * If the subsystem is a ladder operator, the population will be just on a
       * diagonal element within the subspace.
       * LOWER is in the if because that is the op in the subsystem list for ladder operators
       */
      if (subsystem_list[i]->my_op_type==LOWER){

        /* Zero out the subspace density matrix */
        MatZeroEntries(subspace_dm);
        i_sub = (int)subsystem_list[i]->initial_pop;
        j_sub = i_sub;
        mat_tmp_val = 1. + 0.0*PETSC_i;
        _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
        //     _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,n_after,n_before,subsystem_list[i]->my_levels);
        _mult_PETSc_init_DM(subspace_dm,rho_mat,(double)1.0);

      }
    }
    /* vectorize the density matrix */

    /* First, get the array where matdense is stored */
    MatDenseGetArray(rho_mat,&rho_mat_array);

    /* Create an index list for, needed for VecSetValues */
    PetscMalloc1(total_levels*total_levels,&index_array);
    for (i=0;i<total_levels;i++){
      for (j=0;j<total_levels;j++){
        index_array[i+j*total_levels] = i+j*total_levels;
      }
    }

    VecSetValues(x,total_levels*total_levels,index_array,rho_mat_array,INSERT_VALUES);
    MatDenseRestoreArray(rho_mat,&rho_mat_array);
    MatDestroy(&rho_mat);
    PetscFree(index_array);
  }
  MatDestroy(&subspace_dm);

  assemble_dm(x);
  return;
}

/*
 * void get_dm_element gets a specific i,j element from the
 * input density matrix - global version; will grab from other
 * cores.
 * NOTE: This should be called by all processors, or the code will hang!
 *
 * Inputs:
 *        Vec new_dm   - density matrix from which to get element
 *        PetscInt row - i location of requested element
 *        PetscInt col - j location of requested element
 * Outpus:
 *        PetscScalar val - requested density matrix element
 *
 */
void get_dm_element(Vec dm,PetscInt row,PetscInt col,PetscScalar *val){
  PetscInt location[1],dm_size,my_start,my_end;
  PetscScalar val_array[1];
  location[0] = row;
  VecGetSize(dm,&dm_size);
  location[0] = sqrt(dm_size)*row + col;

  VecGetOwnershipRange(dm,&my_start,&my_end);
  if (location[0]>=my_start&&location[0]<my_end) {
    VecGetValues(dm,1,location,val_array);
  } else{
    val_array[0] = 0.0 + 0.0*PETSC_i;
  }
  MPI_Allreduce(MPI_IN_PLACE,val_array,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

  *val = val_array[0];
}

/*
 * void get_dm_element_local gets a specific i,j element from the
 * input density matrix - local version; only gets values local
 * to the current core
 * NOTE: This should be only be called with an i,j such that the
 *       calling core owns that position!
 *
 * Inputs:
 *        Vec new_dm   - density matrix from which to get element
 *        PetscInt row - i location of requested element
 *        PetscInt col - j location of requested element
 * Outpus:
 *        PetscScalar val - requested density matrix element
 *
 */

void get_dm_element_local(Vec dm,PetscInt row,PetscInt col,PetscScalar *val){
  PetscInt location[1],dm_size,my_start,my_end;
  PetscScalar val_array[1];
  location[0] = row;
  VecGetSize(dm,&dm_size);
  location[0] = sqrt(dm_size)*row + col;
  VecGetOwnershipRange(dm,&my_start,&my_end);
  if (location[0]>=my_start&&location[0]<my_end) {
    VecGetValues(dm,1,location,val_array);
  } else{
    if (nid==0){
      printf("ERROR! Can only get elements local to a core in !\n");
      printf("       get_dm_element_local.\n");
      exit(0);
    }
  }
  *val = val_array[0];
}


/*
 * void destroy_dm frees the memory from a previously created dm object
 *
 * Inputs:
 *         Vec dm - memory to free
 * Outpus:
 *         None, but frees the memory from dm
 *
 */

void destroy_dm(Vec dm){
  /* destroy the density matrix */
  VecDestroy(&dm);
}

/*
 * void add_value_to_dm adds the specified value to the density matrix
 *
 * Inputs:
 *         Vec dm - dm to add to
 *         PetscInt row - row location of value
 *         PetscInt col - column location of value
 *         PetscScalr val - value to add at row,col
 * Outpus:
 *         None, but adds the value to the dm
 *
 * NOTE: You MUST call assemble_dm after adding all values.
 *
 */

void add_value_to_dm(Vec dm,PetscInt row,PetscInt col,PetscScalar val){
  PetscInt location,dm_size,low,high;

  /* Get information about the dm */
  VecGetSize(dm,&dm_size);
  VecGetOwnershipRange(dm,&low,&high);
  location = sqrt(dm_size)*row + col;
  /* If I own it, set the value */
  if (location>=low&&location<high) {
    VecSetValue(dm,location,val,ADD_VALUES);
  }
}

/*
 * void assemble_dm puts all the cached values in the right place and
 * allows for the dm to be used
 *
 * Inputs:
 *         Vec dm - dm to assemble
 * Outpus:
 *         None, but assembles the dm
 *
 */
void assemble_dm(Vec dm){
  VecAssemblyBegin(dm);
  VecAssemblyEnd(dm);
}

void partial_trace_over_one(Vec full_dm,Vec ptraced_dm,PetscInt nbef,PetscInt nop,PetscInt naf,PetscInt cur_levels){
  PetscInt ibef,jbef,iaf,jaf,iop,loc_full,loc_sub,full_low,full_high;
  PetscScalar val;
  const PetscScalar *full_dm_array;

  /* Get the full_dm information */
  VecGetOwnershipRange(full_dm,&full_low,&full_high);
  VecGetArrayRead(full_dm,&full_dm_array);

  for (ibef=0;ibef<nbef;ibef++){
    for (jbef=0;jbef<nbef;jbef++){
      for (iaf=0;iaf<naf;iaf++){
        for (jaf=0;jaf<naf;jaf++){
          for (iop=0;iop<nop;iop++){
            /* Location in full hilbert space */
            loc_full = cur_levels*(naf*nop*ibef+naf*iop+iaf) + naf*nop*jbef+naf*iop+jaf;

            /* Only partial trace a processor's own values */
            if (loc_full>=full_low&&loc_full<full_high) {
              val     = full_dm_array[loc_full-full_low]; //Offset because we need local location

              /* Location in ptraced Hilbert space */
              /* naf*nbef = total_levels in subspace. Needed to get linearized position */
              loc_sub = naf*nbef*(naf*ibef+iaf)+naf*jbef+jaf;

              /* Add values */
              VecSetValue(ptraced_dm,loc_sub,val,ADD_VALUES);
            }
          }
        }
      }
    }
  }

  VecRestoreArrayRead(full_dm,&full_dm_array);

  /* Assemble array */
  VecAssemblyBegin(ptraced_dm);
  VecAssemblyEnd(ptraced_dm);
}


/*
 * int get_num_populations calculates the number of populations of all operators previously declared
  * Inputs:
  *        None
  * Returns:
 *         int num_pop - Number of populations = number of regular operators + num vec levels
 */
int get_num_populations() {
  int               num_pop,i;
  /*
   * Loop through operators to see how many populations we need to
   * calculate, because VEC need a population for each level.
   * We also set an array that translates i_subsystem to i_pop for normal ops.
   */
  num_pop = 0;
  for (i=0;i<num_subsystems;i++){
    if (subsystem_list[i]->my_op_type==VEC){
      num_pop += subsystem_list[i]->my_levels;
    } else {
      num_pop += 1;
    }
  }
  return num_pop;
}

/*
 * void get_populations calculates the populations of all operators previously declared
 * and returns the number of populations and the populations
 * For normal operators, this is Tr(C^\dagger C rho). For vec_op's, we instead calculate
 * the probability of each level, i.e. Tr(|i><i| rho) for i=0,num_levels.
 *
 * Inputs:
 *         Vec x   - density matrix with which to find the populations
 *                    Note: Must represent the full density matrix, with all states
  * Outputs:
 *         double **populations - an array of those populations
 */
void get_populations(Vec x,double **populations) {
  int               j,my_levels,n_after,cur_state,num_pop;
  int               *i_sub_to_i_pop;
  PetscInt          x_low,x_high,i,dm_size,diag_index,dim;
  const PetscScalar *xa;
  PetscReal         tmp_real,tmp_imag;
  if(_lindblad_terms) {
    dim = total_levels*total_levels;
  } else {
    dim = total_levels;
  }
  VecGetSize(x,&dm_size);

  if (dm_size!=dim){
    if (nid==0){
      printf("ERROR! The input density matrix does not seem to be the full one!\n");
      printf("       Populations cannot be calculated.\n");
      exit(0);
    }
  }

  VecGetOwnershipRange(x,&x_low,&x_high);
  VecGetArrayRead(x,&xa);

  /*
   * Loop through operators to see how many populations we need to
   * calculate, because VEC need a population for each level.
   * We also set an array that translates i_subsystem to i_pop for normal ops.
   */
  i_sub_to_i_pop = malloc(num_subsystems*sizeof(int));
  num_pop = 0;
  for (i=0;i<num_subsystems;i++){
    if (subsystem_list[i]->my_op_type==VEC){
      i_sub_to_i_pop[i] = num_pop;
      num_pop += subsystem_list[i]->my_levels;
    } else {
      i_sub_to_i_pop[i] = num_pop;
      num_pop += 1;
    }
  }

  /* Initialize population arrays */
  for (i=0;i<num_pop;i++){
    (*populations)[i] = 0.0;
  }


  for (i=0;i<total_levels;i++){
    if (_lindblad_terms) {
      diag_index = i*total_levels+i;
    } else {
      /* If we are using the schrodinger solver, then i is the diag index */
      diag_index = i;
    }
    if (diag_index>=x_low&&diag_index<x_high) {
      /* Get the diagonal entry of rho */
      tmp_real = (double)PetscRealPart(xa[diag_index-x_low]);
      tmp_imag = (double)PetscImaginaryPart(xa[diag_index-x_low]);
      //      printf("%e \n",(double)PetscRealPart(xa[i*(total_levels)+i-x_low]));
      for(j=0;j<num_subsystems;j++){
        /*
         * We want to calculate the populations. To do that, we need
         * to know what the state of the number operator for a specific
         * subsystem is for a given i. To accomplish this, we make use
         * of the fact that we can take the diagonal index i from the
         * full space and get which diagonal index it is in the subspace
         * by calculating:
         * i_subspace = floor(i/n_a) % l
         * For regular operators, these are just number operators, and we count from 0,
         * so, cur_state = i_subspace
         *
         * For VEC ops, we can use the same technique. Once we get cur_state (i_subspace),
         * we use that to go the appropriate location in the population array.
         */
        if (subsystem_list[j]->my_op_type==VEC){
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels));
          if (_lindblad_terms) {
            (*populations)[i_sub_to_i_pop[j]+cur_state] += tmp_real;
          } else {
            // If we are using the Schrodinger solver, we need to use
            // a^* a (that is, complex conjugate of a times a)
            (*populations)[i_sub_to_i_pop[j]+cur_state] += tmp_real*tmp_real + tmp_imag*tmp_imag;
          }
        } else {
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels));
          if (_lindblad_terms) {
            (*populations)[i_sub_to_i_pop[j]] += tmp_real*cur_state;
          } else {
            // If we are using the Schrodinger solver, we need to use
            // a^* a (that is, complex conjugate of a times a)
            (*populations)[i_sub_to_i_pop[j]] += cur_state * (tmp_real*tmp_real + tmp_imag*tmp_imag);
          }
        }
      }
    }
  }

  /* Reduce results across cores */
  if(nid==0) {
    MPI_Reduce(MPI_IN_PLACE,(*populations),num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce((*populations),(*populations),num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }

  /* Print results */
  /* if(nid==0) { */
  /*   /\* A negative time means we were called from steadystate, so print to stdout*\/ */
  /*   if (time<0){ */
  /*     printf("Final (*populations): "); */
  /*     for(i=0;i<num_pop;i++){ */
  /*       printf(" %e ",(*populations)[i]); */
  /*     } */
  /*     printf("\n"); */
  /*   } else if (time==0){ */
  /*     /\* If time is 0, we should overwrite the old file *\/ */
  /*     FILE *f = fopen("pop","w"); */
  /*     fprintf(f,"%e ",time); */
  /*     for(i=0;i<num_pop;i++){ */
  /*       fprintf(f," %e ",(*populations)[i]); */
  /*     } */
  /*     fprintf(f,"\n"); */
  /*     fclose(f); */
  /*   } else { */
  /*     /\* Normal printing, append to file *\/ */
  /*     FILE *f = fopen("pop","a"); */
  /*     fprintf(f,"%e ",time); */
  /*     for(i=0;i<num_pop;i++){ */
  /*       fprintf(f," %e ",(*populations)[i]); */
  /*     } */
  /*     fprintf(f,"\n"); */
  /*     fclose(f); */
  /*   } */
  /* } */


  /* Put the array back in Petsc's hands */
  VecRestoreArrayRead(x,&xa);
  /* Vec ptraced_dm; */
  /* PetscReal pop_tmp = 0; */
  /* create_dm(&ptraced_dm,subsystem_list[0]->my_levels); */
  /* partial_trace_over(x,ptraced_dm,1,subsystem_list[1]); */
  /* VecGetArrayRead(ptraced_dm,&xa); */
  /* for (i=0;i<subsystem_list[0]->my_levels;i++) { */
  /*   pop_tmp = pop_tmp + i*xa[i*subsystem_list[0]->my_levels+i]; */
  /* } */
  /* printf("pop_tmp: %f\n",pop_tmp); */
  /* VecRestoreArrayRead(ptraced_dm,&xa); */
  /* destroy_dm(ptraced_dm); */

  /* create_dm(&ptraced_dm,subsystem_list[1]->my_levels); */
  /* partial_trace_over(x,ptraced_dm,1,subsystem_list[0]); */
  /* VecGetArrayRead(ptraced_dm,&xa); */
  /* pop_tmp = 0.0; */

  /* for (i=0;i<subsystem_list[1]->my_levels;i++) { */
  /*   pop_tmp = pop_tmp + i*xa[i*subsystem_list[1]->my_levels+i]; */
  /* } */
  /* printf("pop_tmp: %f\n",pop_tmp); */
  /* VecRestoreArrayRead(ptraced_dm,&xa); */
  /* destroy_dm(ptraced_dm); */

  /* Free memory */
  free(i_sub_to_i_pop);
  return;
}

/*
 * void get_expectation_value calculates the expectation value of the multiplication
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
 *         Vec dm            - full Hilbert space density matrix
 *         int number_of_ops - number of operators in the list
 *          ...              - list of operators
 * Outputs:
 *         PetscScalar *trace_val - the expectation value of the multiplied operators
 *
 * An example calling this function:
 *      get_expectation_value(dm,&expect,4,ph[0]->dag,ph[1]->dag,ph[0],ph[1]);
 *
 */

void get_expectation_value(Vec rho,PetscScalar *trace_val,int number_of_ops,...){
  va_list ap;
  operator *op;
  PetscInt i,j,this_i,this_j,my_j_start,my_j_end,my_start,my_end,dim,dm_size;
  PetscInt this_loc;
  PetscScalar dm_element,val,op_val;

  va_start(ap,number_of_ops);
  op = malloc(number_of_ops*sizeof(struct operator));
  /* Loop through passed in ops and store in list */
  for (i=0;i<number_of_ops;i++){
    op[i] = va_arg(ap,operator);
  }
  va_end(ap);

  if(_lindblad_terms) {
    dim = total_levels*total_levels;
  } else {
    dim = total_levels;

    _get_expectation_value_psi(rho,trace_val,number_of_ops,op);
    return;
    if (nid==0){
      printf("ERROR! Expectation values does not support the Schrodinger solver!\n");
      exit(0);
    }

  }
  VecGetSize(rho,&dm_size);

  if (dm_size!=dim){
    if (nid==0){
      printf("ERROR! The input density matrix does not seem to be the full one!\n");
      printf("       Populations cannot be calculated.\n");
      exit(0);
    }
  }


  /*
   * Calculate Tr(ABC...*rho) using the following observations:
   *     Tr(A*rho) = sum_i (A*rho)_ii = sum_i sum_k A_ik rho_ki
   *          i.e., we do not need a j loop.
   *     Each operator (ABCD...) are very sparse - having less than
   *          1 value per row. This allows us to efficiently do the
   *          multiplication of ABCD... by just calculating the value
   *          for one of the indices (i); if there is no matching j,
   *          the value is 0.
   *
   * Since the matrix is stored in C format, we use the complex
   * conjugate of Rho instead of rho directly, so that we can
   * minimize communication
   */
  *trace_val = 0.0 + 0.0*PETSC_i;
  VecGetOwnershipRange(rho,&my_start,&my_end);

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
    for (j=0;j<number_of_ops;j++){
      _get_val_j_from_global_i(this_i,op[j],&this_j,&val,-1); // Get the corresponding j and val
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
     * NOTE the transpose here and in get_dm_element_local!
     */
    this_loc = total_levels*i + this_i;
    if (this_loc>=my_start&&this_loc<my_end) {
      get_dm_element_local(rho,i,this_i,&dm_element);

      /*
       * Take complex conjugate of dm_element (since we relied on the fact
       * that rho was hermitian to get better data locality)
       */
      *trace_val = *trace_val + op_val*(PetscRealPart(dm_element) + PetscImaginaryPart(dm_element)*PETSC_i);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE,trace_val,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

  free(op);
  return;
}

/*
 * void get_bipartite_concurrence calculates the bipartite concurrence of a density matrix
 * bipartite concurrence is defined as:
 *              C(rho) = max(0,lamba_1 - lambda_2 - lambda_3 - lambda_4)
 * where lambda_i's are the square roots of the eigenvalues (in decreasing order) of:
 *              rho*(sigma_y cross sigma_y)*conj(rho)*(sigma_y cross sigma_y)
 * where sigma_y is the y pauli spin matrix
 *
 * Inputs:
 *         Vec dm   - density matrix with which to find the concurrence
 *                    Note: Must represent a 4x4 density matrix!
 * Outputs:
 *         double *concurrence - the concurrence of the state
 *
 */

void get_bipartite_concurrence(Vec dm,double *concurrence) {
  VecScatter        ctx_dm;
  PetscInt          i,dm_size,levels;
  PetscScalar       *dm_a,val;
  PetscReal         max;
  Mat               dm_mat,dm_tilde,result_mat,sigy_sigy;
  Vec               dm_local;
  /* Variables needed for LAPACK */
  PetscScalar  *work,*eigs,sdummy,*array;
  PetscReal    *rwork;
  PetscBLASInt idummy,lwork,lierr,nb;

  levels = 4;//4 is hardcoded because this is bipartite concurrence

  VecGetSize(dm,&dm_size);
  if (dm_size!=levels*levels){
    if (nid==0){
      printf("ERROR! The input density matrix is not 4x4!\n");
      printf("       Concurrence cannot be calculated.\n");
      exit(0);
    }
  }

  /* Collect DM onto master core */
  VecScatterCreateToZero(dm,&ctx_dm,&dm_local);

  VecScatterBegin(ctx_dm,dm,dm_local,INSERT_VALUES,SCATTER_FORWARD);
  VecScatterEnd(ctx_dm,dm,dm_local,INSERT_VALUES,SCATTER_FORWARD);
  /* Rank 0 now has a local copy of the matrices, so it does the calculations */
  if (nid==0){
    levels = sqrt(dm_size);
    /*
     * We want to work with the density matrices as matrices directly,
     * so that we can get eigenvalues, etc.
     */
    VecGetArray(dm_local,&dm_a);
    MatCreateSeqDense(PETSC_COMM_SELF,levels,levels,dm_a,&dm_mat);
    MatCreateSeqDense(PETSC_COMM_SELF,levels,levels,NULL,&sigy_sigy);

    /* Fill in the sigy_sigy matrix */
    val = -1.0;
    MatSetValue(sigy_sigy,0,3,val,INSERT_VALUES);
    MatSetValue(sigy_sigy,3,0,val,INSERT_VALUES);
    val = 1.0;
    MatSetValue(sigy_sigy,1,2,val,INSERT_VALUES);
    MatSetValue(sigy_sigy,2,1,val,INSERT_VALUES);

    MatAssemblyBegin(sigy_sigy,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(sigy_sigy,MAT_FINAL_ASSEMBLY);
    MatDuplicate(dm_mat,MAT_COPY_VALUES,&dm_tilde);

    /* Calculate conjugate of dm */
    MatConjugate(dm_tilde);

    /* Calculate dm_tilde = (sigy cross sigy) conj(dm) (sigy cross sigy)*/
    MatMatMult(sigy_sigy,dm_tilde,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&result_mat);
    MatMatMult(result_mat,sigy_sigy,MAT_REUSE_MATRIX,PETSC_DEFAULT,&dm_tilde);

    /* Calculate dm * dm_tilde */
    MatMatMult(dm_mat,dm_tilde,MAT_REUSE_MATRIX,PETSC_DEFAULT,&result_mat);

      /* We need the actual array so that we can pass it to lapack */
    MatDenseGetArray(result_mat,&array);

    /* Setup lapack things */
    idummy = levels;
    lwork  = 5*levels;
    PetscMalloc1(5*levels,&work);
    PetscMalloc1(2*levels,&rwork);
    PetscMalloc1(levels,&eigs);
    PetscBLASIntCast(levels,&nb);

    /* Call LAPACK through PETSc to ensure portability */
    LAPACKgeev_("N","N",&nb,array,&nb,eigs,&sdummy,&idummy,&sdummy,&idummy,work,&lwork,rwork,&lierr);
    /*
     * We want eig_max - sum(other_eigs), so we do
     * 2*eig_max - sum(all_eigs)
     */

    /* Find max eigenvalue and sum up all eigs */

    for (i=0;i<levels;i++){
      max = PetscRealPart(eigs[i]);
      if(max<1e-14){
        eigs[i] = 0.0;
      }
    }
    max = -1; //The values should all be positive, so we can set the base comparison to negative
    *concurrence = 0;
    for (i=0;i<levels;i++){
      *concurrence = *concurrence + sqrt(PetscRealPart(eigs[i]));
      if (sqrt(PetscRealPart(eigs[i]))>max) {
        max = sqrt(PetscRealPart(eigs[i]));
      }
    }
    *concurrence = 2*max - *concurrence;

    /* Clean up memory */
    VecRestoreArray(dm_local,&dm_a);
    MatDenseRestoreArray(result_mat,&array);
    MatDestroy(&dm_mat);
    MatDestroy(&sigy_sigy);
    MatDestroy(&dm_tilde);
    MatDestroy(&result_mat);
    PetscFree(work);
    PetscFree(rwork);
    PetscFree(eigs);

  }

  /* Broadcast the value to all cores */
  MPI_Bcast(concurrence,1,MPI_DOUBLE,0,PETSC_COMM_WORLD);

  VecDestroy(&dm_local);
  VecScatterDestroy(&ctx_dm);
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
 */
void get_fidelity(Vec dm,Vec dm_r,double *fidelity) {
  VecScatter        ctx_dm,ctx_dm_r;
  PetscInt          i,dm_size,dm_r_size,levels;
  PetscScalar       *dm_a,*dm_r_a;;
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
}

void _get_expectation_value_psi(Vec psi,PetscScalar *trace_val,int number_of_ops,operator *ops){
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
    for (j=0;j<number_of_ops;j++){
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
