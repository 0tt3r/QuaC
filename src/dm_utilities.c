#include "dm_utilities.h"
#include <stdlib.h>
#include <stdio.h>
#include <petscblaslapack.h>

/*
 * partial_trace_over does the partial trace over a list of operators,
 * leaving the operators not listed.
 *
 * Inputs:
 *     Vec full_dm: the full Hilbert space density matrix to trace ove.
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
  PetscInt i,j,current_total_levels,*nbef_prev,*nop_prev,dm_size,nbef,naf;
  va_start(ap,number_of_ops);
  
  /* Check that the full_dm is of size total_levels */

  VecGetSize(full_dm,&dm_size);

  if (dm_size!=pow(total_levels,2)){
    if (nid==0){
      printf("ERROR! You need to use the full Hilbert space sized DM in \n");
      printf("       partial_trace_over!");
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
    current_total_levels = current_total_levels/op->my_levels;
    /* Creat a smaller, temporary DM to store the current partial trace */
    create_dm(&tmp_dm,current_total_levels);

    nbef = op->n_before;
    naf  = total_levels/(op->my_levels*nbef);

    /* Update nbef and naf with the already removed operators */
    for (j=0;j<i;j++) {
      if (nbef_prev[j]==op->n_before){
        if (nid==0){
          printf("ERROR! Partial tracing the same operator twice does not make sense!");
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
    partial_trace_over_one(tmp_full_dm,tmp_dm,nbef,op->my_levels,naf);
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

  if (dm_size<current_total_levels){
    if (nid==0){
      printf("ERROR! ptraced_dm is not large enough to store the traced over density matrix!\n");
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
  VecCreate(PETSC_COMM_WORLD,new_dm);
  VecSetType(*new_dm,VECMPI);
  VecSetSizes(*new_dm,PETSC_DECIDE,pow(size,2));
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
  PetscInt    i,j,init_row_op=0,n_after,i_sub,j_sub;
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
    }

    init_row_op = total_levels*init_row_op + init_row_op;
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
        _mult_PETSc_init_DM(subspace_dm,rho_mat,(double)1.0);

      } else if (subsystem_list[i]->my_op_type==VEC){
        /*
         * If the subsystem is a vec type operator, we loop over each
         * state and set the initial population.
         */
        n_after = total_levels/(subsystem_list[i]->my_levels*subsystem_list[i]->n_before);
        /* Zero out the subspace density matrix */
        MatZeroEntries(subspace_dm);
        vec_pop = 0.0;
        for (j=0;j<subsystem_list[i]->my_levels;j++){
          i_sub   = subsystem_list[i]->vec_op_list[j]->position;
          j_sub   = i_sub;
          vec_pop += subsystem_list[i]->vec_op_list[j]->initial_pop;
          mat_tmp_val = subsystem_list[i]->vec_op_list[j]->initial_pop + 0.0*PETSC_i;
          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,i_sub,j_sub,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
        }

        if (vec_pop==(double)0.0){
          printf("WARNING! No initial population set for a vector operator!\n");
          printf("         Defaulting to all population in the 0th element\n");
          mat_tmp_val = 1.0 + 0.0*PETSC_i;
          vec_pop     = 1.0;
          _add_PETSc_DM_kron_ij(mat_tmp_val,subspace_dm,rho_mat,0,0,subsystem_list[i]->n_before,n_after,subsystem_list[i]->my_levels);
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
 * void get_dm_element gets a specific i,j element from the 
 * input density matrix.
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
  PetscInt location[1],dm_size;
  PetscScalar val_array[1];
  location[0] = row;
  VecGetSize(dm,&dm_size);
  location[0] = sqrt(dm_size)*row + col;

  VecGetValues(dm,1,location,val_array);
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

void partial_trace_over_one(Vec full_dm,Vec ptraced_dm,PetscInt nbef,PetscInt nop,PetscInt naf){
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
            loc_full = total_levels*(naf*nop*ibef+naf*iop+iaf) + naf*nop*jbef+naf*iop+jaf;
            
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
 * void get_populations calculates the populations of all operators previously declared
 * and prints it to a file
 * For normal operators, this is Tr(C^\dagger C rho). For vec_op's, we instead calculate
 * the probability of each level, i.e. Tr(|i><i| rho) for i=0,num_levels. 
 *
 * Inputs: 
 *         Vec x   - density matrix with which to find the populations
 *                    Note: Must represent the full density matrix, with all states
 *         PetscReal time - the current time of the timestep, or negative to print to stdout
 * Outputs:
 *         None, but prints to file or stdout
 */

void get_populations(Vec x,PetscReal time) {
  int               j,my_levels,n_after,cur_state,num_pop;
  int               *i_sub_to_i_pop;
  PetscInt          x_low,x_high,i,dm_size;
  const PetscScalar *xa;
  PetscReal         tmp_real;
  double            *populations;

  VecGetSize(x,&dm_size);

  if (dm_size!=total_levels*total_levels){
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
  populations = malloc(num_pop*sizeof(double));
  for (i=0;i<num_pop;i++){
    populations[i] = 0.0;
  }


  for (i=0;i<total_levels;i++){
    if ((i*total_levels+i)>=x_low&&(i*total_levels+i)<x_high) {
      /* Get the diagonal entry of rho */
      tmp_real = (double)PetscRealPart(xa[i*(total_levels)+i-x_low]);
      //      printf("%e \n",(double)PetscRealPart(xa[i*(total_levels)+i-x_low]));
      for(j=0;j<num_subsystems;j++){
        /*
         * We want to calculate the populations. To do that, we need 
         * to know what the state of the number operator for a specific
         * subsystem is for a given i. To accomplish this, we make use
         * of the fact that we can take the diagonal index i from the
         * full space and get which diagonal index it is in the subspace
         * by calculating:
         * i_subspace = mod(mod(floor(i/n_a),l*n_a),l)
         * For regular operators, these are just number operators, and we count from 0,
         * so, cur_state = i_subspace
         *
         * For VEC ops, we can use the same technique. Once we get cur_state (i_subspace),
         * we use that to go the appropriate location in the population array.
         */
        if (subsystem_list[j]->my_op_type==VEC){
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]+cur_state] += tmp_real;
        } else {
          my_levels = subsystem_list[j]->my_levels;
          n_after   = total_levels/(my_levels*subsystem_list[j]->n_before);
          cur_state = ((int)floor(i/n_after)%(my_levels*n_after))%my_levels;
          populations[i_sub_to_i_pop[j]] += tmp_real*cur_state;
        }
      }
    }
  } 

  /* Reduce results across cores */
  if(nid==0) {
    MPI_Reduce(MPI_IN_PLACE,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(populations,populations,num_pop,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }

  /* Print results */
  /* FIXME? Possibly move the printing to user? */
  if(nid==0) {
    /* A negative time means we were called from steadystate, so print to stdout*/
    if (time<0){
      printf("Final populations: ");
      for(i=0;i<num_pop;i++){
        printf(" %e ",populations[i]);
      }
      printf("\n");
    } else if (time==0){
      /* If time is 0, we should overwrite the old file */
      FILE *f = fopen("pop","w");
      fprintf(f,"%e ",time);
      for(i=0;i<num_pop;i++){
        fprintf(f," %e ",populations[i]);
      }
      fprintf(f,"\n");
      fclose(f);
    } else {
      /* Normal printing, append to file */
      FILE *f = fopen("pop","a");
      fprintf(f,"%e ",time);
      for(i=0;i<num_pop;i++){
        fprintf(f," %e ",populations[i]);
      }
      fprintf(f,"\n");
      fclose(f);
    }
  }

  
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
  free(populations);
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
    max = -1; //The values should all be positive, so we can set the base comparison to negative
    *concurrence = 0;
    for (i=0;i<levels;i++){
      *concurrence = *concurrence + PetscRealPart(eigs[i]);
      if (PetscRealPart(eigs[i])>max) {
        max = PetscRealPart(eigs[i]);
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
  MatHermitianTranspose(V,MAT_REUSE_MATRIX,&V);
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
