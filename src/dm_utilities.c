
#include "dm_utilities.h"
#include <stdlib.h>
#include <stdio.h>



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

void create_dm(Vec* new_dm,PetscInt size){
  /* Create the dm, partition with PETSc */
  VecCreate(PETSC_COMM_WORLD,new_dm);
  VecSetType(*new_dm,VECMPI);
  VecSetSizes(*new_dm,PETSC_DECIDE,pow(size,2));
  /* Set all elements to 0 */
  VecSet(*new_dm,0.0);
}

void destroy_dm(Vec dm){
  /* destroy the density matrix */
  VecDestroy(&dm);
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


void get_populations(Vec x,PetscReal time) {
  int               j,my_levels,n_after,cur_state,num_pop;
  int               *i_sub_to_i_pop;
  PetscInt          x_low,x_high,i;
  const PetscScalar *xa;
  PetscReal         tmp_real;
  double            *populations;
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
  if(nid==0) {
    /* A negative time means we were called from steadystate, so print to stdout*/
    if (time<0){
      printf("Final populations: ");
      for(i=0;i<num_pop;i++){
        printf(" %e ",populations[i]);
      }
      printf("\n");
    } else {
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
