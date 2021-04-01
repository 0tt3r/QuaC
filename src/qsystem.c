#include "kron_p.h" //Includes petscmat.h and operators_p.h
#include "quac_p.h"
#include "operators.h"
#include "qsystem.h"
#include "quantum_gates.h"
#include "quantum_circuits.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

PetscErrorCode _RHS_time_dep_ham_sys(TS,PetscReal,Vec,Mat,Mat,void*); // Move to header?
PetscErrorCode _RHS_time_dep_ham_mf_sys(TS,PetscReal,Vec,Mat,Mat,void*); // Move to header?


void initialize_system(qsystem *qsys){
  qsystem temp = NULL;
  PetscInt num_init_alloc = 150;
  int tmp_nid,tmp_np;
  if (!petsc_initialized){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to call QuaC_initialize before creating\n");
    PetscPrintf(PETSC_COMM_WORLD,"       any systems!\n");
    exit(0);
  }

  temp = malloc(sizeof(struct qsystem));
  temp->hspace_frozen = 0;
  temp->dm_equations   = PETSC_FALSE; //Assume no density matrix at beginning
  temp->mcwf_solver    = PETSC_FALSE; //Assume deterministic solver
  temp->time_step_called = PETSC_FALSE;
  temp->total_levels   = 1;
  temp->dim = 1;

  temp->min_time_dep = 1e-10; //Minimum

  temp->num_subsystems = 0;
  temp->alloc_subsystems = num_init_alloc;
  temp->subsystem_list = malloc(num_init_alloc*sizeof(struct operator));
  //temp->stiff_solver could go here if we used it
  temp->mat_allocated = 0;
  //Alloc some space for the mat terms initially
  temp->num_time_indep = 0;
  temp->num_time_dep = 0;

  temp->alloc_time_dep = num_init_alloc;
  temp->alloc_time_indep = num_init_alloc;

  temp->time_indep = malloc(num_init_alloc*sizeof(mat_term));
  temp->time_dep = malloc(num_init_alloc*sizeof(mat_term));

  temp->ts_monitor = NULL;
  //Distribution info
  /* Get core's id */
  MPI_Comm_rank(PETSC_COMM_WORLD,&(tmp_nid));
  /* Get number of processors */
  MPI_Comm_size(PETSC_COMM_WORLD,&(tmp_np));
  temp->nid = tmp_nid;
  temp->np = tmp_np;
  temp->my_num = -1;
  temp->Istart = -1;
  temp->Iend   = -1;

  //Circuit info
  temp->num_circuits = -1;
  temp->current_circuit = 0;


  *qsys = temp;
  return;
}

void destroy_system(qsystem *qsys){
  PetscInt i;
  free((*qsys)->subsystem_list);

  clear_mat_terms_sys(*qsys);
  free((*qsys)->time_dep); /***/
  free((*qsys)->time_indep); /**/

  if((*qsys)->mcwf_solver==PETSC_TRUE && (*qsys)->time_step_called==PETSC_TRUE){
    VecDestroy(&((*qsys)->mcwf_work_vec));
    VecDestroy(&((*qsys)->mcwf_backup_vec));
    free((*qsys)->rand_number);
  }


  if((*qsys)->num_circuits>0){
    free((*qsys)->circuit_list);
  }
  free((*qsys));
}

void _create_single_op(PetscInt total_levels,PetscInt number_of_levels,
                       op_type my_op_type,operator *op){
  operator temp = NULL;
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->initial_exc = 0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = my_op_type;
  /* Since this is a basic operator, not a vec, set positions to -1 */
  temp->position    = -1;
  *op           = temp;

  return;
}

void create_op_sys(qsystem sys,PetscInt number_of_levels,operator *new_op){

  operator temp = NULL,temp1 = NULL;
  operator *tmp_list;
  PetscInt i=0;
  if (sys->hspace_frozen){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You cannot add more operators after\n");
    PetscPrintf(PETSC_COMM_WORLD,"       the creating a dm or constructing the matrix!\n");
    exit(0);
  }

  /* First make the annihilation operator */
  _create_single_op(sys->total_levels,number_of_levels,LOWER,&temp);
  *new_op           = temp;
  temp1             = temp;

  /* Make creation operator */
  _create_single_op(sys->total_levels,number_of_levels,RAISE,&temp);
  temp->dag         = temp1; //Point dagger operator to LOWER op
  (*new_op)->dag    = temp;

  /* Make number operator */
  _create_single_op(sys->total_levels,number_of_levels,NUMBER,&temp);
  (*new_op)->n      = temp;
  temp->dag         = temp; //NUMBER operator is hermitian

  /* Make identity operator */
  _create_single_op(sys->total_levels,number_of_levels,IDENTITY,&temp);
  (*new_op)->eye    = temp;
  temp->dag         = temp; //IDENTITY operator is hermitian

  /* Make SIGMA_X operator (only valid for qubits, made for every system) */
  _create_single_op(sys->total_levels,number_of_levels,SIGMA_X,&temp);
  (*new_op)->sig_x      = temp;
  temp->dag         = temp; //pauli operators are hermitian

  /* Make SIGMA_Z operator (only valid for qubits, made for every system) */
  _create_single_op(sys->total_levels,number_of_levels,SIGMA_Z,&temp);
  (*new_op)->sig_z      = temp;
  temp->dag         = temp; //pauli operators are hermitian

  /* Make SIGMA_Y operator (only valid for qubits, made for every system) */
  _create_single_op(sys->total_levels,number_of_levels,SIGMA_Y,&temp);
  (*new_op)->sig_y      = temp;
  temp->dag         = temp; //pauli operators are hermitian

  /* Increase total_levels */
  sys->total_levels = sys->total_levels*number_of_levels;

  /* Add to list */
  if (sys->num_subsystems==sys->alloc_subsystems){
    /* Realloc array */
    sys->alloc_subsystems = 2*sys->alloc_subsystems;
    tmp_list = malloc(sys->num_subsystems*sizeof(struct operator));
    for (i=0;i<sys->num_subsystems;i++){
      tmp_list[i] = sys->subsystem_list[i];
    }
    free(sys->subsystem_list);
    sys->subsystem_list = malloc(sys->alloc_subsystems*sizeof(struct operator));
    for (i=0;i<sys->num_subsystems;i++){
      sys->subsystem_list[i] = tmp_list[i];
    }
    free(tmp_list);
  }

  /* Save position in hilbert space */
  (*new_op)->pos_in_sys_hspace = sys->num_subsystems;

  sys->subsystem_list[sys->num_subsystems] = (*new_op);
  sys->num_subsystems++;

  return;
}


void _create_single_vec(PetscInt total_levels,PetscInt number_of_levels,
                        PetscInt position,operator *op){
  operator temp = NULL;
  temp              = malloc(sizeof(struct operator));
  temp->initial_pop = (double) 0.0;
  temp->initial_exc = 0;
  temp->n_before    = total_levels;
  temp->my_levels   = number_of_levels;
  temp->my_op_type  = VEC;
  /* This is a VEC operator; set its position */
  temp->position    = position;
  *op           = temp;

  return;
}

void create_vec_op_sys(qsystem qsys,PetscInt number_of_levels,vec_op *new_vec){

  operator temp = NULL,temp1 = NULL;
  operator *tmp_list;
  PetscInt i=0;
  if (qsys->hspace_frozen){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You cannot add more vec_ops after\n");
    PetscPrintf(PETSC_COMM_WORLD,"       the creating a dm or constructing the matrix!\n");
    exit(0);
  }

  (*new_vec) = malloc(number_of_levels*(sizeof(struct operator*)));

  for (i=0;i<number_of_levels;i++){
    _create_single_vec(qsys->total_levels,number_of_levels,i,&temp);
    (*new_vec)[i]     = temp;
  }

  /*
   * Store the top of the array in vec[0], so we can access it later,
   * through subsystem_list.
   */
  (*new_vec)[0]->vec_op_list = (*new_vec);

  /* Increase total_levels */
  qsys->total_levels = qsys->total_levels*number_of_levels;

  /* Add to list */
  if (qsys->num_subsystems==qsys->alloc_subsystems){
    /* Realloc array */
    qsys->alloc_subsystems = 2*qsys->alloc_subsystems;
    tmp_list = malloc(qsys->num_subsystems*sizeof(struct operator));
    for (i=0;i<qsys->num_subsystems;i++){
      tmp_list[i] = qsys->subsystem_list[i];
    }
    free(qsys->subsystem_list);
    qsys->subsystem_list = malloc(qsys->alloc_subsystems*sizeof(struct operator));
    for (i=0;i<qsys->num_subsystems;i++){
      qsys->subsystem_list[i] = tmp_list[i];
    }
    free(tmp_list);
  }

  /* Save position in hilbert space */
  (*new_vec)[0]->pos_in_sys_hspace = qsys->num_subsystems;

  /*
   * We store just the first VEC in the subsystem list, since it has
   * enough information to define all others
   */
  qsys->subsystem_list[qsys->num_subsystems] = (*new_vec);
  qsys->num_subsystems++;

  return;
}

void destroy_vec_op_sys(vec_op *vop){
  PetscInt i,levels;
  levels = (*vop)[0]->my_levels;
  for(i=0;i<levels;i++){
    free((*vop)[i]);
  }
  free(*vop);

  return;
}

void destroy_op_sys(operator *op){

  free((*op)->dag);
  free((*op)->n);
  free((*op)->eye);
  free((*op)->sig_x);
  free((*op)->sig_z);
  free((*op)->sig_y);
  free(*op);

  return;
}

/*
 * Set the solver to use the MCWF solver
 */
void use_mcwf_solver(qsystem qsys,PetscInt num_tot_trajs,PetscInt seed){
  PetscInt i;

  if(qsys->hspace_frozen==PETSC_TRUE){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! use_mcwf_solver must be called before construct_matrix or create_qvec_sys!\n");
    exit(9);
  }

  if(qsys->dm_equations==PETSC_FALSE){
    //We want a lindblad term, or else there is no reason to have an ensemble, as there will be no randomness
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! use_mcwf_solver must be used with a Lindblad term!\n");
    exit(9);
  }

  if(num_tot_trajs<2){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! use_mcwf_solver must be called with more than 2 num_tot_trajs!\n");
    exit(9);
  }

  if(seed==NULL){
    //Generate a random seed from sprng
    seed = make_sprng_seed();
  }
  qsys->hspace_frozen = PETSC_TRUE;
  qsys->mcwf_solver = PETSC_TRUE;
  qsys->seed = seed;
  if(num_tot_trajs<1){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! num_tot_trajs must be greater than 0!\n");
    exit(9);
  }
  qsys->num_tot_trajs = num_tot_trajs;
  qsys->num_local_trajs = num_tot_trajs/qsys->np; //Distribute this properyl?
  //Initialize sprng random number generator
  init_sprng(seed,SPRNG_DEFAULT);

  //Cache the matrices for the jump operators
  //Time independent terms
  for(i=0;i<qsys->num_time_indep;i++){
    if(qsys->time_indep[i].my_term_type==LINDBLAD){
      construct_op_matrix_wf_list(qsys,1.0,&(qsys->time_indep[i].mat_A),qsys->time_indep[i].num_ops,qsys->time_indep[i].ops);
    }
  }
  //TODO Time dependent terms

  return;
}



void add_ham_term(qsystem sys,PetscScalar a,PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  if(sys->num_time_indep>sys->alloc_time_indep){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Asking for more terms than were allocated in add_ham_term!\n");
    exit(9);
  }

  sys->time_indep[sys->num_time_indep].my_term_type = HAM;
  sys->time_indep[sys->num_time_indep].a = a;
  sys->time_indep[sys->num_time_indep].num_ops = num_ops;
  sys->time_indep[sys->num_time_indep].ops = malloc(num_ops*sizeof(struct operator));

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_indep[sys->num_time_indep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);
  sys->num_time_indep = sys->num_time_indep+1;

  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void add_lin_term(qsystem sys,PetscScalar a,PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  if(sys->num_time_indep>sys->alloc_time_indep){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Asking for more terms than were allocated in add_lin_term!\n");
    exit(9);
  }
  sys->dm_equations = 1;//Lindblad equation
  sys->time_indep[sys->num_time_indep].my_term_type = LINDBLAD;
  sys->time_indep[sys->num_time_indep].a = a;
  sys->time_indep[sys->num_time_indep].num_ops = num_ops;
  sys->time_indep[sys->num_time_indep].ops = malloc(num_ops*sizeof(struct operator));

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_indep[sys->num_time_indep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);
  sys->num_time_indep = sys->num_time_indep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void add_lin_term_list(qsystem sys,PetscScalar a,PetscInt num_ops,operator *in_ops){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  if(sys->num_time_indep>sys->alloc_time_indep){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Asking for more terms than were allocated in add_lin_term_list!\n");
    exit(9);
  }
  sys->dm_equations = 1;//Lindblad equation
  sys->time_indep[sys->num_time_indep].my_term_type = LINDBLAD;
  sys->time_indep[sys->num_time_indep].a = a;
  sys->time_indep[sys->num_time_indep].num_ops = num_ops;
  sys->time_indep[sys->num_time_indep].ops = malloc(num_ops*sizeof(struct operator));

  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_indep[sys->num_time_indep].ops[i] = in_ops[i];
  }

  sys->num_time_indep = sys->num_time_indep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void add_ham_term_time_dep(qsystem sys,PetscScalar a,void *ctx,PetscScalar (*time_dep_func)(double),
                           PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */
  if(sys->num_time_dep>sys->alloc_time_dep){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Asking for more terms than were allocated in add_ham_term_time_dep!\n");
    exit(9);
  }

  sys->time_dep[sys->num_time_dep].my_term_type = TD_HAM;
  sys->time_dep[sys->num_time_dep].a = a;
  sys->time_dep[sys->num_time_dep].num_ops = num_ops;
  sys->time_dep[sys->num_time_dep].ops = malloc(num_ops*sizeof(struct operator));
  sys->time_dep[sys->num_time_dep].time_dep_func = time_dep_func;
  sys->time_dep[sys->num_time_dep].ctx = ctx;

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_dep[sys->num_time_dep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);
  sys->num_time_dep = sys->num_time_dep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

void add_lin_term_time_dep(qsystem sys,PetscScalar a,void *ctx,PetscScalar (*time_dep_func)(double),
                           PetscInt num_ops,...){
  va_list  ap;
  operator *ops;
  int      i;
  //FIXME This gives a segfault for some reason?
  /* PetscLogEventBegin(add_to_ham_event,0,0,0,0); */

  if(sys->num_time_dep>sys->alloc_time_dep){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Asking for more terms than were allocated in add_lin_term_time_dep!\n");
    exit(9);
  }

  sys->dm_equations = 1;//Lindblad equation
  sys->time_dep[sys->num_time_dep].my_term_type = TD_LINDBLAD;
  sys->time_dep[sys->num_time_dep].a = a;
  sys->time_dep[sys->num_time_dep].num_ops = num_ops;
  sys->time_dep[sys->num_time_dep].ops = malloc(num_ops*sizeof(struct operator));
  sys->time_dep[sys->num_time_dep].time_dep_func = time_dep_func;
  sys->time_dep[sys->num_time_dep].ctx = ctx;

  va_start(ap,num_ops);
  //Loop through operators, store them
  for (i=0;i<num_ops;i++){
    sys->time_dep[sys->num_time_dep].ops[i] = va_arg(ap,operator);
  }
  va_end(ap);

  sys->num_time_dep = sys->num_time_dep+1;
  /* PetscLogEventEnd(add_to_ham_event,0,0,0,0); */
  return;
}

/*
 * Clear the mat_terms so that the system can be reused with a different H / L
 */
void clear_mat_terms_sys(qsystem qsys){
  PetscInt i;

  for(i=0;i<qsys->num_time_dep;i++){
    free(qsys->time_dep[i].ops);
    if(qsys->mat_allocated){
      //Free each time dep term
      MatDestroy(&(qsys->time_dep[i].mat_A));
    }
  }

  for(i=0;i<qsys->num_time_indep;i++){
    free(qsys->time_indep[i].ops);
    if(qsys->mcwf_solver==PETSC_TRUE){
      if(qsys->time_indep[i].my_term_type==LINDBLAD){
        MatDestroy(&(qsys->time_indep[i].mat_A));
      }
    }
  }


  if (qsys->mat_allocated){
    //Free the matrix
    MatDestroy(&(qsys->mat_A));
    free(qsys->o_nnz);
    free(qsys->d_nnz);
    qsys->mat_allocated = 0;
  }

  //Reset our counters
  qsys->num_time_indep = 0;
  qsys->num_time_dep = 0;

  return;
}

void construct_matrix (qsystem qsys){
  PetscInt    i,j;
  PetscScalar tmp_a;
  qsys->hspace_frozen = 1;

  /* Check to make sure some operators were created */
  if (qsys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to create operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a matrix!\n");
    exit(0);
  }

  /* Check to make sure some terms were added*/
  if ((qsys->num_time_dep+qsys->num_time_indep)==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to add some terms before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a matrix!\n");
    exit(0);
  }

  if(qsys->dm_equations){
    //If we had Lindblads or are wanting to timestep a DM, we need to use the larger space
    qsys->dim = qsys->total_levels*qsys->total_levels;
    if(qsys->mcwf_solver){
      //Except when we use the MCWF solver, where we work in the wf space
      qsys->dim = qsys->total_levels;
    }
  } else {
    if(qsys->mcwf_solver){
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! It does not make sense to use the mcwf_solver with no Lindblad terms.\n");
      exit(9);
    }
    //Just the schrodinger solver
    qsys->dim = qsys->total_levels;
  }

  _preallocate_qsys_matrix(qsys);
  qsys->mat_allocated = 1;
  //Loop over time independent terms
  for(i=0;i<qsys->num_time_indep;i++){
    _add_ops_to_mat(qsys->time_indep[i].a,qsys->mat_A,qsys->time_indep[i].my_term_type,qsys->total_levels,
                    qsys->dm_equations,qsys->mcwf_solver,qsys->time_indep[i].num_ops,qsys->time_indep[i].ops);
  }

  /*
   * Loop over time dependent terms, using 1.0 as the scalar. This
   * pre-caches the time-dependent matrices. We will later add them to A during
   * time stepping.
   *
   * We also add the time dep matrix terms to A so that A has the right nonzero structure
   * when we add those time dependent terms
   */

  for(i=0;i<qsys->num_time_dep;i++){
    tmp_a = 1.0;
    _add_ops_to_mat(tmp_a,qsys->time_dep[i].mat_A,qsys->time_dep[i].my_term_type,qsys->total_levels,
                    qsys->dm_equations,qsys->mcwf_solver,qsys->time_dep[i].num_ops,qsys->time_dep[i].ops);
    tmp_a = 0.0;
    _add_ops_to_mat(tmp_a,qsys->mat_A,qsys->time_dep[i].my_term_type,qsys->total_levels,
                    qsys->dm_equations,qsys->mcwf_solver,qsys->time_dep[i].num_ops,qsys->time_dep[i].ops);

  }

  //Loop over diagonal to specifically add 0 to it.
  tmp_a = 0.0;
  for(i=qsys->Istart;i<qsys->Iend;i++){
    MatSetValue(qsys->mat_A,i,i,tmp_a,ADD_VALUES);
    for(j=0;j<qsys->num_time_dep;j++){
      MatSetValue(qsys->time_dep[j].mat_A,i,i,tmp_a,ADD_VALUES);
    }
  }
  MatAssemblyBegin(qsys->mat_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(qsys->mat_A,MAT_FINAL_ASSEMBLY);

  for(i=0;i<qsys->num_time_dep;i++){
    MatAssemblyBegin(qsys->time_dep[i].mat_A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(qsys->time_dep[i].mat_A,MAT_FINAL_ASSEMBLY);
  }
  if(qsys->dm_equations==PETSC_FALSE||qsys->mcwf_solver==PETSC_TRUE){
    //We scale the matrix by -i, because we did NOT account for that in the kron routines
    MatScale(qsys->mat_A,-PETSC_i);
    //Time dep matrices are taken care of in _RHS_time_dep_ham_sys
  }
  return;
}

void _setup_distribution(PetscInt nid,PetscInt np,PetscInt dim,PetscInt *my_num,PetscInt *Istart,PetscInt *Iend){
  PetscInt    count,remainder,tmp_nid,tmp_np;

  count = dim / np;
  remainder = dim % np;

  if (nid < remainder){
      *Istart = nid * (count + 1);
      *Iend   = (*Istart) + count + 1;
      *my_num = count + 1;
    } else {
      *Istart = nid * count + remainder;
      *Iend   = (*Istart) + (count - 1) + 1;
      *my_num = count;
    }

  return;
}

void _preallocate_qsys_matrix(qsystem qsys){
  PetscInt i,j,count,remainder,tmp_nid,tmp_np;
  PetscInt *o_nnz_td,*d_nnz_td; //Time dependent nnzs

  if(qsys->my_num<0){//Only setup the distribution once. Could be created otherwise
    _setup_distribution(qsys->nid,qsys->np,qsys->dim,&(qsys->my_num),&(qsys->Istart),&(qsys->Iend));
  }

  MatCreate(PETSC_COMM_WORLD,&(qsys->mat_A));
  MatSetType(qsys->mat_A,MATMPIAIJ);
  MatSetSizes(qsys->mat_A,qsys->my_num,qsys->my_num,qsys->dim,qsys->dim);
  MatSetFromOptions(qsys->mat_A);

  qsys->o_nnz = malloc(qsys->my_num*sizeof(PetscInt));
  qsys->d_nnz = malloc(qsys->my_num*sizeof(PetscInt));

  o_nnz_td = malloc(qsys->my_num*sizeof(PetscInt));
  d_nnz_td = malloc(qsys->my_num*sizeof(PetscInt));

  for(i=0;i<qsys->my_num;i++){
    qsys->o_nnz[i] = 0;
    //Start with assuming the diagonal has values
    //And a bit of a buffer incase we are adding many in the same spot at one time
    qsys->d_nnz[i] = 1;
    o_nnz_td[i] = 0;
    d_nnz_td[i] = 1;
  }


  /*
   * This counting can double count. For instance, if you added
   * several copies of the same exact term, it counts them separately,
   * counting the 1,2 element many times, for instance.
   *
   * FIXME: Possible minor solution: Have a counter which checks if basic
   * operators of a certain nonzero pattern have yet been added. Would fix
   * double counting of basic operators at least.
   */
  //Loop over time independent terms
  for(i=0;i<qsys->num_time_indep;i++){
    _count_ops_in_mat(qsys->d_nnz,qsys->o_nnz,qsys->total_levels,qsys->Istart,qsys->Iend,
                      qsys->mat_A,qsys->time_indep[i].my_term_type,
                      qsys->dm_equations,qsys->mcwf_solver,
                      qsys->time_indep[i].num_ops,qsys->time_indep[i].ops);

  }

  //Loop over time dependent terms
  for(i=0;i<qsys->num_time_dep;i++){
    //Create matrices for the time dependent parts; one for each
    MatCreate(PETSC_COMM_WORLD,&(qsys->time_dep[i].mat_A));
    MatSetType(qsys->time_dep[i].mat_A,MATMPIAIJ);
    MatSetSizes(qsys->time_dep[i].mat_A,qsys->my_num,qsys->my_num,qsys->dim,qsys->dim);
    MatSetFromOptions(qsys->time_dep[i].mat_A);

    //Matrix qsys->mat_A is only used for the size, n, and A and time_dep[i] have
    //the same size
    // FIXME: Perhaps it should be the matrix size, if that is really all that it is used for?
    _count_ops_in_mat(d_nnz_td,o_nnz_td,qsys->total_levels,qsys->Istart,qsys->Iend,
                      qsys->mat_A,qsys->time_dep[i].my_term_type,
                      qsys->dm_equations,qsys->mcwf_solver,
                      qsys->time_dep[i].num_ops,qsys->time_dep[i].ops);

    //Set the preallocation for time_dep[i] to only include time_dep[i]'s nonzers'
    MatMPIAIJSetPreallocation(qsys->time_dep[i].mat_A,-1,d_nnz_td,-1,o_nnz_td);

    //Add the nnzs to the mat_As nnzs and zero out for the next count
    for(j=0;j<qsys->my_num;j++){
      qsys->o_nnz[j] += o_nnz_td[j];
      qsys->d_nnz[j] += d_nnz_td[j];
      o_nnz_td[j] = 0;
      d_nnz_td[j] = 1; //always need 1 to ensure the diagonal has some elements
    }

  }

  //-1s are ignored
  MatMPIAIJSetPreallocation(qsys->mat_A,-1,qsys->d_nnz,-1,qsys->o_nnz);
  free(o_nnz_td);
  free(d_nnz_td);
  return;
}

void _preallocate_op_matrix(Mat *mat_A,PetscInt nid,PetscInt np,PetscInt dim,PetscInt tot_levels,mat_term_type my_mat_type,
                         PetscBool dm_equations,PetscBool mcwf_solver,PetscInt num_ops,operator *ops){
  PetscInt    i,count,remainder,tmp_nid,tmp_np,my_num,Istart,Iend,*o_nnz,*d_nnz;

  _setup_distribution(nid,np,dim,&my_num,&Istart,&Iend);

  MatCreate(PETSC_COMM_WORLD,mat_A);
  MatSetType(*mat_A,MATMPIAIJ);
  MatSetSizes(*mat_A,my_num,my_num,dim,dim);
  MatSetFromOptions(*mat_A);

  o_nnz = malloc((my_num)*sizeof(PetscInt));
  d_nnz = malloc((my_num)*sizeof(PetscInt));

  for(i=0;i<my_num;i++){
    o_nnz[i] = 0;
    //Start with assuming the diagonal has values
    //And a bit of a buffer incase we are adding many in the same spot at one time
    d_nnz[i] = 1;
  }


  /*
   * This counting can double count. For instance, if you added
   * several copies of the same exact term, it counts them separately,
   * counting the 1,2 element many times, for instance.
   *
   * FIXME: Possible minor solution: Have a counter which checks if basic
   * operators of a certain nonzero pattern have yet been added. Would fix
   * double counting of basic operators at least.
   */
  _count_ops_in_mat(d_nnz,o_nnz,tot_levels,Istart,Iend,
                    (*mat_A),my_mat_type,dm_equations,mcwf_solver,
                    num_ops,ops);

  //-1s are ignored
  MatMPIAIJSetPreallocation((*mat_A),-1,d_nnz,-1,o_nnz);

  free(o_nnz);
  free(d_nnz);
  return;
}

/*
 * Construct the matrix defined by ops[0]*ops[1]*ops[2]*... for the NxN wavefunction space
 * defined by qsys. qsys and ops must be consistent.
 */
void construct_op_matrix_wf_list(qsystem qsys,PetscScalar prefactor_a,Mat *mat_A,PetscInt num_ops,operator *ops){
  PetscInt    i,Istart,Iend;
  PetscScalar tmp_a;
  if(qsys->num_subsystems==0){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! You need to create operators before you construct\n");
    PetscPrintf(PETSC_COMM_WORLD,"       a matrix!\n");
    exit(0);
  }

  /*
   * We use total_levels instead of dim because dim is total_levels**2 when using DM equations
   * and we specifically want the matrix in the WF space. We set dm_equations = PETSC_FALSE
   * and mcwf_solver = PETSC_FALSE for similar reasons.
   */

  _preallocate_op_matrix(mat_A,qsys->nid,qsys->np,qsys->total_levels,qsys->total_levels,
                         HAM,PETSC_FALSE,PETSC_FALSE,num_ops,ops);

  _add_ops_to_mat(prefactor_a,*mat_A,HAM,qsys->total_levels,PETSC_FALSE,PETSC_FALSE,num_ops,ops);

  MatGetOwnershipRange(*mat_A,&Istart,&Iend);
  //Loop over diagonal to specifically add 0 to it.
  tmp_a = 0.0;
  for(i=Istart;i<Iend;i++){
    MatSetValue(*mat_A,i,i,tmp_a,ADD_VALUES);
  }
  MatAssemblyBegin(*mat_A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*mat_A,MAT_FINAL_ASSEMBLY);

  return;
}


/*
 *
 * set_ts_monitor accepts a user function which can calculate observables, print output, etc
 * at each time step.
 *
 * Inputs:
 *      PetscErrorCode *monitor - function pointer for user ts_monitor function
 *      void           *ctx     - user-defined struct to store data used in tsmonitor
 */
void set_ts_monitor_sys(qsystem sys,PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void *ctx){
  sys->ts_monitor = (*monitor);
  sys->ts_ctx     = ctx;
  return;
}

/*
 * time_step solves for the time_dependence of the system
 * that was previously setup using the add_to_ham and add_lin
 * routines. Solver selection and parameters can be controlled via PETSc
 * command line options. Default solver is TSRK3BS
 *
 * Inputs:
 *       qvec     x:             The density matrix/wavefunction, with appropriate inital conditions
 *       PetscReal init_time:    initial tim
 *       PetscReal dt:           initial timestep. For certain explicit methods, this timestep
 *                               can be changed, as those methods have adaptive time steps
 *       PetscReal time_max:     the maximum time to integrate to
 *       PetscInt steps_max:     max number of steps to take
 */
void time_step_sys(qsystem qsys,qvec x, PetscReal init_time, PetscReal time_max,
               PetscReal dt,PetscInt steps_max){
  PetscViewer    mat_view;
  TS             ts; /* timestepping context */
  Mat            AA;
  PetscInt       nevents=0,direction[2],i,j,retry,m_local,n_local;
  PetscBool      terminate[2]; //We have two types of events for now, hence [2]
  PetscReal      solve_time,atol=5e-7,rtol=5e-7; //These tolerances are from a simple T1 test. May not be general?
  TSConvergedReason reason;
  PetscScalar norm;
  PetscErrorCode ierr;
  PetscLogStagePop();
  PetscLogStagePush(solve_stage);

  qsys->time_step_called = PETSC_TRUE;
  qsys->solution_qvec = x;
  if(qsys->mat_allocated!=1){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! Matrix does not seem to be constructed! You need to\n");
    PetscPrintf(PETSC_COMM_WORLD,"       call construct_matrix!\n");
    exit(0);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*
   *       Create the timestepping solver and set various options       *
   *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
   * Create timestepping solver context
   */
  TSCreate(PETSC_COMM_WORLD,&ts);
  TSSetProblemType(ts,TS_LINEAR);

  /*
   * Set function to get information at every timestep
   */
  if (qsys->ts_monitor!=NULL){
    TSMonitorSet(ts,qsys->ts_monitor,qsys->ts_ctx,NULL);
  }

  /*
   * Set up ODE qsystem
   */
  TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,qsys);
  if (qsys->num_time_dep>0){
    VecDuplicate(x->data,&(qsys->work_vec));
    //Duplicate matrix for time dependent runs
    MatDuplicate(qsys->mat_A,MAT_COPY_VALUES,&AA);
    MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);
    TSSetRHSJacobian(ts,AA,AA,_RHS_time_dep_ham_sys,qsys);

    /* ierr = MatGetLocalSize(qsys->mat_A,&m_local,&n_local); */
    /* ierr = MatCreateShell(PETSC_COMM_WORLD,m_local,n_local,PETSC_DETERMINE,PETSC_DETERMINE,qsys,&AA);CHKERRQ(ierr); */
    /* ierr = MatShellSetOperation(AA,MATOP_MULT,(void (*)(void))_time_dep_mat_mult);CHKERRQ(ierr); */

    /* TSSetRHSJacobian(ts,AA,AA,_RHS_time_dep_ham_mf_sys,qsys); */

  } else {
    //Time indep
    TSSetRHSJacobian(ts,qsys->mat_A,qsys->mat_A,TSComputeRHSJacobianConstant,qsys);
  }
  /* Print information about the matrix. */
  //TODO Put verbose flag?
  /* PetscViewerASCIIOpen(PETSC_COMM_WORLD,NULL,&mat_view); */
  /* PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_INFO); */
  /* /\* PetscViewerPushFormat(mat_view,PETSC_VIEWER_ASCII_MATLAB); *\/ */
  /* MatView(qsys->mat_A,mat_view); */
  /* PetscViewerPopFormat(mat_view); */
  /* PetscViewerDestroy(&mat_view); */
  if(qsys->mcwf_solver==PETSC_TRUE){
    if(qsys->dm_equations==PETSC_FALSE){
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! It does not make sense to use the MCWF solver without DM equations.\n");
    }
    /*
     * If we are using the mcwf solver, we need to setup an event to catch if we have passed the
     * time for a quantum jump.
     */
    direction[nevents] = -1; //We only want to count an event if we go from positive to negative
    terminate[nevents] = PETSC_FALSE; //Keep time stepping after we passed our event
    qsys->post_event_functions[nevents] = _sys_MCWF_PostEventFunction;
    nevents   =  nevents + 1;

    if(x->ens_spawned==PETSC_FALSE){
      VecDuplicate(x->data,&(qsys->mcwf_work_vec));
      VecDuplicate(x->data,&(qsys->mcwf_backup_vec));
      qsys->rand_number = malloc(qsys->num_local_trajs*sizeof(PetscReal));
      //Copy initial data into the ensemble
      for(i=0;i<qsys->num_local_trajs;i++){
        VecCopy(x->data,x->ens_datas[i]);
        //Pick an initial random number
        qsys->rand_number[i] = sprng();

      }
      x->ens_spawned=PETSC_TRUE;
    }
  }

  /*
   * Set default options, can be changed at runtime
   */
  TSSetTimeStep(ts,dt);
  TSSetMaxSteps(ts,steps_max);
  TSSetMaxTime(ts,time_max);
  TSSetTime(ts,init_time);
  /* TSSetExactFinalTime(ts,TS_EXACTFINALTIME_INTERPOLATE); */
  TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);

  TSSetType(ts,TSRK);
  TSRKSetType(ts,TSRK3BS);
  TSSetTolerances(ts,atol,NULL,rtol,NULL);
  /* If we have a circuit to apply, set up the event handler */
  if(qsys->num_circuits>0){
    direction[nevents] = -1; //We only want to count an event if we go from positive to negative
    terminate[nevents] = PETSC_FALSE; //Keep time stepping after we passed our event
    qsys->post_event_functions[nevents] = _sys_QC_PostEventFunction;
    nevents   =  nevents + 1;
  }
  /* Arguments are: ts context, nevents, direction of zero crossing, whether to terminate,
   * a function to check event status, a function to apply events, private data context.
   */
  if(nevents>0){
    TSSetEventHandler(ts,nevents,&direction,&terminate,_sys_EventFunction,_sys_PostEventFunction,qsys);
    TSSetEventTolerances(ts,1e-9,NULL);
  }

  //Store a reference to the solution vector in the qsystem
  TSSetFromOptions(ts);
  if(qsys->mcwf_solver){
    for(i=0;i<qsys->num_local_trajs;i++){
      retry=0;
      VecCopy(x->ens_datas[i],qsys->mcwf_backup_vec);
      if(retry<10){
        //Reset some time stepping things
        x->ens_i = i;
        qsys->ens_i = i;
        TSSetTime(ts,init_time);
        TSSetStepNumber(ts,0);
        TSSetTimeStep(ts,dt);

        if(qsys->num_circuits>0){
          //Reset circuit stuff
          qsys->current_circuit=0;
          for(j=0;j<qsys->num_circuits;j++){
            qsys->circuit_list[qsys->current_circuit].current_gate = 0;
            qsys->circuit_list[qsys->current_circuit].current_layer = 0;
          }
        }
        //old_rand can likely be removed because we fixed the corner cases
        if(retry>0){
          //draw a new random number
          qsys->rand_number[qsys->ens_i] = sprng();
        }
        qsys->old_rand = qsys->rand_number[qsys->ens_i];
        VecCopy(qsys->mcwf_backup_vec,x->ens_datas[i]);
        TSSolve(ts,x->ens_datas[i]);
        VecDot(x->ens_datas[i],x->ens_datas[i],&norm);
        if(PetscRealPart(norm)<qsys->rand_number[qsys->ens_i]){
          //Reset random number, because we overshot the time_max, interpolated back
          //This should not happen because we correctly set tolerances. Left for error checking purposes
          PetscPrintf("ERROR! There was a reset: %d %f %.20f %.20f\n",i,PetscRealPart(norm),qsys->old_rand,qsys->rand_number[qsys->ens_i]);
          exit(0);
          qsys->rand_number[qsys->ens_i] = qsys->old_rand;
        }

        TSGetSolveTime(ts,&solve_time);
        if(abs(solve_time-time_max)>1e-8){
          TSGetConvergedReason(ts,&reason);
          PetscPrintf(PETSC_COMM_WORLD,"ERROR! Did not reach final time! Retrying. Reason: %d\n",reason);
          retry=retry+1;
        } else {
          retry = 100;
        }
      } else {
        PetscPrintf(PETSC_COMM_WORLD,"ERROR! Did not reach final time! Exhausted retries.\n");
        exit(9);
      }
    }
  } else {
    TSSolve(ts,x->data);
    TSGetSolveTime(ts,&solve_time);
    if(abs(solve_time-time_max)>1e-8){
      TSGetConvergedReason(ts,&reason);
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! Did not reach final time! Reason: %d\n",reason);
      //      exit(9);
    }
  }

  /* Free work space */
  TSDestroy(&ts);
  if(qsys->num_time_dep>0){
    VecDestroy(&(qsys->mcwf_work_vec));
    MatDestroy(&AA);
  }

  PetscLogStagePop();
  PetscLogStagePush(post_solve_stage);

  return;
}


PetscErrorCode _sys_EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx) {
  qsystem qsys = (qsystem) ctx;
  PetscInt event_num=0;
  PetscScalar tmp_fvalue;

  //Check MCWF event
  if(qsys->mcwf_solver==PETSC_TRUE){
    //Check mcwf event
    _sys_MCWF_EventFunction(qsys,U,&tmp_fvalue);
    fvalue[event_num] = tmp_fvalue;
    event_num = event_num+1;
  }

  //Check circuit event
  if(qsys->num_circuits>0){
    _sys_QC_EventFunction(qsys,t,&tmp_fvalue);
    fvalue[event_num] = tmp_fvalue;
    event_num = event_num+1;
  }

  return(0);
}


PetscErrorCode _sys_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],
                                           PetscReal t,Vec psi_data,PetscBool forward,void* ctx) {
  qsystem qsys = (qsystem) ctx;
  PetscInt i;


  for(i=0;i<nevents;i++){
    qsys->post_event_functions[event_list[i]](ts,nevents,event_list,t,psi_data,forward,ctx);
  }

  return(0);
}

PetscErrorCode _sys_MCWF_EventFunction(qsystem qsys,Vec U,PetscScalar *fvalue){
  PetscScalar norm;
  //PetscLogEventBegin(_qc_event_function_event,0,0,0,0);

  //Check if the norm of the wavefunction is below
  //Get norm = <psi | psi>
  VecDot(U,U,&norm);
  //if the norm goes below the rand_number, fvalue will be negative and we trigger the event
  //PETSc will then backtrack until it goes below TSEventTolerance
  *fvalue = PetscRealPart(norm - qsys->rand_number[qsys->ens_i]);
  //PetscLogEventEnd(_qc_event_function_event,0,0,0,0);
  return(0);
}



/* PostEventFunction is the other step in Petsc. If an event has happend, petsc will call this function
 * to apply that event.
 */
PetscErrorCode _sys_MCWF_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],
                                           PetscReal t,Vec psi_data,PetscBool forward,void* ctx) {
  qsystem qsys = (qsystem) ctx;
  PetscInt i,j,i_jump_op,num_ops;
  PetscBool found_jump_op=PETSC_FALSE;
  PetscReal rand_num_op,*probs_of_jump_op,total_prob_jump_op,tmp_sum_prob;
  PetscScalar trace_val;
  //PetscLogEventBegin(_qc_postevent_function_event,0,0,0,0);

  if (nevents) {
    //A jump has occurred, but we have to pick which one
    //Loop through all Lindblads and store prob in array
    //Array won't be larger than num_time_indep + num_time_dep
    probs_of_jump_op = malloc((qsys->num_time_indep + qsys->num_time_dep)*sizeof(PetscReal));
    total_prob_jump_op = 0.0;
    i_jump_op = 0;
    //Time independent terms
    for(i=0;i<qsys->num_time_indep;i++){
      if(qsys->time_indep[i].my_term_type==LINDBLAD){
        //Get the state C_i * |psi>
        MatMult(qsys->time_indep[i].mat_A,psi_data,qsys->mcwf_work_vec);
        //Now get <psi | C_i^\dag C_i | psi> by using VecDot
        VecDot(qsys->mcwf_work_vec,qsys->mcwf_work_vec,&trace_val);
        //Multiply the trace_val by gamma and store in array
        //Should be explicitly real? Maybe check?
        probs_of_jump_op[i_jump_op] = PetscRealPart(qsys->time_indep[i].a*trace_val);
        total_prob_jump_op = total_prob_jump_op + PetscRealPart(qsys->time_indep[i].a*trace_val);
        i_jump_op = i_jump_op+1;
      }
    }
    //TODO: time dependent, don't forget to use callback

    //Pick which jump has occurred
    rand_num_op = sprng();
    tmp_sum_prob = 0.0;
    i_jump_op = 0;
    //Time independent terms
    for(i=0;i<qsys->num_time_indep;i++){
      if(qsys->time_indep[i].my_term_type==LINDBLAD){
        tmp_sum_prob = tmp_sum_prob + probs_of_jump_op[i_jump_op]/total_prob_jump_op;
        if(tmp_sum_prob>=rand_num_op){
          //This is the jump that occurred, so we break
          found_jump_op=PETSC_TRUE;
          break;
        }
        i_jump_op = i_jump_op+1;
      }
    }

    //Apply the jump and renormalize the state if we found it in the time indep terms, or go through time_dep terms
    if(found_jump_op==PETSC_TRUE){


      //Apply jump C | psi>
      MatMult(qsys->time_indep[i].mat_A,psi_data,qsys->mcwf_work_vec);

      //Get renormalization value
      trace_val = 1/PetscSqrtReal(probs_of_jump_op[i_jump_op]); // sqrt(a * <psi|C_i^\dag C_i|psi>
      trace_val = trace_val * PetscSqrtComplex(qsys->time_indep[i].a); //need sqrt(a) on the numerator for units consistency

    } else {
    //TODO: time dependent, don't forget to use callback
    }

    VecScale(qsys->mcwf_work_vec,trace_val);
    //Copy the result into psi_data
    VecCopy(qsys->mcwf_work_vec,psi_data);

    //Draw a new random number
    qsys->old_rand = qsys->rand_number[qsys->ens_i];
    qsys->rand_number[qsys->ens_i] = sprng();
    free(probs_of_jump_op);
  }


  //PetscLogEventEnd(_qc_postevent_function_event,0,0,0,0);
  return(0);
}


/*
 * _RHS_time_dep_ham_sys adds the (user created) time dependent functions
 * to the time independent hamiltonian. It is used internally by PETSc
 * during time stepping.
 */

PetscErrorCode _RHS_time_dep_ham_sys(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx){
  double time_dep_val;
  PetscScalar time_dep_scalar;
  int i,j;
  operator op;
  qsystem qsys = (qsystem) ctx;

  MatZeroEntries(AA);
  MatCopy(qsys->mat_A,AA,SAME_NONZERO_PATTERN);

  PetscLogEventBegin(_RHS_time_dep_event,0,0,0,0);
  for(i=0;i<qsys->num_time_dep;i++){
    if(qsys->dm_equations==PETSC_FALSE||qsys->mcwf_solver==PETSC_TRUE){
      //We multiply by -i, because we did NOT account for that in the kron routines
      time_dep_scalar = -PETSC_i*qsys->time_dep[i].a*qsys->time_dep[i].time_dep_func(t,qsys->time_dep[i].ctx);
    } else {
      time_dep_scalar = qsys->time_dep[i].a*qsys->time_dep[i].time_dep_func(t,qsys->time_dep[i].ctx);
    }

    //Only add the term if it is above a threshold
    //Do A = A + s(t)*B
    if(PetscAbsScalar(time_dep_scalar)>PetscAbsScalar(qsys->min_time_dep)){
      MatAXPY(AA,time_dep_scalar,qsys->time_dep[i].mat_A,SUBSET_NONZERO_PATTERN);
    }
    /* _add_ops_to_mat(time_dep_scalar,AA,qsys->time_dep[i].my_term_type,qsys->total_levels, */
    /*                 qsys->dm_equations,qsys->mcwf_solver,qsys->time_dep[i].num_ops,qsys->time_dep[i].ops); */
  }
  PetscLogEventEnd(_RHS_time_dep_event,0,0,0,0);
  MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);

  if(AA!=BB) {
    MatAssemblyBegin(AA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA,MAT_FINAL_ASSEMBLY);
  }

  PetscFunctionReturn(0);
}


/*
 * _RHS_time_dep_ham_mf_sys caches the current time. It will be used in the custom matmult function
 */

PetscErrorCode _RHS_time_dep_ham_mf_sys(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx){
  double time_dep_val;
  PetscScalar time_dep_scalar;
  int i,j;
  operator op;
  qsystem qsys = (qsystem) ctx;

  qsys->current_time = t;
  PetscFunctionReturn(0);
}


 PetscErrorCode _time_dep_mat_mult(Mat A_shell,Vec X,Vec Y){
   qsystem qsys;
   PetscErrorCode ierr;
   PetscScalar time_dep_scalar;
   PetscReal t;
   PetscInt i;

   PetscLogEventBegin(_RHS_time_dep_event,0,0,0,0);
   ierr = MatShellGetContext(A_shell,(void**)&qsys);CHKERRQ(ierr);
   t = qsys->current_time;
   MatMult(qsys->mat_A,X,Y);
   for(i=0;i<qsys->num_time_dep;i++){
     if(qsys->dm_equations==PETSC_FALSE||qsys->mcwf_solver==PETSC_TRUE){
       //We multiply by -i, because we did NOT account for that in the kron routines
       time_dep_scalar = -PETSC_i*qsys->time_dep[i].a*qsys->time_dep[i].time_dep_func(t,qsys->time_dep[i].ctx);
     } else {
       time_dep_scalar = qsys->time_dep[i].a*qsys->time_dep[i].time_dep_func(t,qsys->time_dep[i].ctx);
     }
     //    PetscPrintf(PETSC_COMM_WORLD,"here i= %d %e %e\n",i,PetscAbsScalar(time_dep_scalar),PetscAbsScalar(qsys->min_time_dep));
     //Only add the term if it is above a threshold
     if(PetscAbsScalar(time_dep_scalar)>PetscAbsScalar(qsys->min_time_dep)){
       MatMult(qsys->time_dep[i].mat_A,X,qsys->work_vec); // work = A * X
       VecAXPY(Y,time_dep_scalar,qsys->work_vec); //dm = dm + a * work
     }
   }

   PetscLogEventEnd(_RHS_time_dep_event,0,0,0,0);

   PetscFunctionReturn(0);

 }
/*
 * set_init_excited_op sets the initial excitation level for a single operator
 * Inputs:
 *       operator op1
 *       PetscInt initial_exc
 * Return:
 *       none
 */
void set_init_excited_op(operator op,PetscInt initial_exc){

  if (initial_exc>=op->my_levels&&op->my_op_type!=VEC){
    PetscPrintf(PETSC_COMM_WORLD,"ERROR! The initial excitation level must be less than the number of levels and can't be used for VEC_OP!\n");
    exit(0);
  }

  op->initial_exc = initial_exc;

  return;
}
