#include "quantum_circuits.h"
#include "quantum_gates.h"
#include "quac_p.h"
#include <stdlib.h>
#include <stdio.h>
#include <petsc.h>
#include <stdarg.h>


//FIXME: This should maybe be accessible by a user to register a new gate?

void (*_get_val_j_functions_gates_sys[MAX_GATES])(qsystem,PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);
/* EventFunction is one step in Petsc to apply some action at a specific time.
 * This function checks to see if an event has happened.
 */
PetscErrorCode _sys_QC_EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx) {
  qsystem sys = (qsystem) ctx;
  /* Check if the time has passed a gate */
  PetscInt current_gate,num_gates;
  PetscLogEventBegin(_qc_event_function_event,0,0,0,0);
  //FIXME: Circuits at time 0 do not work
  if (sys->current_circuit<sys->num_circuits) {
    current_gate = sys->circuit_list[sys->current_circuit].current_gate;
    num_gates    = sys->circuit_list[sys->current_circuit].num_gates;
    if (current_gate<num_gates) {
      /* We signal that we passed the time by returning a negative number */
      fvalue[0] = sys->circuit_list[sys->current_circuit].gate_list[current_gate].time
        +sys->circuit_list[sys->current_circuit].start_time - t;
    } else {
      if (nid==0){
        printf("ERROR! current_gate should never be larger than num_gates in _QC_EventFunction\n");
        exit(0);
      }
    }
  } else {
    fvalue[0] = t;
  }
  PetscLogEventEnd(_qc_event_function_event,0,0,0,0);
  return(0);
}

/* PostEventFunction is the other step in Petsc. If an event has happend, petsc will call this function
 * to apply that event.
*/
PetscErrorCode _sys_QC_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],
                                     PetscReal t,Vec U,PetscBool forward,void* ctx) {
  qsystem sys = (qsystem) ctx;
  PetscInt current_gate,num_gates;
  PetscReal gate_time;
   /* We only have one event at the moment, so we do not need to branch.
    * If we had more than one event, we would put some logic here.
    */

  PetscLogEventBegin(_qc_postevent_function_event,0,0,0,0);
  if (nevents) {
    num_gates    = sys->circuit_list[sys->current_circuit].num_gates;
    current_gate = sys->circuit_list[sys->current_circuit].current_gate;
    gate_time = sys->circuit_list[sys->current_circuit].gate_list[current_gate].time;
    /* Apply all gates at a given time incrementally  */
    while (current_gate<num_gates && sys->circuit_list[sys->current_circuit].gate_list[current_gate].time == gate_time){
      /* apply the current gate */
      _apply_gate_sys(sys,sys->circuit_list[sys->current_circuit].gate_list[current_gate],U);

      /* Increment our gate counter */
      sys->circuit_list[sys->current_circuit].current_gate = sys->circuit_list[sys->current_circuit].current_gate + 1;
      current_gate = sys->circuit_list[sys->current_circuit].current_gate;
    }
    if(sys->circuit_list[sys->current_circuit].current_gate>=sys->circuit_list[sys->current_circuit].num_gates){
      /* We've exhausted this circuit; move on to the next. */
      sys->current_circuit = sys->current_circuit + 1;
    }

  }

  TSSetSolution(ts,U);
  PetscLogEventEnd(_qc_postevent_function_event,0,0,0,0);
  return(0);
}


/*
 * Add a gate to a circuit.
 * Inputs:
 *        circuit circ: circuit to add to
 *        PetscReal time: time that gate would be applied, counting from 0 at
 *                        the start of the circuit
 *        gate_type my_gate_type: which gate to add
 *        ...:   list of qubit gate will act on, other (U for controlled_U?)
 */
void add_gate_to_circuit_sys(circuit *circ,PetscReal time,gate_type my_gate_type,...){
  PetscReal theta,phi,lambda;
  int num_qubits=0,qubit,i;
  va_list ap;

  if (_gate_array_initialized==0){
    //Initialize the array of gate function pointers
    _initialize_gate_function_array_sys();
    _gate_array_initialized = 1;
  }

  _check_gate_type(my_gate_type,&num_qubits);

  if ((*circ).num_gates==(*circ).gate_list_size){
    //FIXME: Put reallocation of array here?
    if (nid==0){
      printf("ERROR! Gate list not large enough!\n");
      exit(1);
    }
  }
  // Store arguments in list
  (*circ).gate_list[(*circ).num_gates].qubit_numbers = malloc(num_qubits*sizeof(int));
  (*circ).gate_list[(*circ).num_gates].time = time;
  (*circ).gate_list[(*circ).num_gates].my_gate_type = my_gate_type;
  (*circ).gate_list[(*circ).num_gates]._get_val_j_from_global_i_sys = _get_val_j_functions_gates_sys[my_gate_type+_min_gate_enum];

  if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ) {
    va_start(ap,num_qubits+1); //FIXME This +1 is probably useless?
  } else if (my_gate_type==U3){
    va_start(ap,num_qubits+3);
  } else {
    va_start(ap,num_qubits);
  }

  // Loop through and store qubits
  for (i=0;i<num_qubits;i++){
    qubit = va_arg(ap,int);
    if (qubit>=num_subsystems) {
      if (nid==0){
        // Disable warning because qasm parser will make the circuit before
        // the qubits are allocated
        // In rewrite, this check is not needed, since circuits are independent
        // of systems.
        //printf("Warning! Qubit number greater than total systems\n");
      }
    }
    (*circ).gate_list[(*circ).num_gates].qubit_numbers[i] = qubit;
  }
  if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ){
    //Get the theta parameter from the last argument passed in
    theta = va_arg(ap,PetscReal);
    (*circ).gate_list[(*circ).num_gates].theta = theta;
    (*circ).gate_list[(*circ).num_gates].phi = 0;
    (*circ).gate_list[(*circ).num_gates].lambda = 0;
  } else if (my_gate_type==U3){
    theta = va_arg(ap,PetscReal);
    (*circ).gate_list[(*circ).num_gates].theta = theta;
    phi = va_arg(ap,PetscReal);
    (*circ).gate_list[(*circ).num_gates].phi = phi;
    lambda = va_arg(ap,PetscReal);
    (*circ).gate_list[(*circ).num_gates].lambda = lambda;
  } else {
    //Set theta to 0
    (*circ).gate_list[(*circ).num_gates].theta = 0;
    (*circ).gate_list[(*circ).num_gates].phi = 0;
    (*circ).gate_list[(*circ).num_gates].lambda = 0;
  }

  (*circ).num_gates = (*circ).num_gates + 1;
  return;
}

void combine_circuit_to_mat_sys(qsystem sys,Mat *matrix_out,circuit circ){
  PetscScalar op_vals[2];
  PetscInt Istart,Iend,i_mat;
  PetscInt i,these_js[2],num_js;
  Mat tmp_mat1,tmp_mat2,tmp_mat3;

  // Should this inherit its stucture from full_A?

  MatCreate(PETSC_COMM_WORLD,&tmp_mat1);
  MatSetType(tmp_mat1,MATMPIAIJ);
  MatSetSizes(tmp_mat1,PETSC_DECIDE,PETSC_DECIDE,sys->total_levels,sys->total_levels);
  MatSetFromOptions(tmp_mat1);

  MatMPIAIJSetPreallocation(tmp_mat1,2,NULL,2,NULL);

  /* Construct the first matrix in tmp_mat1 */
  MatGetOwnershipRange(tmp_mat1,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    circ.gate_list[0]._get_val_j_from_global_i_sys(sys,i,circ.gate_list[0],&num_js,these_js,op_vals,-1); // Get the corresponding j and val
    MatSetValues(tmp_mat1,1,&i,num_js,these_js,op_vals,ADD_VALUES);
  }

  MatAssemblyBegin(tmp_mat1,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tmp_mat1,MAT_FINAL_ASSEMBLY);

  for (i_mat=1;i_mat<circ.num_gates;i_mat++){
    // Create the next matrix
    MatCreate(PETSC_COMM_WORLD,&tmp_mat2);
    MatSetType(tmp_mat2,MATMPIAIJ);
    MatSetSizes(tmp_mat2,PETSC_DECIDE,PETSC_DECIDE,sys->total_levels,sys->total_levels);
    MatSetFromOptions(tmp_mat2);

    MatMPIAIJSetPreallocation(tmp_mat2,2,NULL,2,NULL);

    MatGetOwnershipRange(tmp_mat2,&Istart,&Iend);
    for (i=Istart;i<Iend;i++){
      circ.gate_list[i_mat]._get_val_j_from_global_i_sys(sys,i,circ.gate_list[i_mat],&num_js,these_js,op_vals,-1); // Get the corresponding j and val
      MatSetValues(tmp_mat2,1,&i,num_js,these_js,op_vals,ADD_VALUES);
    }
    MatAssemblyBegin(tmp_mat2,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tmp_mat2,MAT_FINAL_ASSEMBLY);

    // Now do matrix matrix multiply
    MatMatMult(tmp_mat2,tmp_mat1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tmp_mat3);
    MatDestroy(&tmp_mat1);
    MatDestroy(&tmp_mat2); //Do I need to destroy it?

    //Store tmp_mat3 into tmp_mat1
    MatConvert(tmp_mat3,MATSAME,MAT_INITIAL_MATRIX,&tmp_mat1);
    MatDestroy(&tmp_mat3);
  }

  //Copy tmp_mat1 into *matrix_out
  MatConvert(tmp_mat1,MATSAME,MAT_INITIAL_MATRIX,matrix_out);

  MatDestroy(&tmp_mat1);
  return;
}


/* register a circuit to be run a specific time during the time stepping */
void apply_circuit_to_sys(qsystem sys,circuit *circ,PetscReal time){
  //FIXME: Circuits must be time ordered and cannot overlap
  (*circ).start_time = time;
  if (sys->num_circuits==-1){
    //First circuit, need to allocate memory, set up variables

    //Initial list of 10 circuits - can be reallocated to be bigger if needed
    sys->circuit_list_size = 10;
    sys->circuit_list = malloc(sys->circuit_list_size*sizeof(circuit));
    sys->num_circuits = 0;
    sys->current_circuit = 0;
  }

  if (sys->num_circuits==sys->circuit_list_size){
    //FIXME: Put reallocation of array here
    if (nid==0){
      printf("ERROR! Circuit list not large enough!\n");
      exit(1);
    }
  }
  sys->circuit_list[sys->num_circuits] = *circ;
  sys->num_circuits = sys->num_circuits + 1;

  return;
}

//FIXME: Move below to quantum_gates.c when rewrite is stable
/* Apply a specific gate */
void _apply_gate_sys(qsystem sys,struct quantum_gate_struct this_gate,Vec rho){
  PetscScalar *op_vals;
  //FIXME Consider having only one static Mat stores in sys for all gates, rather than creating new ones every time
  Mat gate_mat;
  Vec tmp_answer;
  PetscInt i,num_js,*these_js,Istart,Iend;
  // FIXME: maybe total_levels*2 is too much or not enough? Consider having a better bound.

  op_vals  = malloc(sys->total_levels*2*sizeof(PetscScalar)); //Up to two elements per row in the DM case
  these_js = malloc(sys->total_levels*2*sizeof(PetscInt)); //Up to two elements per row in the DM case
  PetscLogEventBegin(_apply_gate_event,0,0,0,0);

  VecDuplicate(rho,&tmp_answer); //Create a new vec with the same size as rho

  //FIXME: Do this allocation once!
  MatCreate(PETSC_COMM_WORLD,&gate_mat);
  MatSetSizes(gate_mat,PETSC_DECIDE,PETSC_DECIDE,sys->dim,sys->dim);
  MatSetFromOptions(gate_mat);
  MatMPIAIJSetPreallocation(gate_mat,4,NULL,4,NULL); //This matrix is incredibly sparse!
  MatSetUp(gate_mat);
  /* Construct the gate matrix, on the fly */
  MatGetOwnershipRange(gate_mat,&Istart,&Iend); //Could be different Istart and Iend than Hamiltonian mat
  for (i=Istart;i<Iend;i++){
    if (sys->dm_equations){
      // Get the corresponding j and val for the superoperator U* cross U
      this_gate._get_val_j_from_global_i_sys(sys,i,this_gate,&num_js,these_js,op_vals,0);
    } else {
      // Get the corresponding j and val for just the matrix U
      this_gate._get_val_j_from_global_i_sys(sys,i,this_gate,&num_js,these_js,op_vals,-1);
    }
    MatSetValues(gate_mat,1,&i,num_js,these_js,op_vals,ADD_VALUES);
  }

  MatAssemblyBegin(gate_mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(gate_mat,MAT_FINAL_ASSEMBLY);
  /* MatView(gate_mat,PETSC_VIEWER_STDOUT_SELF); */
  MatMult(gate_mat,rho,tmp_answer);
  VecCopy(tmp_answer,rho); //Copy our tmp_answer array into rho

  VecDestroy(&tmp_answer); //Destroy the temp answer
  MatDestroy(&gate_mat);
  free(op_vals);
  free(these_js);
  PetscLogEventEnd(_apply_gate_event,0,0,0,0);
}



void _change_basis_ij_pair_sys(qsystem sys,PetscInt *i_op,PetscInt *j_op,PetscInt system1,PetscInt system2){
  PetscInt na1,na2,lev1,lev2;

  /*
   * To apply our change of basis we use the neat trick that the row number
   * in a given basis can be calculated similar to how a binary number is
   * calculated (but generalized in that some bits can have more than two
   * states. e.g. with three qubits
   *  i(| 0 1 0 >) = 0*4 + 1*2 + 0*1 = 2
   * where i() is the index, in this ordering, of the ket.
   * Another example, with 1 2level, 1 3levels, and 1 4 level system:
   *  i(| 0 1 0 >) = 0*12 + 1*4 + 0*1 = 4
   * that is,
   *  i(| a b c >) = a*n_af^a + b*n_af^b + c*n_af^c
   * where n_af^a is the Hilbert space before system a, etc.
   *
   * Given a specific i, and only switching two systems,
   * we can calculate i's partner in the switched basis
   * by subtracting off the part from the current basis and
   * adding in the part from the desired basis. This leaves everything
   * else the same, but switches the two systems of interest.
   *
   * We need to be able to go from i to a specific subsystem's state.
   * This is accomplished with the formula:
   * (i/n_a % l)
   * Take our example above:
   * three qubits:  2 -> 2/4 % 2 = 0$2 = 0
   *                2 -> 2/2 % 2 = 1%2 = 1
   *                2 -> 2/1 % 2 = 2%2 = 0
   * Or, the other example: 4 -> 4/12 % 2 = 0
   *                        4 -> 4/4 % 3  = 1
   *                        4 -> 4/1 % 4  = 0
   * Note that this depends on integer division - 4/12 = 0
   *
   * Using this, we can precisely calculate a system's part of the sum,
   * subtract that off, and then add the new basis.
   *
   * For example, let's switch our qubits from before around:
   * i(| 1 0 0 >) = 1*4 + 0*2 + 0*1 = 4
   * Now, switch back to the original basis. Note we swapped q3 and q2
   * first, subtract off the contributions from q3 and q2
   * i_new = 4 - 1*4 - 0*2 = 0
   * Now, add on the contributions in the original basis
   * i_new = 0 + 0*4 + 1*2 = 2
   * Algorithmically,
   * i_new = i - (i/na1)%lev1 * na1 - (i/na2)%lev2 * na2
   *           + (i/na2)%lev1 * na1 + (i/na1)%lev2 * na2
   * Note that we use our formula above to calculate the qubits
   * state in this basis, given this specific i.
   */


  lev1  = sys->subsystem_list[system1]->my_levels;
  na1   = sys->total_levels/(lev1*sys->subsystem_list[system1]->n_before);

  lev2  = sys->subsystem_list[system2]->my_levels;
  na2   = sys->total_levels/(lev2*sys->subsystem_list[system2]->n_before); // Changed from lev1->lev2

  *i_op = *i_op - ((*i_op/na1)%lev1)*na1 - ((*i_op/na2)%lev2)*na2 +
    ((*i_op/na1)%lev2)*na2 + ((*i_op/na2)%lev1)*na1;

  *j_op = *j_op - ((*j_op/na1)%lev1)*na1 - ((*j_op/na2)%lev2)*na2 +
    ((*j_op/na1)%lev2)*na2 + ((*j_op/na2)%lev1)*na1;

  return;
}

/*
 * No issue for js_i* = -1 because every row is guaranteed to have a 0
 * in all the gates implemented below.
 * See commit: 9956c78171fdac1fa0ef9e2f0a39cbffd4d755dc where this was an issue
 * in _get_val_j_from_global_i for raising / lowering / number operators, where
 * -1 was used to say there was no nonzero in that row.
 */


void CNOT_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                  PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1=0,num_js_i2=0,js_i1[2],js_i2[2];
  PetscInt control,i_tmp,my_levels,j_sub,moved_system,j1;
  PetscScalar vals_i1[2],vals_i2[2];

  /* The controlled NOT gate has two inputs, a target and a control.
   * the target output is equal to the target input if the control is
   * |0> and is flipped if the control input is |1> (Marinescu 146)
   * As a matrix, for a two qubit system:
   *     1 0 0 0        I2 0
   *     0 1 0 0   =    0  sig_x
   *     0 0 0 1
   *     0 0 1 0
   * Of course, when there are other qubits, tensor products and such
   * must be applied to get the full basis representation.
   */

  if (tensor_control!= 0) {

    /* 4 is hardcoded because 2 qubits with 2 levels each */
    my_levels   = 4;
    // Get the correct hilbert space information
    i_tmp = i;
    _get_n_after_2qbit_sys(sys,&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);

    *num_js = 1;
    if (i_sub==0){
      // Same, regardless of control
      // Diagonal
      vals[0] = 1.0;
      /*
       * We shouldn't need to deal with any permutation here;
       * i_sub is in the permuted basis, but we know that a
       * diagonal element is diagonal in all bases, so
       * we just use the computational basis value.
       p         */
      js[0]  = i;

    } else if (i_sub==1){
      // Check which is the control bit
      vals[0] = 1.0;
      if (control==0){
        // Diagonal
        js[0]   = i;
      } else {
        // Off diagonal
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 3;
        j1   = (j_sub) * n_after + k1 + k2*my_levels*n_after; // 3 = j_sub

        /* Permute back to computational basis */
        _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
        js[0] = j1;
      }

    } else if (i_sub==2){
      vals[0] = 1.0;
      if (control==0){
        // Off diagonal
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 3;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;
        /* Permute back to computational basis */
        _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
        js[0] = j1;
      } else {
        // Diagonal
        js[0]   = i;
      }
    } else if (i_sub==3){
      vals[0]   = 1.0;
      if (control==0){
        // Off diagonal element
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 2;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;
      } else {
        // Off diagonal element
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 1;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;
      }
      /* Permute back to computational basis */
      _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
      js[0] = j1;
    } else {
      if (nid==0){
        printf("ERROR! CNOT gate is only defined for 2 qubits!\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CNOT_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CNOT_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void CXZ_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                  PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1=0,num_js_i2=0,js_i1[2],js_i2[2];
  PetscInt control,i_tmp,my_levels,j_sub,moved_system,j1;
  PetscScalar vals_i1[2],vals_i2[2];

  /* The controlled-XZ gate has two inputs, a target and a control.
   * As a matrix, for a two qubit system
   *     1 0 0 0        I2 0
   *     0 1 0 0   =    0  sig_x * sig_z
   *     0 0 0 -1
   *     0 0 1 0
   * Of course, when there are other qubits, tensor products and such
   * must be applied to get the full basis representation.
   *
   * Note that this is a temporary gate; i.e., we will create a more
   * general controlled-U gate at a later time that will replace this.
   */

  if (tensor_control!= 0) {

    /* 4 is hardcoded because 2 qubits with 2 levels each */
    my_levels   = 4;
    // Get the correct hilbert space information
    i_tmp = i;
    _get_n_after_2qbit_sys(sys,&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);
    *num_js = 1;
    if (i_sub==0){
      // Same, regardless of control
      // Diagonal
      vals[0] = 1.0;
      /*
       * We shouldn't need to deal with any permutation here;
       * i_sub is in the permuted basis, but we know that a
       * diagonal element is diagonal in all bases, so
       * we just use the computational basis value.
       p         */
      js[0]  = i;

    } else if (i_sub==1){
      // Check which is the control bit
      if (control==0){
        // Diagonal
        vals[0] = 1.0;
        js[0]   = i;
      } else {
        // Off diagonal
        vals[0] = -1.0;
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 3;
        j1   = (j_sub) * n_after + k1 + k2*my_levels*n_after; // 3 = j_sub

        /* Permute back to computational basis */
        _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
        js[0] = j1;
      }

    } else if (i_sub==2){
      if (control==0){
        vals[0] = -1.0;
        // Off diagonal
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 3;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;

        /* Permute back to computational basis */
        _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
        js[0] = j1;
      } else {
        // Diagonal
        vals[0] = 1.0;
        js[0]   = i;
      }
    } else if (i_sub==3){
      vals[0]   = 1.0;
      if (control==0){
        // Off diagonal element
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 2;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;
      } else {
        // Off diagonal element
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 1;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;
      }
      /* Permute back to computational basis */
      _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
      js[0] = j1;
    } else {
      if (nid==0){
        printf("ERROR! CXZ gate is only defined for 2 qubits!\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CXZ_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CXZ_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}

void CZ_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                  PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,i1,i2,num_js_i1=0,num_js_i2=0,js_i1[2],js_i2[2];
  PetscInt control,i_tmp,my_levels,moved_system;
  PetscScalar vals_i1[2],vals_i2[2];

  /* The controlled-Z gate has two inputs, a target and a control.
   * As a matrix, for a two qubit system
   *     1 0 0 0        I2 0
   *     0 1 0 0   =    0  sig_z
   *     0 0 1 0
   *     0 0 0 -1
   * Of course, when there are other qubits, tensor products and such
   * must be applied to get the full basis representation.
   *
   * Note that this is a temporary gate; i.e., we will create a more
   * general controlled-U gate at a later time that will replace
   *
   * Controlled-z is the same for both possible controls
   */

  if (tensor_control!= 0) {

    /* 4 is hardcoded because 2 qubits with 2 levels each */
    my_levels   = 4;
    // Get the correct hilbert space information
    i_tmp = i;
    _get_n_after_2qbit_sys(sys,&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);
    *num_js = 1;
    if (i_sub==0){
      // Same, regardless of control
      // Diagonal
      vals[0] = 1.0;
      /*
       * We shouldn't need to deal with any permutation here;
       * i_sub is in the permuted basis, but we know that a
       * diagonal element is diagonal in all bases, so
       * we just use the computational basis value.
       p         */
      js[0]  = i;

    } else if (i_sub==1){
      // Diagonal
      vals[0] = 1.0;
      js[0]   = i;
    } else if (i_sub==2){
      // Diagonal
      vals[0] = 1.0;
      js[0]   = i;
    } else if (i_sub==3){
      vals[0] = -1.0;
      js[0]   = i;
    } else {
      if (nid==0){
        printf("ERROR! CZ gate is only defined for 2 qubits!\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CZ_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CZ_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void CmZ_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                  PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,i1,i2,num_js_i1=0,num_js_i2=0,js_i1[2],js_i2[2];
  PetscInt control,i_tmp,my_levels,moved_system;
  PetscScalar vals_i1[2],vals_i2[2];

  /* The controlled-mZ gate has two inputs, a target and a control.
   * As a matrix, for a two qubit system
   *     1 0 0 0        I2 0
   *     0 1 0 0   =    0  -sig_z
   *     0 0 -1 0
   *     0 0 0 1
   * Of course, when there are other qubits, tensor products and such
   * must be applied to get the full basis representation.
   *
   * Note that this is a temporary gate; i.e., we will create a more
   * general controlled-U gate at a later time that will replace
   *
   */

  if (tensor_control!= 0) {

    /* 4 is hardcoded because 2 qubits with 2 levels each */
    my_levels   = 4;
    // Get the correct hilbert space information
    i_tmp = i;
    _get_n_after_2qbit_sys(sys,&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);

    *num_js = 1;
    if (i_sub==0){
      // Same, regardless of control
      // Diagonal
      vals[0] = 1.0;
      /*
       * We shouldn't need to deal with any permutation here;
       * i_sub is in the permuted basis, but we know that a
       * diagonal element is diagonal in all bases, so
       * we just use the computational basis value.
       p         */
      js[0]  = i;

    } else if (i_sub==1){
      // Diagonal
      js[0]   = i;
      if (control==0) {
        vals[0] = 1.0;
      } else {
        vals[0] = -1.0;
      }
    } else if (i_sub==2){
      // Diagonal
      js[0]   = i;
      if (control==0) {
        vals[0] = -1.0;
      } else {
        vals[0] = 1.0;
      }
    } else if (i_sub==3){
      vals[0] = 1.0;
      js[0]   = i;
    } else {
      if (nid==0){
        printf("ERROR! CmZ gate is only defined for 2 qubits!\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CmZ_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CmZ_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void CZX_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                  PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1=0,num_js_i2=0,js_i1[2],js_i2[2];
  PetscInt control,i_tmp,my_levels,j_sub,moved_system,j1;
  PetscScalar vals_i1[2],vals_i2[2];

  /* The controlled-ZX gate has two inputs, a target and a control.
   * As a matrix, for a two qubit system
   *     1 0 0 0        I2 0
   *     0 1 0 0   =    0  sig_z * sig_x
   *     0 0 0 1
   *     0 0 -1 0
   * Of course, when there are other qubits, tensor products and such
   * must be applied to get the full basis representation.
   *
   * Note that this is a temporary gate; i.e., we will create a more
   * general controlled-U gate at a later time that will replace this.
   */

  if (tensor_control!= 0) {

    /* 4 is hardcoded because 2 qubits with 2 levels each */
    my_levels   = 4;
    // Get the correct hilbert space information
    i_tmp = i;
    _get_n_after_2qbit_sys(sys,&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);

    *num_js = 1;
    if (i_sub==0){
      // Same, regardless of control
      // Diagonal
      vals[0] = 1.0;
      /*
       * We shouldn't need to deal with any permutation here;
       * i_sub is in the permuted basis, but we know that a
       * diagonal element is diagonal in all bases, so
       * we just use the computational basis value.
       p         */
      js[0]  = i;

    } else if (i_sub==1){
      // Check which is the control bit
      vals[0] = 1.0;
      if (control==0){
        // Diagonal
        js[0]   = i;
      } else {
        // Off diagonal
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 3;
        j1   = (j_sub) * n_after + k1 + k2*my_levels*n_after; // 3 = j_sub

        /* Permute back to computational basis */
        _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
        js[0] = j1;
      }

    } else if (i_sub==2){
      vals[0] = 1.0;
      if (control==0){
        // Off diagonal
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 3;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;

        /* Permute back to computational basis */
        _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
        js[0] = j1;
      } else {
        // Diagonal
        js[0]   = i;
      }
    } else if (i_sub==3){
      vals[0]   = -1.0;
      if (control==0){
        // Off diagonal element
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 2;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;
      } else {
        // Off diagonal element
        tmp_int = i_tmp - i_sub * n_after;
        k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
        k1      = tmp_int%(my_levels*n_after);
        j_sub   = 1;
        j1     = j_sub * n_after + k1 + k2*my_levels*n_after;
      }
      /* Permute back to computational basis */
      _change_basis_ij_pair_sys(sys,&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
      js[0] = j1;
    } else {
      if (nid==0){
        printf("ERROR! CZX gate is only defined for 2 qubits!\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CZX_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CZX_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void HADAMARD_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];

  /*
   * HADAMARD gate
   *
   * 1/sqrt(2) | 1  1 |
   *           | 1 -1 |
   * Hadamard gates have two values per row,
   * with both diagonal anad off diagonal elements
   *
   */
  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 2;
    if (i_sub==0) {
      // Diagonal element
      js[0]   = i;
      vals[0] = pow(2,-0.5);

      // Off diagonal element
      tmp_int = i - 0 * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]   = (0 + 1) * n_after + k1 + k2*my_levels*n_after;
      vals[1] = pow(2,-0.5);

    } else if (i_sub==1){
      // Diagonal element
      js[0]   = i;
      vals[0] = -pow(2,-0.5);

      // Off diagonal element
      tmp_int = i - (0+1) * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]   = 0 * n_after + k1 + k2*my_levels*n_after;
      vals[1] = pow(2,-0.5);

    } else {
      if (nid==0){
        printf("ERROR! Hadamard gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    HADAMARD_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    HADAMARD_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }

  return;
}

void U3_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];
  PetscReal theta,lambda,phi;
  /*
   * u3 gate
   *
   * u3(theta,phi,lambda)  = | cos(theta/2)              -e^(i lambda) * sin(theta/2)    |
   *                         | e^(i phi) sin(theta/2)    e^(i (lambda+phi)) cos(theta/2) |
   * the u3 gate is a general one qubit transformation.
   * the u2 gate is u3(pi/2,phi,lambda)
   * the u1 gate is u3(0,0,lambda)
   * The u3 gate has two elements per row,
   * with both diagonal anad off diagonal elements
   *
   */
  theta = gate.theta;
  phi = gate.phi;
  lambda = gate.lambda;
  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 2;
    if (i_sub==0) {
      // Diagonal element
      js[0]   = i;
      vals[0] = PetscCosReal(theta/2);

      // Off diagonal element
      tmp_int = i - 0 * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]   = (0 + 1) * n_after + k1 + k2*my_levels*n_after;
      vals[1] = -PetscExpComplex(PETSC_i*lambda)*PetscSinReal(theta/2);

    } else if (i_sub==1){
      // Diagonal element
      js[0]   = i;
      vals[0] = PetscExpComplex(PETSC_i*(lambda+phi))*PetscCosReal(theta/2);
      // Off diagonal element
      tmp_int = i - (0+1) * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]   = 0 * n_after + k1 + k2*my_levels*n_after;
      vals[1] = PetscExpComplex(PETSC_i*phi)*PetscSinReal(theta/2);

    } else {
      if (nid==0){
        printf("ERROR! u3 gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (pi1) by calling this function */
    U3_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    U3_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);


    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }

  return;
}

void EYE_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2];
  PetscScalar vals_i1[2],vals_i2[2];

  /*
   * Identity (EYE) gate
   *
   *   | 1   0 |
   *   | 0   1 |
   *
   */

  if (tensor_control!= 0) {
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 1;
    js[0] = i;
    vals[0] = 1.0;
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    EYE_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    EYE_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void SIGMAZ_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];

  /*
   * SIGMAZ gate
   *
   *   | 1   0 |
   *   | 0  -1 |
   *
   */

  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 1;
    if (i_sub==0) {
      // Diagonal element
      js[0] = i;
      vals[0] = 1.0;

    } else if (i_sub==1){
      // Diagonal element
      js[0] = i;
      vals[0] = -1.0;

    } else {
      if (nid==0){
        printf("ERROR! sigmaz gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    SIGMAZ_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    SIGMAZ_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}

void RZ_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];
  PetscReal theta;
  /*
   * RZ gate
   *
   *   | exp(i*theta/2)   0               |
   *   | 0                -exp(i*theta/2) |
   *
   */

  theta = gate.theta;
  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 1;
    if (i_sub==0) {
      // Diagonal element
      js[0] = i;
      vals[0] = PetscExpComplex(PETSC_i*theta/2);

    } else if (i_sub==1){
      // Diagonal element
      js[0] = i;
      vals[0] = PetscExpComplex(-PETSC_i*theta/2);

    } else {
      if (nid==0){
        printf("ERROR! rz gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    RZ_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    RZ_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}

void RY_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];
  PetscReal theta;
  /*
   * RY gate
   *
   *   | cos(theta/2)   sin(theta/2)  |
   *   | -sin(theta/2)  cos(theta/2)  |
   *
   */

  theta = gate.theta;
  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 2;
    if (i_sub==0) {
      // Diagonal element
      js[0] = i;
      vals[0] = PetscCosReal(theta/2.0);
      // Off diagonal element
      tmp_int = i - 0 * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]     = (0 + 1) * n_after + k1 + k2*my_levels*n_after;
      vals[1]   = PetscSinReal(theta/2);


    } else if (i_sub==1){
      // Diagonal element
      js[0] = i;
      vals[0] = PetscCosReal(theta/2);

      // Off diagonal element
      tmp_int = i - (0+1) * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]   = 0 * n_after + k1 + k2*my_levels*n_after;
      vals[1]   = -PetscSinReal(theta/2);


    } else {
      if (nid==0){
        printf("ERROR! rz gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    RY_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    RY_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void RX_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];
  PetscReal theta;
  /*
   * RX gate
   *
   *   | cos(theta/2)    i*sin(theta/2) |
   *   | i*sin(theta/2)  cos(theta/2)   |
   *
   */

  theta = gate.theta;
  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 2;
    if (i_sub==0) {
      // Diagonal element
      js[0] = i;
      vals[0] = PetscCosReal(theta/2);

      // Off diagonal element
      tmp_int = i - 0 * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]     = (0 + 1) * n_after + k1 + k2*my_levels*n_after;
      vals[1]   = PETSC_i * PetscSinReal(theta/2);


    } else if (i_sub==1){
      // Diagonal element
      js[0] = i;
      vals[0] = PetscCosReal(theta/2);

      // Off diagonal element
      tmp_int = i - (0+1) * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[1]   = 0 * n_after + k1 + k2*my_levels*n_after;
      vals[1]   = PETSC_i * PetscSinReal(theta/2);


    } else {
      if (nid==0){
        printf("ERROR! rz gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    RX_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    RX_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void SIGMAY_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];

  /*
   * SIGMAY gate
   *
   *   | 0  -1.j |
   *   | 1.j  0 |
   *
   */
  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 1;
    if (i_sub==0) {

      // Off diagonal element
      tmp_int = i - 0 * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[0]   = (0 + 1) * n_after + k1 + k2*my_levels*n_after;
      vals[0] = -1.0*PETSC_i;

    } else if (i_sub==1){

      // Off diagonal element
      tmp_int = i - (0+1) * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[0]   = 0 * n_after + k1 + k2*my_levels*n_after;
      vals[0] = 1.0*PETSC_i;

    } else {
      if (nid==0){
        printf("ERROR! sigmay gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    SIGMAY_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    SIGMAY_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void SIGMAX_get_val_j_from_global_i_sys(qsystem sys,PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
  PetscScalar vals_i1[2],vals_i2[2];

  /*
   * SIGMAX gate
   *
   *   | 0  1 |
   *   | 1  0 |
   *
   */
  if (tensor_control!= 0) {
    my_levels = 2; //Hardcoded becase single qubit gate
    _get_n_after_1qbit_sys(sys,i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
    *num_js = 1;
    if (i_sub==0) {

      // Off diagonal element
      tmp_int = i - 0 * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[0]     = (0 + 1) * n_after + k1 + k2*my_levels*n_after;
      vals[0]   = 1.0;

    } else if (i_sub==1){

      // Off diagonal element
      tmp_int = i - (0+1) * n_after;
      k2      = tmp_int/(my_levels*n_after);//Use integer arithmetic to get floor function
      k1      = tmp_int%(my_levels*n_after);
      js[0]   = 0 * n_after + k1 + k2*my_levels*n_after;
      vals[0] = 1.0;

    } else {
      if (nid==0){
        printf("ERROR! sigmax gate is only defined for qubits\n");
        exit(0);
      }
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
    i1 = i/sys->total_levels;
    i2 = i%sys->total_levels;

    /* Now, get js for U* (i1) by calling this function */
    SIGMAX_get_val_j_from_global_i_sys(sys,i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    SIGMAX_get_val_j_from_global_i_sys(sys,i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = sys->total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}

void _get_n_after_2qbit_sys(qsystem sys,PetscInt *i,int qubit_numbers[],PetscInt tensor_control,PetscInt *n_after, PetscInt *control, PetscInt *moved_system, PetscInt *i_sub){
  operator this_op1,this_op2;
  PetscInt n_before1,n_before2,extra_after,my_levels=4,j1; //4 is hardcoded because 2 qbits
  if (tensor_control==1) {
    extra_after = sys->total_levels;
  } else {
    extra_after = 1;
  }

  //Two qubit gates
  this_op1 = sys->subsystem_list[qubit_numbers[0]];
  this_op2 = sys->subsystem_list[qubit_numbers[1]];
  if (this_op1->my_levels * this_op2->my_levels != 4) {
    //Check that it is a two level system
    if (nid==0){
      printf("ERROR! Two qubit gates can only affect two 2-level systems (global_i)\n");
      exit(0);
    }
  }

  n_before1  = this_op1->n_before;
  n_before2  = this_op2->n_before;

  *control = 0;
  *moved_system = qubit_numbers[1];

  /* 2 is hardcoded because CNOT gates are for qubits, which have 2 levels */
  /* 4 is hardcoded because 2 qubits with 2 levels each */
  *n_after   = sys->total_levels/(4*n_before1)*extra_after;

  /*
   * Check which is the control and which is the target,
   * flip if need be.
   */
  if (n_before2<n_before1) {
    *n_after   = sys->total_levels/(4*n_before2);
    *control   = 1;
    *moved_system = qubit_numbers[0];
    n_before1 = n_before2;
  }
  /*
   * Permute to temporary basis
   * Get the i_sub in the permuted basis
   */
  _change_basis_ij_pair_sys(sys,i,&j1,qubit_numbers[*control]+1,*moved_system); // j1 useless here

  *i_sub = *i/(*n_after)%my_levels; //Use integer arithmetic to get floor function

  return;
}

void _get_n_after_1qbit_sys(qsystem sys,PetscInt i,int qubit_number,PetscInt tensor_control,PetscInt *n_after,PetscInt *i_sub){
  operator this_op1;
  PetscInt extra_after;
  if (tensor_control==1) {
    extra_after = sys->total_levels;
  } else {
    extra_after = 1;
  }

  //Get the system this is affecting
  this_op1 = sys->subsystem_list[qubit_number];
  if (this_op1->my_levels!=2) {
    //Check that it is a two level system
    if (nid==0){
      printf("ERROR! Single qubit gates can only affect 2-level systems\n");
      exit(0);
    }
  }
  *n_after = sys->total_levels/(this_op1->my_levels*this_op1->n_before)*extra_after;
  *i_sub = i/(*n_after)%this_op1->my_levels; //Use integer arithmetic to get floor function

  return;
}

/*
 * Put the gate function pointers into an array
 */
void _initialize_gate_function_array_sys(){
  _get_val_j_functions_gates_sys[CZX+_min_gate_enum] = CZX_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[CmZ+_min_gate_enum] = CmZ_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[CZ+_min_gate_enum] = CZ_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[CXZ+_min_gate_enum] = CXZ_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[CNOT+_min_gate_enum] = CNOT_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[HADAMARD+_min_gate_enum] = HADAMARD_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[SIGMAX+_min_gate_enum] = SIGMAX_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[SIGMAY+_min_gate_enum] = SIGMAY_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[SIGMAZ+_min_gate_enum] = SIGMAZ_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[EYE+_min_gate_enum] = EYE_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[RX+_min_gate_enum] = RX_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[RY+_min_gate_enum] = RY_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[RZ+_min_gate_enum] = RZ_get_val_j_from_global_i_sys;
  _get_val_j_functions_gates_sys[U3+_min_gate_enum] = U3_get_val_j_from_global_i_sys;
}
