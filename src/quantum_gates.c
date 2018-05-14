#include "quantum_gates.h"
#include "quac_p.h"
#include <stdlib.h>
#include <stdio.h>
#include <petsc.h>
#include <stdarg.h>


int _num_quantum_gates = 0;
int _current_gate = 0;
struct quantum_gate_struct _quantum_gate_list[MAX_GATES];
int _min_gate_enum = 5; // Minimum gate enumeration number
int _gate_array_initialized = 0;
int _num_circuits    = 0;
int _current_circuit = 0;
circuit _circuit_list[MAX_GATES];
void (*_get_val_j_functions_gates[MAX_GATES])(PetscInt,struct quantum_gate_struct,PetscInt*,PetscInt[],PetscScalar[],PetscInt);

/* EventFunction is one step in Petsc to apply some action at a specific time.
 * This function checks to see if an event has happened.
 */
PetscErrorCode _QG_EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx) {
  /* Check if the time has passed a gate */

  if (_current_gate<_num_quantum_gates) {
    /* We signal that we passed the time by returning a negative number */
    fvalue[0] = _quantum_gate_list[_current_gate].time - t;
  } else {
    fvalue[0] = t;
  }

  return(0);
}

/* PostEventFunction is the other step in Petsc. If an event has happend, petsc will call this function
 * to apply that event.
*/
PetscErrorCode _QG_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,
                                     Vec U,PetscBool forward,void* ctx) {

   /* We only have one event at the moment, so we do not need to branch.
    * If we had more than one event, we would put some logic here.
    */
  if (nevents) {
    /* Apply the current gate */
    //Deprecated?
    /* _apply_gate(_quantum_gate_list[_current_gate].my_gate_type,_quantum_gate_list[_current_gate].qubit_numbers,U); */
    /* Increment our gate counter */
    _current_gate = _current_gate + 1;
  }

  TSSetSolution(ts,U);
  return(0);
}

/* EventFunction is one step in Petsc to apply some action at a specific time.
 * This function checks to see if an event has happened.
 */
PetscErrorCode _QC_EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx) {
  /* Check if the time has passed a gate */
  PetscInt current_gate,num_gates;
  PetscLogEventBegin(_qc_event_function_event,0,0,0,0);
  if (_current_circuit<_num_circuits) {
    current_gate = _circuit_list[_current_circuit].current_gate;
    num_gates    = _circuit_list[_current_circuit].num_gates;
    if (current_gate<num_gates) {
      /* We signal that we passed the time by returning a negative number */
      fvalue[0] = _circuit_list[_current_circuit].gate_list[current_gate].time
        +_circuit_list[_current_circuit].start_time - t;
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
PetscErrorCode _QC_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],
                                     PetscReal t,Vec U,PetscBool forward,void* ctx) {
  PetscInt current_gate,num_gates;
  PetscReal gate_time;
   /* We only have one event at the moment, so we do not need to branch.
    * If we had more than one event, we would put some logic here.
    */

  PetscLogEventBegin(_qc_postevent_function_event,0,0,0,0);

  if (nevents) {
    num_gates    = _circuit_list[_current_circuit].num_gates;
    current_gate = _circuit_list[_current_circuit].current_gate;
    gate_time = _circuit_list[_current_circuit].gate_list[current_gate].time;
    /* Apply all gates at a given time incrementally  */
    while (current_gate<num_gates && _circuit_list[_current_circuit].gate_list[current_gate].time == gate_time){
      /* apply the current gate */
      _apply_gate(_circuit_list[_current_circuit].gate_list[current_gate],U);

      /* Increment our gate counter */
      _circuit_list[_current_circuit].current_gate = _circuit_list[_current_circuit].current_gate + 1;
      current_gate = _circuit_list[_current_circuit].current_gate;
    }
    if(_circuit_list[_current_circuit].current_gate>=_circuit_list[_current_circuit].num_gates){
      /* We've exhausted this circuit; move on to the next. */
      _current_circuit = _current_circuit + 1;
    }

  }

  TSSetSolution(ts,U);
  PetscLogEventEnd(_qc_postevent_function_event,0,0,0,0);
  return(0);
}

/* Add a gate to the list */
void add_gate(PetscReal time,gate_type my_gate_type,...) {
  int num_qubits=0,qubit,i;
  va_list ap;

  if (my_gate_type==HADAMARD) {
    num_qubits = 1;
  } else if (my_gate_type==CNOT){
    num_qubits = 2;
  } else {
    if (nid==0){
      printf("ERROR! Gate type not recognized!\n");
      exit(0);
    }
  }

  // Store arguments in list
  _quantum_gate_list[_num_quantum_gates].qubit_numbers = malloc(num_qubits*sizeof(int));
  _quantum_gate_list[_num_quantum_gates].time = time;
  _quantum_gate_list[_num_quantum_gates].my_gate_type = my_gate_type;
  _quantum_gate_list[_num_quantum_gates]._get_val_j_from_global_i = HADAMARD_get_val_j_from_global_i;

  // Loop through and store qubits
  for (i=0;i<num_qubits;i++){
    qubit = va_arg(ap,int);
    _quantum_gate_list[_num_quantum_gates].qubit_numbers[i] = qubit;
  }

  _num_quantum_gates = _num_quantum_gates + 1;
}


/* Apply a specific gate */
void _apply_gate(struct quantum_gate_struct this_gate,Vec rho){
  PetscScalar op_vals[total_levels*2];
  Mat gate_mat; //FIXME Consider having only one static Mat for all gates, rather than creating new ones every time
  Vec tmp_answer;
  PetscInt dim,i,Istart,Iend,num_js,these_js[total_levels*2];
  // FIXME: maybe total_levels*2 is too much or not enough? Consider having a better bound.

  PetscLogEventBegin(_apply_gate_event,0,0,0,0);

  if (_lindblad_terms){
    dim = total_levels*total_levels;
  } else {
    dim = total_levels;
  }

  VecDuplicate(rho,&tmp_answer); //Create a new vec with the same size as rho

  MatCreate(PETSC_COMM_WORLD,&gate_mat);
  MatSetSizes(gate_mat,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
  MatSetFromOptions(gate_mat);
  MatMPIAIJSetPreallocation(gate_mat,4,NULL,4,NULL); //This matrix is incredibly sparse!
  MatSetUp(gate_mat);
  /* Construct the gate matrix, on the fly */
  MatGetOwnershipRange(gate_mat,&Istart,&Iend);

  for (i=Istart;i<Iend;i++){
    if (_lindblad_terms){
      // Get the corresponding j and val for the superoperator U* cross U
      this_gate._get_val_j_from_global_i(i,this_gate,&num_js,these_js,op_vals,0);
    } else {
      // Get the corresponding j and val for just the matrix U
      this_gate._get_val_j_from_global_i(i,this_gate,&num_js,these_js,op_vals,-1);
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

  PetscLogEventEnd(_apply_gate_event,0,0,0,0);
}

/*z
 * _construct_gate_mat constructs the matrix needed for the quantum
 * computing gates.
 *
 * Inputs:
 *     gate_type my_gate_type  type of quantum gate
 *     int *s
 * Outputs:
 *      Mat gate_mat: the expanded, superoperator matrix for that gate
 */

void _construct_gate_mat(gate_type my_gate_type,int *systems,Mat gate_mat){
  PetscInt i,j,i_mat,j_mat,k1,k2,k3,k4,n_before1,n_before2,my_levels,n_after;
  PetscInt i1,j1,i2=0,j2=0,comb_levels,control=-1,moved_system;
  PetscReal    val1,val2;
  PetscScalar add_to_mat;

  if (my_gate_type == CNOT) {

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


    /* Figure out which system is first in our basis
     * 0 and 1 is hardcoded because CNOT gates have only 2 qubits */
    n_before1  = subsystem_list[systems[0]]->n_before;
    n_before2  = subsystem_list[systems[1]]->n_before;
    control    = 0;
    moved_system = systems[1];
    /* 2 is hardcoded because CNOT gates are for qubits, which have 2 levels */
    /* 4 is hardcoded because 2 qubits with 2 levels each */
    n_after   = total_levels/(4*n_before1);

    /* Check which is the control and which is the target */
    if (n_before2<n_before1) {
      n_after   = total_levels/(4*n_before2);
      control   = 1;
      moved_system = systems[0];
      n_before1 = n_before2;
    }
    /* 4 is hardcoded because 2 qubits with 2 levels each */
    my_levels   = 4;
    for (k1=0;k1<n_after;k1++){
      for (k2=0;k2<n_before1;k2++){
        for (i=0;i<4;i++){ //4 is hardcoded because there are only 4 entries
          val1 = _get_val_in_subspace_gate(i,my_gate_type,control,&i1,&j1);

          /* Get I_b cross CNOT cross I_a in the temporary basis */
          i1 = i1*n_after + k1 + k2*my_levels*n_after;
          j1 = j1*n_after + k1 + k2*my_levels*n_after;

          /* Permute to computational basis
           * To aid in the computation, we generated the matrix in the 'temporary basis',
           * where we exchanged the subsystem immediately after the first qubit with
           * the second qubit of interest. I.e., doing a CNOT(q3,q7)
           * Computational basis: q1 q2 q3 q4 q5 q6 q7 q8 q9 q10
           * Temporary basis:     q1 q2 q3 q7 q5 q6 q4 q8 q9 q10
           * This allows us to calculate CNOT easily - we just need to change
           * the basis back to the computational one.
           *
           * Since the control variable tells us which qubit was the first,
           * we switched the system immediately before that one with the
           * stored variable moved_system
           */
           _change_basis_ij_pair(&i1,&j1,systems[control]+1,moved_system);
          for (k3=0;k3<n_after;k3++){
            for (k4=0;k4<n_before1;k4++){
              for (j=0;j<4;j++){ //4 is hardcoded because there are only 4 entries
                val2 = _get_val_in_subspace_gate(j,my_gate_type,control,&i2,&j2);

                /* Get I_b cross CNOT cross I_a in the temporary basis */
                i2 = i2*n_after + k3 + k4*my_levels*n_after;
                j2 = j2*n_after + k3 + k4*my_levels*n_after;

                /* Permute to computational basis */
                _change_basis_ij_pair(&i2,&j2,systems[control]+1,moved_system);

                add_to_mat = val1*val2;
                /* Do the normal kron product expansion */
                i_mat = my_levels*n_before1*n_after*i1 + i2;
                j_mat = my_levels*n_before1*n_after*j1 + j2;
                MatSetValue(gate_mat,i_mat,j_mat,add_to_mat,ADD_VALUES);
              }
            }
          }
        }
      }
    }
  } else if (my_gate_type == HADAMARD) {

    /*
     * The Hadamard gate is a one qubit gate defined as:
     *
     *    H = 1/sqrt(2) 1  1
     *                  1 -1
     *
     * Find the necessary Hilbert space dimensions for constructing the
     * full space matrix.
     */

    n_before1  = subsystem_list[systems[0]]->n_before;
    my_levels  = subsystem_list[systems[0]]->my_levels; //Should be 2, because qubit
    n_after   = total_levels/(my_levels*n_before1);
    comb_levels = my_levels*my_levels*n_before1*n_after;

    for (k4=0;k4<n_before1*n_after;k4++){
      for (i=0;i<4;i++){ // 4 hardcoded because there are 4 values in the hadamard
        val1 = _get_val_in_subspace_gate(i,my_gate_type,control,&i1,&j1);
        for (j=0;j<4;j++){
          val2 = _get_val_in_subspace_gate(j,my_gate_type,control,&i2,&j2);
          i2 = i2 + k4*my_levels;
          j2 = j2 + k4*my_levels;
          /*
           * We need my_levels*n_before*n_after because we are taking
           * H cross (Ia cross Ib cross H), so the the size of the second operator
           * is my_levels*n_before*n_after
           */
          add_to_mat = val1*val2;
          i_mat = my_levels*n_before1*n_after*i1 + i2;
          j_mat = my_levels*n_before1*n_after*j1 + j2;
          _add_to_PETSc_kron_ij(gate_mat,add_to_mat,i_mat,j_mat,n_before1,n_after,comb_levels);
        }
      }
    }
  } else if (my_gate_type == SIGMAX || my_gate_type == SIGMAY || my_gate_type == SIGMAZ) {

    /*
     * The pauli matrices are two qubit gates, sigmax, sigmay, sigmaz
     */

    n_before1  = subsystem_list[systems[0]]->n_before;
    my_levels  = subsystem_list[systems[0]]->my_levels; //Should be 2, because qubit
    n_after   = total_levels/(my_levels*n_before1);
    comb_levels = my_levels*my_levels*n_before1*n_after;

    for (k4=0;k4<n_before1*n_after;k4++){
      for (i=0;i<2;i++){// 2 hardcoded because there are 4 values in the hadamard
        val1 = _get_val_in_subspace_gate(i,my_gate_type,control,&i1,&j1);
        for (j=0;j<2;j++){
          val2 = _get_val_in_subspace_gate(j,my_gate_type,control,&i2,&j2);
          i2 = i2 + k4*my_levels;
          j2 = j2 + k4*my_levels;
          /*
           * We need my_levels*n_before*n_after because we are taking
           * H cross (Ia cross Ib cross H), so the the size of the second operator
           * is my_levels*n_before*n_after
           */
          add_to_mat = val1*val2;
          i_mat = my_levels*n_before1*n_after*i1 + i2;
          j_mat = my_levels*n_before1*n_after*j1 + j2;
          _add_to_PETSc_kron_ij(gate_mat,add_to_mat,i_mat,j_mat,n_before1,n_after,comb_levels);
        }
      }
    }
  }

  return;
}



void _change_basis_ij_pair(PetscInt *i_op,PetscInt *j_op,PetscInt system1,PetscInt system2){
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


  lev1  = subsystem_list[system1]->my_levels;
  na1   = total_levels/(lev1*subsystem_list[system1]->n_before);

  lev2  = subsystem_list[system2]->my_levels;
  na2   = total_levels/(lev2*subsystem_list[system2]->n_before); // Changed from lev1->lev2

  *i_op = *i_op - ((*i_op/na1)%lev1)*na1 - ((*i_op/na2)%lev2)*na2 +
    ((*i_op/na1)%lev2)*na2 + ((*i_op/na2)%lev1)*na1;

  *j_op = *j_op - ((*j_op/na1)%lev1)*na1 - ((*j_op/na2)%lev2)*na2 +
    ((*j_op/na1)%lev2)*na2 + ((*j_op/na2)%lev1)*na1;

  return;
}



/*
 * _get_val_in_subspace_gate is a simple function that returns the
 * i_op,j_op pair and val for a given index;
 * Inputs:
 *      int i:              current index
 *      gate_type my_gate_type the gate type
 * Outputs:
 *      int *i_op:          row value in subspace
 *      int *j_op:          column value in subspace
 * Return value:
 *      PetscScalar val:         value at i_op,j_op
 */

PetscScalar _get_val_in_subspace_gate(PetscInt i,gate_type my_gate_type,PetscInt control,PetscInt *i_op,PetscInt *j_op){
  PetscScalar val=0.0;
  if (my_gate_type == CNOT) {
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
    if (control==0) {
      if (i==0){
        *i_op = 0; *j_op = 0;
        val = 1.0;
      } else if (i==1) {
        *i_op = 1; *j_op = 1;
        val = 1.0;
      } else if (i==2) {
        *i_op = 2; *j_op = 3;
        val = 1.0;
      } else if (i==3) {
        *i_op = 3; *j_op = 2;
        val = 1.0;
      }
    } else if (control==1) {

      if (i==0){
        *i_op = 0; *j_op = 0;
        val = 1.0;
      } else if (i==1) {
        *i_op = 1; *j_op = 3;
        val = 1.0;
      } else if (i==2) {
        *i_op = 2; *j_op = 2;
        val = 1.0;
      } else if (i==3) {
        *i_op = 3; *j_op = 1;
        val = 1.0;
      }
    }

  } else if (my_gate_type == HADAMARD) {
    /*
     * The Hadamard gate is a one qubit gate defined as:
     *
     *    H = 1/sqrt(2) 1  1
     *                  1 -1
     *
     * Find the necessary Hilbert space dimensions for constructing the
     * full space matrix.
     */
    if (i==0){
      *i_op = 0; *j_op = 0;
      val = 1.0/sqrt(2);
    } else if (i==1) {
      *i_op = 0; *j_op = 1;
      val = 1.0/sqrt(2);
    } else if (i==2) {
      *i_op = 1; *j_op = 0;
      val = 1.0/sqrt(2);
    } else if (i==3) {
      *i_op = 1; *j_op = 1;
      val = -1.0/sqrt(2);
    }
  } else if (my_gate_type == SIGMAX){
    /*
     * SIGMAX gate
     *
     *   | 0  1 |
     *   | 1  0 |
     *
     */
    if (i==0){
      *i_op = 0; *j_op = 1;
      val = 1.0;
    } else if (i==1) {
      *i_op = 1; *j_op = 0;
      val = 1.0;
    } else if (i==2){
      *i_op = 1; *j_op = 1;
      val = 0.0;
    } else if (i==2) {
      *i_op = 0; *j_op = 0;
      val = 0.0;
    }
  } else if (my_gate_type == SIGMAX){
    /*
     * SIGMAY gate
     *
     *   | 0    -1.j |
     *   | 1.j    0 |
     *
     */
    if (i==0){
      *i_op = 0; *j_op = 1;
      val = -PETSC_i;
    } else if (i==1) {
      *i_op = 1; *j_op = 0;
      val = PETSC_i;
    } else if (i==2){
      *i_op = 1; *j_op = 1;
      val = 0.0;
    } else if (i==2) {
      *i_op = 0; *j_op = 0;
      val = 0.0;
    }
  } else if (my_gate_type == SIGMAZ){
    /*
     * SIGMAZ gate
     *
     *   | 1  0 |
     *   | 0 -1 |
     *
     */
    if (i==0){
      *i_op = 0; *j_op = 0;
      val = 1.0;
    } else if (i==1) {
      *i_op = 1; *j_op = 1;
      val = 1.0;
    }else if (i==2){
      *i_op = 1; *j_op = 0;
      val = 0.0;
    } else if (i==2) {
      *i_op = 0; *j_op = 1;
      val = 0.0;
    }
  }

  return val;

}

/*
 * create_circuit initializez the circuit struct. Gates can be added
 * later.
 *
 * Inputs:
 *        circuit circ: circuit to be initialized
 *        PetscIn num_gates_est: an estimate of the number of gates in
 *                               the circuit; can be negative, if no
 *                               estimate is known.
 * Outputs:
 *       operator *new_op: lowering op (op), raising op (op->dag), and number op (op->n)
 */

void create_circuit(circuit *circ,PetscInt num_gates_est){
  (*circ).start_time   = 0.0;
  (*circ).num_gates    = 0;
  (*circ).current_gate = 0;
  /*
   * If num_gates_est was positive when passed in, use that
   * as the initial gate_list size, otherwise set to
   * 100. gate_list will be dynamically resized when needed.
   */
  if (num_gates_est>0) {
    (*circ).gate_list_size = num_gates_est;
  } else {
    // Default gate_list_size
    (*circ).gate_list_size = 100;
  }
  // Allocate gate list
  (*circ).gate_list = malloc((*circ).gate_list_size * sizeof(struct quantum_gate_struct));
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
void add_gate_to_circuit(circuit *circ,PetscReal time,gate_type my_gate_type,...){
  PetscReal theta;
  int num_qubits=0,qubit,i;
  va_list ap;

  if (_gate_array_initialized==0){
    //Initialize the array of gate function pointers
    _initialize_gate_function_array();
    _gate_array_initialized = 1;
  }

  _check_gate_type(my_gate_type,&num_qubits);

  if ((*circ).num_gates==(*circ).gate_list_size){
    if (nid==0){
      printf("ERROR! Gate list not large enough!\n");
      exit(1);
    }
  }
  // Store arguments in list
  (*circ).gate_list[(*circ).num_gates].qubit_numbers = malloc(num_qubits*sizeof(int));
  (*circ).gate_list[(*circ).num_gates].time = time;
  (*circ).gate_list[(*circ).num_gates].my_gate_type = my_gate_type;
  (*circ).gate_list[(*circ).num_gates]._get_val_j_from_global_i = _get_val_j_functions_gates[my_gate_type+_min_gate_enum];

  if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ) {
    va_start(ap,num_qubits+1);
  } else {
    va_start(ap,num_qubits);
  }

  // Loop through and store qubits
  for (i=0;i<num_qubits;i++){
    qubit = va_arg(ap,int);
    if (qubit>=num_subsystems) {
      if (nid==0){
        // Disable warning because of qasm parser will make the circuit before
        // the qubits are allocated
        //printf("Warning! Qubit number greater than total systems\n");
      }
    }
    (*circ).gate_list[(*circ).num_gates].qubit_numbers[i] = qubit;
  }
  if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ){
    //Get the theta parameter from the last argument passed in
    theta = va_arg(ap,PetscReal);
    (*circ).gate_list[(*circ).num_gates].theta = theta;
  } else {
    //Set theta to 0
    (*circ).gate_list[(*circ).num_gates].theta = 0;
  }

  (*circ).num_gates = (*circ).num_gates + 1;
  return;
}


/*
 * Add a circuit to another circuit.
 * Assumes whole circuit happens at time
 */
void add_circuit_to_circuit(circuit *circ,circuit circ_to_add,PetscReal time){
  int num_qubits=0,qubit,i,j;
  va_list ap;

  // Check that we can fit the circuit in
  if (((*circ).num_gates+circ_to_add.num_gates-1)==(*circ).gate_list_size){
    if (nid==0){
      printf("ERROR! Gate list not large enough to add this circuit!\n");
      exit(1);
    }
  }

  for (i=0;i<circ_to_add.num_gates;i++){
    // Copy gate information over
    (*circ).gate_list[(*circ).num_gates].time = time;
    if (circ_to_add.gate_list[i].my_gate_type<0){
      num_qubits = 2;
    } else {
      num_qubits = 1;
    }
    (*circ).gate_list[(*circ).num_gates].qubit_numbers = malloc(num_qubits*sizeof(int));
    for (j=0;j<num_qubits;j++){
      (*circ).gate_list[(*circ).num_gates].qubit_numbers[j] = circ_to_add.gate_list[i].qubit_numbers[j];
    }

    (*circ).gate_list[(*circ).num_gates].my_gate_type = circ_to_add.gate_list[i].my_gate_type;
    (*circ).gate_list[(*circ).num_gates]._get_val_j_from_global_i = circ_to_add.gate_list[i]._get_val_j_from_global_i;
    (*circ).gate_list[(*circ).num_gates].theta = circ_to_add.gate_list[i].theta;
    (*circ).num_gates = (*circ).num_gates + 1;
  }



  return;
}

/* register a circuit to be run a specific time during the time stepping */
void start_circuit_at_time(circuit *circ,PetscReal time){
  (*circ).start_time = time;
  _circuit_list[_num_circuits] = *circ;
  _num_circuits = _num_circuits + 1;

}

/*
 *
 * tensor_control - switch on which superoperator to compute
 *                  -1: I cross G or just G (the difference is controlled by the passed in i's, but
 *                                           the internal logic is exactly the same)
 *                   0: G* cross G
 *                   1: G* cross I
 */

void _get_val_j_from_global_i_gates(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                    PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  operator this_op1,this_op2;
  PetscInt n_after,i_sub,tmp_int,control,moved_system,my_levels,num_js_i1=0,num_js_i2=0;
  PetscInt k1,k2,n_before1,n_before2,i_tmp,j_sub,extra_after,i1,i2,j1,js_i1[2],js_i2[2];
  PetscScalar vals_i1[2],vals_i2[2];
  //2 is hardcoded because 2 is the largest number of js from 1 i (HADAMARD)
  /*
   * We store our gates as a type and affected systems;
   * we use the stored information to calculate the global j(s) location
   * and nonzero value(s) for a give global i
   *
   * Fo all 2-qubit gates, we use the fact that
   * diagonal elements are diagonal, even in global space
   * and that off-diagonal elements can be worked out from the
   * following:
   * Off diagonal elements:
   *    if (i_sub==1)
   *       i = 1 * n_af + k1 + k2*n_me*n_af
   *       j = 0 * n_af + k1 + k2*n_me*n_af
   *    if (i_sub==0)
   *       i = 0 * n_af + k1 + k2*n_l*n_af
   *       j = 1 * n_af + k1 + k2*n_l*n_af
   * We work out k1 and k2 from i to get j.
   *
   */
  if (tensor_control!= 0) {
    if (tensor_control==1) {
      extra_after = total_levels;
    } else {
      extra_after = 1;
    }

    if (gate.my_gate_type > 0) { // Single qubit gates are coded as positive numbers

      //Get the system this is affecting
      this_op1 = subsystem_list[gate.qubit_numbers[0]];
      if (this_op1->my_levels!=2) {
        //Check that it is a two level system
        if (nid==0){
          printf("ERROR! Single qubit gates can only affect 2-level systems\n");
          exit(0);
        }
      }
      n_after = total_levels/(this_op1->my_levels*this_op1->n_before)*extra_after;
      i_sub = i/n_after%this_op1->my_levels; //Use integer arithmetic to get floor function


      //Branch on the gate types
      if (gate.my_gate_type == HADAMARD){
        /*
         * HADAMARD gate
         *
         * 1/sqrt(2) | 1  1 |
         *           | 1 -1 |
         * Hadamard gates have two values per row,
         * with both diagonal anad off diagonal elements
         *
         */
        *num_js = 2;
        if (i_sub==0) {
          // Diagonal element
          js[0]   = i;
          vals[0] = pow(2,-0.5);

          // Off diagonal element
          tmp_int = i - 0 * n_after;
          k2      = tmp_int/(this_op1->my_levels*n_after);//Use integer arithmetic to get floor function
          k1      = tmp_int%(this_op1->my_levels*n_after);
          js[1]   = (0 + 1) * n_after + k1 + k2*this_op1->my_levels*n_after;
          vals[1] = pow(2,-0.5);

        } else if (i_sub==1){
          // Diagonal element
          js[0]   = i;
          vals[0] = -pow(2,-0.5);

          // Off diagonal element
          tmp_int = i - (0+1) * n_after;
          k2      = tmp_int/(this_op1->my_levels*n_after);//Use integer arithmetic to get floor function
          k1      = tmp_int%(this_op1->my_levels*n_after);
          js[1]   = 0 * n_after + k1 + k2*this_op1->my_levels*n_after;
          vals[1] = pow(2,-0.5);

        } else {
          if (nid==0){
            printf("ERROR! Hadamard gate is only defined for qubits\n");
            exit(0);
          }
        }
      } else if (gate.my_gate_type == SIGMAX){
        /*
         * SIGMAX gate
         *
         *   | 0  1 |
         *   | 1  0 |
         *
         */
        *num_js = 1;
        if (i_sub==0) {

          // Off diagonal element
          tmp_int = i - 0 * n_after;
          k2      = tmp_int/(this_op1->my_levels*n_after);//Use integer arithmetic to get floor function
          k1      = tmp_int%(this_op1->my_levels*n_after);
          js[0]     = (0 + 1) * n_after + k1 + k2*this_op1->my_levels*n_after;
          vals[0]   = 1.0;

        } else if (i_sub==1){

          // Off diagonal element
          tmp_int = i - (0+1) * n_after;
          k2      = tmp_int/(this_op1->my_levels*n_after);//Use integer arithmetic to get floor function
          k1      = tmp_int%(this_op1->my_levels*n_after);
          js[0]   = 0 * n_after + k1 + k2*this_op1->my_levels*n_after;
          vals[0] = 1.0;

        } else {
          if (nid==0){
            printf("ERROR! sigmax gate is only defined for qubits\n");
            exit(0);
          }
        }
      } else if (gate.my_gate_type == SIGMAY){
        /*
         * SIGMAY gate
         *
         *   | 0  -1.j |
         *   | 1.j  0 |
         *
         */
        *num_js = 1;
        if (i_sub==0) {

          // Off diagonal element
          tmp_int = i - 0 * n_after;
          k2      = tmp_int/(this_op1->my_levels*n_after);//Use integer arithmetic to get floor function
          k1      = tmp_int%(this_op1->my_levels*n_after);
          js[0]   = (0 + 1) * n_after + k1 + k2*this_op1->my_levels*n_after;
          vals[0] = -1.0*PETSC_i;

        } else if (i_sub==1){

          // Off diagonal element
          tmp_int = i - (0+1) * n_after;
          k2      = tmp_int/(this_op1->my_levels*n_after);//Use integer arithmetic to get floor function
          k1      = tmp_int%(this_op1->my_levels*n_after);
          js[0]   = 0 * n_after + k1 + k2*this_op1->my_levels*n_after;
          vals[0] = 1.0*PETSC_i;

        } else {
          if (nid==0){
            printf("ERROR! sigmax gate is only defined for qubits\n");
            exit(0);
          }
        }
      } else if (gate.my_gate_type == SIGMAZ){
        /*
         * SIGMAZ gate
         *
         *   | 1   0 |
         *   | 0  -1 |
         *
         */
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
            printf("ERROR! sigmax gate is only defined for qubits\n");
            exit(0);
          }
        }

      } else if (gate.my_gate_type == EYE){
        /*
         * Identity (EYE) gate
         *
         *   | 1   0 |
         *   | 0   1 |
         *
         */
        *num_js = 1;
        if (i_sub==0) {
          // Diagonal element
          js[0] = i;
          vals[0] = 1.0;

        } else if (i_sub==1){
          // Diagonal element
          js[0] = i;
          vals[0] = 1.0;

        } else {
          if (nid==0){
            printf("ERROR! sigmax gate is only defined for qubits\n");
            exit(0);
          }
        }

      } else {


        if (nid==0){
          printf("ERROR! Gate type not understood!\n");
          exit(0);
        }
      }
    } else {
      //Two qubit gates
      this_op1 = subsystem_list[gate.qubit_numbers[0]];
      this_op2 = subsystem_list[gate.qubit_numbers[1]];
      if (this_op1->my_levels * this_op2->my_levels != 4) {
        //Check that it is a two level system
        if (nid==0){
          printf("ERROR! Two qubit gates can only affect two 2-level systems (global_i)\n");
          exit(0);
        }
      }

      n_before1  = this_op1->n_before;
      n_before2  = this_op2->n_before;

      control = 0;
      moved_system = gate.qubit_numbers[1];

      /* 2 is hardcoded because CNOT gates are for qubits, which have 2 levels */
      /* 4 is hardcoded because 2 qubits with 2 levels each */
      n_after   = total_levels/(4*n_before1)*extra_after;

      /*
       * Check which is the control and which is the target,
       * flip if need be.
       */
      if (n_before2<n_before1) {
        n_after   = total_levels/(4*n_before2);
        control   = 1;
        moved_system = gate.qubit_numbers[0];
        n_before1 = n_before2;
      }

      /* 4 is hardcoded because 2 qubits with 2 levels each */
      my_levels   = 4;

      /*
       * Permute to temporary basis
       * Get the i_sub in the permuted basis
       */
      i_tmp = i;
      _change_basis_ij_pair(&i_tmp,&j1,gate.qubit_numbers[control]+1,moved_system); // j1 useless here

      i_sub = i_tmp/n_after%my_levels; //Use integer arithmetic to get floor function

      if (gate.my_gate_type == CNOT) {
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
            _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
            _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
          _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
          js[0] = j1;
        } else {
          if (nid==0){
            printf("ERROR! CNOT gate is only defined for 2 qubits!\n");
            exit(0);
          }
        }
      } else if (gate.my_gate_type == CXZ) {
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
            _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
            _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
          _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
          js[0] = j1;
        } else {
          if (nid==0){
            printf("ERROR! CXZ gate is only defined for 2 qubits!\n");
            exit(0);
          }
        }
      } else if (gate.my_gate_type == CZX) {
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
            _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
            _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
          _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
          js[0] = j1;
        } else {
          if (nid==0){
            printf("ERROR! CZX gate is only defined for 2 qubits!\n");
            exit(0);
          }
        }
      } else if (gate.my_gate_type == CZ) {
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
      } else if (gate.my_gate_type == CmZ) {
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
        if (nid==0){
          printf("ERROR! Gate type not understood! %d\n",gate.my_gate_type);
          exit(0);
        }
      }
    }
    if (tensor_control==1){
      //Take complex conjugate of answer to get U* cross I
      for (i=0;i<*num_js;i++){
        vals[i] = PetscConjComplex(vals[i]);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    _get_val_j_from_global_i_gates(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    _get_val_j_from_global_i_gates(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void combine_circuit_to_mat2(Mat *matrix_out,circuit circ){
  PetscScalar op_val,op_vals[total_levels],vals[2]={0};
  PetscInt Istart,Iend;
  PetscInt i,j,k,l,this_i,these_js[total_levels],js[2]={0},num_js_tmp=0,num_js,num_js_current;

  // Should this inherit its stucture from full_A?
  MatCreate(PETSC_COMM_WORLD,matrix_out);
  MatSetType(*matrix_out,MATMPIAIJ);
  MatSetSizes(*matrix_out,PETSC_DECIDE,PETSC_DECIDE,total_levels,total_levels);
  MatSetFromOptions(*matrix_out);

  MatMPIAIJSetPreallocation(*matrix_out,16,NULL,16,NULL);

  /*
   * Calculate G1*G2*G3*.. using the following observation:
   *     Each gate is very sparse - having no more than
   *          2 values per row. This allows us to efficiently do the
   *          multiplication by just touching the nonzero values
   */
  MatGetOwnershipRange(*matrix_out,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    this_i = i; // The leading index which we check
    // Reset the result for the row
    num_js = 1;
    these_js[0] = i;
    op_vals[0]  = 1.0;
    for (j=0;j<circ.num_gates;j++){
      num_js_current = num_js;
      for (k=0;k<num_js_current;k++){
        // Loop through all of the js from the previous gate multiplications
        this_i = these_js[k];
        op_val = op_vals[k];

        _get_val_j_from_global_i_gates(this_i,circ.gate_list[j],&num_js_tmp,js,vals,-1); // Get the corresponding j and val
        /*
         * Assume there is always at least 1 nonzero per row. This is a good assumption
         * because all basic quantum gates have at least 1 nonzero per row
         */
        // WARNING! CODE NOT FINISHED
        // WILL NOT WORK FOR HADAMARD * HADAMARD

        these_js[k] = js[0];
        op_vals[k]  = op_val*vals[0];

        for (l=1;l<num_js_tmp;l++){
          //If we have more than 1 num_js_tmp, we append to the end of the list
          these_js[num_js+l-1] = js[l];
          op_vals[num_js+l-1]  = op_val*vals[l];
        }
         num_js = num_js + num_js_tmp - 1; //If we spawned an extra j, add it here
      }
    }
    MatSetValues(*matrix_out,1,&i,num_js,these_js,op_vals,ADD_VALUES);
  }

  MatAssemblyBegin(*matrix_out,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(*matrix_out,MAT_FINAL_ASSEMBLY);

  return;
}


void combine_circuit_to_mat(Mat *matrix_out,circuit circ){
  PetscScalar op_vals[2];
  PetscInt Istart,Iend,i_mat;
  PetscInt i,these_js[2],num_js;
  Mat tmp_mat1,tmp_mat2,tmp_mat3;

  // Should this inherit its stucture from full_A?

  MatCreate(PETSC_COMM_WORLD,&tmp_mat1);
  MatSetType(tmp_mat1,MATMPIAIJ);
  MatSetSizes(tmp_mat1,PETSC_DECIDE,PETSC_DECIDE,total_levels,total_levels);
  MatSetFromOptions(tmp_mat1);

  MatMPIAIJSetPreallocation(tmp_mat1,2,NULL,2,NULL);

  /* Construct the first matrix in tmp_mat1 */
  MatGetOwnershipRange(tmp_mat1,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    circ.gate_list[0]._get_val_j_from_global_i(i,circ.gate_list[0],&num_js,these_js,op_vals,-1); // Get the corresponding j and val
    MatSetValues(tmp_mat1,1,&i,num_js,these_js,op_vals,ADD_VALUES);
  }

  MatAssemblyBegin(tmp_mat1,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tmp_mat1,MAT_FINAL_ASSEMBLY);

  for (i_mat=1;i_mat<circ.num_gates;i_mat++){
    // Create the next matrix
    MatCreate(PETSC_COMM_WORLD,&tmp_mat2);
    MatSetType(tmp_mat2,MATMPIAIJ);
    MatSetSizes(tmp_mat2,PETSC_DECIDE,PETSC_DECIDE,total_levels,total_levels);
    MatSetFromOptions(tmp_mat2);

    MatMPIAIJSetPreallocation(tmp_mat2,2,NULL,2,NULL);

    /* Construct new matrix */
    MatGetOwnershipRange(tmp_mat2,&Istart,&Iend);
    for (i=Istart;i<Iend;i++){
      _get_val_j_from_global_i_gates(i,circ.gate_list[i_mat],&num_js,these_js,op_vals,-1); // Get the corresponding j and val
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
  MatConvert(tmp_mat1,MATSAME,MAT_INITIAL_MATRIX,matrix_out);;

  MatDestroy(&tmp_mat1);
  return;
}


void combine_circuit_to_super_mat(Mat *matrix_out,circuit circ){
  PetscScalar val1=0,val2=0,op_vals[4];
  PetscInt Istart,Iend,i_mat,dim;
  PetscInt i,this_j1=0,this_j2=0,these_js[4],num_js;
  Mat tmp_mat1,tmp_mat2,tmp_mat3;

  // Should this inherit its stucture from full_A?
  dim = total_levels*total_levels;
  MatCreate(PETSC_COMM_WORLD,&tmp_mat1);
  MatSetType(tmp_mat1,MATMPIAIJ);
  MatSetSizes(tmp_mat1,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
  MatSetFromOptions(tmp_mat1);

  MatMPIAIJSetPreallocation(tmp_mat1,8,NULL,8,NULL);

  /* Construct the first matrix in tmp_mat1 */
  MatGetOwnershipRange(tmp_mat1,&Istart,&Iend);
  for (i=Istart;i<Iend;i++){
    _get_val_j_from_global_i_gates(i,circ.gate_list[0],&num_js,these_js,op_vals,0); // Get the corresponding j and val
    MatSetValues(tmp_mat1,1,&i,num_js,these_js,op_vals,ADD_VALUES);
  }

  MatAssemblyBegin(tmp_mat1,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tmp_mat1,MAT_FINAL_ASSEMBLY);

  for (i_mat=1;i_mat<circ.num_gates;i_mat++){
    // Create the next matrix
    MatCreate(PETSC_COMM_WORLD,&tmp_mat2);
    MatSetType(tmp_mat2,MATMPIAIJ);
    MatSetSizes(tmp_mat2,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
    MatSetFromOptions(tmp_mat2);

    MatMPIAIJSetPreallocation(tmp_mat2,8,NULL,8,NULL);

    /* Construct new matrix */
    MatGetOwnershipRange(tmp_mat2,&Istart,&Iend);
    for (i=Istart;i<Iend;i++){
      _get_val_j_from_global_i_gates(i,circ.gate_list[i_mat],&num_js,these_js,op_vals,0); // Get the corresponding j and val
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
  MatConvert(tmp_mat1,MATSAME,MAT_INITIAL_MATRIX,matrix_out);;

  MatDestroy(&tmp_mat1);
  return;
}


void CNOT_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_2qbit(&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);

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
        _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
        _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
      _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CNOT_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CNOT_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void CXZ_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_2qbit(&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);
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
        _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
        _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
      _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CXZ_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CXZ_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}

void CZ_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                  PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1=0,num_js_i2=0,js_i1[2],js_i2[2];
  PetscInt control,i_tmp,my_levels,j_sub,moved_system,j1;
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
    _get_n_after_2qbit(&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CZ_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CZ_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void CmZ_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                  PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1=0,num_js_i2=0,js_i1[2],js_i2[2];
  PetscInt control,i_tmp,my_levels,j_sub,moved_system,j1;
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
    _get_n_after_2qbit(&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);

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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CmZ_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CmZ_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void CZX_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_2qbit(&i_tmp,gate.qubit_numbers,tensor_control,&n_after,&control,&moved_system,&i_sub);

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
        _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
        _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1); // i_tmp useless here
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
      _change_basis_ij_pair(&i_tmp,&j1,moved_system,gate.qubit_numbers[control]+1);//i_tmp useless here
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    CZX_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    CZX_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }

  }

  return;
}


void HADAMARD_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    HADAMARD_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    HADAMARD_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }

  return;
}


void EYE_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2];
  PetscScalar vals_i1[2],vals_i2[2];

  /*
   * Identity (EYE) gate
   *
   *   | 1   0 |
   *   | 0   1 |
   *
   */

  if (tensor_control!= 0) {
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    EYE_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    EYE_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void SIGMAZ_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
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
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    SIGMAZ_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    SIGMAZ_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}

void RZ_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
                                      PetscInt js[],PetscScalar vals[],PetscInt tensor_control){
  PetscInt n_after,i_sub,k1,k2,tmp_int,i1,i2,num_js_i1,num_js_i2,js_i1[2],js_i2[2],my_levels;
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
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    RZ_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    RZ_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}

void RY_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    RY_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    RY_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void RX_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    RX_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    RX_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void SIGMAY_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    SIGMAY_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    SIGMAY_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}


void SIGMAX_get_val_j_from_global_i(PetscInt i,struct quantum_gate_struct gate,PetscInt *num_js,
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
    _get_n_after_1qbit(i,gate.qubit_numbers[0],tensor_control,&n_after,&i_sub);
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
    i1 = i/total_levels;
    i2 = i%total_levels;

    /* Now, get js for U* (i1) by calling this function */
    SIGMAX_get_val_j_from_global_i(i1,gate,&num_js_i1,js_i1,vals_i1,-1);

    /* Now, get js for U (i2) by calling this function */
    SIGMAX_get_val_j_from_global_i(i2,gate,&num_js_i2,js_i2,vals_i2,-1);

    /*
     * Combine j's to get U* cross U
     * Must do all possible permutations
     */
    *num_js = 0;
    for(k1=0;k1<num_js_i1;k1++){
      for(k2=0;k2<num_js_i2;k2++){
        js[*num_js] = total_levels * js_i1[k1] + js_i2[k2];
        //Need to take complex conjugate to get true U*
        vals[*num_js] = PetscConjComplex(vals_i1[k1])*vals_i2[k2];

        *num_js = *num_js + 1;
      }
    }
  }
  return;
}

void _get_n_after_2qbit(PetscInt *i,int qubit_numbers[],PetscInt tensor_control,PetscInt *n_after, PetscInt *control, PetscInt *moved_system, PetscInt *i_sub){
  operator this_op1,this_op2;
  PetscInt n_before1,n_before2,extra_after,i_tmp,my_levels=4,j1; //4 is hardcoded because 2 qbits
  if (tensor_control==1) {
    extra_after = total_levels;
  } else {
    extra_after = 1;
  }

  //Two qubit gates
  this_op1 = subsystem_list[qubit_numbers[0]];
  this_op2 = subsystem_list[qubit_numbers[1]];
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
  *n_after   = total_levels/(4*n_before1)*extra_after;

  /*
   * Check which is the control and which is the target,
   * flip if need be.
   */
  if (n_before2<n_before1) {
    *n_after   = total_levels/(4*n_before2);
    *control   = 1;
    *moved_system = qubit_numbers[0];
    n_before1 = n_before2;
  }
  /*
   * Permute to temporary basis
   * Get the i_sub in the permuted basis
   */
  _change_basis_ij_pair(i,&j1,qubit_numbers[*control]+1,*moved_system); // j1 useless here

  *i_sub = *i/(*n_after)%my_levels; //Use integer arithmetic to get floor function

  return;
}

void _get_n_after_1qbit(PetscInt i,int qubit_number,PetscInt tensor_control,PetscInt *n_after,PetscInt *i_sub){
  operator this_op1;
  PetscInt extra_after;
  if (tensor_control==1) {
    extra_after = total_levels;
  } else {
    extra_after = 1;
  }

  //Get the system this is affecting
  this_op1 = subsystem_list[qubit_number];
  if (this_op1->my_levels!=2) {
    //Check that it is a two level system
    if (nid==0){
      printf("ERROR! Single qubit gates can only affect 2-level systems\n");
      exit(0);
    }
  }
  *n_after = total_levels/(this_op1->my_levels*this_op1->n_before)*extra_after;
  *i_sub = i/(*n_after)%this_op1->my_levels; //Use integer arithmetic to get floor function

  return;
}

// Check that the gate type is valid and set the number of qubits
void _check_gate_type(gate_type my_gate_type,int *num_qubits){

  if (my_gate_type==HADAMARD||my_gate_type==SIGMAX||my_gate_type==SIGMAY||my_gate_type==SIGMAZ||my_gate_type==EYE||
      my_gate_type==RZ||my_gate_type==RX||my_gate_type==RY) {
    *num_qubits = 1;
  } else if (my_gate_type==CNOT||my_gate_type==CXZ||my_gate_type==CZ||my_gate_type==CmZ||my_gate_type==CZX){
    *num_qubits = 2;
  } else {
    if (nid==0){
      printf("ERROR! Gate type not recognized\n");
      exit(0);
    }
  }

}
/*
 * Put the gate function pointers into an array
 */
void _initialize_gate_function_array(){
  _get_val_j_functions_gates[CZX+_min_gate_enum] = CZX_get_val_j_from_global_i;
  _get_val_j_functions_gates[CmZ+_min_gate_enum] = CmZ_get_val_j_from_global_i;
  _get_val_j_functions_gates[CZ+_min_gate_enum] = CZ_get_val_j_from_global_i;
  _get_val_j_functions_gates[CXZ+_min_gate_enum] = CXZ_get_val_j_from_global_i;
  _get_val_j_functions_gates[CNOT+_min_gate_enum] = CNOT_get_val_j_from_global_i;
  _get_val_j_functions_gates[HADAMARD+_min_gate_enum] = HADAMARD_get_val_j_from_global_i;
  _get_val_j_functions_gates[SIGMAX+_min_gate_enum] = SIGMAX_get_val_j_from_global_i;
  _get_val_j_functions_gates[SIGMAY+_min_gate_enum] = SIGMAY_get_val_j_from_global_i;
  _get_val_j_functions_gates[SIGMAZ+_min_gate_enum] = SIGMAZ_get_val_j_from_global_i;
  _get_val_j_functions_gates[EYE+_min_gate_enum] = EYE_get_val_j_from_global_i;
  _get_val_j_functions_gates[RX+_min_gate_enum] = RX_get_val_j_from_global_i;
  _get_val_j_functions_gates[RY+_min_gate_enum] = RY_get_val_j_from_global_i;
  _get_val_j_functions_gates[RZ+_min_gate_enum] = RZ_get_val_j_from_global_i;
}
