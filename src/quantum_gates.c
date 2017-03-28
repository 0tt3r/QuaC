#include "quantum_gates.h"
#include <stdlib.h>
#include <stdio.h>
#include <petsc.h>
#include <stdarg.h>


int _num_quantum_gates = 0;
int _current_gate = 0;
struct quantum_gate_struct _quantum_gate_list[MAX_GATES];


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
PetscErrorCode _QG_PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,void* ctx) {

   /* We only have one event at the moment, so we do not need to branch.
    * If we had more than one event, we would put some logic here.
    */
  if (nevents) {
    /* Apply the current gate */
    _apply_gate(_quantum_gate_list[_current_gate].my_gate_type,_quantum_gate_list[_current_gate].qubit_numbers,U);
    /* Increment our gate counter */
    _current_gate = _current_gate + 1;
  }

  TSSetSolution(ts,U);
  return(0);
}

/* Add a gate to the list */
void add_gate(PetscReal time,gate_type my_gate_type,...) {
  int num_qubits,qubit,i;
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

  va_start(ap,num_qubits);

  // Loop through and store qubits
  for (i=0;i<num_qubits;i++){
    qubit = va_arg(ap,int);
    _quantum_gate_list[_num_quantum_gates].qubit_numbers[i] = qubit;
  }

  _num_quantum_gates = _num_quantum_gates + 1;
}


/* Apply a specific gate */
void _apply_gate(gate_type my_gate_type,int *systems,Vec rho){
  Mat gate_mat; //FIXME Consider having only one static array for all gates, rather than creating new ones every time
  Vec tmp_answer;
  PetscInt dim;

  dim = total_levels*total_levels;

  VecDuplicate(rho,&tmp_answer); //Create a new vec with the same size as rho

  MatCreate(PETSC_COMM_WORLD,&gate_mat);
  MatSetSizes(gate_mat,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
  MatSetFromOptions(gate_mat);
  MatMPIAIJSetPreallocation(gate_mat,2,NULL,2,NULL); //This matrix is incredibly sparse!
  MatSetUp(gate_mat);
  /* Construct the gate matrix, on the fly */

  _construct_gate_mat(my_gate_type,systems,gate_mat);

  MatAssemblyBegin(gate_mat,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(gate_mat,MAT_FINAL_ASSEMBLY);
  //  MatView(gate_mat,PETSC_VIEWER_STDOUT_SELF);
  MatMult(gate_mat,rho,tmp_answer);
  VecCopy(tmp_answer,rho); //Copy our tmp_answer array into rho

  VecDestroy(&tmp_answer); //Destroy the temp answer
  MatDestroy(&gate_mat);
}


/*
 * _construct_gate_mat constructs the matrix needed for the quantum
 * computing gates.
 *
 * Inputs:
 *     gate_type my_gate_type  type of quantum gate
 *     int *s
 * Outputs:
 *      none, but adds to PETSc matrix full_A
 */

void _construct_gate_mat(gate_type my_gate_type,int *systems,Mat gate_mat){
  PetscInt i,j,i_mat,j_mat,k1,k2,k3,k4,n_before1,n_before2,my_levels,n_after;
  PetscInt i1,j1,i2,j2,comb_levels,control,moved_system;
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
      for (i=0;i<4;i++){
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
  na2   = total_levels/(lev1*subsystem_list[system2]->n_before);

  //  printf("nb1,nb2,lev1,lev2: %d %d %d %d\n",nb1,nb2,lev1,lev2);
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

PetscScalar _get_val_in_subspace_gate(int i,gate_type my_gate_type,int control,int *i_op,int *j_op){
  PetscScalar val;
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


  }

  return val;

}
