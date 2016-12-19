




void _apply_gate(gate_type my_gate_type,int *systems,Mat AA){
  Mat gate_mat,tmp_answer;

  /* Construct the gate matrix, on the fly */
  _construct_gate_mat(my_gate_type,systems,gate_mat);
  MatMatMult(gate_mat,AA,MAT_INITAL_MATRIX,&tmp_answer);

}
/*                                              
 * _add_to_PETSc_kron expands an operator given a Hilbert space size
 * before and after and adds that to the Petsc matrix full_A
 *
 * Inputs:
 *      PetscScalar a       scalar to multiply operator (can be complex)
 *      int n_before:       Hilbert space size before
 *      int my_levels:      number of levels for operator
 *      op_type my_op_type: operator type
 *      int position:       vec operator's position variable
 *      int extra_before:   extra Hilbert space size before
 *      int extra_after:    extra Hilbert space size after
 * Outputs:
 *      none, but adds to PETSc matrix full_A
 */

void _construct_gate_matrix(gate_type my_gate_type,int *systems,Mat AA){
  long i,i_mat,j_mat,n_after,n_before1,n_before2,tmp_switch,n_between,my_levels,k3;
  PetscReal    val;
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
    n_before1  = subsystem_list[systems[0]].n_before;
    n_before2  = subsystem_list[systems[1]].n_before;
    /* 2 is hardcoded because CNOT gates are for qubits, which have 2 levels */
    n_between = n_before2/(n_before1*2); 
    /* 4 is hardcoded because 2 qubits with 2 levels each */
    n_after   = total_levels/(4*n_before1);
    if (n_before2<n_before1) {
      n_between = n_before1/(n_before*2);
      n_after   = total_levels/(4*n_before2);
    }
    /* 4 is hardcoded because 2 qubits with 2 levels each */
    my_levels = 4*n_between;
    for (k3=0;k3<n_between;k++){

      if (n_before1<n_before1) {
        /* The first qubit is the Control qubit */
        /* Identity part */
        i_mat = 0; j_mat = 0; add_to_mat = 1.0; 
        i_mat = i_mat + k3*2;
        j_mat = j_mat + k3*2;
        _add_to_mat_kron_ij(AA,add_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);

        i_mat = 1; j_mat = 1; add_to_mat = 1.0;
        i_mat = i_mat + k3*2;
        j_mat = j_mat + k3*2;
        _add_to_mat_kron_ij(AA,add_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);

        /* Sig_x part */
        i_mat = 2; j_mat = 3; add_to_mat = 1.0;
        i_mat = i_mat + (n_between-1)*2 + k3*2;
        j_mat = j_mat + (n_between-1)*2 + k3*2;
        _add_to_mat_kron_ij(AA,add_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);
        i_mat = 3; j_mat = 2; add_to_mat = 1.0;
        i_mat = i_mat + (n_between-1)*2 + k3*2;
        j_mat = j_mat + (n_between-1)*2 + k3*2;
        _add_to_mat_kron_ij(AA,add_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);

      } else {

        /* The second qubit is the Control qubit */
        /* Sig x part */
        i_mat = 0; j_mat = 1; add_to_mat = 1.0;
        i_mat = i_mat + k3*2;
        j_mat = j_mat + k3*2;
        _add_to_mat_kron_ij(AA,add_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);

        i_mat = 1; j_mat = 0; add_to_mat = 1.0;
        i_mat = i_mat + k3*2;
        j_mat = j_mat + k3*2;
        _add_to_mat_kron_ij(AA,add_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);

        /* Identity part */
        i_mat = 2; j_mat = 2; add_to_mat = 1.0;
        i_mat = i_mat + (n_between-1)*2 + k3*2;
        j_mat = j_mat + (n_between-1)*2 + k3*2;
        _add_to_mat_kron_ij(AA,add_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);
        i_mat = 3; j_mat = 3; add_to_mat = 1.0;
        i_mat = i_mat + (n_between-1)*2 + k3*2;
        j_mat = j_mat + (n_between-1)*2 + k3*2;
        _add_to_mat_kron_ij(AA,padd_to_mat,i_mat,j_mat,n_before1,n_after,my_levels);

      } 
    } else if (my_gate_type == HADAMARD) {
      
    }

  return;
}
