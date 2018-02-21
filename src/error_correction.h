#ifndef ERROR_CORRECTION_H_
#define ERROR_CORRECTION_H_

#include "quantum_gates.h"
typedef enum {
  NONE     = 0,
  BIT      = 1,
  PHASE    = 2,
  FIVE     = 3
} encoder_type;

typedef struct stabilizer{
  int n_ops;
  operator* ops;
} stabilizer;

typedef struct encoded_qubit{
  PetscInt *qubits,num_qubits;
  encoder_type my_encoder_type;
  circuit encoder_circuit,decoder_circuit;
} encoded_qubit;

void build_recovery_lin(Mat*,operator,char[],int,...);
void add_lin_recovery(PetscScalar,PetscInt,operator,char[],int,...);
void create_stabilizer(stabilizer*,int,...);
void destroy_stabilizer(stabilizer*);
void _get_row_nonzeros(PetscScalar[],PetscInt[],PetscInt*,PetscInt,operator,char[],int,stabilizer[]);
void _get_this_i_and_val_from_stab(PetscInt*, PetscScalar*,stabilizer,char,PetscInt);
void create_encoded_qubit(encoded_qubit*,encoder_type,...);
void add_encoded_gate_to_circuit(circuit*,PetscReal,gate_type,...);
void encode_state(Vec,PetscInt,...);
void decode_state(Vec,PetscInt,...);
void add_continuous_error_correction(encoded_qubit,PetscReal);
PetscErrorCode _DQEC_PostEventFunction(TS,PetscInt,PetscInt[],PetscReal,Vec,void*);
PetscErrorCode _DQEC_EventFunction(TS,PetscReal,Vec,PetscScalar*,void*);
extern int _discrete_ec;
#endif
