#ifndef QUAC_P_H_
#define QUAC_P_H_
#include <petsc.h>
extern int  petsc_initialized;
PetscLogEvent add_lin_event,add_to_ham_event,add_lin_recovery_event,add_encoded_gate_to_circuit_event;
PetscLogEvent _qc_event_function_event,_qc_postevent_function_event,_apply_gate_event;
PetscClassId quac_class_id;
PetscLogStage pre_solve_stage,solve_stage,post_solve_stage;
#endif
