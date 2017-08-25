#ifndef OPERATORS_P_H_
#define OPERATORS_P_H_

#include <petscmat.h>

typedef enum {
    RAISE  = -1,
    NUMBER = 0,
    LOWER  = 1,
    VEC    = 2
  } op_type;


void _check_initialized_A();
void _check_initialized_op();

extern int  _num_time_dep;
extern Mat  full_A,full_stiff_A;
extern Mat  ham_A,ham_stiff_A;
extern int  op_finalized;
extern int  _lindblad_terms;
extern int  _stiff_solver;
extern PetscInt total_levels;
extern int  num_subsystems;
extern int  op_initialized;
extern PetscScalar **_hamiltonian;
extern int _print_dense_ham;
#endif
