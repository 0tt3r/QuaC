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
extern Mat  full_A;
extern int  op_finalized;
extern PetscInt total_levels;
extern int  num_subsystems;
extern double **_hamiltonian;
extern int _print_dense_ham;
#endif
