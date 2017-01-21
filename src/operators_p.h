#ifndef OPERATORS_P_H_
#define OPERATORS_P_H_

#include <petscmat.h>

typedef enum {
    RAISE  = -1,
    NUMBER = 0,
    LOWER  = 1,
    VEC    = 2
  } op_type;

typedef struct time_dep_struct{
  double (*time_dep_func)(double);
  Mat mat;
} time_dep_struct;



void _check_initialized_A();
void _check_initialized_op();

extern int  _num_time_dep;
extern Mat  full_A;
extern int  op_finalized;
extern PetscInt total_levels;
extern int  num_subsystems;
extern double **_hamiltonian;
extern int _print_dense_ham;
#endif
