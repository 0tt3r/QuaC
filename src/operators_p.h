#ifndef OPERATORS_P_H_
#define OPERATORS_P_H_

#include <petscmat.h>


void check_initialized_A();
Mat  full_A;
int  op_finalized;
long total_levels;
int  num_subsystems;

#endif
