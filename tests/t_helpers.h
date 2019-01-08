#ifndef T_HELPERS_H_
#define T_HELPERS_H_

#include "petsc.h"

#define DELTA 1e-9

void _get_mat_and_diff_norm(char*,Mat,PetscReal*);
void _get_mat_combine_and_diff_norm(char*,Mat,PetscReal*);

#endif
