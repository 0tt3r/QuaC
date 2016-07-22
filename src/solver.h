#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscksp.h>
#include <petscts.h>

void steady_state();
void get_populations(Vec);// Move to private?
void time_step(PetscReal,PetscReal,PetscInt);

#endif
