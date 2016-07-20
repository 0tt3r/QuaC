#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscksp.h>
#include <petscts.h>

void steady_state();
void get_populations(Vec);// Move to private?
void time_step();
PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*); // Move to private?
//PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
#endif
