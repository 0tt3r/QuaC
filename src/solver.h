#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscksp.h>
#include <petscts.h>

void steady_state(Vec);
void time_step(Vec,PetscReal,PetscReal,PetscReal,PetscInt);
void set_ts_monitor(PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*));

#endif
