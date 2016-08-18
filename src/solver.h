#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscksp.h>
#include <petscts.h>

void steady_state();
void time_step(PetscReal,PetscReal,PetscInt);
void set_ts_monitor(PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*));

#endif
