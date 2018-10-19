#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscksp.h>
#include <petscts.h>
#include <slepceps.h>

void steady_state(Vec);
void time_step(Vec,PetscReal,PetscReal,PetscReal,PetscInt);
void diagonalize(PetscInt*,Vec**,PetscScalar**);
void destroy_diagonalize(PetscInt,Vec**,PetscScalar**);
void set_ts_monitor(PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*));
void set_ts_monitor_ctx(PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*),void*);
void g2_correlation(PetscScalar ***,Vec,PetscInt,PetscReal,PetscInt,PetscReal,PetscInt,...);
PetscErrorCode _g2_ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
typedef struct {
  Mat I_cross_A;
  PetscInt i_tau,i_st,tau_evolve;
  Vec tmp_dm,tmp_dm2;
  PetscScalar **g2_values;
} TSCtx;

#endif
