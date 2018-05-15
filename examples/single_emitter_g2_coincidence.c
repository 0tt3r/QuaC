
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
double pulse(double);

/* Declared globally so that we can access this in ts_monitor */
FILE *f_pop;
operator qd;
double amp,tp,td,u_qd,we,td2,pe;
PetscInt i_g2;
int tau_evolve;
PetscReal *g2_values;

int main(int argc,char **args){
  PetscReal gamma,dep,st_dt,st_max,previous_start_time,this_start_time;
  PetscReal dt,tau_t_max,tau_max,dt_tau,ave;
  PetscScalar val,pe_l;
  PetscInt  steps_max,n_st,n_tau,i;
  Vec dm0,init_dm;

  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  td2 = 3200;
  PetscOptionsGetReal(NULL,NULL,"-td2",&td2,NULL);


  /* Define scalars to add to Ham */
  we = 0.0735;
  gamma=3.3e-3; //Purcell enhanced
  dep = 0.0000735;

  amp = 2e-5;
  tp = 1000;
  td = 1800;
  u_qd = 3.93;

  n_tau = 5000;
  tau_max = 10000;
  n_st = 5000;
  st_max = 10000;

  PetscMalloc1((n_st+1)*(n_tau+1),&g2_values);
  create_op(2,&qd);

  /* Add terms to the hamiltonian */
  add_to_ham(we,qd->n);
  add_to_ham_time_dep(pulse,2,qd->dag,qd);

  add_lin(gamma,qd);
  add_lin(dep,qd->n);

  create_full_dm(&dm0);
  create_full_dm(&init_dm);
  val = 1.0;
  add_value_to_dm(dm0,0,0,val);
  assemble_dm(dm0);
  assemble_dm(init_dm);

  steps_max = 2100;
  dt        = 0.025;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor);
  /* Open file that we will print to in ts_monitor */
  if (nid==0){
    f_pop = fopen("pop","w");
    fprintf(f_pop,"#Time Pop Sig_x, Sig_y, Sig_z\n");
  }

  //Starting from 0, we time step to the next 'start time' to get the
  //Initial dm for our tau sweep
  st_dt = st_max/n_st;
  previous_start_time = 0;
  for (i_g2=0;i_g2<n_tau+1;i_g2++){
    g2_values[i_g2] = 0.0;
  }
  for (this_start_time=st_dt;this_start_time<=st_max;this_start_time+=st_dt){
    //Go from previous start time to this_start_time
    tau_evolve = 0;
    dt = (this_start_time - previous_start_time)/2000;
    time_step(dm0,previous_start_time,this_start_time,dt,steps_max);
    //Copy the timestepped dm into our init_dm for tau sweep
    VecCopy(dm0,init_dm);
    //Get the expectation value of q->dag q for this init_dm
    get_expectation_value(init_dm,&pe_l,1,qd->n);
    pe = PetscRealPart(pe_l);
    //Force an emission
    measure_dm(init_dm,qd);
    //Timestep through taus
    tau_t_max = this_start_time + tau_max;
    dt_tau = (tau_t_max - this_start_time)/n_tau;
    tau_evolve = 1;
    time_step(init_dm,this_start_time,tau_t_max,dt_tau,steps_max);
    previous_start_time = this_start_time;
    /* if(i_g2>250){ */
    /*   return 0; */
    /* } */
  }
  //Calculate average of g2
  ave = 0;
  for (i=0;i<(n_tau+1)*(n_st+1);i++){
    ave = ave + g2_values[i];
  }
  ave = ave/((n_tau+1)*(n_st+1));
  printf("ave: %e\n",ave);
  destroy_op(&qd);
  destroy_dm(dm0);
  destroy_dm(init_dm);
  if (nid==0) fclose(f_pop);
  QuaC_finalize();
  return 0;
}

double pulse(double t){
  double pulse_value;
  pulse_value = (amp*exp(-2 * log(2) * pow((t-td)/(tp),2)) * cos(we*(t-td))
                 + amp*exp(-2*log(2)*pow((t-td2)/(tp),2))*cos(we*(t-td2)));
  pulse_value = -u_qd * pulse_value;
  return pulse_value;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  PetscScalar ev;
  if (tau_evolve==1){
    get_expectation_value(dm,&ev,1,qd->n);
    g2_values[i_g2] = pe*PetscRealPart(ev);
    i_g2 = i_g2 + 1;
  }
  PetscFunctionReturn(0);
}
