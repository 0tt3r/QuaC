
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
PetscInt i_g2,i_st,i_tau;
int tau_evolve;
PetscReal **g2_values_2d;

int main(int argc,char **args){
  PetscReal gamma,dep,st_dt,st_max,previous_start_time,this_start_time;
  PetscReal dt,tau_t_max,tau_max,dt_tau,ave,ave2,total_pe;
  PetscScalar val,pe_l;
  PetscInt  steps_max,n_st,n_tau,i;
  Vec dm0,init_dm;
  PetscBool run_fl = PETSC_FALSE,print_map = PETSC_FALSE;
  FILE *file;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  td2 = 3200;
  run_fl = PETSC_FALSE;
  gamma=3.3e-3; //Purcell enhanced
  amp = 2e-5;
  dep = 0.0000735;

  PetscOptionsGetReal(NULL,NULL,"-td2",&td2,NULL);
  PetscOptionsGetReal(NULL,NULL,"-amp",&amp,NULL);
  PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);
  PetscOptionsGetReal(NULL,NULL,"-dep",&dep,NULL);
  PetscOptionsGetBool(NULL,NULL,"-run_fl",&run_fl,NULL);
  PetscOptionsGetBool(NULL,NULL,"-map",&print_map,NULL);


  /* Define scalars to add to Ham */
  we = 0.0735;


  tp = 1000;
  td = 1800;
  u_qd = 3.93;

  n_tau = 4000; //was 2000
  tau_max = 7500;
  n_st = 4000; //was 2000
  st_max = 7500;

  g2_values_2d = (PetscReal **)malloc((n_st+1)*sizeof(PetscReal *));
  for (i=0;i<n_st+1;i++){
    g2_values_2d[i] = (PetscReal *)malloc((n_tau+1)*sizeof(PetscReal));
  }
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

  steps_max = 5100;
  dt        = 0.025;

  /* Set the ts_monitor to print results at each time step */
  set_ts_monitor(ts_monitor);

  if (run_fl==PETSC_TRUE){
    st_dt = st_max/n_st;
    previous_start_time = 0;
    total_pe = 0;
    i_st = i_st + 1;
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
      total_pe = total_pe + pe;
      previous_start_time = this_start_time;
      i_st = i_st + 1;
      /* if(i_g2>250){ */
      /*   return 0; */
      /* } */
    }
    printf("fl: %e\n",total_pe/i_st);

  } else {
    //Starting from 0, we time step to the next 'start time' to get the
    //Initial dm for our tau sweep
    st_dt = st_max/n_st;
    previous_start_time = 0;
    i_st = 0;
    for (i_g2=0;i_g2<n_tau+1;i_g2++){
      g2_values_2d[i_st][i_g2] = 0.0;
    }
    i_st = i_st + 1;
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
      total_pe = total_pe + pe;
      //Force an emission
      measure_dm(init_dm,qd);
      //Timestep through taus
      tau_t_max = this_start_time + tau_max;
      dt_tau = (tau_t_max - this_start_time)/n_tau;
      tau_evolve = 1;
      i_tau = 0;
      time_step(init_dm,this_start_time,tau_t_max,dt_tau,steps_max);
      previous_start_time = this_start_time;
      i_st = i_st + 1;
      /* if(i_g2>250){ */
      /*   return 0; */
      /* } */
    }
    //Calculate average of g2
    ave2 = 0;
    for (i_st=0;i_st<(n_st+1);i_st++){
      for (i_tau=0;i_tau<(n_tau+1);i_tau++){
        ave2 = ave2 + g2_values_2d[i_st][i_tau];
      }
    }
    ave2 = ave2/((n_tau+1)*(n_st+1));
    printf("ave: %e %e\n",ave2,total_pe/i_st);
  }
  if (print_map){
    file = fopen("map.dat","w");
    for (i_st=0;i_st<(n_st+1);i_st++){
      for (i_tau=0;i_tau<(n_tau+1);i_tau++){
        fprintf(file,"%e\n",g2_values_2d[i_st][i_tau]);
      }
    }
    fclose(file);
  }
  /* for (i=0;i<(n_tau+1)*(n_st+1);i++){ */
  /*   printf("%e\n",g2_values[i]); */
  /* } */

  destroy_op(&qd);
  destroy_dm(dm0);
  destroy_dm(init_dm);
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
    g2_values_2d[i_st][i_tau] = pe*PetscRealPart(ev);
    i_tau = i_tau + 1;
  }
  PetscFunctionReturn(0);
}
