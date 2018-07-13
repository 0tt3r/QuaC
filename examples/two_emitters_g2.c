
#include <math.h>
#include <stdlib.h>
#include "quac.h"
#include "operators.h"
#include "solver.h"
#include "dm_utilities.h"
#include "quantum_gates.h"
#include "petsc.h"

PetscErrorCode ts_monitor(TS,PetscInt,PetscReal,Vec,void*);
double pulse1(double);
double pulse2(double);

/* Declared globally so that we can access this in ts_monitor */
FILE *f_pop;
operator qd1,qd2;
double amp1,amp2,tp,td,u_qd,we1,we2,we_p,td2;
PetscScalar pe;
PetscInt i_g2,i_st,i_tau;
int tau_evolve;
PetscScalar **g2_values_2d_b,**g2_values_2d_1,**g2_values_2d_2;

int main(int argc,char **args){
  PetscReal gamma,dep,st_dt,st_max,previous_start_time,this_start_time;
  PetscReal dt,tau_t_max,tau_max,dt_tau,ave,total_pe;
  PetscScalar val,pe_l,ave1,ave2,aveb;
  PetscInt  steps_max,n_st,n_tau,i,j;
  Vec dm0,init_dm;
  PetscBool print_map = PETSC_FALSE;
  FILE *file1,*file2,*fileb;
  /* Initialize QuaC */
  QuaC_initialize(argc,args);

  td2 = 5000;
  amp1 = 2e-5;
  amp2 = 2e-5;
  dep = 0.0000735;
  gamma = 3.3e-3;
  we1  = 0.0735;
  we2  = 0.0735;
  we_p = 0.0735;
  PetscOptionsGetReal(NULL,NULL,"-td2",&td2,NULL);
  PetscOptionsGetReal(NULL,NULL,"-we1",&we1,NULL);
  PetscOptionsGetReal(NULL,NULL,"-we2",&we2,NULL);
  PetscOptionsGetReal(NULL,NULL,"-amp1",&amp1,NULL);
  PetscOptionsGetReal(NULL,NULL,"-amp2",&amp2,NULL);
  PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL);
  PetscOptionsGetReal(NULL,NULL,"-dep",&dep,NULL);
  PetscOptionsGetBool(NULL,NULL,"-map",&print_map,NULL);

  /* Define scalars to add to Ham */
  tp = 1000;
  td = 1800;
  u_qd = 3.93;

  n_tau = 4000; //was 2000
  tau_max = 7500;
  n_st = 4000; //was 2000
  st_max = 7500;

  g2_values_2d_b = (PetscScalar **)malloc((n_st+1)*sizeof(PetscScalar *));
  g2_values_2d_1 = (PetscScalar **)malloc((n_st+1)*sizeof(PetscScalar *));
  g2_values_2d_2 = (PetscScalar **)malloc((n_st+1)*sizeof(PetscScalar *));
  for (i=0;i<n_st+1;i++){
    g2_values_2d_b[i] = (PetscScalar *)malloc((n_tau+1)*sizeof(PetscScalar));
    g2_values_2d_1[i] = (PetscScalar *)malloc((n_tau+1)*sizeof(PetscScalar));
    g2_values_2d_2[i] = (PetscScalar *)malloc((n_tau+1)*sizeof(PetscScalar));
    for (j=0;j<n_tau+1;j++){
      g2_values_2d_b[i][j] = 0.0;
      g2_values_2d_1[i][j] = 0.0;
      g2_values_2d_2[i][j] = 0.0;
    }
  }
  create_op(2,&qd1);
  create_op(2,&qd2);

  /* Add terms to the hamiltonian */
  add_to_ham_p(we1,1,qd1->n);
  add_to_ham_p(we2,1,qd2->n);

  add_to_ham_time_dep_p(pulse1,1,qd1->dag);
  add_to_ham_time_dep_p(pulse1,1,qd1);

  add_to_ham_time_dep_p(pulse2,1,qd2->dag);
  add_to_ham_time_dep_p(pulse2,1,qd2);

  add_lin(gamma,qd1);
  add_lin(dep,qd1->n);

  add_lin(gamma,qd2);
  add_lin(dep,qd2->n);

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

  //Starting from 0, we time step to the next 'start time' to get the
  //Initial dm for our tau sweep
  st_dt = st_max/n_st;
  previous_start_time = 0;
  i_st = 0;
  i_st = i_st + 1;
  for (this_start_time=st_dt;this_start_time<=st_max;this_start_time+=st_dt){
    //Go from previous start time to this_start_time
    tau_evolve = 0;
    dt = (this_start_time - previous_start_time)/2000;
    time_step(dm0,previous_start_time,this_start_time,dt,steps_max);

    //Get the expectation value of q->dag q for this init_dm
    get_expectation_value(dm0,&pe_l,1,qd1->n);
    printf("pe_l1: %e %e\n",pe_l);
    pe = pe_l;
    get_expectation_value(dm0,&pe_l,1,qd2->n);
    pe = pe + pe_l;
    total_pe = total_pe + PetscRealPart(pe);

    //Timestep through taus
    tau_t_max = this_start_time + tau_max;
    dt_tau = (tau_t_max - this_start_time)/n_tau;

    /*
     * Force an emission to get \sig_1 \rho \sig_1^\dag terms
     */
    tau_evolve = 1;
    //Copy the timestepped dm into our init_dm for tau sweep
    VecCopy(dm0,init_dm);
    mult_dm_left_right(init_dm,qd1,qd1);
    i_tau = 0;
    time_step(init_dm,this_start_time,tau_t_max,dt_tau,steps_max);

    /*
     * Force an emission to get \sig_2 \rho \sig_2^\dag terms
     */
    tau_evolve = 2;
    VecCopy(dm0,init_dm);
    mult_dm_left_right(init_dm,qd2,qd2);
    i_tau = 0;
    time_step(init_dm,this_start_time,tau_t_max,dt_tau,steps_max);

    /*
     * Now get terms from \sig_1 \rho \sig_2^\dag
     */
    tau_evolve = 3;
    VecCopy(dm0,init_dm);
    mult_dm_left_right(init_dm,qd1,qd2);
    i_tau = 0;
    time_step(init_dm,this_start_time,tau_t_max,dt_tau,steps_max);

    /*
     * Now get terms from \sig_2 \rho \sig_1^\dag
     */
    tau_evolve = 3;
    VecCopy(dm0,init_dm);
    mult_dm_left_right(init_dm,qd2,qd1);
    i_tau = 0;
    time_step(init_dm,this_start_time,tau_t_max,dt_tau,steps_max);

    previous_start_time = this_start_time;
    i_st = i_st + 1;
  }
  //Calculate average of g2
  ave2 = 0;
  ave1 = 0;
  aveb = 0;
  for (i_st=0;i_st<(n_st+1);i_st++){
    for (i_tau=0;i_tau<(n_tau+1);i_tau++){
      ave1 = ave1 + g2_values_2d_1[i_st][i_tau];
      ave2 = ave2 + g2_values_2d_2[i_st][i_tau];
      aveb = aveb + g2_values_2d_b[i_st][i_tau];
    }
  }
  ave1 = ave1/((n_tau+1)*(n_st+1));
  ave2 = ave2/((n_tau+1)*(n_st+1));
  aveb = aveb/((n_tau+1)*(n_st+1));


  printf("ave1: %e %e ave2: %e %e aveb: %e %e pe: %e\n",ave1,ave2,aveb,total_pe/i_st);

  if (print_map){
    file1 = fopen("map1.dat","w");
    file2 = fopen("map2.dat","w");
    fileb = fopen("mapb.dat","w");
    for (i_st=0;i_st<(n_st+1);i_st++){
      for (i_tau=0;i_tau<(n_tau+1);i_tau++){
        fprintf(file1,"%e %e\n",g2_values_2d_1[i_st][i_tau]);
        fprintf(file2,"%e %e\n",g2_values_2d_2[i_st][i_tau]);
        fprintf(fileb,"%e %e\n",g2_values_2d_b[i_st][i_tau]);
      }
    }
    fclose(file1);
    fclose(file2);
    fclose(fileb);
  }

  destroy_op(&qd1);
  destroy_op(&qd2);
  destroy_dm(dm0);
  destroy_dm(init_dm);
  QuaC_finalize();
  return 0;
}

double pulse1(double t){
  double pulse_value;
  pulse_value = (amp1*exp(-2 * log(2) * pow((t-td)/(tp),2)) * cos(we_p*(t-td))
                 + amp1*exp(-2*log(2)*pow((t-td2)/(tp),2))*cos(we_p*(t-td2)));
  pulse_value = -u_qd * pulse_value;
  return pulse_value;
}

double pulse2(double t){
  double pulse_value;
  pulse_value = (amp2*exp(-2 * log(2) * pow((t-td)/(tp),2)) * cos(we_p*(t-td))
                 + amp2*exp(-2*log(2)*pow((t-td2)/(tp),2))*cos(we_p*(t-td2)));
  pulse_value = -u_qd * pulse_value;
  return pulse_value;
}

PetscErrorCode ts_monitor(TS ts,PetscInt step,PetscReal time,Vec dm,void *ctx){
  PetscScalar ev1,ev2,ev3,ev4,tmp;
  if (tau_evolve==1){
    get_expectation_value(dm,&ev1,1,qd1->n);
    get_expectation_value(dm,&ev2,1,qd2->n);
    get_expectation_value(dm,&ev3,2,qd1->dag,qd2);
    get_expectation_value(dm,&ev4,2,qd2->dag,qd1);
    tmp =(ev1 + ev2 + ev3 + ev4);
    g2_values_2d_b[i_st][i_tau] += tmp;
    g2_values_2d_1[i_st][i_tau] += ev1;
    i_tau = i_tau + 1;
  } else if (tau_evolve==2){
    get_expectation_value(dm,&ev1,1,qd1->n);
    get_expectation_value(dm,&ev2,1,qd2->n);
    get_expectation_value(dm,&ev3,2,qd1->dag,qd2);
    get_expectation_value(dm,&ev4,2,qd2->dag,qd1);
    tmp =(ev1 + ev2 + ev3 + ev4);
    g2_values_2d_b[i_st][i_tau] += tmp;
    g2_values_2d_2[i_st][i_tau] += ev2;
    i_tau = i_tau + 1;
  } else if (tau_evolve==3){
    get_expectation_value(dm,&ev1,1,qd1->n);
    get_expectation_value(dm,&ev2,1,qd2->n);
    get_expectation_value(dm,&ev3,2,qd1->dag,qd2);
    get_expectation_value(dm,&ev4,2,qd2->dag,qd1);
    tmp =(ev1 + ev2 + ev3 + ev4);
    g2_values_2d_b[i_st][i_tau] += tmp;
    i_tau = i_tau + 1;
  }
  PetscFunctionReturn(0);
}
