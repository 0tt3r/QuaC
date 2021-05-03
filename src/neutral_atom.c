#include "neutral_atom.h"
#include "qvec_utilities.h"
#include "operators_p.h"
#include "operators.h"
#include "quantum_circuits.h"
#include "kron_p.h"
#include <stdlib.h>
#include <stdio.h>
#include <petscblaslapack.h>
#include <string.h>



void apply_1q_na_gate_to_qvec(qvec q,gate_type this_gate,operator op){
  PetscScalar **gate_mat,op_vals[2];
  PetscInt i,j,these_js[2],num_js;
  qsystem qsys;
  struct quantum_gate_struct gate;
  operator qb_op;

  gate_mat = (PetscScalar **)malloc(3*sizeof(PetscScalar*));
  for(i=0;i<3;i++){ //3 is hard coded because of neutral atom 3 lvl system
    gate_mat[i] = malloc(3*sizeof(PetscScalar));
  }

  //Make a fake single qubit system
  initialize_system(&qsys);
  create_op_sys(qsys,2,&qb_op);

  //Fill up the gate mat with zeros initially
  for(i=0;i<3;i++){
    for(j=0;j<3;j++){
      gate_mat[i][j] = 0.0;
    }
  }
  _initialize_gate_function_array_sys();
  //Make a fake gate
  gate.qubit_numbers = malloc(1*sizeof(int));
  gate.qubit_numbers[0] = 0; //Single qubit, so qubit num is 0
  gate.my_gate_type=this_gate;

  //Get qubit matrix values
  for(i=0;i<2;i++){
    _get_val_j_functions_gates_sys[this_gate+_min_gate_enum](qsys,i,gate,&num_js,&these_js,&op_vals,TENSOR_IG);
    for(j=0;j<num_js;j++){
      gate_mat[i][these_js[j]] = op_vals[j];
    }
  }

  apply_op_to_qvec_mat(q,gate_mat,op);

  for(i=0;i<3;i++){
    free(gate_mat[i]);
  }
  free(gate_mat);
  destroy_system(&qsys);
  return;

}

void apply_projective_measurement_tensor(qvec q,PetscScalar *meas_val,PetscInt num_ops,...){
  va_list ap;
  PetscInt i;
  operator *ops;

  ops = malloc(num_ops*sizeof(struct operator));

  va_start(ap,num_ops);
  for (i=0;i<num_ops;i++){
    ops[i] = va_arg(ap,operator);
  }
  va_end(ap);

  apply_projective_measurement_tensor_list(q,meas_val,num_ops,ops);
  return;
}

void apply_projective_measurement_tensor_list(qvec q,PetscScalar *meas_val,PetscInt num_ops,operator *ops){
  PetscInt i,j,k,i_evec,*op_loc_list;
  PetscScalar probs[3],**evec_mat,this_meas_value;
  PetscReal rand_num,current_prob;
  qvec ptrace_dm;

  op_loc_list = malloc(q->n_ops*sizeof(PetscInt));
  evec_mat = (PetscScalar **)malloc(3*sizeof(PetscScalar*));
  for(i=0;i<3;i++){
    evec_mat[i] = malloc(3*sizeof(PetscScalar));
  }
  *meas_val = 1;
  //Loop over the ops
  for(i=0;i<num_ops;i++){
    //Ptrace down to just the system of interest
    k=0;
    for(j=0;j<q->n_ops-1;j++){
      // ptrace away every system EXCEPT i
      if(j==i){
        k=k+1;
      }
      op_loc_list[j] = k;

      k = k+1;
    }
    //ptrace_dm will be the 3x3 one qubit density matrix
    ptrace_over_list_qvec(q,q->n_ops-1,op_loc_list,&ptrace_dm);

    //Get probabilities of different states
    get_probs_pauli_1sys(ptrace_dm,ops[i],probs);

    //randomly pick one
    rand_num = sprng();

    current_prob = 0;
    for(j=0;j<3;j++){//3 is hardcoded because qubit + 'other'
      current_prob = current_prob + probs[j];
      if(rand_num<current_prob){
        i_evec = j;
        break;//we found the right one
      }
    }

    //Build the matrix M_i = |i><i|
    //Assuming Pauli and neutral atom system here ([q0,q1,other])
    for(j=0;j<3;j++){
      for(k=0;k<3;k++){
        evec_mat[j][k] = 0.0;
      }
    }

    //The leakage measurement mat
    if(i_evec==2){
      evec_mat[2][2] = 1.0;
      //We are leaked. There is a 2/3 chance of leak showing up as 1 and 1/3 of leak showing up as 0
      rand_num = sprng();
      if(rand_num>2.0/3.0){
        this_meas_value = -1; //check -1 here
      } else {
        this_meas_value = 1;
      }
    } else {
      //The
      for(j=0;j<2;j++){//3 hardcoded because neutral atom
        for(k=0;k<2;k++){
          evec_mat[j][k] = ops[i]->evecs[i_evec][j]*PetscConjComplex(ops[i]->evecs[i_evec][k]);
        }
      }
      this_meas_value = ops[i]->evals[i_evec];
    }


    apply_op_to_qvec_mat(q,evec_mat,ops[i]);
    //renormalize!!
    if(q->my_type==DENSITY_MATRIX){
      VecScale(q->data,1/probs[i_evec]);
    } else {
      VecScale(q->data,1/sqrt(probs[i_evec]));
    }

    //Record the measurement result
    *meas_val = *meas_val*this_meas_value;
    destroy_qvec(&ptrace_dm);
  }

  for(i=0;i<3;i++){
    free(evec_mat[i]);
  }
  free(evec_mat);

  return;
}


void get_probs_pauli_1sys(qvec dm,operator op,PetscScalar probs[3]){
  PetscInt i,j,k,i_evec;
  PetscScalar this_prob,Mij,rhojk,Mik,total_prob=0;
  //Assumes [0,1,r] ordering
  //Calculate tr(M * rho * M^\dag)
  //These are super, super small (size < 10, probably always), so we do them by hand, in serial

  //We know that the eigenvectors have only 2 nonzeros because we are only allowing pauli operators at the moment
  for(i_evec=0;i_evec<2;i_evec++){
    this_prob = 0;

    for(i=0;i<2;i++){//2 is hardcoded because of Pauli matrices
      for(j=0;j<2;j++){
        for(k=0;k<2;k++){
          Mij = op->evecs[i_evec][i]*PetscConjComplex(op->evecs[i_evec][j]);
          get_dm_element_qvec_local(dm,j,k,&rhojk);
          Mik = op->evecs[i_evec][i]*PetscConjComplex(op->evecs[i_evec][k]);
          //Tr(M*rho*M^\dag)_ii = Mij * rhojk * Mik*
          //          printf("Mij %f %f rhojk %f %f Mik* %f %f \n",Mij,rhojk,PetscConjComplex(Mik));
          //          printf("prod %f %f\n",Mij * rhojk * PetscConjComplex(Mik));
          this_prob = this_prob + Mij * rhojk * PetscConjComplex(Mik);
          //          printf("this prob: %f %f\n",this_prob);
        }
      }
    }
    //    printf("aaathis prob[ %d ]: %f %f\n",i_evec,this_prob);
    probs[i_evec] = this_prob;
    //    printf("probs[ %d ] = %f %f\n",i_evec,probs[i_evec]);
    total_prob = total_prob + this_prob;
  }

  //We assume we can only measure the qubit states - if we do not measure a qubit state, we assume it went somewhere else
  // which will be the evec(0,0,1,1,1,1,1,1,..)
  probs[2] = 1 - total_prob;
  return;
}

//Define time dependent pulses
PetscScalar omega_arp(PetscReal time,void *ctx){
  PetscScalar pulse_value;
  PetscReal tau,t1,t2,ts,p,a;
  PulseParams *pulse_params = (PulseParams*) ctx;   /* user-defined struct */


  t1 = pulse_params->length/4.0;
  t2 = 3*pulse_params->length/4.0;
  ts = pulse_params->stime;
  tau = 0.175*pulse_params->length;
  a = exp(-pow(t1,4)/pow(tau,4));
  if(time-ts>0 && time-ts<pulse_params->length/2){
    p = (exp(-pow((time-ts-t1),4)/pow(tau,4)) - a)/(1-a);
  } else if(time-ts>pulse_params->length/2 && time-ts<pulse_params->length){
    p = (exp(-pow((time-ts-t2),4)/pow(tau,4)) - a)/(1-a);
  } else{
    p = 0.0;
  }

  pulse_value = pulse_params->omega/2.0*p;

  return pulse_value;
}


PetscScalar delta_arp(PetscReal time,void *ctx){
  PetscScalar pulse_value;
  PulseParams *pulse_params = (PulseParams*) ctx;   /* user-defined struct */
  PetscReal sigma,p,ts;

  ts = pulse_params->stime;

  if(time-ts>0 && time-ts<pulse_params->length/2){
    p = sin(2*PETSC_PI*(time-ts + 3*pulse_params->length/4)/pulse_params->length);
  } else if(time-ts>pulse_params->length/2 && time-ts<pulse_params->length){
    p = sin(2*PETSC_PI*(time-ts + pulse_params->length/4)/pulse_params->length);
  } else {
    p=0.0;
  }
  pulse_value = pulse_params->delta * p;
  return pulse_value;
}

PetscScalar omega_sp(PetscReal time,void *ctx){
  PetscScalar pulse_value;
  PetscReal tau,dt,p,a,ts;
  PulseParams *pulse_params = (PulseParams*) ctx;   /* user-defined struct */

  dt = pulse_params->deltat;
  ts = pulse_params->stime;

  p=exp(-pow((time-ts-5*dt),2)/pow(dt,2));

  pulse_value = pulse_params->omega/2.0*p;

  return pulse_value;
}

