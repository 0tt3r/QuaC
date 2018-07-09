#include "quantum_gates.h"
#include "quac_p.h"
#include <stdlib.h>
#include <stdio.h>
#include <petsc.h>
#include <stdarg.h>
#include "qasm_parser.h"
#include "error_correction.h"
#include "dm_utilities.h"
#include <ctype.h>

void quil_read(char filename[],PetscInt *num_qubits,circuit *circ){
  FILE *fp;
  int ch = 0,lines=0,finished_allocate=0,num_pragma=0,this_qubit,max_qubit=-1;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  PetscReal time=1.0;

  fp = fopen(filename,"r");

  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in projectq_qasm_read!\n");
    }
  }

  //Count number of lines
  while(!feof(fp)){
    ch = fgetc(fp);
    if(ch == '\n'){
      lines++;
    }
  }
  //Rewind file
  rewind(fp);

  *num_qubits = 0;
  num_pragma  = 0;
  this_qubit  = -1;
  while ((read = getline(&line, &len, fp)) != -1){
    if (strstr(line,"PRAGMA")){
      //We will skip pragma lines
      num_pragma = num_pragma + 1;
    } else {
      this_qubit = line[strlen(line) - 2] - '0';
      if (this_qubit>max_qubit){
        max_qubit = this_qubit;
      }
    }
  }
  *num_qubits = max_qubit+1;

  lines = lines - num_pragma;

  create_circuit(circ,lines);
  rewind(fp); //Rewind the file (again)

  while ((read = getline(&line, &len, fp)) != -1){
    if (strstr(line,"PRAGMA")){
      //We will skip pragma lines
    } else if (strlen(line)>1){
      _quil_add_gate(line,circ,time);
      time = time + 1.0;
    }
  }

  fclose(fp);
  if (line) free(line);
  return;
}

void _quil_get_angle_pi(char angle_pi[32],PetscReal *angle){
  char numerator[32],denominator[32];
  int found_denom=1,i_n,i_d,i,denom,numer,factor;
  //Gate was printed with 'pi' or 'pi/2'
  angle_pi[strlen(angle_pi)-1] = 0;
  factor = 1;
  //Search for digits
  i_d = 0;   i_n = 0;
  denom = 1; numer = 1;
  for (i=0;angle_pi[i] != '\0'; i++){
    if (angle_pi[i]=='-'){
      //Found negative sign
      factor = -1;
    } else if (angle_pi[i]=="/") {
      // Found division; all digits after this
      // belong to the denominator
      found_denom = 1;
    } if (isdigit(angle_pi[i])){
      if (found_denom==0){
        //Numerator
        numerator[i_n] = angle_pi[i];
        i_n = i_n + 1;
      } else {
        //Denominator
        denominator[i_n] = angle_pi[i];
        i_d = i_d + 1;
      }
    }
  }
  if (i_d>0){
    denom = atoi(denominator);
  }
  if (i_n>0){
    numer = atoi(numerator);
  }
  *angle = factor * numer * PETSC_PI / denom;
  return;
}


void _quil_add_gate(char *line,circuit *circ,PetscReal time){
  char *token=NULL,*ptr=NULL;
  const char s[2] = " ";
  char angle_pi[32];
  int qubit1=-1,qubit2=-1;
  size_t i,j;
  PetscReal angle;
  gate_type my_gate_type;
  // Split string on " " to separate the gate and the qubits
  while (token=strsep(&line," ")) {
    for (i=0, j=0; token[j]=token[i]; j+=!isspace(token[i++]));
    //FIXME: Not exhaustive
    if (isdigit(token[0])){
      // check qubit numbers
      if (qubit1<0){
        qubit1 = atoi(token);
      } else {
        qubit2 = atoi(token);
      }
    } else {
      // gate types
      if (strcmp(token,"CNOT")==0){
        my_gate_type = CNOT;
      } else if (strcmp(token,"I")==0){
        my_gate_type = EYE;
      } else if (strcmp(token,"CZ")==0){
        my_gate_type = CZ;;
      } else if (strcmp(token,"H")==0) {
        my_gate_type = HADAMARD;
      } else if (strcmp(token,"Z")==0) {
        my_gate_type = SIGMAZ;
      } else if (strcmp(token,"X")==0) {
        my_gate_type = SIGMAX;
      } else if (strcmp(token,"Y")==0) {
        my_gate_type = SIGMAY;
      } else if (strstr(token,"RX")) {
        my_gate_type = RX;
        if (strstr(token,"pi")){
          sscanf(token,"RX(%s)",&angle_pi);
          _quil_get_angle_pi(angle_pi,&angle);
        } else {
          //Not sure if quil prints like this
          sscanf(token,"RX(%lf)",&angle);
        }
      } else if (strstr(token,"RY")) {
        my_gate_type = RY;
        if (strstr(token,"pi")){
          sscanf(token,"RY(%s)",&angle_pi);
          _quil_get_angle_pi(angle_pi,&angle);
        } else {
          //Not sure if quil prints like this
          sscanf(token,"RY(%lf)",&angle);
        }
      } else if (strstr(token,"RZ")) {
        if (strstr(token,"pi")){
          sscanf(token,"RZ(%s)",&angle_pi);
          _quil_get_angle_pi(angle_pi,&angle);
        } else {
          //Not sure if quil prints like this
          sscanf(token,"RZ(%lf)",&angle);
        }
        my_gate_type = RZ;
      }
    }
  }

  if (qubit2>0){
    //Multiqubit gate
    add_gate_to_circuit(circ,time,my_gate_type,qubit1,qubit2);
  } else {
    //Single qubit gate
    if (my_gate_type==6||my_gate_type==7||my_gate_type==8){
      //Rotation gate
      add_gate_to_circuit(circ,time,my_gate_type,qubit1,angle);
    } else {
      add_gate_to_circuit(circ,time,my_gate_type,qubit1);
    }
  }
  return;
}

void projectq_qasm_read(char filename[],PetscInt *num_qubits,circuit *circ){
  FILE *fp;
  int ch = 0,lines=0,finished_allocate=0;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  PetscReal time=1.0;

  fp = fopen(filename,"r");

  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in projectq_qasm_read!\n");
    }
  }

  //Count number of lines
  while(!feof(fp)){
    ch = fgetc(fp);
    if(ch == '\n')
      {
        lines++;
      }
  }
  //Rewind file
  rewind(fp);

  *num_qubits = 0;
  while ((read = getline(&line, &len, fp)) != -1){
    if (!finished_allocate){
      if (strstr(line,"Allocate")){
        //Get number of qubits by reading number of lines with 'Allocate'
        *num_qubits = *num_qubits+1;
      } else {
        /*Subtract off the allocate lines to get the total number
         * of gates.
         * Factor of 2 because the end has DEALLOCATE statements
         * WARNING! Comments will be included, too - not too big of a problem,
         * just will overallocate a bit */
        lines = lines - 2*(*num_qubits);
        //Allocate the circuit
        create_circuit(circ,lines);
        finished_allocate = 1;
        //Add the first gate to list
        _projectq_qasm_add_gate(line,circ,time);
        time = time + 1.0;
        //        gate_list[0]
      }
    } else if (strstr(line,"Deallocate")){
      //Breakout since we've finished going through the circuit
      break;
    } else {
      _projectq_qasm_add_gate(line,circ,time);
      time = time + 1.0;
    }

  }

  fclose(fp);
  if (line) free(line);
  return;
}


void _projectq_qasm_add_gate(char *line,circuit *circ,PetscReal time){
  char *token=NULL;
  int qubit1,qubit2;
  PetscReal angle;
  gate_type my_gate_type;
  size_t i,j;
  // Split string on | to separate gate type and qubits
  while (token=strsep(&line,"|")) {
    //Strip whitespace
    for (i=0, j=0; token[j]=token[i]; j+=!isspace(token[i++]));
    //Do direct strcmp for some, strstr for others
    //FIXME: Not exhaustive
    if (strstr(token,"Qureg")){
      // qubit numbers
      if (my_gate_type<0){
        //Multiqubit gate
        sscanf(token,"(Qureg[%d],Qureg[%d])",&qubit1,&qubit2);
        add_gate_to_circuit(circ,time,my_gate_type,qubit1,qubit2);
      } else {
        //Single qubit gate
        sscanf(token,"Qureg[%d]",&qubit1);
        if (my_gate_type==6||my_gate_type==7||my_gate_type==8){
          //Rotation gate
          add_gate_to_circuit(circ,time,my_gate_type,qubit1,angle);
        } else {
          add_gate_to_circuit(circ,time,my_gate_type,qubit1);
        }
      }
    } else {
      // gate types
      if (strcmp(token,"CX")==0){
        my_gate_type = CNOT;
      } else if (strcmp(token,"H")==0) {
        my_gate_type = HADAMARD;
      } else if (strcmp(token,"Z")==0) {
        my_gate_type = SIGMAZ;
      } else if (strcmp(token,"X")==0) {
        my_gate_type = SIGMAX;
      } else if (strcmp(token,"Y")==0) {
        my_gate_type = SIGMAY;
      } else if (strstr(token,"Rx")) {
        my_gate_type = RX;
        sscanf(token,"Rx(%lf)",&angle);
      } else if (strstr(token,"Ry")) {
        my_gate_type = RY;
        sscanf(token,"Ry(%lf)",&angle);
      } else if (strstr(token,"Rz")) {
        sscanf(token,"Rz(%lf)",&angle);
        my_gate_type = RZ;
      }
    }
  }
}

void projectq_vqe_get_expectation(char filename[],Vec rho,PetscScalar *trace_val){
  FILE *fp;
  char *token=NULL,*token2=NULL;
  char *line = NULL,gate_char;
  size_t len = 0,i,j;
  ssize_t read;
  int token_number,num_ops,qubit_number;
  operator ops[100];
  PetscReal scalar_multiply;
  PetscScalar temp_trace_val;
  fp = fopen(filename,"r");
  *trace_val = 0.0;
  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in projectq_vqe_get_expectation!\n");
    }
  }

  while ((read = getline(&line, &len, fp)) != -1){
    token_number = 0;
    while (token=strsep(&line,"[")) {
      if(token_number==0){
        //Strip whitespace
        for (i=0, j=0; token[j]=token[i]; j+=!isspace(token[i++]));
        scalar_multiply = atof(token);
        token_number = 1;
      } else {
        token2=strsep(&token,"]");
        num_ops = 0;
        while (token=strsep(&token2," ")) {
          if (strcmp(token,"")==0){
            *trace_val = *trace_val + scalar_multiply;
          } else {
            //Assume qubit number in file is global system number
            //FIXME: Put logical->physical qubit mapping here
            sscanf(token,"%c%d",&gate_char,&qubit_number);
            if (gate_char=='X'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_x;
            } else if (gate_char=='Y'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_y;
            } else if (gate_char=='Z'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_z;
            }
            num_ops = num_ops + 1;
          }
        }
        if (num_ops!=0){
          //This is a hack where I pass many ops, even though many of them
          //may not exist or be valid; they won't be accessed, at least.
          //Consider passing the array instead, and iterating inside?
          get_expectation_value(rho,&temp_trace_val,num_ops,
                                ops[0],ops[1],ops[2],ops[3],
                                ops[4],ops[5],ops[6],ops[7],
                                ops[8],ops[9],ops[10],ops[11],
                                ops[12],ops[13],ops[14],ops[15],
                                ops[16],ops[17],ops[18],ops[19]);
          temp_trace_val = temp_trace_val * scalar_multiply;
          *trace_val = *trace_val + temp_trace_val;
        }
      }
    }
  }
  return;
}

void projectq_vqe_get_expectation_squared(char filename[],Vec rho,PetscScalar *trace_val){
  FILE *fp;
  char *token=NULL,*token2=NULL;
  char *line = NULL,gate_char;
  size_t len = 0,i,j;
  ssize_t read;
  int token_number,num_ops,qubit_number;
  operator ops[100];
  PetscReal scalar_multiply;
  PetscScalar temp_trace_val;
  fp = fopen(filename,"r");
  *trace_val = 0.0;
  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in projectq_vqe_get_expectation!\n");
    }
  }

  while ((read = getline(&line, &len, fp)) != -1){
    token_number = 0;
    while (token=strsep(&line,"[")) {
      if(token_number==0){
        //Strip whitespace
        for (i=0, j=0; token[j]=token[i]; j+=!isspace(token[i++]));
        scalar_multiply = atof(token);
        token_number = 1;
      } else {
        token2=strsep(&token,"]");
        num_ops = 0;
        while (token=strsep(&token2," ")) {
          if (strcmp(token,"")==0){
            *trace_val = *trace_val + scalar_multiply;
          } else {
            //Assume qubit number in file is global system number
            //FIXME: Put logical->physical qubit mapping here
            sscanf(token,"%c%d",&gate_char,&qubit_number);
            if (gate_char=='X'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_x;
            } else if (gate_char=='Y'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_y;
            } else if (gate_char=='Z'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_z;
            }
            num_ops = num_ops + 1;
          }
        }
        if (num_ops!=0){
          //This is a hack where I pass many ops, even though many of them
          //may not exist or be valid; they won't be accessed, at least.
          //Consider passing the array instead, and iterating inside?
          get_expectation_value(rho,&temp_trace_val,num_ops,
                                ops[0],ops[1],ops[2],ops[3],
                                ops[4],ops[5],ops[6],ops[7],
                                ops[8],ops[9],ops[10],ops[11],
                                ops[12],ops[13],ops[14],ops[15],
                                ops[16],ops[17],ops[18],ops[19]);
          temp_trace_val = temp_trace_val * scalar_multiply;
          *trace_val = *trace_val + temp_trace_val;
        }
      }
    }
  }
  return;
}

void projectq_vqe_get_expectation_encoded(char filename[],Vec rho,PetscScalar *trace_val,
                                          PetscInt num_encoders,...){
  FILE *fp;
  char *token=NULL,*token2=NULL;
  char *line = NULL,gate_char;
  size_t len = 0,i_s,j;
  ssize_t read;
  int token_number,num_ops,qubit_number;
  va_list ap;
  operator ops[100];
  PetscInt i;
  PetscReal scalar_multiply;
  PetscScalar temp_trace_val;
  encoded_qubit encoders[50]; //50 is more systems than we will be able to do

  va_start(ap,num_encoders);
  for (i=0;i<num_encoders;i++){
    encoders[i] = va_arg(ap,encoded_qubit);
  }

  fp = fopen(filename,"r");
  *trace_val = 0.0;
  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in projectq_vqe_get_expectation!\n");
    }
  }

  while ((read = getline(&line, &len, fp)) != -1){
    token_number = 0;
    while (token=strsep(&line,"[")) {
      if(token_number==0){
        //Strip whitespace
        for (i_s=0, j=0; token[j]=token[i]; j+=!isspace(token[i++]));
        scalar_multiply = atof(token);
        token_number = 1;
      } else {
        token2=strsep(&token,"]");
        num_ops = 0;
        while (token=strsep(&token2," ")) {
          if (strcmp(token,"")==0){
            *trace_val = *trace_val + scalar_multiply;
          } else {
            //Assume qubit number in file is global system number
            //FIXME: Put logical->physical qubit mapping here
            sscanf(token,"%c%d",&gate_char,&qubit_number);
            qubit_number = encoders[qubit_number].qubits[0];
            if (gate_char=='X'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_x;
            } else if (gate_char=='Y'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_y;
            } else if (gate_char=='Z'){
              ops[num_ops] = subsystem_list[qubit_number]->sig_z;
            }
            num_ops = num_ops + 1;
          }
        }
        if (num_ops!=0){
          if (num_ops>4){
            if (nid==0) {
              printf("ERROR! vqe_get_expectation only supports 4 ops for now\n");
              exit(0);
            }
          }
          get_expectation_value(rho,&temp_trace_val,num_ops,ops[0],ops[1],ops[2],ops[3]);
          temp_trace_val = temp_trace_val * scalar_multiply;
          *trace_val = *trace_val + temp_trace_val;
        }
      }
    }
  }
  return;
}
