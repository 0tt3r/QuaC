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
      exit(1);
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
      exit(1);
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
      exit(1);
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
      exit(1);
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
      exit(1);
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

void _qasm_get_angles_from_string(char angle_string[256],PetscReal *angle1,PetscReal *angle2,PetscReal *angle3){
  char numerator[256],denominator[256],*token=NULL,*token2=NULL;
  int found_denom=1,i_n,i_d,i,denom,numer,factor;
  PetscReal this_angle=0,mult=0,pos_neg=1;
  angle_string[strlen(angle_string)-1] = 0; //Remove trailing )
  //Assume either pi*num or num*pi or pi/num or num/pi, but not, say num*pi*num*num*pi, etc
  //First, split string on ',', in case this is a multi-angle gate
  i=0;

  while (token=strsep(&angle_string,",")) {
    //Token should be one angle, possibly with pi
    //Search for 'pi*'
    if (token[0]=='-'){
      token2 = token+1;
      pos_neg = -1;
    } else {
      token2 = token;
      pos_neg = 1;
    }
    if (strstr(token2,"pi*")){
      //We have a 'pi*' then some number
      sscanf(token2,"pi*%lf",&this_angle);
      this_angle = PETSC_PI * this_angle;
    } else if(strstr(token2,"*pi/")){
      sscanf(token2,"%lf*pi/%lf",&this_angle,&mult);
      this_angle = PETSC_PI * this_angle/mult;
    } else if(strstr(token2,"pi/")){
      //We have a 'pi/' then some number
      sscanf(token2,"pi/%lf",&this_angle);
      this_angle = PETSC_PI/(this_angle);
    } else if(strstr(token2,"*pi")){
      sscanf(token2,"%lf*pi",&this_angle);
      this_angle = PETSC_PI * this_angle;
    } else if(strstr(token2,"pi/")){
      //We have a '/pi' then some number -- Is this necessaty?
      sscanf(token2,"%lf/pi",&this_angle);
      this_angle = this_angle/PETSC_PI;
    } else if(strstr(token2,"pi")){
        //We have just 'pi'
        this_angle = PETSC_PI;
    } else {
      //No pi found, just read the angle
      sscanf(token2,"%lf",&this_angle);
    }

    if(i==0){
      *angle1 = pos_neg*this_angle;
    } else if(i==1){
      *angle2 = pos_neg*this_angle;
    } else if (i==2){
      *angle3 = pos_neg*this_angle;
    } else{
      PetscPrintf(PETSC_COMM_WORLD,"ERROR! Too many angle parameters in gate!\n");
      exit(9);
    }
    i++;
  }

  return;
}


void qiskit_qasm_read(char filename[],PetscInt *num_qubits,circuit *circ){
  FILE *fp;
  int ch = 0,lines=0,found_qubits=0,blank_lines=0,comment_lines=0;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  PetscReal time=1.0;

  fp = fopen(filename,"r");

  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in qiskit_qasm_read!\n");
      exit(1);
    }
  }

  //Count number of lines
  while(!feof(fp)){
    ch = fgetc(fp);
    if(ch == '\n'){
        lines++;
      }
    if (ch  ==  '\n'){
      if ((ch = fgetc(fp))  ==  '\n'){
        fseek(fp, -1, 1);
        blank_lines++;
      }
    }
  }
  fseek(fp, 0, 0);
  while ((ch = fgetc(fp)) != EOF){
    if (ch  ==  '/'){
      if ((ch = fgetc(fp))  ==  '/'){
        comment_lines++;
      }
    }
  }

  //Rewind file
  rewind(fp);
  //Subtract off 3 because of the header, subtract comments and blank lines, too
  lines = lines - 3 - blank_lines - comment_lines;
  //Allocate the circuit

  create_circuit(circ,lines);
  found_qubits = 0;
  *num_qubits = 0;
  while ((read = getline(&line, &len, fp)) != -1){
    if (strstr(line,"qreg")){
      //Get number of qubits by reading qreg line
      sscanf(line,"qreg qr[%d];",num_qubits);
      found_qubits = 1;
    } else if(found_qubits==1){
      //Skip lines that are only a newline - 'blank lines'
      //FIXME: Will fail if there are spaces somewhere, or if the first character is a space, etc
      if(!isspace(line[0])){
        //Add the first gate to list
        _qiskit_qasm_add_gate(line,circ,time,*num_qubits,1);
        time = time + 1.0;
      }
    } //else skip the header
  }

  fclose(fp);
  if (line) free(line);
  return;
}

void qiskit_tqasm_read(char filename[],PetscInt *num_qubits,circuit *circ){
  FILE *fp;
  int ch = 0,lines=0,found_qubits=0,blank_lines=0,comment_lines=0;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(filename,"r");

  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in qiskit_tqasm_read!\n");
      exit(1);
    }
  }

  //Count number of lines
  while(!feof(fp)){
    ch = fgetc(fp);
    if(ch == '\n'){
        lines++;
      }
    if (ch  ==  '\n'){
      if ((ch = fgetc(fp))  ==  '\n'){
        fseek(fp, -1, 1);
        blank_lines++;
      }
    }
  }
  fseek(fp, 0, 0);
  while ((ch = fgetc(fp)) != EOF){
    if (ch  ==  '/'){
      if ((ch = fgetc(fp))  ==  '/'){
        comment_lines++;
      }
    }
  }

  //Rewind file
  rewind(fp);
  //Subtract off 3 because of the header, subtract comments and blank lines, too
  //3 is not safe, e.g., if there is a creg
  lines = lines - 3 - blank_lines - comment_lines;
  //Allocate the circuit

  create_circuit(circ,lines);
  found_qubits = 0;
  *num_qubits = 0;
  while ((read = getline(&line, &len, fp)) != -1){
    if (strstr(line,"qreg")){
      //Get number of qubits by reading qreg line
      sscanf(line,"qreg qr[%d];",num_qubits);
      found_qubits = 1;
    } else if(found_qubits==1){
      //Skip lines that are only a newline - 'blank lines'
      //FIXME: Will fail if there are spaces somewhere, or if the first character is a space, etc
      if(!isspace(line[0])){
        //Add the first gate to list
        _qiskit_tqasm_add_gate(line,circ,*num_qubits,0);
      }
    } //else skip the header
  }

  fclose(fp);
  if (line) free(line);
  return;
}

void cirq_qasm_read(char filename[],PetscInt *num_qubits,circuit *circ){
  FILE *fp;
  int ch = 0,lines=0,found_qubits=0,blank_lines=0,comment_lines=0;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  PetscReal time=1.0;

  fp = fopen(filename,"r");

  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in qiskit_qasm_read!\n");
      exit(1);
    }
  }

  //Count number of lines
  while(!feof(fp)){
    ch = fgetc(fp);
    if(ch == '\n'){
        lines++;
      }
    if (ch  ==  '\n'){
      if ((ch = fgetc(fp))  ==  '\n'){
        fseek(fp, -1, 1);
        blank_lines++;
      }
    }
  }
  fseek(fp, 0, 0);
  while ((ch = fgetc(fp)) != EOF){
    if (ch  ==  '/'){
      if ((ch = fgetc(fp))  ==  '/'){
        comment_lines++;
      }
    }
  }

  //Rewind file
  rewind(fp);
  //Subtract off 3 because of the header, subtract comments and blank lines, too
  lines = lines - 3 - blank_lines - comment_lines;
  //Allocate the circuit

  create_circuit(circ,lines);
  found_qubits = 0;
  *num_qubits = 0;
  while ((read = getline(&line, &len, fp)) != -1){
    if (strstr(line,"qreg")){
      //Get number of qubits by reading qreg line
      sscanf(line,"qreg qr[%d];",num_qubits);
      found_qubits = 1;
    } else if(found_qubits==1){
      //Skip lines that are only a newline - 'blank lines'
      //FIXME: Will fail if there are spaces somewhere, or if the first character is a space, etc
      if(!isspace(line[0])){
        //Add the first gate to list
        _qiskit_qasm_add_gate(line,circ,time,*num_qubits,0);
        time = time + 1.0;
      }
    } //else skip the header
  }

  fclose(fp);
  if (line) free(line);
  return;
}

void _qiskit_qasm_add_gate(char *line,circuit *circ,PetscReal time,PetscInt num_qubits,PetscInt qiskit_ordering){
  char *token=NULL,pi_string[256];
  int qubit1=-1,qubit2=-1,skip_gate=0;
  PetscReal angle,angle2,angle3;
  gate_type my_gate_type=NULL_GATE;
  size_t i,j;
  //qiskit_ordering reorders the qubits so that the wavefunction the is printed out is consistent with qiskit
  // Split string on ' ' to separate gate type and qubits
  while (token=strsep(&line," ")) {
    //Strip whitespace
    for (i=0, j=0; token[j]=token[i]; j+=!isspace(token[i++]));
    //Do direct strcmp for some, strstr for others
    //FIXME: Not exhaustive
    if (strstr(token,"qr[")){ //can't have q named gates
      // qubit numbers
      if (skip_gate==0){
        if (my_gate_type==NULL_GATE){
          PetscPrintf(PETSC_COMM_WORLD,"ERROR! NULL_GATE type encounterd!\n");
          exit(0);
        } else if (my_gate_type<0){
          //Multiqubit gate
          sscanf(token,"qr[%d],qr[%d];",&qubit1,&qubit2);
          if(qiskit_ordering==1){
            qubit1 = num_qubits - qubit1 - 1;
            qubit2 = num_qubits - qubit2 - 1;
          }
          add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1,qubit2);
        } else {
          //Single qubit gate
          sscanf(token,"qr[%d];",&qubit1);
          if(qiskit_ordering==1){
            qubit1 = num_qubits - qubit1 - 1;
          }
          if (my_gate_type==U3||my_gate_type==U2||my_gate_type==U1){
            //U3 gate
            add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1,angle,angle2,angle3);
          } else if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ){
            //Single parameter rotations
            add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1,angle);
          } else {
            //No parameter gates (X, H, Y, Z, etc)
            add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1);
          }
        }
      } else {
        //Skipped a barrier
        skip_gate = 0;
      }
    } else {
      // gate types
      // FIXME: Only u1,u2,u3,cx,cz,h,rx,ry,rzfor now!
      // Assume pi was stripped out
      if (strcmp(token,"cx")==0){//strcmp because no angle
        my_gate_type = CNOT;
      } else if (strcmp(token,"cz")==0){//strcmp because no angle
        my_gate_type = CZ;
      } else if (strcmp(token,"h")==0){//strcmp because no angle
        my_gate_type = HADAMARD;
      } else if (strstr(token,"rx")) {//strstr because changing angle
        my_gate_type = RX;
        sscanf(token,"rx(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
      } else if (strstr(token,"ry")) {//strstr because changing angle
        my_gate_type = RY;
        sscanf(token,"ry(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
      } else if (strstr(token,"rz")) {//strstr because changing angle
        my_gate_type = RZ;
        sscanf(token,"rz(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
      } else if (strstr(token,"u1")) {//strstr because changing angle
        my_gate_type = U1;
        sscanf(token,"u1(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
        //        sscanf(token,"u1(%lf)",&angle3);
        //We read angle in first, but we want that to be angle3 because
        //of the definition of U1
        angle3 = angle;
        angle = 0;
        angle2 = 0;
      } else if (strstr(token,"u2")) {
        my_gate_type = U2;
        //u2(phi,lambda) = u3(pi/2,phi,lambda)
        //        sscanf(token,"u2(%lf,%lf)",&angle2,&angle3);
        sscanf(token,"u2(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
        //reassign angles to fit u3 definition
        angle3 = angle2;
        angle2 = angle;
        angle = PETSC_PI/2;
      } else if (strstr(token,"u3")) {
        my_gate_type = U3;
        sscanf(token,"u3(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
        /* sscanf(token,"u3(%lf,%lf,%lf)",&angle,&angle2,&angle3); */
      } else if (strstr(token,"x")) {
        my_gate_type = SIGMAX;
      } else if (strstr(token,"barrier")){
        //Skip barrier
        skip_gate = 1;
      } else {
        PetscPrintf(PETSC_COMM_WORLD,"%s\n",token);
        PetscPrintf(PETSC_COMM_WORLD,"ERROR! Gate type not recognized in qiskit_qasm!\n");
        exit(0);
      }
    }
  }

  return;
}

void _qiskit_tqasm_add_gate(char *line,circuit *circ,PetscInt num_qubits,PetscInt qiskit_ordering){
  char *token=NULL,pi_string[256];
  int qubit1=-1,qubit2=-1,skip_gate=0;
  PetscReal angle,angle2,angle3,time;
  gate_type my_gate_type=NULL_GATE;
  size_t i,j;
  //qiskit_ordering reorders the qubits so that the wavefunction the is printed out is consistent with qiskit
  // Split string on ' ' to separate gate type and qubits
  while (token=strsep(&line," ")) {
    //Strip whitespace
    for (i=0, j=0; token[j]=token[i]; j+=!isspace(token[i++]));
    //Do direct strcmp for some, strstr for others
    //FIXME: Not exhaustive
    if (strstr(token,"@")){ //The time the gate should be applied
      sscanf(token,"@%lf;",&time);
      if (skip_gate==0){
        if (my_gate_type==NULL_GATE){
          PetscPrintf(PETSC_COMM_WORLD,"ERROR! NULL_GATE type encounterd!\n");
          exit(0);
        } else if (my_gate_type<0){
          add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1,qubit2);
        } else {
          if (my_gate_type==U3||my_gate_type==U2||my_gate_type==U1){
            //U3 gate
            add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1,angle,angle2,angle3);
          } else if (my_gate_type==RX||my_gate_type==RY||my_gate_type==RZ){
            //Single parameter rotations
            add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1,angle);
          } else {
            //No parameter gates (X, H, Y, Z, etc)
            add_gate_to_circuit_sys(circ,time,my_gate_type,qubit1);
          }
        }
      } else {
        //Skipped a barrier
        skip_gate = 0;
      }

    } else if (strstr(token,"qr[")){ //can't have q named gates
      // qubit numbers
      if (skip_gate==0){
        if (my_gate_type==NULL_GATE){
          PetscPrintf(PETSC_COMM_WORLD,"ERROR! NULL_GATE type encounterd!\n");
          exit(0);
        } else if (my_gate_type<0){
          //Multiqubit gate
          sscanf(token,"qr[%d],qr[%d];",&qubit1,&qubit2);
          if(qiskit_ordering==1){
            qubit1 = num_qubits - qubit1 - 1;
            qubit2 = num_qubits - qubit2 - 1;
          }
        } else {
          //Single qubit gate
          sscanf(token,"qr[%d];",&qubit1);
          if(qiskit_ordering==1){
            qubit1 = num_qubits - qubit1 - 1;
          }
        }
      }
    } else {
      // gate types
      // FIXME: Only u1,u2,u3,cx,cz,h,rx,ry,rzfor now!
      // Assume pi was stripped out
      if (strcmp(token,"cx")==0){//strcmp because no angle
        my_gate_type = CNOT;
      } else if (strcmp(token,"cz")==0){//strcmp because no angle
        my_gate_type = CZ;
      } else if (strcmp(token,"h")==0){//strcmp because no angle
        my_gate_type = HADAMARD;
      } else if (strcmp(token,"id")==0){//strcmp because no angle
        my_gate_type = EYE;
       } else if (strstr(token,"rx")) {//strstr because changing angle
        my_gate_type = RX;
        sscanf(token,"rx(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
      } else if (strstr(token,"ry")) {//strstr because changing angle
        my_gate_type = RY;
        sscanf(token,"ry(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
      } else if (strstr(token,"rz")) {//strstr because changing angle
        my_gate_type = RZ;
        sscanf(token,"rz(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
      } else if (strstr(token,"u1")) {//strstr because changing angle
        my_gate_type = U1;
        sscanf(token,"u1(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
        //        sscanf(token,"u1(%lf)",&angle3);
        //We read angle in first, but we want that to be angle3 because
        //of the definition of U1
        angle3 = angle;
        angle = 0;
        angle2 = 0;
      } else if (strstr(token,"u2")) {
        my_gate_type = U2;
        //u2(phi,lambda) = u3(pi/2,phi,lambda)
        //        sscanf(token,"u2(%lf,%lf)",&angle2,&angle3);
        sscanf(token,"u2(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
        //reassign angles to fit u3 definition
        angle3 = angle2;
        angle2 = angle;
        angle = PETSC_PI/2;
      } else if (strstr(token,"u3")) {
        my_gate_type = U3;
        sscanf(token,"u3(%s)",&pi_string);
        _qasm_get_angles_from_string(pi_string,&angle,&angle2,&angle3);
        /* sscanf(token,"u3(%lf,%lf,%lf)",&angle,&angle2,&angle3); */
      } else if (strstr(token,"x")) {
        my_gate_type = SIGMAX;
      } else if (strstr(token,"barrier")){
        //Skip barrier
        skip_gate = 1;
      } else {
        PetscPrintf(PETSC_COMM_WORLD,"%s\n",token);
        PetscPrintf(PETSC_COMM_WORLD,"ERROR! Gate type not recognized in qiskit_qasm!\n");
        exit(0);
      }
    }
  }

  return;
}

void qiskit_vqe_get_expectation(char filename[],qvec rho,PetscScalar *trace_val,PetscInt *num_evs_r,PetscScalar **evs,qsystem sys){
  FILE *fp;
  char *token=NULL,*token2=NULL;
  char *line = NULL,gate_char;
  size_t len = 0,i,j;
  ssize_t read;
  int token_number,num_ops,qubit_number,num_evs=0;
  PetscScalar temp_ev_vals[1000]; //FIXME Temp over allocation
  operator ops[100];
  PetscReal scalar_multiply;
  PetscScalar temp_trace_val;
  fp = fopen(filename,"r");
  *trace_val = 0.0;
  if (fp == NULL){
    if (nid==0){
      printf("ERROR! File not found in qiskit_vqe_get_expectation!\n");
      exit(1);
    }
  }

  while ((read = getline(&line, &len, fp)) != -1){
    token_number = 0;
    while (token=strsep(&line,"\t")) {
      //Strip whitespace
      /* for (i=0, j=0; token[j]=token[i]; j+=!isspace(token[i++])); */
      if(token_number!=0){
        token2=strsep(&token,"+");
        token2++;
        //Scalar multiply before pauli string
        scalar_multiply = atof(token2);
      } else {
        num_ops = strlen(token);
        for (i=0;i<num_ops;i++){
          qubit_number = num_ops-i-1; //Might need to be num_ops - i because qiskit has reverse order
          gate_char = token[i];
          if (gate_char=='I'){
            ops[i] = sys->subsystem_list[qubit_number]->eye;
          } else if (gate_char=='X'){
            ops[i] = sys->subsystem_list[qubit_number]->sig_x;
          } else if (gate_char=='Y'){
            ops[i] = sys->subsystem_list[qubit_number]->sig_y;
          } else if (gate_char=='Z'){
            ops[i] = sys->subsystem_list[qubit_number]->sig_z;
          }
        }
        token_number = 1;
      }
    }
    /*
     * At this point, we have read in the scalar_multiply
     * and the operator. Now get the value.
     */
    if (num_ops!=0){
      //This is a hack where I pass many ops, even though many of them
      //may not exist or be valid; they won't be accessed, at least.
      //Consider passing the array instead, and iterating inside?
      /* get_expectation_value_qvec(rho,&temp_trace_val,num_ops,ops[0],ops[1],ops[2],ops[3], */
      /*                            ops[4],ops[5],ops[6],ops[7],ops[8],ops[9],ops[10],ops[11]); */
      get_expectation_value_qvec_list(rho,&temp_trace_val,num_ops,ops);
      temp_ev_vals[num_evs] = temp_trace_val;
      temp_trace_val = temp_trace_val * scalar_multiply;
      *trace_val = *trace_val + temp_trace_val;

      num_evs = num_evs + 1;

    }
  }
  *evs = malloc(num_evs*sizeof(PetscScalar));
  for(i=0;i<num_evs;i++){
    (*evs)[i] = temp_ev_vals[i];
  }
  *num_evs_r = num_evs;
  return;

}
