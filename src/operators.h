#ifndef OPERATORS_H_
#define OPERATORS_H_

#include "operators_p.h"
struct operator;

typedef struct operator{
  double  initial_pop;
  int     n_before;
  int     my_levels;
  op_type my_op_type;
  int     position; /* For vec operators only */
  struct operator *vec_array;
  struct operator *dag; /* Raising operator */
  struct operator *n;   /* number operator */
  
} *operator;

typedef operator *vec_op;
/* /\* Treat vec_op as an array of operators *\/ */


void create_op(int,operator*);
void create_vec(int,vec_op*);
void add_to_ham(double,operator);
void add_to_ham_mult2(double,operator,operator);
void add_to_ham_mult3(double,operator,operator,operator);
int  _check_op_type2(operator,operator);
int  _check_op_type3(operator,operator,operator);
void add_lin(double,operator);
void add_lin_mult2(double,operator,operator);
void print_dense_ham();
void set_initial_pop_op(operator,int);
extern int nid; /* a ranks id */
extern int np; /* number of processors */
#define MAX_SUB 100  //Consider making this not a define
extern operator subsystem_list[MAX_SUB];

#endif
