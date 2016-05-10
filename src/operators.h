#ifndef OPERATORS_H_
#define OPERATORS_H_


typedef enum {
    RAISE  = -1,
    NUMBER = 0,
    LOWER  = 1,
    VEC    = 2
  } op_type;

struct operator;

typedef struct operator{

  int     n_before;
  int     my_levels;
  op_type my_op_type;
  int     positions; /* For vec operators only */

  struct operator *dag; /* Raising operator */
  struct operator *n;   /* number operator */
  
} *operator;

/* Treat vec_op as an array of operators */
typedef vec_op operator[];

void create_op(int,operator);
void add_to_ham(double,operator);
void add_to_ham_comb(double,operator,operator);
void add_lin(double,operator);
void print_ham();
int nid;

#endif
