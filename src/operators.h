#ifndef OPERATORS_H_
#define OPERATORS_H_

#include "operators_p.h"

struct operator;

typedef struct operator{
  double  initial_pop;
  PetscInt     initial_exc;
  PetscInt     n_before;
  PetscInt     my_levels;
  op_type my_op_type;
  PetscInt pos_in_sys_hspace; //Position in Hilbert space
  /* For ladder operators only */
  struct operator *dag;
  struct operator *n;
  struct operator *sig_x;
  struct operator *sig_y;
  struct operator *sig_z;
  struct operator *eye;
  struct operator *other;
  /* For vec operators only */
  PetscInt     position;
  /* Stores a pointer to the top of the list. Used in vec[0] only*/
  struct operator **vec_op_list;

  //Eigenvalue/vector information, used for projective measurements and only defined in the local space
  PetscScalar evals[2]; //2 is hardcoded because these are pauli matrices
  PetscScalar evecs[2][2];
} *operator;

typedef operator *vec_op; /* Treat vec_op as an array of operators  */

typedef struct time_dep_struct{
  double (*time_dep_func)(double);
  operator *ops;
  int num_ops;
  Mat mat;
} time_dep_struct;


void create_op(int,operator*);
void create_vec(int,vec_op*);

void no_lindblad_terms();

void add_to_ham(PetscScalar,operator);
void add_to_ham_stiff(PetscScalar,operator);
void add_to_ham_time_dep(double(*pulse)(double),int,...);
void add_to_ham_mult2(PetscScalar,operator,operator);
void add_to_ham_stiff_mult2(PetscScalar,operator,operator);
void add_to_ham_mult3(PetscScalar,operator,operator,operator);
int  _check_op_type2(operator,operator);
int  _check_op_type3(operator,operator,operator);
void add_lin(PetscScalar,operator);
void add_lin_mat(PetscScalar,Mat);
void add_lin_mult2(PetscScalar,operator,operator);
void print_dense_ham();
void set_initial_pop(operator,double);
void combine_ops_to_mat(Mat*,PetscInt,int,...);


extern int nid; /* a ranks id */
extern int np; /* number of processors */
#define MAX_SUB 100  //Consider making this not a define
extern operator subsystem_list[MAX_SUB];

extern time_dep_struct _time_dep_list[MAX_SUB];
extern time_dep_struct _time_dep_list_lin[MAX_SUB];

#endif
