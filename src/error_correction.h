typedef struct stabilizer{
  int n_ops;
  operator* ops;
} stabilizer;

void build_recovery_lin(Mat*,operator,char[],int,...);
void add_lin_recovery(PetscScalar,operator,char[],int,...);
void create_stabilizer(stabilizer*,int,...);
void destroy_stabilizer(stabilizer*);
void _get_row_nonzeros(PetscScalar[],PetscInt[],PetscInt*,PetscInt,operator,char[],int,stabilizer[]);
void _get_this_i_and_val_from_stab(PetscInt*, PetscScalar*,stabilizer,char);
