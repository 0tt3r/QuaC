#ifndef QUAC_H_
#define QUAC_H_

#define SIMPLE_SPRNG		/* simple interface                        */
#define USE_MPI			/* use MPI to find number of processes     */
#include "sprng.h"
void QuaC_initialize(int,char**);
void QuaC_finalize();
void QuaC_clear();
void destroy_op();
void destroy_vec();

#endif
