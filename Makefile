CFLAGS	         =


ODIR=obj
SRCDIR=src

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

_DEPS = operators.h solver.h operators_p.h
DEPS  = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

_OBJ  = operators.o solver.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB}

nv_cooling_2state: $(SRCDIR)/nv_cooling_2state.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

simple_jaynes_cummings: $(SRCDIR)/simple_jaynes_cummings.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

# .PHONY: clean

# clean:
# 	rm -f $(ODIR)/*.o
