CFLAGS	         =

ODIR=obj
SRCDIR=src

include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules

_DEPS = operators.h solver.h operators_p.h quac.h quac_p.h kron_p.h
DEPS  = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

_OBJ  = quac.o operators.o solver.o kron.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	@mkdir -p $(@D)
	${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES} 

nv_cooling_7state: $(ODIR)/nv_cooling_7state.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

nv_mech_polarization: $(ODIR)/nv_mech_polarization.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

qd_plasmon: $(ODIR)/qd_plasmon.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

nv_cooling_2state_tc_test: $(ODIR)/nv_cooling_2state_tc_test.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

nv_cooling_2state: $(ODIR)/nv_cooling_2state.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

simple_jc_test: $(ODIR)/simple_jc_test.o $(OBJ) $(DEPS)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES}

simple_jc_test_vec: $(ODIR)/simple_jc_test_vec.o $(OBJ) $(DEPS)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES}

.PHONY: clean

clean:
	rm -f $(OBJ)
	rm -f simple_jc_test nv_cooling_2state
