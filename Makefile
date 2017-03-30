CFLAGS =

TESTS=integration_tests dm_utilities_tests
ODIR=obj
SRCDIR=src
TESTDIR=tests
UNITYDIR=external/Unity/src
CPPFLAGS= -isystem $(UNITYDIR)
CPPFLAGS += -isystem $(SRCDIR)


include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules

_DEPS = quantum_gates.h dm_utilities.h operators.h solver.h operators_p.h quac.h quac_p.h kron_p.h
DEPS  = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

_OBJ  = quac.o operators.o solver.o kron.o dm_utilities.o quantum_gates.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_TEST_OBJ  = unity.o timedep_test.o
TEST_OBJ = $(patsubst %,$(ODIR)/%,$(_TEST_OBJ))

_TEST_DEPS = tests.h
TEST_DEPS  = $(patsubst %,$(SRCDIR)/%,$(_TEST_DEPS))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	@mkdir -p $(@D)
	${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES}

$(ODIR)/%.o: $(TESTDIR)/%.c $(DEPS) $(TEST_DEPS)
	@mkdir -p $(@D)
	${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES} $(CPPFLAGS)

$(ODIR)/%.o: $(UNITYDIR)/%.c
	@mkdir -p $(@D)
	${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES} $(CPPFLAGS)


all: coupled_qds quant_tele nv_cooling_7state nv_mech_polarization qd_plasmon nv_cooling_2state_tc_test rpurcell_osci rpurcell nv_cooling_2state simple_jc_test simple_jc_test_vec timedep_test

dm_utilities_tests: CFLAGS += -DUNIT_TEST
dm_utilities_tests: $(ODIR)/dm_utilities_tests.o $(OBJ) $(TEST_OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) $(CPPFLAGS) ${PETSC_KSP_LIB}
	-./$@

.PHONY: clean_test test

clean_test:
	rm -f $(TEST_OBJ)

integration_tests: CFLAGS += -DUNIT_TEST
integration_tests: $(ODIR)/integration_tests.o $(OBJ) $(TEST_OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) $(CPPFLAGS) ${PETSC_KSP_LIB}
	-./$@

test: clean_test $(TESTS)


nv_cooling_7state: $(ODIR)/nv_cooling_7state.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

nv_mech_polarization: $(ODIR)/nv_mech_polarization.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

qd_plasmon: $(ODIR)/qd_plasmon.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

timedep_test: $(ODIR)/timedep_test.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

nv_cooling_2state_tc_test: $(ODIR)/nv_cooling_2state_tc_test.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

rpurcell_osci: $(ODIR)/rpurcell_osci.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

rpurcell: $(ODIR)/rpurcell.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

nv_cooling_2state: $(ODIR)/nv_cooling_2state.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

simple_jc_test: $(ODIR)/simple_jc_test.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

simple_jc_test_vec: $(ODIR)/simple_jc_test_vec.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

coupled_qds: $(ODIR)/coupled_qds.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

quant_tele: $(ODIR)/quant_tele.o $(OBJ)
	-${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}


.PHONY: clean

clean:
	rm -f $(OBJ)
	rm -f nv_cooling_7state nv_mech_polarization qd_plasmon nv_cooling_2state_tc_test rpurcell_osci rpurcell nv_cooling_2state simple_jc_test simple_jc_test_vec timedep_test
