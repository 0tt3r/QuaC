CFLAGS =


ODIR=obj
SRCDIR=src
EXAMPLESDIR=examples
EXAMPLES=$(basename $(notdir $(wildcard $(EXAMPLESDIR)/*.c)))
TESTDIR=tests
TESTS=$(basename $(notdir $(wildcard $(TESTDIR)/*test*.c)))
CFLAGS += -isystem $(SRCDIR)

include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules

_DEPS = quantum_gates.h dm_utilities.h operators.h solver.h operators_p.h quac.h quac_p.h kron_p.h
DEPS  = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

_OBJ  = quac.o operators.o solver.o kron.o dm_utilities.o quantum_gates.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_TEST_OBJ  = unity.o timedep_test.o imag_ham.o
TEST_OBJ = $(patsubst %,$(ODIR)/%,$(_TEST_OBJ))

_TEST_DEPS = tests.h
TEST_DEPS  = $(patsubst %,$(TESTDIR)/%,$(_TEST_DEPS))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	@mkdir -p $(@D)
	${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES}

$(ODIR)/%.o: $(EXAMPLESDIR)/%.c $(DEPS)
	@mkdir -p $(@D)
	${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES}

$(ODIR)/%.o: $(TESTDIR)/%.c $(DEPS) $(TEST_DEPS)
	@mkdir -p $(@D)
	@${PETSC_COMPILE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES}

all: examples

examples: clean_test $(EXAMPLES)

$(TESTS) : CFLAGS += -DUNIT_TEST
$(TESTS) : % : $(ODIR)/%.o $(OBJ) $(TEST_OBJ)
	@${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}
	@-./$@ -ts_adapt_type none > tmp_test_results
	@echo 'running '$@
	-@grep FAIL tmp_test_results || true
	@cat tmp_test_results >> test_results
	@rm tmp_test_results

.phony: clean_test test count_fails

count_fails:
	@echo "All failures listed below"
	@grep FAIL test_results || true

clean_test:
	rm -f $(TEST_OBJ)
	@rm -f test_results

test: clean_test $(TESTS) count_fails

$(EXAMPLES) : % : $(ODIR)/%.o $(OBJ)
	${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB}

.PHONY: clean

clean:
	rm -f $(ODIR)/*
	rm -f $(EXAMPLES)
	rm -f $(TESTS)
