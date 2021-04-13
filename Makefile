CFLAGS =  -g -Wuninitialized -O3


ODIR=obj
SRCDIR=src
EXAMPLESDIR=examples
EXAMPLES=$(basename $(notdir $(wildcard $(EXAMPLESDIR)/*.c)))
TESTDIR=tests
TESTS=$(basename $(notdir $(wildcard $(TESTDIR)/*test*na*.c)))
MPI_TESTS=$(addprefix mpi_,$(TESTS))
CFLAGS += -isystem $(SRCDIR)

include ${SLEPC_DIR}/lib/slepc/conf/slepc_variables
#include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules

_DEPS = quantum_gates.h dm_utilities.h operators.h solver.h operators_p.h quac.h quac_p.h kron_p.h qasm_parser.h error_correction.h qsystem.h qsystem_p.h qvec_utilities.h quantum_circuits.h neutral_atom.h
DEPS  = $(patsubst %,$(SRCDIR)/%,$(_DEPS))

_OBJ  = quac.o operators.o solver.o kron.o dm_utilities.o quantum_gates.o error_correction.o qasm_parser.o qsystem.o qvec_utilities.o quantum_circuits.o neutral_atom.o
 OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

_TEST_OBJ  = unity.o t_helpers.o
TEST_OBJ = $(patsubst %,$(ODIR)/%,$(_TEST_OBJ))

_TEST_DEPS = t_helpers.h
TEST_DEPS  = $(patsubst %,$(TESTDIR)/%,$(_TEST_DEPS))

$(ODIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	@mkdir -p $(@D)
	${PETSC_COMPILE_SINGLE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES} ${SLEPC_EPS_LIB}

$(ODIR)/%.o: $(EXAMPLESDIR)/%.c $(DEPS)
	@mkdir -p $(@D)
	${PETSC_COMPILE_SINGLE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES} ${SLEPC_EPS_LIB}

$(ODIR)/%.o: $(TESTDIR)/%.c $(DEPS) $(TEST_DEPS)
	@mkdir -p $(@D)
	@${PETSC_COMPILE_SINGLE} -c -o $@ $< $(CFLAGS) ${PETSC_KSP_LIB} ${PETSC_CC_INCLUDES} ${SLEPC_EPS_LIB}

all: examples

examples: clean_test $(EXAMPLES)

$(TESTS) : CFLAGS += -DUNIT_TEST
$(TESTS) : % : $(ODIR)/%.o $(OBJ) $(TEST_OBJ)
	${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB} ${SLEPC_EPS_LIB}
	@echo 'running '$@
	@-./$@ |tee  tmp_test_results
	-@grep FAIL tmp_test_results || true
	@cat tmp_test_results >> test_results
	@rm tmp_test_results

$(MPI_TESTS) : CFLAGS += -DUNIT_TEST
$(MPI_TESTS) : $(TESTS)
	@$(eval tmp=$(subst mpi_,,$@))
	@echo 'running '$@
	@-mpiexec -np 2 ./$(tmp) -ts_adapt_type none > tmp_test_results
	-@grep FAIL tmp_test_results || true
	@cat tmp_test_results >> test_results
	@rm tmp_test_results

.phony: clean_test test count_fails

count_fails:
	@echo "All failures listed below"
	@grep FAIL test_results || true

clean_test:
	@rm -f test_results

test: clean_test $(TESTS) count_fails

mpi_test: clean_test $(MPI_TESTS) count_fails

$(EXAMPLES) : % : $(ODIR)/%.o $(OBJ)
	${CLINKER} -o $@ $^ $(CFLAGS) ${PETSC_KSP_LIB} ${SLEPC_EPS_LIB}

.PHONY: clean

clean:
	rm -f $(ODIR)/*
	rm -f $(EXAMPLES)
	rm -f $(TESTS)
