# Compiler and hypre location
CC        = h5pcc
ifndef HYPRE_DIR
$(info HYPRE_DIR not defined, trying '/usr/local/hypre')
HYPRE_DIR = /usr/local/hypre
endif

ifndef INOISY_DIR
$(info INOISY_DIR not defined, trying current directory)
INOISY_DIR = $(CURRDIR)
endif

# Local directories
INC_DIR = $(INOISY_DIR)/include
SRC_DIR = $(INOISY_DIR)/src
OBJ_DIR = $(INOISY_DIR)/obj

# Compiling and linking options
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include -I$(INC_DIR)
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)

LINKOPTS = $(COPTS)
LDFLAGS  = -L$(HYPRE_DIR)/lib
LIBS     = -lHYPRE -lm -lgsl -lgslcblas -shlib -lstdc++
LFLAGS   = $(LINKOPTS) $(LIBS)

# List of all programs to be compiled
EXE = poisson disk_logr disk_xy noisy_unif noisy_disk general_xy

SRC := $(addprefix $(SRC_DIR)/,main.c hdf5_utils.c model_%.c param_%.c)
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
SRCM := $(addprefix $(SRC_DIR)/,main_matrices.c hdf5_utils.c model_%.c param_%.c)
OBJM := $(SRCM:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(EXE)

$(EXE): %: $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LFLAGS) -o $@

matrices: %: $(OBJM)
	$(CC) $(LDFLAGS) $^ $(LFLAGS) -o $(INOISY_DIR)/$@


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir $@

default: all

# Clean up

clean:
	$(RM) -r $(OBJ_DIR)

distclean: clean
	$(RM) -f matrices
	$(RM) $(EXE)
