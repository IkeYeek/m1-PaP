### config section ###

ENABLE_TRACE	:= 1

CC			:= gcc
CFLAGS		:= -O3 -march=native

######################

ARCH		:= $(shell uname -s | tr a-z A-Z)

PSOURCES	:= $(wildcard test/*.c)
POBJECTS	:= $(PSOURCES:test/%.c=obj/%.o)
PDEPENDS	:= $(PSOURCES:test/%.c=deps/%.d)
PROGRAMS	:= $(PSOURCES:test/%.c=bin/%)

LIB			:= lib/libezm.a

SOURCES		:= $(wildcard src/*.c)

ifneq ($(ENABLE_TRACE), 1)
SOURCES			:= $(filter-out src/ezm_tracerec.c, $(SOURCES))
endif

OBJECTS		:= $(SOURCES:src/%.c=obj/%.o)
DEPENDS		:= $(SOURCES:src/%.c=deps/%.d)

MAKEFILES	:= Makefile

CFLAGS		+= -Wall -Wno-unused-function
CFLAGS		+= -I./include
LDLIBS		+= -lm

PACKAGES	:= sdl2 cglm

ifeq ($(ENABLE_TRACE), 1)
# Right now, only fxt is supported
CFLAGS			+= -DENABLE_TRACE -DENABLE_FUT
CUDA_CFLAGS		+= -DENABLE_TRACE -DENABLE_FUT
PACKAGES		+= fxt
endif

# OpenMP
CFLAGS			+= -fopenmp
LDFLAGS			+= -fopenmp

# Query CFLAGS and LDLIBS for all packages
PKG_CHECK	:= $(shell if pkg-config --print-errors --exists $(PACKAGES); then echo 0; else echo 1; fi)
ifeq ($(PKG_CHECK), 1)
$(error Installation problem: missing package)
endif
CFLAGS		+= $(shell pkg-config --cflags $(PACKAGES))
LDFLAGS		+= $(shell pkg-config --libs-only-L $(PACKAGES))
LDLIBS		+= $(shell pkg-config --libs-only-l $(PACKAGES))

# EZV lib
EZV_DIR			:= ../ezv
EZV_LIB 		:= $(EZV_DIR)/lib/libezv.a
LDFLAGS			+= -L$(EZV_DIR)/lib
LDLIBS			+= -lezv -lscotch
CFLAGS			+= -I$(EZV_DIR)/include
CUDA_CFLAGS		+= -I$(EZV_DIR)/include

# OpenGL
ifeq ($(ARCH),DARWIN)
LDLIBS		+= -framework OpenGL
else
CFLAGS			+= -DUSE_GLAD
LDLIBS			+= -lGL
endif

# Must be the first rule
.PHONY: default lib exec
default: lib

lib: $(LIB)

exec: $(PROGRAMS)

$(OBJECTS) $(POBJECTS): $(MAKEFILES)


# Règle pour compiler les programmes
$(PROGRAMS): bin/%: obj/%.o $(LIB) $(EZV_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OBJECTS): obj/%.o: src/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

$(POBJECTS): obj/%.o: test/%.c
	$(CC) -o $@ $(CFLAGS) -c $<

$(LIB): $(OBJECTS)
	ar rcs $@ $^

.PHONY: depend
depend: $(DEPENDS) $(PDEPENDS)

$(DEPENDS) $(PDEPENDS): $(MAKEFILES)

$(DEPENDS): deps/%.d: src/%.c
	$(CC) -o $@ $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $<

$(PDEPENDS): deps/%.d: test/%.c
	$(CC) -o $@ $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $<

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPENDS) $(PDEPENDS)
endif

.PHONY: clean
clean:
	rm -f $(PROGRAMS) obj/*.o deps/*.d lib/*
