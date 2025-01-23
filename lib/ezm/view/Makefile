
PROGRAM	:= bin/easyview

# Must be the first rule
.PHONY: default
default: $(PROGRAM)

### config section ###

ENABLE_FUT	=	1
######################

ARCH		:= $(shell uname -s | tr a-z A-Z)

SOURCES		:= $(wildcard src/*.c)

OBJECTS		:= $(SOURCES:src/%.c=obj/%.o)
DEPENDS		:= $(SOURCES:src/%.c=deps/%.d)

MAKEFILES	:= Makefile

CFLAGS		:= -O3 -march=native -Wall -Wno-unused-function -DGL_SILENCE_DEPRECATION #-Wno-deprecated-declarations
CFLAGS		+= -I./include -I../../../include
LDLIBS		+= -lm

PACKAGES	:= SDL2_image SDL2_ttf cglm

ifeq ($(ENABLE_FUT), 1)
CFLAGS		+= -DENABLE_FUT
PACKAGES	+= fxt
endif

ifeq ($(ARCH),DARWIN)
#CFLAGS		+= -DUSE_GLAD
LDLIBS		+= -framework OpenGL
else
CFLAGS		+= -DUSE_GLAD -rdynamic
LDFLAGS		+= -export-dynamic
LDLIBS		+= -lGL -lpthread -ldl
endif

# Query CFLAGS and LDLIBS for all packages
PKG_CHECK	:= $(shell if pkg-config --print-errors --exists $(PACKAGES); then echo 0; else echo 1; fi)
ifeq ($(PKG_CHECK), 1)
$(error Installation problem: missing package)
endif
CFLAGS		+= $(shell pkg-config --cflags $(PACKAGES))
LDFLAGS		+= $(shell pkg-config --libs-only-L $(PACKAGES))
LDLIBS		+= $(shell pkg-config --libs-only-l $(PACKAGES))

# EZV lib
EZV_DIR		:= ../../ezv
EZV_LIB 	:= $(EZV_DIR)/lib/libezv.a
LDFLAGS		+= -L$(EZV_DIR)/lib
LDLIBS		+= -lezv -lscotch
CFLAGS		+= -I$(EZV_DIR)/include

$(OBJECTS): $(MAKEFILES)

$(PROGRAM): $(OBJECTS) $(EZV_LIB)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(OBJECTS): obj/%.o: src/%.c
	$(CC) -o $@ $(CFLAGS) -DTHIS_FILE=\"$(CURDIR)/$<\" -c $<


.PHONY: depend
depend: $(DEPENDS)

$(DEPENDS): $(MAKEFILES)

$(DEPENDS): deps/%.d: src/%.c
	$(CC) $(CFLAGS) -MM -MT "deps/$*.d obj/$*.o" $< > $@

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPENDS)
endif

.PHONY: clean
clean:
	rm -f $(PROGRAM) obj/*.o deps/*.d
