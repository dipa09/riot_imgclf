
# Path to the RIOT application
RIOT_DIR = "../RIOT/examples/imgclf"

# Valid options are: EMLEARN, M2CGEN, MICROMLGEN
LIB = EMLEARN

CC = gcc
cflags = -Wall -Wundef -O2 -fwrapv -fno-strict-aliasing

CFLAGS = $(cflags) -DRIOT=0

# Disable this because emlearn doesn't cast values properly leading to a lot of warnings
CFLAGS += -Wno-overflow

# Define a macro to recognize which ML library has been used from the RIOT application
CFLAGS += -D$(LIB)

# In case emlearn is used, the path to the emlearn library must be specified
ifeq ($(LIB),EMLEARN)
	CFLAGS += "-I../emlearn-env/lib/python3.11/site-packages/emlearn"
endif


all: main

main: main.c
	$(CC) $(CFLAGS) main.c -o $@

driver: driver.c
	$(CC) $(cflags) -o $@ $^

liblbp.so: lbp.c
	$(CC) $(cflags) -Winline -shared -fPIC -o $@ $^

sync:
	cp main.c model.h labels.h features.h $(RIOT_DIR)

clean:
	rm -f main driver features features.h model model.h labels.h liblbp.so

.PHONY: clean sync
