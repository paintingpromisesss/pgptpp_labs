# Paths must be absolute: the checking server uses /usr/bin/nvcc and /usr/bin/mpic++.
CC      = /usr/bin/nvcc
CFLAGS  = -Werror cross-execution-space-call -lm

SOURCES = lab1.cu
BIN     = lab1

all: $(BIN)

$(BIN): $(SOURCES)
	$(CC) $(CFLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -f $(BIN)

.PHONY: all clean