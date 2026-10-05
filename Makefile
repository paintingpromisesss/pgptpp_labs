CC      = /usr/bin/nvcc
CFLAGS  = -Werror cross-execution-space-call -lm

SOURCES = main.cu
BIN     = main

all: $(BIN)

$(BIN): $(SOURCES)
	$(CC) $(CFLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -f $(BIN) $(BIN).exe

run:
	./$(BIN)

TARGET = $(word 2,$(MAKECMDGOALS))

build:
	$(CC) $(CFLAGS) $(TARGET) -o $(basename $(TARGET))

sign:
	gpg -ab $(TARGET)

$(TARGET):
	@true

.PHONY: all clean build run sign $(TARGET)
