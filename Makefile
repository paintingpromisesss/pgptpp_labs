CC      = /usr/bin/nvcc
CFLAGS  = -Werror cross-execution-space-call -lm

SOURCES = lab1.cu
BIN     = lab1

all: $(BIN)

$(BIN): $(SOURCES)
	$(CC) $(CFLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -f $(BIN)

TARGET = $(word 2,$(MAKECMDGOALS))

build:
	g++ $(TARGET) -o $(basename $(TARGET)).exe

sign:
	gpg -ab $(TARGET)

$(TARGET):
	@rem

run:
	$(wildcard *.exe)

.PHONY: all clean build run sign $(TARGET)
