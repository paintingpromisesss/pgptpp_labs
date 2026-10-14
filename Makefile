CC      = /usr/bin/nvcc
CFLAGS  = -Werror cross-execution-space-call -lm

SOURCES = main.cu
BIN     = main

all: $(BIN)

$(BIN): $(SOURCES)
	$(CC) $(CFLAGS) -o $(BIN) $(SOURCES)

clean:
	rm -f $(BIN) $(BIN).exe lab1 lab1.exe report/*.aux report/*.log report/*.out report/*.toc docs/*.aux docs/*.log docs/*.out docs/*.toc

run:
	./$(BIN)

TARGET = $(word 2,$(MAKECMDGOALS))

LAB ?= $(if $(filter-out report build clean run sign all $(BIN),$(TARGET)),$(TARGET),1)
REPORT_SRC  = report/report.tex
REPORT_NAME = Отчет_ЛР$(LAB)_Фокин

build:
	$(CC) $(CFLAGS) $(TARGET) -o $(basename $(TARGET))

sign:
	gpg -ab $(TARGET)

report: $(REPORT_SRC)
	pdflatex -jobname="$(REPORT_NAME)" -interaction=nonstopmode -output-directory report $(REPORT_SRC)
	pdflatex -jobname="$(REPORT_NAME)" -interaction=nonstopmode -output-directory report $(REPORT_SRC)
	rm -f report/*.aux report/*.log report/*.out report/*.toc

$(TARGET):
	@true

.PHONY: all clean build run sign report $(TARGET)
