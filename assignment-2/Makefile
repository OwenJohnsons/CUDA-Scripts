CC = gcc
CFLAGS = -Wall

SRCDIR = ./code
SOURCE = $(SRCDIR)/radiator_main.c

BUILDDIR = ./build
OBJECT = $(BUILDDIR)/radiator_main.o

TARGET = radiator_simulator

all: $(TARGET)

$(OBJECT): $(SOURCE) | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJECT)
	$(CC) $(CFLAGS) $^ -o $@

$(BUILDDIR):
	mkdir $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)/*.o $(TARGET)

.PHONY: clean
