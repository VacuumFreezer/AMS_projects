CC = cc

#Note: Change -O to -g when using debugger.
# CFLAGS = -g -Wall
CFLAGS = -O -Wall

all:	hw5_6

hw5_6.o: hw5_6.c
	$(CC) $(CFLAGS) -c $<

hw5_6: hw5_6.o
	$(CC) $< -o $@ -llapack -lblas -lm

clean:
	rm -f hw5_6.o hw5_6
