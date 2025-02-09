CC = g++

CFLAGS = -Wall -Wextra -std=c++17

TARGET = test
SRCS = test.cpp vec.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)