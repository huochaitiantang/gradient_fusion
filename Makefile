CC = g++
TARGET = main
INCLUDE_DIRS+=-I/home/sensetime/libs/include
LD_LIBRARIE_DIRS+=-L/home/sensetime/libs/lib

#SRCS = main.cpp lmath.cpp tools.cpp poisson.cpp

SRCS:=$(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
DLIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_photo -lopencv_imgcodecs
$(TARGET):$(OBJS)
	$(CC) $(LD_LIBRARIE_DIRS) -o $@ $^ $(DLIBS)  
clean:
	rm -rf $(TARGET) $(OBJS)
%.o:%.cpp
	$(CC) $(INCLUDE_DIRS) -o $@ -c $<
