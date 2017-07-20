CC = g++
TARGET = main
SRCS = main.cpp lmath.cpp tools.cpp poisson.cpp
OBJS = $(SRCS:.cpp=.o)
DLIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui
$(TARGET):$(OBJS)
	$(CC) -o $@ $^ $(DLIBS)  
clean:
	rm -rf $(TARGET) $(OBJS)
%.o:%.cpp
	$(CC) -o $@ -c $<
