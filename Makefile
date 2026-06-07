CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4) -I./onnxruntime-linux-x64-1.18.0/include
OPENCV_LDFLAGS = $(shell pkg-config --libs-only-L opencv4)
OPENCV_LIBS = -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_dnn -lopencv_ximgproc -lopencv_cudastereo -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_imgproc -lopencv_calib3d -lopencv_core
LDFLAGS = $(OPENCV_LDFLAGS) $(OPENCV_LIBS) -lpthread -ldlib -lblas -llapack -L./onnxruntime-linux-x64-1.18.0/lib -lonnxruntime -Wl,-rpath,'$$ORIGIN/onnxruntime-linux-x64-1.18.0/lib'

TARGET = main_stereo
SRCS = main_stereo.cpp CameraStream.cpp StereoProcessor.cpp FaceSwapper.cpp YOLODetector.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o $(TARGET)
