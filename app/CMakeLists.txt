cmake_minimum_required(VERSION 3.4.1)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(src/main/cpp/include
                    src/main/cpp/)

add_library(MNN SHARED IMPORTED)
add_library(MNN_CL SHARED IMPORTED)

set_target_properties(MNN
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN.so)
set_target_properties(MNN_CL
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libMNN_CL.so)

add_library(facedetect SHARED src/main/cpp/jni.cpp
                              src/main/cpp/Face.cpp
        )

find_library(log-lib log)
find_library( jnigraphics-lib jnigraphics )
add_definitions(-DMNN_USE_LOGCAT)
target_link_libraries(facedetect
                       MNN
					   MNN_CL
                       jnigraphics
                       z
                       ${log-lib})


