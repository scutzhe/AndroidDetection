cmake_minimum_required(VERSION 3.4.1)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(src/main/cpp/include
                    src/main/cpp/)

add_library(lib_opencv STATIC IMPORTED )
set_target_properties(lib_opencv
		PROPERTIES
		IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so
		)

add_library(lib_tflite STATIC IMPORTED )
set_target_properties(lib_tflite
		PROPERTIES
		IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libtensorflowLite.so
		)

add_library(face_detection SHARED src/main/cpp/jni.cpp
                              src/main/cpp/Face.cpp
        )

find_library(log-lib log)
find_library( jnigraphics-lib jnigraphics )
add_definitions(-DMNN_USE_LOGCAT)
target_link_libraries(face_detection
					   lib_opencv
		               lib_tflite
                       jnigraphics
                       z
                       ${log-lib})


