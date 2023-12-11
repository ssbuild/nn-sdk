cmake_minimum_required(VERSION 3.5)
project(engine_csdk LANGUAGES CXX)
message(${PROJECT_NAME})

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

#add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(BUILD_FLAGS "-DSDK_CC_EXPORT_ -DCBC=1 -DCTR=1 -DECB=1 -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")
set(BUILD_FLAGS  "${BUILD_FLAGS} -DPY_SSIZE_T_CLEAN")



EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} --version TIMEOUT 5 OUTPUT_VARIABLE py_ver OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX REPLACE  "Python " "" py_ver ${py_ver})
message("py_ver:" ${py_ver})
find_package(PythonLibs ${py_ver} EXACT)

message(STATUS ${PYTHON_INCLUDE_DIRS})
message(STATUS ${PYTHON_LIBRARIES})


include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/findnumpy.cmake)

if(NUMPY_FOUND)
    include_directories("${NUMPY_INCLUDE_DIRS}")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/rapidjson")


set(PROJECT_VERSION_MAJOR "1")
set(PROJECT_VERSION_MINOR "8")
set(PROJECT_VERSION_PATCH "26")


set(NN_SDK_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(NN_SDK_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(NN_SDK_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(NN_SDK_VERSION_PATCH ${PROJECT_VERSION_PATCH})

if(CMAKE_CROSSCOMPILING)
    set(BUILD_CROSSCOMPILING "1")
else()
    set(BUILD_CROSSCOMPILING "0")
endif()


execute_process(
 COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/py_nn_sdk ${CMAKE_BINARY_DIR}/
)

configure_file (
    "${PROJECT_SOURCE_DIR}/version_config.h.in"
    "${PROJECT_SOURCE_DIR}/version_config.h"
)
configure_file (
    "${PROJECT_SOURCE_DIR}/version_config.py.in"
    "${CMAKE_BINARY_DIR}/py_nn_sdk/version_config.py"
)

#jni
# 通过 JAVA_HOME环境变量定位 jni.h和jni_md.h

if(WIN32)
	if(NOT DEFINED ENV{JAVA_HOME})
		message(FATAL_ERROR "not defined environment variable:JAVA_HOME")  
	endif()
	set(BUILD_FLAGS "${BUILD_FLAGS} -DJNI_FOUND")

	set(JNI_INCLUDE "$ENV{JAVA_HOME}/include")
	include_directories(${JNI_INCLUDE})
    include_directories("${JNI_INCLUDE}/win32")

     set(JNI_FOUND TRUE)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")

    if(NOT CROSSCOMPILING)
        find_package(JNI)
    else()
        set(JNI_FOUND TRUE)
        set(JNI_INCLUDE_DIRS "/home/build/jdk/aarch64/jdk1.8.0_311/include" "/home/build/jdk/aarch64/jdk1.8.0_311/include/linux")
        set(JNI_LIBRARIES "")
    endif()

	message(STATUS "JNI_FOUND: ${JNI_FOUND}")
	if(JNI_FOUND)
	    message(STATUS ${JNI_INCLUDE_DIRS} )
        message(STATUS ${JNI_LIBRARIES} )

	    set(BUILD_FLAGS "${BUILD_FLAGS} -DJNI_FOUND")
        include_directories(${JNI_INCLUDE_DIRS})
        #list(APPEND  ${JNI_LIBRARIES})
    endif(JNI_FOUND)
endif(WIN32)




include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB java_src "${CMAKE_CURRENT_SOURCE_DIR}/jni_sdk/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/jni_sdk/*.cpp")

#file(GLOB_RECURSE rapidjson_src "${CMAKE_CURRENT_SOURCE_DIR}/rapidjson/*.*")
set(rapidjson_src "")


file(GLOB_RECURSE pybind_src "${CMAKE_CURRENT_SOURCE_DIR}/pybind11/*.*")

file(GLOB MY_SRC ${CMAKE_CURRENT_LIST_DIR}/*.h 
				 ${CMAKE_CURRENT_LIST_DIR}/*.hpp
				 ${CMAKE_CURRENT_LIST_DIR}/*.cpp
				 ${CMAKE_CURRENT_LIST_DIR}/*.c)

file(GLOB DYLOADER_SRC ${CMAKE_CURRENT_LIST_DIR}/dyloader/*.*)
file(GLOB TF_V1_SRC ${CMAKE_CURRENT_LIST_DIR}/tf_v1/*.*)
file(GLOB TF_V2_SRC ${CMAKE_CURRENT_LIST_DIR}/tf_v2/*.*)
file(GLOB TRT_SRC ${CMAKE_CURRENT_LIST_DIR}/trt/*.*)
file(GLOB ONNX_SRC ${CMAKE_CURRENT_LIST_DIR}/onnx/*.*)
file(GLOB FT_SRC ${CMAKE_CURRENT_LIST_DIR}/ft/*.*)

list(REMOVE_ITEM MY_SRC ${CMAKE_CURRENT_LIST_DIR}/test_cc.cpp)
IF (WIN32)
	MESSAGE(STATUS "Now is windows")
ELSEIF (UNIX)
	#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wl,-rpath-link=.")
	#add_executable(test_csdk ${CMAKE_CURRENT_LIST_DIR}/test_cc.cpp)
	#target_link_libraries(test_csdk dl ${PYTHON_LIBRARIES})
ENDIF ()

list(APPEND ALL_SRC ${MY_SRC} ${TF_V1_SRC} ${TF_V2_SRC} ${TRT_SRC} ${ONNX_SRC}  ${rapidjson_src} ${pybind_src} ${FT_SRC} ${DYLOADER_SRC})
if(JNI_FOUND)
    list(APPEND ALL_SRC "${java_src}")
endif(JNI_FOUND)

add_library(${PROJECT_NAME} SHARED ${ALL_SRC})
set_target_properties(${PROJECT_NAME}  PROPERTIES PREFIX "")
if(WIN32)
	set_target_properties(${PROJECT_NAME}  PROPERTIES SUFFIX ".pyd")
endif()
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES})


set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})



#source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}\\rapidjson" PREFIX "rapidjson" FILES ${rapidjson_src})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}\\pybind11" PREFIX "pybind11" FILES ${pybind_src})
source_group("src" FILES ${MY_SRC})

source_group("tf_v1" FILES ${TF_V1_SRC})
source_group("tf_v2" FILES ${TF_V2_SRC})
source_group("trt" FILES ${TRT_SRC})
source_group("onnx" FILES ${ONNX_SRC})
source_group("ft" FILES ${FT_SRC})
source_group("dyloader" FILES ${DYLOADER_SRC})

if(JNI_FOUND)
    source_group("jni" FILES ${java_src})
endif(JNI_FOUND)



add_custom_command(TARGET ${PROJECT_NAME}
						POST_BUILD 
						COMMAND ${CMAKE_COMMAND} -E echo compile finish !!!
						COMMAND cd ${CMAKE_BINARY_DIR} && ${PYTHON_EXECUTABLE} py_nn_sdk/setup.py bdist_wheel
						VERBATIM
					   )
