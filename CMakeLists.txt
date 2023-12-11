cmake_minimum_required(VERSION 3.5)


if(NOT CROSSCOMPILING)
    set(CROSSCOMPILING FALSE)
endif()
if( (NOT WIN32 ) AND CROSSCOMPILING)
    set(CMAKE_HOST_SYSTEM_NAME Linux)
    set(CMAKE_HOST_SYSTEM_PROCESSOR "x86_64")

    SET(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR  "aarch64")

endif()


project(nn_csdk)

if(PYTHON_EXE)
	set(PYTHON_EXECUTABLE ${PYTHON_EXE})
endif()
if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to Release")
endif()

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -g2 -fpic")
if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W2 /WX-")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})



#set(CMAKE_DEBUG_POSTFIX "_d")
#set(CMAKE_RELEASE_POSTFIX "_r") 
#set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "_d")
#set_target_properties(${TARGET_NAME} PROPERTIES RELEASE_POSTFIX "_r") 
#add_subdirectory(easy_tokenizer)

add_subdirectory(engine_csdk)