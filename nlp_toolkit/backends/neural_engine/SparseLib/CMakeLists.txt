cmake_minimum_required(VERSION 3.11)
project(kernellibs)
set(HOST_LIBRARY_NAME "kernellibs")

option(SPARSE_LIB_USE_VTUNE OFF)
option(CMAKE_VTUNE_HOME "whether use vtune" OFF)

set(EXTERNAL_INCLUDE_PATH "")
set(EXTERNAL_LIBRARY_PAYH "")

# include external cmake
if (SPARSE_LIB_USE_VTUNE)
    include(cmake/FindVTune.cmake)
    message( STATUS "VTune enbled" )
endif()



if(OpenMP_FOUND)
    #message(${CMAKE_CXX_FLAGS})
else()
    find_package(OpenMP REQUIRED)
    if(WIN32)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:experimental")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()


file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/kernels/*.cpp
  ${PROJECT_SOURCE_DIR}/src/jit_domain/*.cpp
)
file(GLOB HEADERS
  ${PROJECT_SOURCE_DIR}/include/*.hpp
  ${PROJECT_SOURCE_DIR}/include/kernels/*.hpp
  ${PROJECT_SOURCE_DIR}/include/jit_domain/*.hpp
)


add_library(${HOST_LIBRARY_NAME} SHARED ${SOURCES} ${HEADERS})

install(TARGETS ${HOST_LIBRARY_NAME})

target_compile_definitions(${HOST_LIBRARY_NAME} PRIVATE SPARSE_KERNEL_BUILD)

set_target_properties(${HOST_LIBRARY_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Header file include path
message( STATUS "EXTERNAL_INCLUDE_PAYH ${EXTERNAL_INCLUDE_PAYH}" )
target_include_directories(${HOST_LIBRARY_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)


## link against the third party libraries
target_link_libraries(${HOST_LIBRARY_NAME}
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        glog
        xbyak
)


