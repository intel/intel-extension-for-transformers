file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_subdirectory(jit_domain)
add_subdirectory(kernels)

add_library(${HOST_LIBRARY_NAME} SHARED ${SOURCES})

# Header file include path
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

