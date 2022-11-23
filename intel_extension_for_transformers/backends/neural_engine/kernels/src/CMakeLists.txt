file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_subdirectory(jit_domain)
add_subdirectory(kernels)

add_library(${HOST_LIBRARY_NAME} SHARED ${SOURCES})

# Header file include path
message( STATUS "EXTERNAL_INCLUDE_PAYH ${EXTERNAL_INCLUDE_PAYH}" )
target_include_directories(${HOST_LIBRARY_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

string(STRIP "${EXTERNAL_INCLUDE_PAYH}" EXTERNAL_INCLUDE_PAYH)
target_include_directories(${HOST_LIBRARY_NAME}
    PUBLIC
        ${EXTERNAL_INCLUDE_PAYH}
)

message( STATUS "EXTERNAL_LIBRARY_PAYH ${EXTERNAL_LIBRARY_PAYH}" )
string(STRIP "${EXTERNAL_LIBRARY_PAYH}" EXTERNAL_LIBRARY_PAYH)
## link against the third party libraries
target_link_libraries(${HOST_LIBRARY_NAME}
    PUBLIC
        ${CMAKE_THREAD_LIBS_INIT}
        glog
        xbyak
        ${EXTERNAL_LIBRARY_PAYH}
)

