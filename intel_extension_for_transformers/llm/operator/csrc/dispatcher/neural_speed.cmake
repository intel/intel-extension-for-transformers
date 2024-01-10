set(NEURAL_SPEED_URL https://github.com/intel/neural-speed.git)
set(NEURAL_SPEED_TAG 18720b319d6921c28e59cc9e003e50cee9a85fcc) # kernel-only release v0.2

FetchContent_Declare(
        neural_speed
        GIT_REPOSITORY ${NEURAL_SPEED_URL}
        GIT_TAG        ${NEURAL_SPEED_TAG}
    )
FetchContent_MakeAvailable(neural_speed)
