set(NEURAL_SPEED_URL https://github.com/intel/neural-speed.git)
set(NEURAL_SPEED_TAG 2f7943681e02c6e87a4c70c3925327f00194c78f)

FetchContent_Declare(
        neural_speed
        GIT_REPOSITORY ${NEURAL_SPEED_URL}
        GIT_TAG        ${NEURAL_SPEED_TAG}
    )
FetchContent_MakeAvailable(neural_speed)
