set(NEURAL_SPEED_URL https://github.com/intel/neural-speed.git)
set(NEURAL_SPEED_TAG bestlav0.1)

FetchContent_Declare(
        neural_speed
        GIT_REPOSITORY ${NEURAL_SPEED_URL}
        GIT_TAG        ${NEURAL_SPEED_TAG}
    )
FetchContent_MakeAvailable(neural_speed)
