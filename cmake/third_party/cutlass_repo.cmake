include(FetchContent)

FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG 7d49e6c7e2f8896c47f586706e67e1fb215529dc
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(cutlass)
