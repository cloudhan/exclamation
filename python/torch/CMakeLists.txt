find_package(Python3 COMPONENTS Interpreter Development)

if (NOT Torch_DIR)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c" "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE torch_cmake_prefix_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ECHO_OUTPUT_VARIABLE
  )
  set(Torch_DIR ${torch_cmake_prefix_path}/Torch)
endif()

if (NOT Torch_FOUND)
  find_package(Torch REQUIRED NO_MODULE)
endif()
find_library(TORCH_PYTHON_LIBRARY torch_python PATH ${TORCH_INSTALL_PREFIX}/lib)

include(${PROJECT_SOURCE_DIR}/cmake/third_party/pybind11_repo.cmake)

set(binding_srcs
  paged_attention_torch.cu
)

# explicitly mark the target as SHARED instead of MODULE so that the so library
# can be depended on by other C++ library, in case
python3_add_library(_exclamation_ops SHARED torch_ops.cc ${binding_srcs})
execute_process(
  COMMAND ${Python3_EXECUTABLE} "-c" "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ECHO_OUTPUT_VARIABLE
)
set_target_properties(_exclamation_ops PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")

target_link_libraries(_exclamation_ops
  PUBLIC
    torch ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY}
    exclamation_cuda exclamation_base nvidia::cutlass::cutlass
)
