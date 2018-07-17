cmake_minimum_required(VERSION 3.8)

# Look for sugar.cmake files in the plugin src directory
sugar_include("src")

# The plugin is a static library that Scope will be linked with
add_library(example_scope ${example_SOURCES} ${example_CUDA_SOURCES} )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
target_include_directories(example_scope PRIVATE 
    ${SRC_DIR}
    ${THIRDPARTY_DIR}
    ${TOP_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    "src"
)

target_compile_features(example_scope PUBLIC cxx_std_11)
set_property(TARGET example_scope PROPERTY CUDA_STANDARD 11)
target_link_libraries(example_scope PRIVATE benchmark::benchmark)
