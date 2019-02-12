# Changing the Compiler

Changing the compiler used for Scope has two parts: the compiler used while building the prerequisites controlled by Hunter, and the compiler used while building Scope.

For example, if a new `CMAKE_CXX_COMPILER` should be used while building scope, it must be changed in *both places*.

## Hunter (via CMake Toolchain File)

Setting `CMAKE_CXX_COMPILER` or other variables that control the build will not be used by Hunter when building the Scope prerequisites.
Instead, a [CMake Toolchain file][cmake-toolchain-file] must be used.
An example is present in cmake/clang.toolchain.cmake

It is used with 

    cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain/file ...


[cmake-toolchain-file]: https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/CrossCompiling#the-toolchain-file

## Scope (via CMake command line args)

While Scope is being built, normal CMake variables such as `-DCMAKE_CXX_COMPILER` are respected.

## Example

To change the compiler to `g++-7`, you could create `cmake/g++7.cmake`

```cmake
# cmake/g++7.cmake
set(CMAKE_CXX_COMPILER g++-7)
```

and then configure Scope with

```
cmake ${scope_src_dir} -DCMAKE_TOOLCHAIN_FILE=${scope_src_dir}/cmake/g++7.cmake -DCMAKE_CXX_COMPILER=g++-7 
```