directories in this folder will be treated as scopes, and automatically enabled.

As an additional requirement, they should use in their `CMakeLists.txt`

```cmake
set(SCOPE_AUTOENABLE_TARGET <target> PARENT_SCOPE)
```

to set the `SCOPE_AUTOENABLE_TARGET` variable in the parent scope to be the value that scope should link against to include this scope.