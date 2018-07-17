# Example|Scope

This is an example benchmark plugin for the [Scope](github.com/rai-project/scopes) benchmark project.

## Adding the plugin to Scope

The plugin is expected to be loaded as an optional git submodule in the Scope repo, to be included in the build via `add_subdirectory(<scope path>)` in Scope's `CMakeLists.txt`.
This means that the plugin will inherit any variables Scope defines.

The plugin should export any libraries it requires so that Scope can link against them during the build step.

## Scope Utilities

The plugin may/should make use of utilities provided by Scope in scope

## Scope Initialization

Scope allows plugins to register initialization callbacks in scope/src/init.hpp.

Callbacks are `void (*fn)(int argc, char **argv)` functions that will be passed the command line flags that Scope is executed with.
Callbacks can be registered with the INIT() macro:

```cpp
// plugin/src/init.cpp
#include "scope/init/init.hpp

void plugin_callback(int argc, char **argv) {
    (void) argc;
    (void) argv;
}

INIT(mycallback);
```

Scope does not guarantee any ordering for callback execution.

## Structure

* `src`

* `docs`

* `CMakeLists.txt`

## Adding Sources

Scope provides a python script for generating `sugar.cmake` files.
It should be invoked like this whenever source files are added or moved in the plugin:

    $ [scope-dir]/tools/generate_sugar_files.py --top [plugin-dir]/src --var plugin-name