# Contributors

* Carl Pearson (pearson@illinois.edu)
* Abdul Dakkak (dakkak@illinois.edu)
* Cheng Li (cli99@illinois.edu)

## Editing a Git Submodule in the scope tree

If you are editing a git submodule in the tree and trying to build to test your changes, you will need

    cmake -DGIT_SUBMODULE=0

Otherwise, CMake will try to update your submodule before building, probably reverting it to the last version that was commited to scope.