# Steps to Release Scope

These are the steps to take to create a release of the module ``scope_plot``:

1. Switch to the master branch

```bash
git checkout master
```

2. Run bump2version

    0. Install bump2version (if needed)
    ```bash
    pip install --user bump2version
    ```
    1. Run bump2version with `patch`, `minor`, or `major`.
    ```bash
    bump2version --verbose patch
    ```
3. Update `CHANGELOG.md` and commit.

```bash
git add CHANGELOG.md
```

4. Add and commit the changes.

```bash
git push && git push --tags
```

4. Wait for [Travis](travis-ci.com/rai-project/scope) to generate docker images.
