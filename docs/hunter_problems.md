# Hunter

## Hunter won't download

    error: downloading 'http://github.com/ruslo/hunter/archive/v0.19.214.tar.gz' failed
         status_code: 1
         status_string: "Unsupported protocol"
    ...
    Protocol "https" not supported or disabled in libcurl

    Closing connection -1

This could be caused by a few things:

### CMake was not built with SSL support

If you built your own CMake, you may not have built it with SSL enabled. Try this [hunter documentation](https://docs.hunter.sh/en/latest/faq/how-to-fix-download-error.html) for more info, but the short version is, you should build cmake like this (described in [hunter issue 328](https://github.com/ruslo/hunter/issues/328)):

    git clone ... cmake
    cd cmake
    ./bootstrap --system-curl
    make
    sudo make install

### libcurl was not built with SSL support

If you're using a system libcurl, it should probably have ssl support. If you built your own, you should probably do something along the lines of

    /CurlExtractFolder$ ./configure --with--ssl
    /CurlExtractFolder$ make
    /CurlExtractFolder$ sudo make install

again, described in [hunter issue 328](https://github.com/ruslo/hunter/issues/328).


