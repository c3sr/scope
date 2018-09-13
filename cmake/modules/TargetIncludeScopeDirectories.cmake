include(ScopeFatal)

function(target_include_scope_directories)

    if(NOT DEFINED SCOPE_TOP_DIR)
        scope_fatal("SCOPE_TOP_DIR should point at scope source root")
    endif()

    target_include_directories(${ARGV0} PRIVATE 
        ${SCOPE_TOP_DIR}/src
        ${SCOPE_TOP_DIR}/include
    )
    target_include_directories(${ARGV0} SYSTEM PRIVATE 
        ${SCOPE_TOP_DIR}/third_party
        ${CUDA_INCLUDE_DIRS}
    )    
endfunction()