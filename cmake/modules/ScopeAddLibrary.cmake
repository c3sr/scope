macro(scope_add_library)
  add_library(${ARGV})
  set(SCOPE_NEW_TARGET ${ARGV0} PARENT_SCOPE)
endmacro()
