#include "scope/utils/version.hpp"

#include <string>
#include <iostream>

const char * scope_git_refspec() 
{
#ifdef SCOPE_GIT_REFSPEC
  return SCOPE_GIT_REFSPEC;
#else
  return "";
#endif
}

const char * scope_git_hash() 
{
#ifdef SCOPE_GIT_HASH
  return SCOPE_GIT_HASH;
#else
  return "";
#endif
}

const char * scope_git_local_changes() 
{
#ifdef SCOPE_GIT_LOCAL_CHANGES
  return SCOPE_GIT_LOCAL_CHANGES;
#else
  return "";
#endif
}

std::string version() {

    std::string refspec = scope_git_refspec();
    std::string hash = scope_git_hash();
    std::string local_changes;
    if (std::string("DIRTY") == std::string(scope_git_local_changes())) {
        local_changes = "-dirty";
    } else {
        local_changes = "";
    }

    if (refspec.rfind("refs/heads/", 0) == 0) {
        return refspec.substr(11, refspec.size() - 11) + std::string("-") + hash + local_changes;
    } else if (refspec.rfind("refs/tags/", 0) == 0) {
        return refspec.substr(10, refspec.size() - 10) + local_changes;
    }

}
