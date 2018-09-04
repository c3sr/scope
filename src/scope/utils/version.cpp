#include "scope/init/logger.hpp"
#include "scope/utils/version.hpp"

#include <string>
#include <iostream>

std::string version() {
    std::string changes_part;
    if (std::string("DIRTY") == std::string(SCOPE_GIT_LOCAL_CHANGES)) {
        changes_part = "-dirty";
    } else {
        changes_part = "";
    }

    std::string refspec = SCOPE_GIT_REFSPEC;
    std::string refspec_part;
    if (refspec.rfind("refs/heads/", 0) == 0) {
        refspec_part = refspec.substr(11, refspec.size() - 11);
    } else if (refspec.rfind("refs/tags/", 0) == 0) {
        refspec_part = refspec.substr(10, refspec.size() - 10);
    } else {
      LOG(debug, "refspec={} was unexpected", refspec);
      refspec_part = std::string("unknown");
    }

    std::string hash_part = SCOPE_GIT_HASH;

    std::string version_part = SCOPE_VERSION;

    return version_part + "-" + refspec_part + "-" + hash_part + changes_part;
}
