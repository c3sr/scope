#include "scope/init/logger.hpp"
#include "scope/utils/version.hpp"
#include <string>
#include <iostream>

std::string version(const std::string &project, 
                    const std::string &version, 
                    const std::string &refspec, 
                    const std::string &hash, 
                    const std::string &changes) {
    std::string project_part = project;
    std::string version_part = version;

    std::string refspec_part;
    if (refspec.rfind("refs/heads/", 0) == 0) {
        refspec_part = refspec.substr(11, refspec.size() - 11);
    } else if (refspec.rfind("refs/tags/", 0) == 0) {
        refspec_part = refspec.substr(10, refspec.size() - 10);
    } else {
      LOG(debug, "refspec={} was unexpected", refspec);
      refspec_part = std::string("unknown");
    }

    std::string hash_part = hash.substr(0,8);

    std::string changes_part;
    if (std::string("DIRTY") == changes) {
        changes_part = "-dirty";
    } else {
        changes_part = "";
    }

    return project_part + " " + version_part + "-" + refspec_part + "-" + hash_part + changes_part;
}
