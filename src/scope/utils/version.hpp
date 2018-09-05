#pragma once

#include <string>

// Construct a version string from arguments
// project version-refspec-hash-changes
std::string version(const std::string &project,
                    const std::string &version,
                    const std::string &refspec,
                    const std::string &hash,
                    const std::string &changes);

