#pragma once

#include <string>
#include <sstream>

#include <boost/optional.hpp>

namespace storm {
    namespace utility {
        
        struct StormVersion {
            /// The major version of Storm.
            const static unsigned versionMajor;
            
            /// The minor version of Storm.
            const static unsigned versionMinor;
            
            /// The patch version of Storm.
            const static unsigned versionPatch;
            
            /// The short hash of the git commit this build is based on
            const static std::string gitRevisionHash;
            
            /// How many commits passed since the tag was last set.
            const static unsigned commitsAhead;
            
            /// 0 iff there no files were modified in the checkout, 1 otherwise.
            const static boost::optional<bool> dirty;
            
            /// The system which has compiled Storm.
            const static std::string systemName;
            
            /// The system version which has compiled Storm.
            const static std::string systemVersion;
            
            /// The compiler version that was used to build Storm.
            const static std::string cxxCompiler;

            /// The flags that were used to build Storm.
            const static std::string cxxFlags;

            static std::string shortVersionString() {
                std::stringstream sstream;
                sstream << versionMajor << "." << versionMinor << "." << versionPatch;
                return sstream.str();
            }
            
            static std::string longVersionString() {
                std::stringstream sstream;
                sstream << "Version " << versionMajor << "." <<  versionMinor << "." << versionPatch;
                if (commitsAhead) {
                    sstream << " (+ " << commitsAhead << " commits)";
                }
                if (!gitRevisionHash.empty()) {
                    sstream << " build from revision " << gitRevisionHash;
                }
                if (dirty) {
                    if (dirty.get()) {
                        sstream << " (dirty)";
                    } else {
                        sstream << " (clean)";
                    }
                } else {
                    sstream << " (potentially dirty)";
                }
                return sstream.str();
            }
            
            static std::string buildInfo() {
                std::stringstream sstream;
                sstream << "Compiled on " << systemName << " " << systemVersion << " using " << cxxCompiler << " with flags '" << cxxFlags << "'";
                return sstream.str();	
            }
        };
    }
}
