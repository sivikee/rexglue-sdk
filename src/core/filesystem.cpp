/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>

#include <rex/filesystem.h>

#if REX_PLATFORM_SWITCH
#include <errno.h>
#include <sys/stat.h>

#include <rex/logging.h>
#endif

namespace rex {
namespace filesystem {

bool CreateParentFolder(const std::filesystem::path& path) {
  if (path.has_parent_path()) {
    auto parent_path = path.parent_path();
    if (!std::filesystem::exists(parent_path)) {
      return CreateDirectories(parent_path);
    }
  }
  return true;
}

bool CreateDirectories(const std::filesystem::path& path) {
#if REX_PLATFORM_SWITCH
  std::string s = path.generic_string();
  while (s.size() > 1 && s.back() == '/') s.pop_back();
  // Skip the device prefix ("sdmc:/"); never mkdir the device root.
  size_t pos = s.find(":/");
  pos = pos == std::string::npos ? 0 : pos + 2;
  while (pos < s.size() && s[pos] == '/') ++pos;
  while (pos <= s.size()) {
    size_t next = s.find('/', pos);
    if (next == std::string::npos) next = s.size();
    if (next > pos) {
      std::string prefix = s.substr(0, next);
      if (mkdir(prefix.c_str(), 0777) != 0 && errno != EEXIST) {
        struct stat st;
        if (stat(prefix.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
          REXFS_ERROR("CreateDirectories: mkdir '{}' failed (errno {})", prefix, errno);
          return false;
        }
      }
    }
    pos = next + 1;
  }
  return true;
#else
  return std::filesystem::create_directories(path);
#endif
}

}  // namespace filesystem
}  // namespace rex
