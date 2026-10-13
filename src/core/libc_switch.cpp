// Nintendo Switch: POSIX functions the shared POSIX sources (and vendored FFmpeg) call that
// devkitA64's newlib / libnx do not provide. Declared by the headers (_GNU_SOURCE) but
// never defined, so they would be undefined references at link time.
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#include <malloc.h>
#include <sys/reent.h>
#include <mutex>

#include <switch.h>

namespace {
// pread/pwrite are emulated with lseek+read/write; the lock keeps the seek+IO pair atomic with
// respect to other pread/pwrite callers (plain read/write on the same fd are not expected).
std::mutex g_pio_mutex;
}  // namespace

extern "C" {

ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
  std::lock_guard<std::mutex> lock(g_pio_mutex);
  off_t saved = lseek(fd, 0, SEEK_CUR);
  if (lseek(fd, offset, SEEK_SET) < 0) {
    return -1;
  }
  ssize_t n = read(fd, buf, count);
  if (saved >= 0) {
    lseek(fd, saved, SEEK_SET);
  }
  return n;
}

ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset) {
  std::lock_guard<std::mutex> lock(g_pio_mutex);
  off_t saved = lseek(fd, 0, SEEK_CUR);
  if (lseek(fd, offset, SEEK_SET) < 0) {
    return -1;
  }
  ssize_t n = write(fd, buf, count);
  if (saved >= 0) {
    lseek(fd, saved, SEEK_SET);
  }
  return n;
}

int creat(const char* path, mode_t mode) {
  return open(path, O_WRONLY | O_CREAT | O_TRUNC, mode);
}

int posix_memalign(void** out, size_t alignment, size_t size) {
  void* p = memalign(alignment, size);
  if (!p) {
    return ENOMEM;
  }
  *out = p;
  return 0;
}

int pthread_setname_np(pthread_t, const char*) {
  return 0;
}

// No signals on libnx: asynchronous thread suspend / user-callback delivery is unsupported
// (see threading_posix.cpp). Reporting failure lets callers degrade instead of crashing.
int pthread_kill(pthread_t, int) {
  return ENOSYS;
}

int _getentropy_r(struct _reent*, void* buf, size_t len) {
  randomGet(buf, len);
  return 0;
}

}  // extern "C"
