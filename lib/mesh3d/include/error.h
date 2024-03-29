
#ifndef ERROR_IS_DEF
#define ERROR_IS_DEF

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>

#define exit_with_error(format, ...)                                  \
  do                                                                  \
  {                                                                   \
    fprintf(stderr, "%s:%d: Error: " format "\n", __FILE__, __LINE__, \
            ##__VA_ARGS__);                                           \
    exit(EXIT_FAILURE);                                               \
  } while (0)

#define check(cond, format, ...)                     \
  do                                                 \
  {                                                  \
    if (!(cond))                                     \
      exit_with_error(format " (%s)", ##__VA_ARGS__, \
                      strerror(errno));              \
                                                     \
  } while (0)

#define check_syscall(ret, format, ...) \
  check((ret) != -1, format, ##__VA_ARGS__)

#endif

// Examples of use:
//
// int fd = open (filename, O_RDONLY);
// check_syscall (fd, "Cannot open file %s", filename);
//
// FILE *f = fopen (filename, "r");
// check (f != NULL, "Cannot open file %s", filename);
