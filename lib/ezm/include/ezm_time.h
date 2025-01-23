#ifndef EZM_TIME_H
#define EZM_TIME_H

#include <sys/time.h>
#ifdef ENABLE_FUT
#define CONFIG_FUT
#include <fut.h>
#endif

#define TIMESPEC2USEC(t) ((uint64_t)(t).tv_sec * 1000000ULL + (t).tv_nsec / 1000)

static inline uint64_t ezm_gettime (void)
{
#ifdef ENABLE_FUT
  return fut_getstamp () / 1000;
#else
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);

	return TIMESPEC2USEC(tp);

#endif
}

#endif
