#ifndef EZPTHREAD_IS_DEF
#define EZPTHREAD_IS_DEF

#include <hwloc.h>
#include <pthread.h>


typedef void * (*ez_pthread_func_t)(void *);

void ez_pthread_settopo (hwloc_topology_t t);
void ez_pthread_init (unsigned ncores);
void ez_pthread_finalize (void);

int ez_pthread_create (pthread_t *thread, const pthread_attr_t *attr, ez_pthread_func_t f, void *arg);
int ez_pthread_join (pthread_t thread, void **value_ptr);


#endif
