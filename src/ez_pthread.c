
#define _GNU_SOURCE
#include <hwloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "error.h"
#include "ez_pthread.h"
#include "global.h"

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_attr_t global_pthread_attr;

static hwloc_topology_t topology;
static unsigned nb_cores = 0;

#define MAX_EZ_PTHREADS 512

static struct ez_pthread_info_t
{
  int started;
  int busy;
  int id;
  pthread_t pid;
  pthread_cond_t wait_work;
  pthread_cond_t wait_end;
  pthread_mutex_t mutex;
  int work_available;
  int work_done;
  ez_pthread_func_t func;
  void *arg;
  void *retval;
} thread_pool[MAX_EZ_PTHREADS];

static hwloc_cpuset_t cpu_sets[MAX_EZ_PTHREADS];
static unsigned nb_pre_bindings = 0;

static void compute_cpu_sets (void)
{
  hwloc_obj_t root = hwloc_get_root_obj (topology);

  nb_pre_bindings = easypap_requested_number_of_threads ();

  hwloc_distrib (topology, &root, 1, cpu_sets, nb_pre_bindings, INT_MAX, 0);

  for (int i = 0; i < nb_pre_bindings; i++) {
    hwloc_bitmap_singlify (cpu_sets[i]);
    /*
    char buffer[128];
    hwloc_bitmap_list_snprintf (buffer, 128, cpu_sets[i]);
    PRINT_DEBUG ('t', "CPU SET for thread %d: %s\n", i, buffer);
    */
  }
}

static unsigned bind_me (unsigned id)
{
  unsigned cpu;

  if (id < nb_pre_bindings) {
    hwloc_set_cpubind (topology, cpu_sets[id], HWLOC_CPUBIND_THREAD);
    cpu = hwloc_bitmap_first (cpu_sets[id]);
  } else {
    hwloc_obj_t obj;
    hwloc_bitmap_t set;

    obj = hwloc_get_obj_by_type (topology, HWLOC_OBJ_PU, (id - nb_pre_bindings) % nb_cores);
    set = obj->cpuset;
    hwloc_set_cpubind (topology, set, HWLOC_CPUBIND_THREAD);
    cpu = hwloc_bitmap_first (set);
  }
  PRINT_DEBUG ('t', "Thread %p bound to core %d\n", (void *)pthread_self (), cpu);
  return cpu;
}

static void ez_pthread_info_init (struct ez_pthread_info_t *p, int id)
{
  p->started        = 1;
  p->busy           = 1;
  p->id             = id;
  p->work_available = 0;
  p->work_done      = 0;
  p->func           = NULL;
  p->arg            = NULL;
  p->retval         = NULL;
  pthread_cond_init (&p->wait_work, NULL);
  pthread_cond_init (&p->wait_end, NULL);
  pthread_mutex_init (&p->mutex, NULL);
}

static void *thread_loop (void *p)
{
  struct ez_pthread_info_t *me = (struct ez_pthread_info_t *)p;

  // Bind thread
  bind_me (me->id);

  while (1) {

    pthread_mutex_lock (&me->mutex);
    {
      while (!me->work_available)
        pthread_cond_wait (&me->wait_work, &me->mutex);

      me->work_available = 0;
    }
    pthread_mutex_unlock (&me->mutex);

    // Check if we should terminate
    if (me->func == NULL) {
      me->started = 0;
      PRINT_DEBUG ('t', "Thread %p exited\n", (void*) me->pid);
      pthread_exit (NULL);
    }

    // Now we are busy again : call the requested function
    me->retval = me->func (me->arg);

    pthread_mutex_lock (&me->mutex);
    {
      me->work_done = 1;
      pthread_cond_signal (&me->wait_end);
    }
    pthread_mutex_unlock (&me->mutex);
  }
}

void ez_pthread_settopo (hwloc_topology_t t)
{
  topology = t;
}

void ez_pthread_init (unsigned ncores)
{
  nb_cores = ncores;
  if (nb_cores > MAX_EZ_PTHREADS)
    exit_with_error ("Oh oh, current implementation does not support so many "
                     "(%d) cores : please increase value of MAX_EZ_PTHREADS "
                     "(currently set to %d) in src/ez_pthread.c",
                     nb_cores, MAX_EZ_PTHREADS);

  for (int i = 0; i < MAX_EZ_PTHREADS; i++)
    thread_pool[i].started = 0;

  compute_cpu_sets ();

  // Bind main thread
  bind_me (0);

  ez_pthread_info_init (thread_pool + 0, 0);
  thread_pool[0].pid = pthread_self ();

  pthread_attr_init (&global_pthread_attr);
  pthread_attr_setdetachstate (&global_pthread_attr, PTHREAD_CREATE_DETACHED);
  pthread_attr_setscope (&global_pthread_attr, PTHREAD_SCOPE_SYSTEM);
}

void ez_pthread_finalize (void)
{
  // Purge the pool by killing all threads
  for (int i = 1; thread_pool[i].started == 1; i++) {
    pthread_mutex_lock (&thread_pool[i].mutex);
    {
      thread_pool[i].func           = NULL;
      thread_pool[i].work_available = 1;
      pthread_cond_signal (&thread_pool[i].wait_work);
    }
    pthread_mutex_unlock (&thread_pool[i].mutex);
  }

  // Destroy topology object
  hwloc_topology_destroy (topology);
}

int ez_pthread_create (pthread_t *thread, const pthread_attr_t *attr,
                       ez_pthread_func_t f, void *arg)
{
  int i;

  pthread_mutex_lock (&mutex);
  for (i = 1; i < MAX_EZ_PTHREADS; i++) {
    if (!thread_pool[i].started) {
      // We need to start a new thread
      ez_pthread_info_init (thread_pool + i, i);

      pthread_create (&thread_pool[i].pid, &global_pthread_attr, thread_loop,
                      thread_pool + i);
      break;
    } else if (!thread_pool[i].busy) {
      // We found a sleeping thread we can reuse
      thread_pool[i].busy = 1;
      break;
    }
  }
  pthread_mutex_unlock (&mutex);

  if (i > MAX_EZ_PTHREADS)
    exit_with_error ("Oh oh, current implementation does not support so many "
                     "threads : please increase value of MAX_EZ_PTHREADS "
                     "(currently set to %d) in src/ez_pthread.c",
                     MAX_EZ_PTHREADS);

  // Ok, now we can wakeup our thread
  *thread = thread_pool[i].pid;

  pthread_mutex_lock (&thread_pool[i].mutex);
  {
    thread_pool[i].func           = f;
    thread_pool[i].arg            = arg;
    thread_pool[i].work_available = 1;
    pthread_cond_signal (&thread_pool[i].wait_work);
  }
  pthread_mutex_unlock (&thread_pool[i].mutex);

  return 0;
}

int ez_pthread_join (pthread_t thread, void **value_ptr)
{
  pthread_mutex_lock (&mutex);
  for (int i = 0; i < MAX_EZ_PTHREADS; i++)
    if (thread_pool[i].started && pthread_equal (thread_pool[i].pid, thread)) {
      // we found our thread
      pthread_mutex_unlock (&mutex);

      pthread_mutex_lock (&thread_pool[i].mutex);
      while (!thread_pool[i].work_done)
        pthread_cond_wait (&thread_pool[i].wait_end, &thread_pool[i].mutex);
      if (value_ptr != NULL)
        *value_ptr = thread_pool[i].retval;
      thread_pool[i].work_done = 0;
      thread_pool[i].busy      = 0;
      pthread_mutex_unlock (&thread_pool[i].mutex);
      return 0;
    }
  pthread_mutex_unlock (&mutex);

  // We didn't find the thread in the pool
  return -EINVAL;
}
