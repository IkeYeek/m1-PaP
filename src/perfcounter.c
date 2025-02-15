
#include <papi.h>
#include <pthread.h>
#include <stdlib.h>

#include "debug.h"
#include "error.h"
#include "omp.h"
#include "perfcounter.h"

#ifdef ENABLE_PAPI

unsigned do_cache = 0;

#ifdef MICROARCH_HASWELL
static perf_event event_list[] = {
    {.is_native = true, .name = "CPU_CLK_UNHALTED.THREAD_P"},       // TOTAL_CYCLES
    {.is_native = true, .name = "RESOURCE_STALLS.ANY"}              // TOTAL_STALLS
};

#else
#ifdef MICROARCH_SKYLAKE
static perf_event event_list[] = {
    {.is_native = true, .name = "CPU_CLK_UNHALTED.THREAD_P"},       // TOTAL_CYCLES
    {.is_native = true, .name = "RESOURCE_STALLS.ANY"}              // TOTAL_STALLS
};

#else
#error Cache monitoring not available for this microarchitecture.
#endif
#endif

size_t n_events = sizeof (event_list) / sizeof (perf_event);

typedef struct
{
  // This array contains the values of counters when a sample is complete
  // value of the eventsets.
  long long counter[EASYPAP_NB_COUNTERS];
  long long total_counter[EASYPAP_NB_COUNTERS];
  int eventSet;
} easypap_perfcounter_sample_t;

// Array of samples, one per cpu
static easypap_perfcounter_sample_t *counters = NULL;
static unsigned nb_cpu;
int *code;

void easypap_perfcounter_init (unsigned nb_cpus, unsigned monitor_flags)
{
  // Initialize PAPI and structures.
  int retval;
  nb_cpu   = nb_cpus;
  code     = malloc (n_events * sizeof (int));
  counters = malloc (nb_cpus * sizeof (easypap_perfcounter_sample_t));
  for (int cpu = 0; cpu < nb_cpu; cpu++) {
    counters[cpu].eventSet = PAPI_NULL;
    for (int c = 0; c < EASYPAP_NB_COUNTERS; c++) {
      counters[cpu].counter[c]       = 0;
      counters[cpu].total_counter[c] = 0;
    }
  }

  if ((retval = PAPI_library_init (PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
    ERROR_RETURN (retval);
  }
  for (size_t i = 0; i < n_events; i++) {
    code[i] = PAPI_NULL;
    if (event_list[i].is_native) {
      if ((retval = PAPI_event_name_to_code (event_list[i].name, &(code[i]))) !=
          PAPI_OK)
        ERROR_RETURN (retval);
    }
  }
  retval = PAPI_thread_init ((unsigned long (*) (void)) (pthread_self));
  if (retval != PAPI_OK)
    ERROR_RETURN (retval);
  PRINT_DEBUG ('p', "Perfcounter initialization done (flag = %d)\n",
               monitor_flags);
}

// Return 0 on success
int easypap_perfcounter_create_event_set (unsigned cpu)
{
  if (counters[cpu].eventSet == PAPI_NULL) {
    int retval;
    if ((retval = PAPI_register_thread ()) != PAPI_OK)
      ERROR_RETURN (retval);
    /* Creating the eventset */
    if ((retval = PAPI_create_eventset (&(counters[cpu].eventSet))) != PAPI_OK)
      ERROR_RETURN (retval);
    for (size_t i = 0; i < n_events; i++) {
      if (event_list[i].is_native) {
        if ((retval = PAPI_add_event (counters[cpu].eventSet, code[i])) !=
            PAPI_OK)
          ERROR_RETURN (retval);
      } else {
        if ((retval = PAPI_add_event (counters[cpu].eventSet,
                                      event_list[i].code)) != PAPI_OK)
          ERROR_RETURN (retval);
      }
    }
    return 0;
  }
  return -1;
}

void easypap_perfcounter_monitor_start_tile (unsigned cpu)
{
  int retval, status = 0;
  /* If the eventset wasn't create by calling
  easypap_perfcounter_create_event_set
  then it will be created during the first tile.
  */
  if (counters[cpu].eventSet == PAPI_NULL)
    easypap_perfcounter_create_event_set (cpu);
  if ((retval = (PAPI_state (counters[cpu].eventSet, &status))) != PAPI_OK)
    ERROR_RETURN (retval);
  if (status != PAPI_RUNNING) {
    if ((retval = PAPI_start (counters[cpu].eventSet)) != PAPI_OK)
      ERROR_RETURN (retval);
  }
}

// Stop PAPI monitoring on this cpu
void easypap_perfcounter_monitor_stop_tile (unsigned cpu)
{
  if (do_trace){
    int retval;
    if ((retval = PAPI_stop (counters[cpu].eventSet, counters[cpu].counter)) != PAPI_OK)
      ERROR_RETURN (retval);

    for (unsigned c = 0; c < EASYPAP_NB_COUNTERS; c++) {
      counters[cpu].total_counter[c] += counters[cpu].counter[c];
      if (counters[cpu].total_counter[c] < 0 || counters[cpu].counter[c] < 0){
        printf("OVERFLOW/NEGATIVE counter %s\n", event_list[c].name);
        exit(1);
      }

    }
  }
}

// nothing
void easypap_perfcounter_monitor_start_iteration ()
{
  // nothing
}

void easypap_perfcounter_monitor_stop_iteration ()
{
  if (!do_trace)
    easypap_perfcounter_monitor_stop_all();
}

void easypap_perfcounter_monitor_stop_all()
{
#pragma omp parallel // for that it's the same thread that started the monitoring that stops it
  {
    int retval, status = 0;

    int thread = omp_get_thread_num();

    if (counters[thread].eventSet != PAPI_NULL){

      if ((retval = PAPI_state(counters[thread].eventSet, &status)) != PAPI_OK)
        ERROR_RETURN(retval);

      if (status == PAPI_RUNNING) {

        if ((retval = PAPI_stop(counters[thread].eventSet, counters[thread].counter)) != PAPI_OK)
          ERROR_RETURN(retval);

        for (unsigned c = 0; c < EASYPAP_NB_COUNTERS; c++) {
          counters[thread].total_counter[c] += counters[thread].counter[c];
          if (counters[thread].total_counter[c] < 0 || counters[thread].counter[c] < 0) {
            printf("OVERFLOW/NEGATIVE counter %s\n", event_list[c].name);
            exit(1);
          }
        }
      }
    }
  }
}

int64_t easypap_perfcounter_get_counter (unsigned cpu,
                                         easypap_perfcounter_counter_t counter)
{
  return (int64_t)counters[cpu].counter[counter];
}

int easypap_perfcounter_get_counters (int64_t *counter_array, unsigned cpu)
{
  if (counter_array == NULL)
    return -1;
  for (uint c = 0; c < EASYPAP_NB_COUNTERS; c++) {
    counter_array[c] = (int64_t) counters[cpu].counter[c];
  }
  return 0;
}

int easypap_perfcounter_get_total_counters (int64_t *total_counters)
{
  if (total_counters == NULL)
    return -1;
  // On pourrait vérifier que les compteurs de toutes les tuiles soient bien
  // stoppés avec PAPI_state.
  for (int i = 0; i < EASYPAP_NB_COUNTERS; i++) {
    total_counters[i] = 0;
  }
  for (int cpu = 0; cpu < nb_cpu; cpu++) {
    for (int i = 0; i < EASYPAP_NB_COUNTERS; i++) {
      total_counters[i] += (int64_t)counters[cpu].total_counter[i];
    }
  }
  return 0;
}

void easypap_perfcounter_finalize (void)
{
  PAPI_shutdown ();
  if (counters != NULL) {
    free (counters);
    counters = NULL;
  }
  if (code != NULL) {
    free (code);
    code = NULL;
  }
}
#endif