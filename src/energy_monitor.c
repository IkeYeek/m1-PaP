#define _GNU_SOURCE /* See feature_test_macros(7) */

#include "energy_monitor.h"
#ifndef __linux__

void frequency_monitor_init (hwloc_topology_t topology)
{
}
void frequency_compute (double *frequencies)
{
}

void energy_monitor_init ()
{
}
uint64_t energy_monitor_get_consumption ()
{
  return 0;
}

#else
// energy_monitor.c
#include <dirent.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#define ENERGY_PATH "/sys/class/powercap"
#define ENERGY_FILE "energy_uj"
#define MAX_DOMAINS 16
#define PATHNAMELEN 512
#define MAX_CORES 256

typedef struct
{
  char path[PATH_MAX];
  uint64_t initial_energy;
} EnergyDomain;

static EnergyDomain domains[MAX_DOMAINS];
static int domain_count = 0;

static int num_threads;
static int perf_fds[MAX_CORES];
static int corekinds[MAX_CORES];
static int thread_corekinds[MAX_CORES];

static uint64_t initial_time[MAX_CORES];
static uint64_t initial_ticks[MAX_CORES];
static int nr_corekinds;

void check_omp_places ()
{
  const char *omp_places = getenv ("OMP_PLACES");

  if (!omp_places)
    fprintf (stderr,
             "Warning: The environment variable OMP_PLACES is not defined."
             "The measurements obtained using the performance counters are "
             "likely to be inaccurate.\n");
}

// Détecter les types de cœurs dans la topologie
void detect_corekinds (hwloc_topology_t topology)
{
  nr_corekinds = hwloc_cpukinds_get_nr (topology, 0);
  if (nr_corekinds < 0) {
    fprintf (stderr,
             "Erreur : Impossible de récupérer les kinds de cœurs (nr = %d).\n",
             nr_corekinds);
    return;
  }

  if (nr_corekinds == 0) {
    for (int cpu = 0; cpu < MAX_CORES; cpu++) {
        corekinds[cpu] = 0;
    }  
    nr_corekinds=1;
    return;
  }
   
  hwloc_bitmap_t bitmap = hwloc_bitmap_alloc ();
  for (int kind = 0; kind < nr_corekinds; kind++) {
    int efficiency = 0;
    if (hwloc_cpukinds_get_info (topology, kind, bitmap, &efficiency, NULL, 0,
                                 0) != 0) {
      fprintf (stderr,
               "Erreur : Impossible de récupérer les informations pour le type "
               "%d.\n",
               kind);
      continue;
    }

    for (int cpu = 0; cpu < MAX_CORES; cpu++) {
      if (hwloc_bitmap_isset (bitmap, cpu)) {
        corekinds[cpu] = efficiency;
      }
    }
  }
  hwloc_bitmap_free (bitmap);
}

// Obtenir le timestamp actuel en nanosecondes
static uint64_t get_time_ns ()
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

// Configurer un compteur matériel via perf_event_open
static int perf_event_open (struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags)
{
  return syscall (SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

void frequency_monitor_init (hwloc_topology_t topology)
{
  check_omp_places ();
  struct perf_event_attr pe_cycles;
  memset (&pe_cycles, 0, sizeof (struct perf_event_attr));
  pe_cycles.type     = PERF_TYPE_HARDWARE;
  pe_cycles.size     = sizeof (struct perf_event_attr);
  pe_cycles.config   = PERF_COUNT_HW_CPU_CYCLES;
  pe_cycles.disabled = 1;

  num_threads = omp_get_max_threads ();
  if (num_threads > MAX_CORES) {
    fprintf (
        stderr,
        "Erreur : Trop de cœurs pour le tableau statique (MAX_CORES = %d).\n",
        MAX_CORES);
    exit (EXIT_FAILURE);
  }

  detect_corekinds (topology);

#pragma omp parallel
  {
    int thread_id               = omp_get_thread_num ();
    int cpu_id                  = sched_getcpu ();
    num_threads                 = omp_get_num_threads ();
    thread_corekinds[thread_id] = corekinds[cpu_id];

    perf_fds[thread_id] = perf_event_open (&pe_cycles, -1, cpu_id, -1, 0);
    if (perf_fds[thread_id] == -1) {
      fprintf (stderr,
               "Erreur : Impossible d'ouvrir perf_event (coeur %d). Essayer la "
               "commande:\n sudo sysctl -w kernel.perf_event_paranoid=0\n",
               cpu_id);
      exit (EXIT_FAILURE);
    }

    // Initialiser les compteurs
    ioctl (perf_fds[thread_id], PERF_EVENT_IOC_RESET, 0);
    ioctl (perf_fds[thread_id], PERF_EVENT_IOC_ENABLE, 0);

    // Capturer les valeurs initiales
    initial_ticks[thread_id] = 0;
    read (perf_fds[thread_id], &initial_ticks[thread_id], sizeof (uint64_t));
    initial_time[thread_id] = get_time_ns ();
  }
}

// Calculer les fréquences moyennes des cœurs
void frequency_compute (double *avg_frequencies)
{
  double frequencies[MAX_CORES];
  uint64_t final_ticks[MAX_CORES];
  uint64_t final_time[MAX_CORES];

#pragma omp parallel
  {
    int thread_id          = omp_get_thread_num ();
    final_ticks[thread_id] = 0;
    read (perf_fds[thread_id], &final_ticks[thread_id], sizeof (uint64_t));
    final_time[thread_id] = get_time_ns ();

    uint64_t tick_diff     = final_ticks[thread_id] - initial_ticks[thread_id];
    uint64_t time_diff_ns  = final_time[thread_id] - initial_time[thread_id];
    frequencies[thread_id] = (double)tick_diff / (double)time_diff_ns;
  }

  double nb_core_per_kind[nr_corekinds];
  memset (nb_core_per_kind, 0, nr_corekinds * sizeof *nb_core_per_kind);
  memset (avg_frequencies, 0, nr_corekinds * sizeof *avg_frequencies);

  for (int i = 0; i < num_threads; i++) {
    avg_frequencies[thread_corekinds[i]] += frequencies[i];
    nb_core_per_kind[thread_corekinds[i]]++;
  }

  for (int i = 0; i < nr_corekinds; i++) {
    avg_frequencies[i] =
        nb_core_per_kind[i] > 0 ? avg_frequencies[i] / nb_core_per_kind[i] : 0;
  }

  for (int i = 0; i < num_threads; i++) {
    close (perf_fds[i]);
  }
}

uint64_t read_energy_value (const char *filepath)
{
  FILE *file = fopen (filepath, "r");
  if (!file) {
    perror ("Erreur lors de l'ouverture du fichier pour l'énergie");
    return 0;
  }

  uint64_t value = 0;
  if (fscanf (file, "%lu", &value) != 1) {
    perror ("Erreur lors de la lecture de l'énergie");
  }

  fclose (file);
  return value;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"

static int discover_energy_domains (EnergyDomain *domains, int max_domains)
{
  DIR *dir = opendir (ENERGY_PATH);
  if (!dir) {
    perror ("Erreur lors de l'ouverture du répertoire powercap");
    return 0;
  }

  struct dirent *entry;
  int count = 0;

  while ((entry = readdir (dir)) != NULL) {
    if (strncmp (entry->d_name, "intel-rapl:", 11) == 0 &&
        strlen (entry->d_name) == 12) {
      char domain_path[PATH_MAX];
      char energy_file_path[PATH_MAX];

      snprintf (domain_path, PATH_MAX, "%s/%s", ENERGY_PATH, entry->d_name);
      snprintf (energy_file_path, PATH_MAX, "%s/%s", domain_path, ENERGY_FILE);
      if (access (energy_file_path, R_OK) == 0 && count < max_domains) {
        snprintf (domains[count].path, PATHNAMELEN, "%s", energy_file_path);
        count++;
      }
    }
  }

  closedir (dir);
  return count;
}

#pragma GCC diagnostic pop

// Initialiser le moniteur d'énergie
void energy_monitor_init ()
{
  domain_count = discover_energy_domains (domains, MAX_DOMAINS);
  for (int i = 0; i < domain_count; i++) {
    domains[i].initial_energy = read_energy_value (domains[i].path);
  }
}

// Obtenir la consommation d'énergie
uint64_t energy_monitor_get_consumption ()
{
  uint64_t total_consumption = 0;
  for (int i = 0; i < domain_count; i++) {
    uint64_t current_energy = read_energy_value (domains[i].path);
    if (current_energy > domains[i].initial_energy) {
      total_consumption += (current_energy - domains[i].initial_energy);
    }
  }

  return total_consumption;
}

#endif
