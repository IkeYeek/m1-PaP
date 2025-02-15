// energy_monitor.h
#ifndef ENERGY_MONITOR_H
#define ENERGY_MONITOR_H
#include <hwloc.h>

#include <hwloc/cpukinds.h>

#include <stdint.h>

void frequency_monitor_init( hwloc_topology_t topology) ;
void frequency_compute(double *frequencies) ;

void energy_monitor_init();
uint64_t energy_monitor_get_consumption();

#endif // ENERGY_MONITOR_H