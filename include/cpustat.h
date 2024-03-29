#ifndef CPUSTAT_IS_DEF
#define CPUSTAT_IS_DEF

#ifdef ENABLE_SDL

void cpustat_init (int x, int y);
void cpustat_reset (long now);
void cpustat_start_work (long now, int who);
long cpustat_finish_work (long now, int who);
void cpustat_deduct_idle (long duration, int who);
void cpustat_freeze (long now);
void cpustat_display_stats (void);
void cpustat_clean (void);

#endif

#endif
