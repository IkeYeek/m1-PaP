#ifndef GMONITOR_IS_DEF
#define GMONITOR_IS_DEF


void gmonitor_init (void);
void gmonitor_clean (void);
void gmonitor_toggle_heat_mode (void);

void __gmonitor_start_iteration (long time);
void __gmonitor_end_iteration (long time);
void __gmonitor_start_tile (long time, int who);
void __gmonitor_end_tile (long time, int who, int x, int y, int width,
                          int height);
void __gmonitor_tile (long start_time, long end_time, int who, int x, int y,
                      int width, int height);

#define gmonitor_start_iteration(t)                                            \
  do {                                                                         \
    if (do_gmonitor)                                                           \
      __gmonitor_start_iteration (t);                                          \
  } while (0)

#define gmonitor_end_iteration(t)                                              \
  do {                                                                         \
    if (do_gmonitor)                                                           \
      __gmonitor_end_iteration (t);                                            \
  } while (0)

#define gmonitor_start_tile(t, c)                                              \
  do {                                                                         \
    if (do_gmonitor)                                                           \
      __gmonitor_start_tile ((t), (c));                                        \
  } while (0)

#define gmonitor_end_tile(t, c, x, y, w, h)                                    \
  do {                                                                         \
    if (do_gmonitor)                                                           \
      __gmonitor_end_tile ((t), (c), (x), (y), (w), (h));                      \
  } while (0)

#define gmonitor_tile(st, et, c, x, y, w, h)                                   \
  do {                                                                         \
    if (do_gmonitor)                                                           \
      __gmonitor_tile ((st), (et), (c), (x), (y), (w), (h));                   \
  } while (0)

extern unsigned do_gmonitor;


#endif
