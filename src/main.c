
#include "constants.h"
#include "cpustat.h"
#include "easypap.h"
#include "events.h"
#include "gpu.h"
#include "hooks.h"
#include "trace_record.h"
#ifdef ENABLE_SHA
#include "hash.h"
#endif
#include "ezp_colors.h"
#include "ezp_ctx.h"
#include "ezv_event.h"

#include <fcntl.h>
#include <hwloc.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/utsname.h>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#define MAX_FILENAME 1024
#define MAX_LABEL 64

int max_iter             = 0;
unsigned refresh_rate    = 0;
unsigned do_display      = 1;
unsigned vsync           = 1;
unsigned picking_enabled = 0;

static char *progname    = NULL;
char *variant_name       = NULL;
char *kernel_name        = NULL;
char *tile_name          = NULL;
char *draw_param         = NULL;
char *config_param       = NULL;
char *easypap_image_file = NULL;
char *easypap_mesh_file  = NULL;

easypap_mode_t easypap_mode = EASYPAP_MODE_UNDEFINED;

static char *output_file           = "./data/perf/data.csv";
static char trace_label[MAX_LABEL] = {0};

unsigned gpu_used                                              = 0;
unsigned use_multiple_gpu                                      = 0;
unsigned easypap_mpirun                                        = 0;
unsigned easypap_gl_buffer_sharing                             = 1;
static int _easypap_mpi_rank                                   = 0;
static int _easypap_mpi_size                                   = 1;
static unsigned master_do_display __attribute__ ((unused))     = 1;
static unsigned start_in_pause                                 = 0;
static unsigned quit_when_done                                 = 0;
static unsigned nb_cores                                       = 1;
unsigned do_first_touch                                        = 0;
static unsigned do_dump __attribute__ ((unused))               = 0;
static unsigned do_thumbs __attribute__ ((unused))             = 0;
static unsigned show_gpu_config                                = 0;
static unsigned list_gpu_variants                              = 0;
static unsigned trace_starting_iteration                       = 1;
static unsigned show_sha256_signature __attribute__ ((unused)) = 0;
static unsigned show_iterations                                = 0;
unsigned use_scotch                                            = 0;
static unsigned data_sync_on_host                              = 1;
static unsigned do_shuffle_cells                               = 0;
static unsigned do_shuffle_partitions                          = 0;

static hwloc_topology_t topology;

unsigned easypap_requested_number_of_threads (void)
{
  char *str = getenv ("OMP_NUM_THREADS");

  if (str == NULL)
    return easypap_number_of_cores ();
  else
    return atoi (str);
}

unsigned easypap_gpu_lane (unsigned gpu_no)
{
  return easypap_requested_number_of_threads () + gpu_no;
}

char *easypap_omp_schedule (void)
{
  char *str = getenv ("OMP_SCHEDULE");
  return (str == NULL) ? "" : str;
}

char *easypap_omp_places (void)
{
  char *str = getenv ("OMP_PLACES");
  return (str == NULL) ? "" : str;
}

unsigned easypap_number_of_cores (void)
{
  return nb_cores;
}

int easypap_mpi_rank (void)
{
  return _easypap_mpi_rank;
}

int easypap_mpi_size (void)
{
  return _easypap_mpi_size;
}

int easypap_proc_is_master (void)
{
  // easypap_mpi_rank == 0 even if !easypap_mpirun
  return easypap_mpi_rank () == 0;
}

void easypap_check_mpi (void)
{
#ifndef ENABLE_MPI
  exit_with_error ("Program was not compiled with -DENABLE_MPI");
#else
  if (!easypap_mpirun)
    exit_with_error ("\n**************************************************\n"
                     "**** MPI variant was not launched using mpirun!\n"
                     "****     Please use --mpi <mpi_run_args>\n"
                     "**************************************************");
#endif
}

void easypap_vec_check (unsigned vec_width_in_bytes, direction_t dir)
{
#ifdef ENABLE_VECTO
  // Order of types must be consistent with that defined in vec_type enum (see
  // api_funcs.h)
  int n = (dir == DIR_HORIZONTAL ? TILE_W : TILE_H);

  if (n < vec_width_in_bytes || n % vec_width_in_bytes)
    exit_with_error ("Tile %s (%d) is too small with respect to vectorization "
                     "requirements and should be a multiple of %d",
                     (dir == DIR_HORIZONTAL ? "width" : "height"), n,
                     vec_width_in_bytes);

#endif
}

static void update_refresh_rate (int p)
{
  static int tab_refresh_rate[] = {1, 2, 5, 10, 100, 1000};
  static int i_refresh_rate     = 0;

  if (easypap_mpirun || (i_refresh_rate == 0 && p < 0) ||
      (i_refresh_rate == 5 && p > 0))
    return;

  i_refresh_rate += p;
  refresh_rate = tab_refresh_rate[i_refresh_rate];
  printf ("< Refresh rate set to: %d >\n", refresh_rate);
}

static void output_perf_numbers (long time_in_us, unsigned nb_iter,
                                 int64_t total_cycles, int64_t total_stalls)
{
  FILE *f = fopen (output_file, "a");
  struct utsname s;

  if (f == NULL)
    exit_with_error ("Cannot open \"%s\" file (%s)", output_file,
                     strerror (errno));

  if (ftell (f) == 0) {
    fprintf (f, "%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n", "machine",
             "size", "tilew", "tileh", "threads", "kernel", "variant", "tiling",
             "iterations", "schedule", "places", "label", "arg", "time",
             "total_cycles", "total_stalls");
  }

  if (uname (&s) < 0)
    exit_with_error ("uname failed (%s)", strerror (errno));

  fprintf (
      f, "%s;%u;%u;%u;%u;%s;%s;%s;%u;%s;%s;%s;%s;%ld;%" PRId64 ";%" PRId64 "\n",
      s.nodename, DIM, TILE_W, TILE_H, easypap_requested_number_of_threads (),
      kernel_name, variant_name, tile_name, nb_iter, easypap_omp_schedule (),
      easypap_omp_places (), trace_label, (draw_param ?: "none"), time_in_us,
      total_cycles, total_stalls);

  fclose (f);
}

static void set_default_trace_label (void)
{
  if (trace_label[0] == '\0') {
    char *str = getenv ("OMP_SCHEDULE");

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
      if (str != NULL)
        snprintf (trace_label, MAX_LABEL, "%s %s %s (%s) %d/%dx%d", kernel_name,
                  variant_name, strcmp (tile_name, "none") ? tile_name : "",
                  str, DIM, TILE_W, TILE_H);
      else
        snprintf (trace_label, MAX_LABEL, "%s %s %s %d/%dx%d", kernel_name,
                  variant_name, strcmp (tile_name, "none") ? tile_name : "",
                  DIM, TILE_W, TILE_H);
    } else {
      if (str != NULL)
        snprintf (trace_label, MAX_LABEL, "%s %s %s (%s) %d/%d", kernel_name,
                  variant_name, strcmp (tile_name, "none") ? tile_name : "",
                  str, NB_CELLS, NB_PATCHES);
      else
        snprintf (trace_label, MAX_LABEL, "%s %s %s %d/%d", kernel_name,
                  variant_name, strcmp (tile_name, "none") ? tile_name : "",
                  NB_CELLS, NB_PATCHES);
    }
  }
}

static void usage (int val);

static void filter_args (int *argc, char *argv[]);

static void generate_log_name (char *dest, size_t size, const char *prefix,
                               const char *extension, const int iterations)
{
  snprintf (dest, size, "%s%s-%s-%s-dim-%d-iter-%d-arg-%s.%s", prefix,
            kernel_name, variant_name, tile_name,
            (easypap_mode == EASYPAP_MODE_2D_IMAGES) ? DIM : NB_CELLS,
            iterations, (draw_param ?: "none"), extension);
}

static void init_phases (void)
{
  if (easypap_mesh_file != NULL) {
    easypap_mode = EASYPAP_MODE_3D_MESHES;
  } else
    easypap_mode = EASYPAP_MODE_2D_IMAGES;

#ifdef ENABLE_MPI
  if (easypap_mpirun) {
    int required = MPI_THREAD_FUNNELED;
    int provided;

    MPI_Init_thread (NULL, NULL, required, &provided);

    if (provided != required)
      PRINT_DEBUG ('M', "Note: MPI thread support level = %d\n", provided);

    MPI_Comm_rank (MPI_COMM_WORLD, &_easypap_mpi_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &_easypap_mpi_size);
    PRINT_DEBUG ('i', "Init phase -1: MPI_Init_thread called (%d/%d)\n",
                 _easypap_mpi_rank, _easypap_mpi_size);
  } else
    PRINT_DEBUG ('i', "Init phase -1: [Process not launched by mpirun]\n");
#endif

  /* Allocate and initialize topology object. */
  hwloc_topology_init (&topology);

  /* Perform the topology detection. */
  hwloc_topology_load (topology);

  nb_cores = hwloc_get_nbobjs_by_type (topology, HWLOC_OBJ_PU);
  PRINT_DEBUG ('t', "%d-core machine detected\n", nb_cores);

#ifdef ENABLE_MONITORING
#ifdef ENABLE_PAPI
  if (do_cache)
    easypap_perfcounter_init (easypap_requested_number_of_threads (),
                              EASYPAP_MONITOR_ALL);
#endif
#endif

  ez_pthread_settopo (topology);

  // Set kernel and variant
  {
    if (kernel_name == NULL)
      kernel_name = DEFAULT_KERNEL;

    if (variant_name == NULL) {
      if (gpu_used)
        variant_name = DEFAULT_GPU_VARIANT;
      else
        variant_name = DEFAULT_VARIANT;
    }
  }

  hooks_establish_bindings (show_gpu_config | list_gpu_variants);

  if (debug_enabled ('d')) {
    picking_enabled = 1;
    if (gpu_used)
      easypap_gl_buffer_sharing =
          0; // Force data sync on host in data debugging mode
  }

  if (easypap_mode == EASYPAP_MODE_3D_MESHES) {
    int flags = 0;

    mesh_data_init ();

    if (use_scotch)
      flags |= MESH3D_PART_USE_SCOTCH;
    if (debug_enabled ('m') | picking_enabled)
      flags |= MESH3D_PART_SHOW_FRONTIERS;

    if (do_shuffle_cells) {
      if (!NB_PATCHES) // force repartitionning
        NB_PATCHES = easypap_mesh_desc.nb_patches;
      mesh3d_shuffle_all_cells (&easypap_mesh_desc);
    }

    // NB_PATCHES == 0  =>  do not repartition
    mesh3d_obj_partition (&easypap_mesh_desc, NB_PATCHES, flags);

    if (do_shuffle_partitions)
      mesh3d_shuffle_partitions (&easypap_mesh_desc);

    if (use_multiple_gpu)
      mesh3d_obj_meta_partition (&easypap_mesh_desc, 2, 0);

    NB_PATCHES = easypap_mesh_desc.nb_patches;

  } else {
    img_data_init ();
  }

  master_do_display = do_display;

  if (!(debug_enabled ('M') || easypap_proc_is_master ()))
    do_display = 0;

  if (!do_display) {
    do_gmonitor     = 0;
    picking_enabled = 0;
  }

  ezp_colors_init ();

  // Create window, initialize rendering, preload image if appropriate
  if (do_display) {
    ezp_ctx_init ();

    ezp_ctx_create ((easypap_mode == EASYPAP_MODE_3D_MESHES)
                        ? EZV_CTX_TYPE_MESH3D
                        : EZV_CTX_TYPE_IMG2D);

    if (easypap_mode == EASYPAP_MODE_3D_MESHES) {
      ezv_mesh3d_set_mesh (ctx[0], &easypap_mesh_desc);

      mesh_data_init_huds (show_iterations);
    } else {
      ezv_img2d_set_img (ctx[0], &easypap_img_desc);

      img_data_init_huds (show_iterations);
    }

    if (picking_enabled) // Use CPU colors to display debug information
      ezv_use_cpu_colors (ctx[0]); // (selected patch, etc.)
  }

  if (the_config != NULL) {
    the_config (config_param);
    PRINT_DEBUG ('i', "Init phase 1: config() hook called\n");
  } else {
    PRINT_DEBUG ('i', "Init phase 1: [no config() hook defined]\n");
  }

  if (do_display) {
    if (easypap_mode == EASYPAP_MODE_3D_MESHES)
      mesh_data_set_default_palette_if_none_defined ();
    else
      img_data_set_default_palette_if_none_defined ();
  }

  if (the_tile_check != NULL)
    the_tile_check ();

  if (gpu_used) {
    gpu_init (show_gpu_config, list_gpu_variants);
    gpu_build_program (list_gpu_variants);
    gpu_alloc_buffers ();
    PRINT_DEBUG ('i', "Init phase 2: GPU initialized\n");
  } else
    PRINT_DEBUG ('i', "Init phase 2: [GPU init not required]\n");

#ifdef ENABLE_MONITORING
#ifdef ENABLE_TRACE
  if (trace_may_be_used) {
    char filename[MAX_FILENAME];

    if (easypap_mpirun)
      snprintf (filename, MAX_FILENAME, "%s/%s.%d%s", DEFAULT_EZV_TRACE_DIR,
                DEFAULT_EZV_TRACE_BASE, easypap_mpi_rank (),
                DEFAULT_EZV_TRACE_EXT);
    else
      strcpy (filename, DEFAULT_EASYVIEW_FILE);

    set_default_trace_label ();

    trace_record_init (filename, easypap_requested_number_of_threads (),
                       easypap_number_of_gpus (), DIM, trace_label,
                       trace_starting_iteration, do_cache);

    if (easypap_mode == EASYPAP_MODE_3D_MESHES) {
      trace_record_set_meshfile (easypap_mesh_file);

      ezv_palette_name_t pal = mesh_data_get_palette ();
      if (pal != EZV_PALETTE_UNDEFINED && pal != EZV_PALETTE_CUSTOM)
        trace_record_set_palette (pal);
    }
  }
#endif
#endif

#ifdef ENABLE_MONITORING
  if (do_gmonitor) {
    gmonitor_init ();
    ezv_ctx_raise (ctx[0]);
  }
#endif

  // OpenCL context is initialized, so we can safely call kernel dependent
  // init() func which may allocate additional buffers.
  if (the_init != NULL) {
    the_init ();
    PRINT_DEBUG ('i', "Init phase 3: init() hook called\n");
  } else {
    PRINT_DEBUG ('i', "Init phase 3: [no init() hook defined]\n");
  }

  // Make sure at leat one task id (0 = anonymous) is stored in the trace
#ifdef ENABLE_TRACE
  if (trace_may_be_used)
    trace_record_commit_task_ids ();
#endif

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    // Allocate memory for cur_img and next_img images
    img_data_alloc ();
  } else {
    mesh_data_alloc ();
  }

  if (do_first_touch) {
    if (the_first_touch != NULL) {
      the_first_touch ();
      PRINT_DEBUG ('i', "Init phase 5: first-touch() hook called\n");
    } else
      PRINT_DEBUG ('i', "Init phase 5: [no first-touch() hook defined]\n");
  } else
    PRINT_DEBUG ('i', "Init phase 5: [first-touch policy not activated]\n");

  // First touch has potentially been performed, so we can load the requested
  // image (if any)
  if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
    img_data_imgload ();

  // Appel de la fonction de dessin spécifique, si elle existe
  if (the_draw != NULL) {
    the_draw (draw_param);
    PRINT_DEBUG ('i', "Init phase 6: kernel-specific draw() hook called\n");
  } else
    PRINT_DEBUG ('i',
                 "Init phase 6: [no kernel-specific draw() hook defined]\n");

  if (gpu_used) {
    gpu_send_data ();
  } else
    PRINT_DEBUG ('i', "Init phase 7: [no GPU data transfer involved]\n");
}

static inline void display_refresh (unsigned iter)
{
  if (do_display) {
    if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
      img_data_refresh (iter);
    else
      mesh_data_refresh (iter);
  }
}

static void handle_events (SDL_Event *event, int pause, int iterations)
{
  int pick;
  ezv_process_event (ctx, nb_ctx, event, NULL, &pick);

  if (easypap_mode == EASYPAP_MODE_3D_MESHES) {
    if (picking_enabled && pick)
      mesh_data_do_pick ();
    if (pause)
      mesh_data_refresh (iterations);
  } else {
    if (picking_enabled && pick)
      img_data_do_pick ();
    if (pause)
      img_data_refresh (iterations);
  }
}

static void force_data_sync (void)
{
  if (!data_sync_on_host) {
    if (!hooks_refresh_img ()) {
      if (gpu_used)
        gpu_retrieve_data ();
    }
  }
  data_sync_on_host = 1;
}

static void do_data_sync_if_required (void)
{
  if (data_sync_on_host)
    return;

  if (gpu_used) {
    if (easypap_gl_buffer_sharing)
      return; // data displayed "in place" on GPU device

    if (!hooks_refresh_img ()) // call refresh_img in priority if defined
      gpu_retrieve_data ();    // fall back to retrieve_data

    data_sync_on_host = 1;
  } else {
    // On the CPU side, just check if refresh_img must be called
    if (hooks_refresh_img ())
      data_sync_on_host = 1;
  }
}

int main (int argc, char **argv)
{
  int stable       = 0;
  int iterations   = 0;
  unsigned iter_no = 1;

  filter_args (&argc, argv);

  if (list_gpu_variants) {
    // bypass complete initialization

    if (kernel_name == NULL)
      kernel_name = DEFAULT_KERNEL;

    TILE_W = TILE_H = DEFAULT_GPU_TILE_SIZE;

    gpu_init (0, list_gpu_variants);
    gpu_build_program (list_gpu_variants);

    exit (0);
  }

  arch_flags_print ();

  init_phases ();

  iter_no = trace_starting_iteration;

  // version graphique
  if (master_do_display) {
    unsigned do_pause     = 0;
    unsigned step_by_step = 0;

    if (!(gpu_used && easypap_gl_buffer_sharing))
      hooks_refresh_img ();

    display_refresh (iterations);

    if (refresh_rate == 0)
      refresh_rate = 1;

    if (start_in_pause && easypap_proc_is_master ())
      do_pause = 1;

    for (int quit = 0; !quit;) {
      int r = 0;

      if (step_by_step)
        do_pause = 1;

      // Récupération éventuelle des événements clavier, souris, etc.
      if (do_display)
        do {
          SDL_Event evt;

          r = easypap_get_event (&evt, do_pause | stable);
          if (r > 0) {
            switch (evt.type) {

            case SDL_QUIT:
              quit = 1;
              break;

            case SDL_KEYDOWN:
              // Si l'utilisateur appuie sur une touche
              switch (evt.key.keysym.sym) {
              case SDLK_ESCAPE:
              case SDLK_q:
                quit = 1;
                break;
              case SDLK_SPACE:
                do_pause ^= 1;
                step_by_step = 0;
                break;
              case SDLK_s:
                do_pause ^= 1;
                step_by_step = 1;
                break;
              case SDLK_DOWN:
                update_refresh_rate (-1);
                break;
              case SDLK_UP:
                update_refresh_rate (1);
                break;
              case SDLK_h:
                gmonitor_toggle_heat_mode ();
                break;
              case SDLK_i:
                ezp_ctx_ithud_toggle ();
                break;
              default:
                handle_events (&evt, do_pause, iterations);
              }
              break;

            case SDL_WINDOWEVENT:
              switch (evt.window.event) {
              case SDL_WINDOWEVENT_CLOSE:
                quit = 1;
                break;
              default:
                handle_events (&evt, do_pause, iterations);
              }
              break;

            default:
              handle_events (&evt, do_pause, iterations);
            }
          }

        } while (do_pause && !quit);

#ifdef ENABLE_MPI
      if (easypap_mpirun)
        MPI_Allreduce (MPI_IN_PLACE, &quit, 1, MPI_INT, MPI_LOR,
                       MPI_COMM_WORLD);
#endif

      if (!stable) {
        if (quit) {
          PRINT_MASTER ("Computation interrupted at iteration %d\n",
                        iterations);
        } else {
          if (max_iter && iterations >= max_iter) {
            PRINT_MASTER ("Computation stopped after %d iterations\n",
                          iterations);
            stable = 1;
          } else {
            int n;

            if (max_iter && iterations + refresh_rate > max_iter)
              refresh_rate = max_iter - iterations;

            monitoring_start_iteration ();

            n = the_compute (refresh_rate);

            monitoring_end_iteration ();

            data_sync_on_host = 0;

            if (n > 0) {
              iterations += n;
              stable = 1;
              PRINT_MASTER ("Computation completed after %d itérations\n",
                            iterations);
            } else
              iterations += refresh_rate;

            // Prepare screen refresh
            do_data_sync_if_required ();

            if (do_thumbs && iterations >= trace_starting_iteration) {
              force_data_sync ();

              if (easypap_proc_is_master ()) {
                if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
                  img_data_save_thumbnail (iter_no++);
                else
                  mesh_data_save_thumbnail (iter_no++);
              }
            }
          }
        }
      }
      display_refresh (iterations);
      if (stable && quit_when_done)
        quit = 1;
    }
  } else {
    // Version non graphique (--no-display)
    long temps;
    struct timeval t1, t2;
    int n;

    if (refresh_rate == 0) {
      if (trace_may_be_used | do_thumbs)
        refresh_rate = 1;
      else if (max_iter)
        refresh_rate = max_iter;
      else
        refresh_rate = INT_MAX;
    }

    gettimeofday (&t1, NULL);

    while (!stable) {
      if (max_iter && iterations >= max_iter) {
        iterations = max_iter;
        stable     = 1;
      } else {
        if (max_iter && iterations + refresh_rate > max_iter)
          refresh_rate = max_iter - iterations;

#ifdef ENABLE_TRACE
        if (trace_may_be_used && (iterations + 1 == trace_starting_iteration))
          do_trace = 1;
#endif

        monitoring_start_iteration ();

        n = the_compute (refresh_rate);

        monitoring_end_iteration ();

        data_sync_on_host = 0;

        if (n > 0) {
          iterations += n;
          stable = 1;
        } else
          iterations += refresh_rate;

        if (do_thumbs && iterations >= trace_starting_iteration) {

          force_data_sync ();

          if (easypap_proc_is_master ()) {
            if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
              img_data_save_thumbnail (iter_no++);
            else
              mesh_data_save_thumbnail (iter_no++);
          }
        }
      }
    }

    gettimeofday (&t2, NULL);

    PRINT_MASTER ("Computation completed after %d iterations\n", iterations);

    temps = TIME_DIFF (t1, t2);

    if (easypap_proc_is_master ()) {
#ifdef ENABLE_PAPI
      if (do_cache) {
        int64_t perfcounters[EASYPAP_NB_COUNTERS];
        if (easypap_perfcounter_get_total_counters (perfcounters) == 0)
          output_perf_numbers (temps, iterations,
                               perfcounters[EASYPAP_TOTAL_CYCLES],
                               perfcounters[EASYPAP_TOTAL_STALLS]);
      } else {
        output_perf_numbers (temps, iterations, -1, -1);
      }
#else
      output_perf_numbers (temps, iterations, -1, -1);
#endif
    }
    PRINT_MASTER ("%ld.%03ld \n", temps / 1000, temps % 1000);
  }

#ifdef ENABLE_SHA
  if (show_sha256_signature) {
    char filename[MAX_FILENAME];

    force_data_sync ();

    if (easypap_proc_is_master ()) {
      generate_log_name (filename, MAX_FILENAME, "data/hash/", "sha256",
                         iterations);
      if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
        build_hash_and_store_to_file (image, DIM * DIM * sizeof (unsigned),
                                      filename);
      else
        build_hash_and_store_to_file (mesh_data, NB_CELLS * sizeof (float),
                                      filename);
    }
  }
#endif

  // Check if final image should be dumped on disk
  if (do_dump) {

    force_data_sync ();

    if (easypap_proc_is_master ()) {
      char filename[MAX_FILENAME];

      if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
        generate_log_name (filename, MAX_FILENAME, "data/dump/", "png",
                           iterations);
        img_data_dump_to_file (filename);
        printf ("Image data dumped to %s\n", filename);
      } else {
        generate_log_name (filename, MAX_FILENAME, "data/dump/", "raw",
                           iterations);
        mesh_data_dump_to_file (filename);
        printf ("Mesh data dumped to %s\n", filename);
      }
    }
  }

#ifdef ENABLE_MONITORING
#ifdef ENABLE_TRACE
  if (trace_may_be_used)
    trace_record_finalize ();
#endif
#endif

  if (the_finalize != NULL)
    the_finalize ();

  img_data_free ();

#ifdef ENABLE_MPI
  if (easypap_mpirun)
    MPI_Finalize ();
#endif

#ifdef ENABLE_MONITORING
#ifdef ENABLE_PAPI
  if (do_cache)
    easypap_perfcounter_finalize ();
#endif
#endif

  return 0;
}

static void usage (int val)
{
  fprintf (stderr, "Usage: %s [options]\n", progname);
  fprintf (
      stderr,
      "options can be:\n"
      "\t-a\t| --arg <string>\t: pass argument <string> to draw function\n"
      "\t-c\t| --config <string>\t: pass config argument <string> to config "
      "function\n"
      "\t-d\t| --debug <flags>\t: enable debug messages (see debug.h)\n"
      "\t-du\t| --dump\t\t: dump final image to disk\n"
      "\t-ft\t| --first-touch\t\t: touch memory on different cores\n"
      "\t-g\t| --gpu\t\t\t: use GPU device\n"
      "\t-h\t| --help\t\t: display help\n"
      "\t-i\t| --iterations <n>\t: stop after n iterations\n"
      "\t-k\t| --kernel <name>\t: use <name> computation kernel\n"
      "\t-lb\t| --label <name>\t: assign name <label> to current run\n"
      "\t-lgv\t| --list-gpu-variants\t: list GPU variants\n"
      "\t-l\t| --load-image <file>\t: use PNG image <file>\n"
      "\t-lm(s)\t| --load-mesh <file>\t: load 3D mesh from <file> (use -lms to "
      "shuffle cells)\n"
      "\t-m \t| --monitoring\t\t: enable graphical thread monitoring\n"
      "\t-mg\t| --multi-gpu\t\t: use multiple GPUs if available\n"
      "\t-mpi\t| --mpirun <args>\t: pass <args> to the mpirun MPI process "
      "launcher\n"
      "\t-n\t| --no-display\t\t: avoid graphical display overhead\n"
      "\t-np\t| --nb-patches <N>\t: use N patches\n"
      "\t-nt\t| --nb-tiles <N>\t: use N x N tiles\n"
      "\t-nbs\t| --no-gl-buffer-share\t: disable OpenGL buffer sharing\n"
      "\t-nvs\t| --no-vsync\t\t: disable vertical sync\n"
      "\t-of\t| --output-file <file>\t: output performance numbers in <file>\n"
      "\t-p\t| --pause\t\t: pause between iterations (press space to "
      "continue)\n"
      "\t-pc\t| --perf-counters\t: collect performance counters \n"
      "\t-q\t| --quit\t\t: exit once iterations are done\n"
      "\t-r\t| --refresh-rate <N>\t: display only 1/Nth of images\n"
      "\t-s\t| --size <DIM>\t\t: use image of size DIM x DIM\n"
      "\t-sh\t| --show-hash\t\t: display SHA256 hash of last image\n"
      "\t-si\t| --show-iterations\t: display iterations in main window\n"
      "\t-sd\t| --show-devices\t: display GPU devices\n"
      "\t-tn\t| --thumbnails\t\t: generate thumbnails\n"
      "\t-tni\t| --thumbnails-iter <n>\t: generate thumbnails starting from "
      "iteration n\n"
      "\t-tw\t| --tile-width <W>\t: use tiles of width W\n"
      "\t-th\t| --tile-height <H>\t: use tiles of height H\n"
      "\t-ts\t| --tile-size <TS>\t: use tiles of size TS x TS\n"
      "\t-t\t| --trace\t\t: enable trace\n"
      "\t-ti\t| --trace-iter <n>\t: enable trace starting from iteration n\n"
      "\t-v\t| --variant <name>\t: select kernel variant <name>\n"
      "\t-wt\t| --with-tile <name>\t: use do_tile_<name> tiling function\n");

  exit (val);
}

static void usage_error (char *msg)
{
  fprintf (stderr, "%s\n", msg);
  usage (1);
}

static void warning (char *option, char *flag1, char *flag2)
{
  if (flag2 != NULL)
    fprintf (
        stderr,
        "Warning: option %s ignored because neither %s nor %s are defined\n",
        option, flag1, flag2);
  else
    fprintf (stderr, "Warning: option %s ignored because %s is not defined\n",
             option, flag1);
}

static void filter_args (int *argc, char *argv[])
{
  progname = argv[0];

  // Filter args
  //
  argv++;
  (*argc)--;
  while (*argc > 0) {
    if (!strcmp (*argv, "--no-vsync") || !strcmp (*argv, "-nvs")) {
      vsync = 0;
    } else if (!strcmp (*argv, "--no-gl-buffer-share") ||
               !strcmp (*argv, "-nbs")) {
      easypap_gl_buffer_sharing = 0;
    } else if (!strcmp (*argv, "--gdb") || !strcmp (*argv, "--lldb")) {
      // ignore the flag
    } else if (!strcmp (*argv, "--no-display") || !strcmp (*argv, "-n")) {
      do_display = 0;
    } else if (!strcmp (*argv, "--pause") || !strcmp (*argv, "-p")) {
      start_in_pause = 1;
    } else if (!strcmp (*argv, "--quit") || !strcmp (*argv, "-q")) {
      quit_when_done = 1;
    } else if (!strcmp (*argv, "--help") || !strcmp (*argv, "-h")) {
      usage (0);
    } else if (!strcmp (*argv, "--show-devices") || !strcmp (*argv, "-sd")) {
#if defined(ENABLE_OPENCL) || defined(ENABLE_CUDA)
      show_gpu_config = 1;
      gpu_used        = GPU_CAN_BE_USED;
#else
      warning (*argv, "ENABLE_OPENCL", "ENABLE_CUDA");
#endif
    } else if (!strcmp (*argv, "--show-iterations") || !strcmp (*argv, "-si")) {
      show_iterations = 1;
    } else if (!strcmp (*argv, "--show-hash") || !strcmp (*argv, "-sh")) {
#ifdef ENABLE_SHA
      show_sha256_signature = 1;
#else
      warning (*argv, "ENABLE_SHA", NULL);
#endif
    } else if (!strcmp (*argv, "--list-gpu-variants") ||
               !strcmp (*argv, "-lgv")) {
#if defined(ENABLE_OPENCL) || defined(ENABLE_CUDA)
      list_gpu_variants = 1;
      gpu_used          = GPU_CAN_BE_USED;
      do_display        = 0;
#else
      warning (*argv, "ENABLE_OPENCL", "ENABLE_CUDA");
#endif
    } else if (!strcmp (*argv, "--first-touch") || !strcmp (*argv, "-ft")) {
      do_first_touch = 1;
    } else if (!strcmp (*argv, "--monitoring") || !strcmp (*argv, "-m")) {
      do_gmonitor = 1;
    } else if (!strcmp (*argv, "--trace") || !strcmp (*argv, "-t")) {
#ifndef ENABLE_TRACE
      warning (*argv, "ENABLE_TRACE", NULL);
#else
      trace_may_be_used = 1;
#endif
    } else if (!strcmp (*argv, "--perf-counters") || !strcmp (*argv, "-pc")) {
#ifndef ENABLE_PAPI
      warning (*argv, "ENABLE_PAPI", NULL);
#else
      do_cache       = 1;
      int perf_event = open ("/proc/sys/kernel/perf_event_paranoid", O_RDONLY);
      if (perf_event == -1) {
        fprintf (
            stdout,
            "Couldn't open /proc/sys/kernel/perf_event_paranoid.\nPerf counter "
            "recording aborted.\n");
        do_cache = 0;
      } else {
        char perf_event_value[2];
        if (read (perf_event, &perf_event_value, 2 * sizeof (char)) <= 0) {
          fprintf (stdout,
                   "Couldn't read value in "
                   "/proc/sys/kernel/perf_event_paranoid.\nPerf counter "
                   "recording aborted.\n");
          do_cache = 0;
        } else {
          if (atoi (perf_event_value) > 0) {
            fprintf (stdout,
                     "Warning: PAPI perf counter record may crash. Please "
                     "set /proc/sys/kernel/perf_event_paranoid to 0 (or -1) "
                     "or run as root.\nPerf counter recording aborted.\n");
            do_cache = 0;
          }
        }
      }
#endif
    } else if (!strcmp (*argv, "--trace-iter") || !strcmp (*argv, "-ti")) {
      if (*argc == 1)
        usage_error ("Error: starting iteration is missing");
      (*argc)--;
      argv++;
#ifndef ENABLE_TRACE
      warning (*argv, "ENABLE_TRACE", NULL);
#else
      trace_starting_iteration = atoi (*argv);
      trace_may_be_used        = 1;
#endif
    } else if (!strcmp (*argv, "--thumbnails") || !strcmp (*argv, "-tn")) {
      do_thumbs = 1;
    } else if (!strcmp (*argv, "--thumbnails-iter") ||
               !strcmp (*argv, "-tni")) {
      if (*argc == 1)
        usage_error ("Error: starting iteration is missing");
      (*argc)--;
      argv++;
      trace_starting_iteration = atoi (*argv);
      do_thumbs                = 1;
    } else if (!strcmp (*argv, "--dump") || !strcmp (*argv, "-du")) {
      do_dump = 1;
    } else if (!strcmp (*argv, "--arg") || !strcmp (*argv, "-a")) {
      if (*argc == 1)
        usage_error ("Error: parameter string is missing");
      (*argc)--;
      argv++;
      draw_param = *argv;
    } else if (!strcmp (*argv, "--config") || !strcmp (*argv, "-c")) {
      if (*argc == 1)
        usage_error ("Error: parameter string is missing");
      (*argc)--;
      argv++;
      config_param = *argv;
    } else if (!strcmp (*argv, "--label") || !strcmp (*argv, "-lb")) {
      if (*argc == 1)
        usage_error ("Error: parameter string is missing");
      (*argc)--;
      argv++;
      snprintf (trace_label, MAX_LABEL, "%s", *argv);
    } else if (!strcmp (*argv, "--mpirun") || !strcmp (*argv, "-mpi")) {
#ifndef ENABLE_MPI
      warning (*argv, "ENABLE_MPI", NULL);
      (*argc)--;
      argv++;
#else
      if (*argc == 1)
        usage_error ("Error: parameter string is missing");
      (*argc)--;
      argv++;
      easypap_mpirun = 1;
#endif
    } else if (!strcmp (*argv, "--gpu") || !strcmp (*argv, "-g")) {
#if defined(ENABLE_OPENCL) || defined(ENABLE_CUDA)
      gpu_used = GPU_CAN_BE_USED;
#else
      warning (*argv, "ENABLE_OPENCL", "ENABLE_CUDA");
#endif
    } else if (!strcmp (*argv, "--multi-gpu") || !strcmp (*argv, "-mg")) {
#if defined(ENABLE_OPENCL) || defined(ENABLE_CUDA)
      gpu_used                  = GPU_CAN_BE_USED;
      use_multiple_gpu          = 1;
      easypap_gl_buffer_sharing = 0; // disable OpenGL buffer sharing
#else
      warning (*argv, "ENABLE_OPENCL", "ENABLE_CUDA");
#endif
    } else if (!strcmp (*argv, "--kernel") || !strcmp (*argv, "-k")) {
      if (*argc == 1)
        usage_error ("Error: kernel name is missing");
      (*argc)--;
      argv++;
      kernel_name = *argv;
    } else if (!strcmp (*argv, "--with-tile") || !strcmp (*argv, "-wt")) {
      if (*argc == 1)
        usage_error ("Error: tile function suffix is missing");
      (*argc)--;
      argv++;
      tile_name = *argv;
    } else if (!strcmp (*argv, "--load-image") || !strcmp (*argv, "-l")) {
      if (easypap_image_file != NULL)
        usage_error ("Error: --load-image used multiple times");
      if (easypap_mesh_file != NULL)
        usage_error ("Error: --load-image and --load-mesh are mutually "
                     "exclusive options");
      if (*argc == 1)
        usage_error ("Error: filename is missing");
      (*argc)--;
      argv++;
      easypap_image_file = *argv;
    } else if (!strcmp (*argv, "--load-mesh") || !strcmp (*argv, "-lm") ||
               !strcmp (*argv, "-lms") || !strcmp (*argv, "-lmsc") || !strcmp (*argv, "-lmsp")) {
      if (easypap_mesh_file != NULL)
        usage_error ("Error: --load-mesh used multiple times");
      if (easypap_image_file != NULL)
        usage_error ("Error: --load-image and --load-mesh are mutually "
                     "exclusive options");
      if (*argc == 1)
        usage_error ("Error: filename is missing");
      if (!strcmp (*argv, "-lms")) { // shuffle cells AND partitions
        do_shuffle_cells = 1;
        do_shuffle_partitions = 1;
      } else if (!strcmp (*argv, "-lmsc"))
        do_shuffle_cells = 1;
      else if (!strcmp (*argv, "-lmsp"))
        do_shuffle_partitions = 1;
      (*argc)--;
      argv++;
      easypap_mesh_file = *argv;
    } else if (!strcmp (*argv, "--size") || !strcmp (*argv, "-s")) {
      if (*argc == 1)
        usage_error ("Error: DIM is missing");
      (*argc)--;
      argv++;
      DIM = atoi (*argv);
    } else if (!strcmp (*argv, "--nb-tiles") || !strcmp (*argv, "-nt")) {
      if (*argc == 1)
        usage_error ("Error: number of tiles is missing");
      (*argc)--;
      argv++;
      NB_TILES_X = atoi (*argv);
      NB_TILES_Y = NB_TILES_X;
    } else if (!strcmp (*argv, "--nb-patches") || !strcmp (*argv, "-np")) {
      if (*argc == 1)
        usage_error ("Error: number of tiles is missing");
      (*argc)--;
      argv++;
      NB_TILES_X = atoi (*argv);
      use_scotch = 1;
    } else if (!strcmp (*argv, "--tile-width") || !strcmp (*argv, "-tw")) {
      if (*argc == 1)
        usage_error ("Error: tile width is missing");
      (*argc)--;
      argv++;
      TILE_W = atoi (*argv);
    } else if (!strcmp (*argv, "--tile-height") || !strcmp (*argv, "-th")) {
      if (*argc == 1)
        usage_error ("Error: tile height is missing");
      (*argc)--;
      argv++;
      TILE_H = atoi (*argv);
    } else if (!strcmp (*argv, "--tile-size") || !strcmp (*argv, "-ts")) {
      if (*argc == 1)
        usage_error ("Error: tile size is missing");
      (*argc)--;
      argv++;
      TILE_W = atoi (*argv);
      TILE_H = TILE_W;
    } else if (!strcmp (*argv, "--variant") || !strcmp (*argv, "-v")) {
      if (*argc == 1)
        usage_error ("Error: variant name is missing");
      (*argc)--;
      argv++;
      variant_name = *argv;
    } else if (!strcmp (*argv, "--iterations") || !strcmp (*argv, "-i")) {
      if (*argc == 1)
        usage_error ("Error: number of iterations is missing");
      (*argc)--;
      argv++;
      max_iter = atoi (*argv);
    } else if (!strcmp (*argv, "--refresh-rate") || !strcmp (*argv, "-r")) {
      if (*argc == 1)
        usage_error ("Error: refresh rate is missing");
      (*argc)--;
      argv++;
      refresh_rate = atoi (*argv);
    } else if (!strcmp (*argv, "--debug") || !strcmp (*argv, "-d")) {
      if (*argc == 1)
        usage_error ("Error: debug flags list is missing");
      (*argc)--;
      argv++;
      debug_init (*argv);
    } else if (!strcmp (*argv, "--output-file") || !strcmp (*argv, "-of")) {
      if (*argc == 1)
        usage_error ("Error: filename is missing");
      (*argc)--;
      argv++;
      output_file = *argv;
    } else {
      fprintf (stderr, "Error: unknown option %s\n", *argv);
      usage (1);
    }

    (*argc)--;
    argv++;
  }

#ifdef ENABLE_TRACE
  if (trace_may_be_used && do_display) {
    fprintf (stderr,
             "Warning: disabling display because tracing was requested\n");
    do_display = 0;
  }
#endif
}
