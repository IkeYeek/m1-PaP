

#include <stdio.h>
#include <string.h>

#include "api_funcs.h"
#include "constants.h"
#include "error.h"
#include "ezm.h"
#include "ezp_ctx.h"
#include "global.h"
#include "img_data.h"
#include "mesh_data.h"
#include "monitoring.h"

ezm_recorder_t ezp_monitor = NULL;

char easypap_trace_label[MAX_LABEL] = {0};

unsigned do_trace          = 0;
unsigned trace_may_be_used = 0;
unsigned do_gmonitor       = 0;

extern unsigned trace_starting_iteration;

static void set_default_trace_label (void)
{
  if (easypap_trace_label[0] == '\0') {
    char *str = getenv ("OMP_SCHEDULE");

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
      if (str != NULL)
        snprintf (easypap_trace_label, MAX_LABEL, "%s %s %s (%s) %d/%dx%d",
                  kernel_name, variant_name,
                  strcmp (tile_name, "none") ? tile_name : "", str, DIM, TILE_W,
                  TILE_H);
      else
        snprintf (easypap_trace_label, MAX_LABEL, "%s %s %s %d/%dx%d",
                  kernel_name, variant_name,
                  strcmp (tile_name, "none") ? tile_name : "", DIM, TILE_W,
                  TILE_H);
    } else {
      if (str != NULL)
        snprintf (easypap_trace_label, MAX_LABEL, "%s %s %s (%s) %d/%d",
                  kernel_name, variant_name,
                  strcmp (tile_name, "none") ? tile_name : "", str, NB_CELLS,
                  NB_PATCHES);
      else
        snprintf (easypap_trace_label, MAX_LABEL, "%s %s %s %d/%d", kernel_name,
                  variant_name, strcmp (tile_name, "none") ? tile_name : "",
                  NB_CELLS, NB_PATCHES);
    }
  }
}

void ezp_monitoring_init (unsigned nb_cpus, unsigned nb_gpus)
{
  ezm_init ("lib/ezm", do_display ? 0 : EZM_NO_DISPLAY);

  ezp_monitor = ezm_recorder_create (nb_cpus, nb_gpus);

#ifdef ENABLE_TRACE
  if (trace_may_be_used) {
    char filename[1024];

    if (easypap_mpirun)
      snprintf (filename, 1024, "data/traces/ezv_trace_current.%d.evt",
                easypap_mpi_rank ());
    else
      strcpy (filename, "data/traces/ezv_trace_current.evt");

    set_default_trace_label ();

    ezm_recorder_attach_tracerec (ezp_monitor, filename, easypap_trace_label);

    if (easypap_mode == EASYPAP_MODE_3D_MESHES) {
      ezm_recorder_store_mesh3d_filename (ezp_monitor, easypap_mesh_file);

      ezv_palette_name_t pal = mesh_data_get_palette ();
      if (pal != EZV_PALETTE_UNDEFINED && pal != EZV_PALETTE_CUSTOM)
        ezm_recorder_store_data_palette (ezp_monitor, pal);
    } else {
      ezm_recorder_store_img2d_dim (ezp_monitor, DIM, DIM);
    }

    if (trace_starting_iteration == 1)
      ezm_recorder_enable (ezp_monitor, 1);

    return;
  }
#endif

  if (do_gmonitor) {
    ezm_set_cpu_palette (ezp_monitor, EZV_PALETTE_EASYPAP, 1);

    // Add footprint window
    ezm_helper_add_footprint (ezp_monitor, ctx, &nb_ctx);
    // Add perfmeter
    ezm_helper_add_perfmeter (ezp_monitor, ctx, &nb_ctx);

    ezm_recorder_enable (ezp_monitor, 1);
  }
}

void ezp_monitoring_cleanup (void)
{
  ezm_recorder_destroy (ezp_monitor);
}
