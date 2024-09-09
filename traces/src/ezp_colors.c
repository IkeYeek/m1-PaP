
#include "ezp_colors.h"
#include "ezv_rgba.h"

uint32_t ezp_cpu_colors[EZP_MAX_COLORS + 1];

unsigned ezp_gpu_index[3] = {
    3, // TASK_TYPE_COMPUTE: magenta
    1, // TASK_TYPE_WRITE:   red
    0  // TASK_TYPE_READ:    yellow
};

void ezp_colors_init (void)
{
  ezp_cpu_colors[0]  = ezv_rgb (0xFF, 0xFF, 0x00); // Yellow
  ezp_cpu_colors[1]  = ezv_rgb (0xFF, 0x00, 0x00); // Red
  ezp_cpu_colors[2]  = ezv_rgb (0x00, 0xFF, 0x00); // Green
  ezp_cpu_colors[3]  = ezv_rgb (0xAE, 0x4A, 0xFF); // Purple
  ezp_cpu_colors[4]  = ezv_rgb (0x00, 0xFF, 0xFF); // Cyan
  ezp_cpu_colors[5]  = ezv_rgb (0xB0, 0xB0, 0xB0); // Grey
  ezp_cpu_colors[6]  = ezv_rgb (0x64, 0x64, 0xFF); // Blue
  ezp_cpu_colors[7]  = ezv_rgb (0xFF, 0xBF, 0xF7); // Pale Pink
  ezp_cpu_colors[8]  = ezv_rgb (0xFF, 0xD5, 0x91); // Cream
  ezp_cpu_colors[9]  = ezv_rgb (0xCF, 0xFF, 0xBF); // Pale Green
  ezp_cpu_colors[10] = ezv_rgb (0xF0, 0x80, 0x80); // Light Coral
  ezp_cpu_colors[11] = ezv_rgb (0xE0, 0x00, 0xE0); // Magenta
  ezp_cpu_colors[12] = ezv_rgb (0x77, 0xB5, 0xFE); // Light Blue
  ezp_cpu_colors[13] = ezv_rgb (0x96, 0x4B, 0x00); // Brown
  ezp_cpu_colors[14] = ezv_rgb (0xFF, 0xFF, 0xFF); // White
}
