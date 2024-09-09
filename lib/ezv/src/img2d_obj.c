#include <stdbool.h>
#include <unistd.h>

#include "error.h"
#include "img2d_obj.h"
#include "stb_image.h"
#include "stb_image_resize2.h"
#include "stb_image_write.h"

void img2d_obj_init (img2d_obj_t *img, unsigned width, unsigned height)
{
  img->width    = width;
  img->height   = height;
  img->channels = 4;
}

void img2d_obj_init_from_file (img2d_obj_t *img, char *filename)
{
  int ok;

  ok = stbi_info (filename, (int *)&img->width, (int *)&img->height,
                  (int *)&img->channels);

  if (!ok)
    exit_with_error ("Cannot get file information for %s (please check that "
                     "the file exists)",
                     filename);
}

unsigned img2d_obj_size (img2d_obj_t *img)
{
  return img->width * img->height * sizeof (unsigned);
}

void img2d_obj_load (img2d_obj_t *img, char *filename, void *buffer)
{
  unsigned char *data = NULL;
  int width, height, nrChannels;

  data = stbi_load (filename, &width, &height, &nrChannels, 4);
  if (data == NULL)
    exit_with_error ("Cannot open %s", filename);

  img->channels = 4;
  memcpy (buffer, data, img2d_obj_size (img));

  stbi_image_free (data);
}

void img2d_obj_store (img2d_obj_t *img, char *filename, void *buffer)
{
  int ok = stbi_write_png (filename, img->width, img->height, img->channels,
                           buffer, img->width * img->channels);
  if (!ok)
    exit_with_error ("Cannot save image into %s file", filename);
}

void img2d_obj_store_resized (img2d_obj_t *img, char *filename, void *buffer,
                              unsigned width, unsigned height)
{
  void *data = NULL;

  data = stbir_resize_uint8_srgb (buffer, img->width, img->height,
                                  img->width * img->channels, NULL, width,
                                  height, width * img->channels, STBIR_RGBA);

  int ok = stbi_write_png (filename, width, height, img->channels, data,
                           width * img->channels);
  if (!ok)
    exit_with_error ("Cannot save image into %s file", filename);

  free (data);
}
