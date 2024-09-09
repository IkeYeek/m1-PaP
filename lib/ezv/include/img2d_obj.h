#ifndef IMG2D_OBJ_H
#define IMG2D_OBJ_H


typedef struct
{
  unsigned width;
  unsigned height;
  unsigned channels;
} img2d_obj_t;

void img2d_obj_init (img2d_obj_t *img, unsigned width, unsigned height);
void img2d_obj_init_from_file (img2d_obj_t *img, char *filename);

unsigned img2d_obj_size (img2d_obj_t *img);

void img2d_obj_load (img2d_obj_t *img, char *filename, void *buffer);
void img2d_obj_store (img2d_obj_t *img, char *filename, void *buffer);
void img2d_obj_store_resized (img2d_obj_t *img, char *filename, void *buffer,
                              unsigned width, unsigned height);


#endif
