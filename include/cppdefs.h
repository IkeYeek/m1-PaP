#ifndef __cplusplus
#define RESTRICT restrict
#define EXTERN
#else
#define RESTRICT __restrict__
#define EXTERN extern "C"
#endif