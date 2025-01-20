#include "easypap.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <omp.h>

#define MAX_NBVILLES 22

typedef int DTab_t[MAX_NBVILLES][MAX_NBVILLES];
typedef int chemin_t[MAX_NBVILLES];

/* dernier minimum trouv� */
int minimum = INT_MAX;

/* tableau des distances */
DTab_t distance;

/* nombre de villes */
int nbVilles = 13;

/* profondeur du parallélisme */
int grain = 1;

#define MAXX 100
#define MAXY 100
typedef struct
{
  int x, y;
} coor_t;

typedef coor_t coortab_t[MAX_NBVILLES];

void initialisation ()
{
  /* initialisation du tableau des distances */
  /* on positionne les villes aléatoirement sur une carte MAXX x MAXY  */
  minimum = INT_MAX;
  coortab_t lesVilles;

  int i, j;
  int dx, dy;

  for (i = 0; i < nbVilles; i++) {
    lesVilles[i].x = rand () % MAXX;
    lesVilles[i].y = rand () % MAXY;
  }

  for (i = 0; i < nbVilles; i++)
    for (j = 0; j < nbVilles; j++) {
      dx             = lesVilles[i].x - lesVilles[j].x;
      dy             = lesVilles[i].y - lesVilles[j].y;
      distance[i][j] = (int)sqrt ((double)((dx * dx) + (dy * dy)));
    }
}

/* résolution du problème du voyageur de commerce */

int present (int ville, int mask)
{
  return mask & (1 << ville);
}

void verifier_minimum (int lg, chemin_t chemin)
{
    if (lg + distance[0][chemin[nbVilles - 1]] < minimum) {
      minimum = lg + distance[0][chemin[nbVilles - 1]];
      printf ("%3d par %d :", minimum, omp_get_thread_num ());
      for (int i = 0; i < nbVilles; i++)
        printf ("%2d ", chemin[i]);
      printf ("\n");
    }
}

static void tsp_monitor (int etape, int lg, chemin_t chemin, int mask);

void tsp_seq (int etape, int lg, chemin_t chemin, int mask)
{
  int ici, dist;

  if (etape == nbVilles)
    verifier_minimum (lg, chemin);
  else {
    ici = chemin[etape - 1];

    for (int i = 1; i < nbVilles; i++) {
      if (!present (i, mask)) {
        chemin[etape] = i;
        dist          = distance[ici][i];
        tsp_seq (etape + 1, lg + dist, chemin, mask | (1 << i));
      }
    }
  }
}

// TD2

void tsp_ompfor (int etape, int lg, chemin_t chemin, int mask)
{

}

void tsp_ompcol4 ()
{
 
}

void tsp_ompcol3 ()
{
 
}

void tsp_ompcol2 ()
{
 
}

// TD3

void tsp_omptaskwait (int etape, int lg, chemin_t chemin, int mask)
{
  
}

void tsp_omptaskpriv (int etape, int lg, chemin_t chemin, int mask)
{

}

void tsp_omptaskdyn (int etape, int lg, chemin_t chemin, int mask)
{
 
}

int tsp_compute_seq (unsigned nb_iter)
{
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    chemin_t chemin;
    chemin[0] = 0;
    tsp_monitor (1, 0, chemin, 1);
    swap_images ();
  }
  return 0;
}

int tsp_compute_taskdyn (unsigned nb_iter)
{
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    chemin_t chemin;
    chemin[0] = 0;
    tsp_omptaskdyn (1, 0, chemin, 1);
    swap_images ();
  }
  return 0;
}

int tsp_compute_taskwait (unsigned nb_iter)
{
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    chemin_t chemin;
    chemin[0] = 0;
    tsp_omptaskwait (1, 0, chemin, 1);
    swap_images ();
  }
  return 0;
}

int tsp_compute_taskpriv (unsigned nb_iter)
{
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    chemin_t chemin;
    chemin[0] = 0;
    tsp_omptaskpriv (1, 0, chemin, 1);
    swap_images ();
  }
  return 0;
}

int tsp_compute_ompfor (unsigned nb_iter)
{
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    omp_set_max_active_levels (grain+1);
    chemin_t chemin;
    chemin[0] = 0;
    tsp_ompfor (1, 0, chemin, 1);
    swap_images ();
  }
  return 0;
}

int tsp_compute_ompcol2 (unsigned nb_iter)
{
  grain = 2;
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    tsp_ompcol2 ();
    swap_images ();
  }
  return 0;
}

int tsp_compute_ompcol3 (unsigned nb_iter)
{
  grain = 3;
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    tsp_ompcol3 ();
    swap_images ();
  }
  return 0;
}

int tsp_compute_ompcol4 (unsigned nb_iter)
{
  grain = 4;
  for (int step = 1; step <= nb_iter; step++) {

    initialisation ();
    tsp_ompcol4 ();
    swap_images ();
  }
  return 0;
}

void tsp_init ()
{
  srand (1234);
}

void tsp_draw (char *param)
{
  sscanf (param, "%d-%d", &nbVilles, &grain);
  if (nbVilles < 5 && nbVilles > MAX_NBVILLES) {
    fprintf (stderr, "nbVilles incorrect");
  }
}

static void emplacement(int *x, int *y, int *largeur, int *hauteur, chemin_t chemin, int etape){
  if (etape > grain)
    return;
  if (*largeur > *hauteur){
    *largeur /= nbVilles - 1;
    *x += (chemin[etape] - 1) * *largeur;
  } else {
    *hauteur /= nbVilles - 1;
    *y += (chemin[etape] - 1) * *hauteur;
  }
  emplacement (x, y, largeur, hauteur, chemin, etape + 1);
}

static void tsp_monitor (int etape, int lg, chemin_t chemin, int mask)
{
  int x = 0, y = 0, largeur = DIM, hauteur = DIM;
  emplacement (&x, &y, &largeur, &hauteur, chemin, 1);
  uint64_t clock = monitoring_start_tile (omp_get_thread_num ());
  tsp_seq (etape, lg, chemin, mask);
  monitoring_end_tile (clock, x, y, largeur, hauteur, omp_get_thread_num ());
}
