
/*******************************************************************************
 * The MIT License (MIT)
 *
 * Copyright (c) 2011, 2017 OpenWorm.
 * http://openworm.org
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License
 * which accompanies this distribution, and is available at
 * http://opensource.org/licenses/MIT
 *
 * Contributors:
 *     	OpenWorm - http://openworm.org/people.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/

#ifdef cl_amd_printf
#pragma OPENCL EXTENSION cl_amd_printf : enable
#define PRINTF_ON // this comment because using printf leads to very slow work
                  // on Radeon r9 290x on my machine
                  // don't know why
#elif defined(cl_intel_printf)
#pragma OPENCL EXTENSION cl_intel_printf : enable
#define PRINTF_ON
#endif

#ifdef _DOUBLE_PRECISION
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define _DOUBLE_PRECISION
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif
#endif

#ifdef _DOUBLE_PRECISION
typedef double T;
#else
typedef float T;
#endif

#ifdef _DOUBLE_PRECISION
typedef double4 T4;
#else
typedef float4 T4;
#endif

#include "inc\\ocl_const.h"
#include "inc\\ocl_struct.h"

//#define NO_PARTICLE_ID -1

//#define MAX_NEIGHBOR_COUNT 32
#define CELL_OUT_OF_BOUNDS -1

typedef struct particle_f {
  float4 pos;
  float4 vel;
  size_t type_;
  size_t cell_id;
  float density;
  float pressure;
} particle_f;
#ifdef _DOUBLE_PRECISION
typedef struct particle_d {
  double4 pos;
  double4 vel;
  size_t cell_id;
  size_t type_;
  double density;
  double pressure;
} particle_d;
#endif

typedef struct particle {
  T4 pos;
  T4 vel;
  size_t type_;
  size_t cell_id;
  T density;
  T pressure;
} particle;

#define DIVIDE(a, b) native_divide(a, b)
#define SQRT(x) native_sqrt(x)

/*
#ifdef _DOUBLE_PRECISION
typedef struct particle {
  double4 pos;
  double4 vel;
  size_t cell_id;
  size_t type_;
  double density;
  double pressure;
} particle;
#else
typedef struct particle {
  float4 pos;
  float4 vel;
  size_t type_;
  size_t cell_id;
  float density;
  float pressure;
} particle;
#endif
*/

/** Just for test
 */
__kernel void work_with_struct(__global struct extendet_particle *ext_particles,
                               __global struct
#ifdef _DOUBLE_PRECISION
                               particle_d
#else
                               particle_f
#endif
                                   *particles) {
  int id = get_global_id(0);
#ifdef PRINTF_ON
  if (id == 0) {
    printf("sizeof() of particles_f is %d\n", sizeof(particle_f));
#ifdef _DOUBLE_PRECISION
    printf("sizeof() of particles_d is %d\n", sizeof(particle_d));
#endif
  }
#endif
#ifdef _DOUBLE_PRECISION

  particles[id].pos = (double4)(id, id, id, id);
  particles[id].vel = (double4)(id, id, id, id);
#else
  particles[id].pos = (float4)(id, id, id, id);
  particles[id].vel = (float4)(id, id, id, id);
#endif
  particles[id].type_ = id + 1;
}

/**
 * Calculating start position in particleIndex for every cell
 * Kernel fill up gridCellIndex buffer empty cell
 * (spatial cell which has no particle inside at this time is filling by -1).
 */
__kernel void _ker_grid_cell_indexing(__global struct particle *particles,
                                      uint PARTICLE_COUNT, uint start_id,
                                      uint end_id,
                                      __global uint *grid_cell_index,
                                      uint grid_cell_count, uint start_cell_id,
                                      uint end_cell_id) {
  int local_buffer_index = get_global_id(0);
  int cell_id_to_find = local_buffer_index + start_cell_id;

  // check for trivial cases
  if (cell_id_to_find == 0) {
    grid_cell_index[cell_id_to_find] = 0;
    return;
  }
  if (cell_id_to_find == grid_cell_count) {
    grid_cell_index[cell_id_to_find] = PARTICLE_COUNT;
    return;
  }

  // check for unexpected and unacceptable cases
  // TODO remove comprehensive cases
  if (PARTICLE_COUNT == 0) {
    /* array is empty */
    return;
  } else if (cell_id_to_find < particles[0].cell_id) {
    /* cell_id_to_find is lower than anyone else */
    return;
  } else if (cell_id_to_find > particles[PARTICLE_COUNT - 1].cell_id) {
    /* cell_id_to_find is greater than anyone else */
    return;
  } else if (cell_id_to_find > grid_cell_count) {
    /* cell_id_to_find does not belong model*/
    return;
  } else if (cell_id_to_find < start_cell_id) {
    /* cell_id_to_find does not belong current partition*/
    return;
  } else if (cell_id_to_find > end_cell_id) {
    /* cell_id_to_find does not belong current partition*/
    return;
  }
  // end check

  // print all particles that sytisfies conditions
  // printf("[K7]\t\tid=%d\n", cell_id_to_find);

  // start binary search
  int low = 0;
  int high = PARTICLE_COUNT - 1;
  bool converged = false;
  bool found = false;
  int particle_id = NO_PARTICLE_ID;
  int cur_part_cell_id = -1;

  while (!converged && !found) {
    if (low > high) {
      converged = true;
      particle_id = NO_PARTICLE_ID;
      continue;
    }

    if (low == high) {
      found = true;
      particle_id = low;
      continue;
    }

    int middle = low + (high - low) / 2;
    cur_part_cell_id = particles[middle].cell_id;
    particle_id = middle;
    if (cell_id_to_find <= cur_part_cell_id) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  // end binary search

  // find lowest particle_id that gives cell_id_to_find and init grid_cell_index
  if (!converged && particles[high].cell_id == cell_id_to_find) {
    particle_id = high;
    if (low == high) {
      particle_id = low;
    }
    // find lowest particle_id that gives cell_id_to_find
    while ((particle_id > 0) &&
           (particles[particle_id - 1].cell_id == cell_id_to_find)) {
      particle_id--;
    }
  } else {
    particle_id = NO_PARTICLE_ID;
  }

  grid_cell_index[local_buffer_index] =
      particle_id == NO_PARTICLE_ID ? particle_id : particle_id + start_id;
}

__kernel void _ker_grid_cell_index_alternative(
    __global struct particle *particles, uint PARTICLE_COUNT,
    __global uint *grid_cell_index, uint grid_cell_count) {
  int id = get_global_id(0);
  if (id >= PARTICLE_COUNT)
    return;
  particle cur_part = particles[id];
  int cur_part_cell_id = cur_part.cell_id;
  grid_cell_index[cur_part_cell_id] += 1;
}

/**Initialization of neighbour list by -1
 * what means that it's no neighbors.
 */
__kernel void
_ker_init_ext_particles(__global struct extendet_particle *ext_particles,
                        uint PARTICLE_COUNT) {
  int id = get_global_id(0);

  if (id >= PARTICLE_COUNT) {
    return;
  }

  ext_particles[id].p_id = id;
  for (int i = 0; i < NEIGHBOR_COUNT; ++i) {
    ext_particles[id].neighbour_list[i] = -1;
  }
}

/** Calc current cell id for each particles
 */
__kernel void _ker_calc_cell_id(__global struct particle *particles,
                                uint PARTICLE_COUNT, T hash_grid_cell_size_inv,
                                uint grid_cells_x, uint grid_cells_y,
                                uint grid_cells_z) {
  int id = get_global_id(0);

  if (id >= PARTICLE_COUNT)
    return;
  // particle *part = &particles[id];
  ////! cast to int
  // int A = (int)part->pos.x * hash_grid_cell_size_inv;
  // int B = (int)part->pos.y * hash_grid_cell_size_inv;
  // int C = (int)part->pos.z * hash_grid_cell_size_inv;
  // part->cell_id = A + B * grid_cells_x + C * grid_cells_x * grid_cells_y;
  int A = (int)particles[id].pos.x * hash_grid_cell_size_inv;
  int B = (int)particles[id].pos.y * hash_grid_cell_size_inv;
  int C = (int)particles[id].pos.z * hash_grid_cell_size_inv;
  particles[id].cell_id =
      A + B * grid_cells_x + C * grid_cells_x * grid_cells_y;
}

/** Caculates discrete cell_id coordinates for spatial particle
 */
int4 cellFactors(particle *particle, T xmin, T ymin, T zmin,
                 T hashGridCellSizeInv) {
  // xmin, ymin, zmin
  int4 result;
  result.x = (int)((particle->pos.x - xmin) * hashGridCellSizeInv);
  result.y = (int)((particle->pos.y - ymin) * hashGridCellSizeInv);
  result.z = (int)((particle->pos.z - zmin) * hashGridCellSizeInv);
  return result;
}

/** Calculation of cellId using its discrete coordinates
 */
int cellId(int4 cellFactors_, uint gridCellsX, uint gridCellsY,
           uint gridCellsZ // don't use
) {
  int cellId_ = cellFactors_.x + cellFactors_.y * gridCellsX +
                cellFactors_.z * gridCellsX * gridCellsY;
  return cellId_;
}

/** Returns cell_id of one of the cells, that lie near the particular cell
 */
int search_cell(int cell_id, int delta_x, int delta_y, int delta_z,
                uint grid_cells_x, uint grid_cells_y, uint grid_cells_z,
                uint grid_cell_count) {
  if (delta_x == 0 && delta_y == 0 && delta_z == 0)
    return CELL_OUT_OF_BOUNDS;
  int dx = delta_x;
  int dy = grid_cells_x * delta_y;
  int dz = grid_cells_x * grid_cells_y * delta_z;
  int target_cell_id = cell_id + dx + dy + dz;
  target_cell_id = target_cell_id < 0
                       ? /*target_cell_id + grid_cell_count*/ CELL_OUT_OF_BOUNDS
                       : target_cell_id;
  target_cell_id = target_cell_id >= grid_cell_count
                       ? /*target_cell_id - grid_cell_count*/ CELL_OUT_OF_BOUNDS
                       : target_cell_id;
  return target_cell_id;
}

/**Gets index of max element of the array
 */
int get_max_index(T *d_array) {
  int result;
  T max_d = -1.0;
  for (int i = 0; i < NEIGHBOR_COUNT; i++) {
    if (d_array[i] > max_d) {
      max_d = d_array[i];
      result = i;
    }
  }
  return result;
}

/** Searching for particular particle's neighbors in particular spatial cell.
 *  It takes every particle of particular cell and checks if it satisfies
 *  the condition that distance between particles is <= closest_distance
 */
int search_for_neighbors_in_cell(
    int cell_to_search_in, __global uint *grid_cell_index, particle cur_part,
    int cur_part_id, __global struct particle *particles,
    __global struct extendet_particle *ext_particles, uint PARTICLE_COUNT,
    int *closest_indexes, T *closest_distances, int last_farthest,
    int *found_count) {
  if (cell_to_search_in == CELL_OUT_OF_BOUNDS) {
    return last_farthest;
  }
  int base_particle_id = grid_cell_index[cell_to_search_in];
  int next_particle_id = grid_cell_index[cell_to_search_in + 1];
  int particle_count_in_cell = next_particle_id - base_particle_id;
  T squared_distance;
  int neighbor_particle_id;
  int farthest_neighbor = last_farthest;
  int i = 0;

  //  if (cur_part_id == 0) {
  //    printf("\t\tclosest_distances[farthest_neighbor] = %f \n",
  //           closest_distances[farthest_neighbor]);
  //    closest_distances[farthest_neighbor] = 1.0;
  //  }

  while (i < particle_count_in_cell) {
    // printf("\t\tclosest_distances[farthest_neighbor] = %f \n",
    // closest_distances[farthest_neighbor]);
    // closest_distances[farthest_neighbor] = 1.0;
    if (cur_part_id == 0) {
      // printf("\t\tclosest_distances[farthest_neighbor] = %f \n",
      // closest_distances[farthest_neighbor]);
      // closest_distances[farthest_neighbor] = 1.0;
    }

    neighbor_particle_id = base_particle_id + i;
    if (cur_part_id != neighbor_particle_id) {
      // d - difference vector of cur particle and its neighbor
      T4 d = cur_part.pos - particles[neighbor_particle_id].pos;
      squared_distance = d.x * d.x + d.y * d.y + d.z * d.z;

      if (squared_distance <= closest_distances[farthest_neighbor]) {

        closest_distances[farthest_neighbor] = squared_distance;
        closest_indexes[farthest_neighbor] = neighbor_particle_id;
        if (*found_count < NEIGHBOR_COUNT - 1) {
          (*found_count)++;
          farthest_neighbor = *found_count;
        } else {
          farthest_neighbor = get_max_index(closest_distances);
        }
      }
    }
    i++;
  }
  return farthest_neighbor;
}

void disp(T4 val) {
  printf("\tx=%f;\n\ty=%f;\n\tz=%f;\n\tw=%f;\n", val.x, val.y, val.z, val.w);
}

/**Calculates signum vector for first 3 coordinates of vector value
 */
int4 signum(T4 val) {
  int4 result;
  result.x = (val.x > 0) - (val.x < 0);
  result.y = (val.y > 0) - (val.y < 0);
  result.z = (val.z > 0) - (val.z < 0);
  return result;
}

/**Gets direction vector and set coordinate to 0 if direction coordinate is
 * invalid (if direction points outside simulation area)
 * @param cell_id - cell_id of particle we search neighbors for (we suppose that
 *	this function is called for existing particle and its cell_id is valid)
 */
int4 validate_direction(uint cell_id, int4 d_vector, uint grid_cells_x,
                        uint grid_cells_y, uint grid_cells_z) {

  int gxgy = grid_cells_x * grid_cells_y;
  // double division = cell_id / gxgy;
  int cell_z = (int)(cell_id / gxgy);
  if (cell_z == 0 && d_vector.z == -1)
    d_vector.z = 0;
  if (cell_z == grid_cells_z - 1 && d_vector.z == 1)
    d_vector.z = 0;

  // a is a remainder of cell_id/(grid_cells_x * grid_cells_y)
  int a = cell_id - cell_z * gxgy;
  int cell_y = (int)(a / grid_cells_x);
  if (cell_y == 0 && d_vector.y == -1)
    d_vector.y = 0;
  if (cell_y == grid_cells_y - 1 && d_vector.y == 1)
    d_vector.y = 0;

  // b is remainder of a/grid_cells_x
  int cell_x = a - cell_y * grid_cells_x;
  if (cell_x == 0 && d_vector.x == -1)
    d_vector.x = 0;
  if (cell_x == grid_cells_x - 1 && d_vector.x == 1)
    d_vector.x = 0;

  return d_vector;
}

/** Searches for neigbours of each particle of partition
 */
__kernel void
_ker_neighbour_search(__global struct extendet_particle *ext_particles,
                      uint partition_size, uint p_start_id, uint p_end_id,
                      __global struct particle *particles, uint PARTICLE_COUNT,
                      __global uint *grid_cell_index, uint grid_cell_count,
                      uint grid_cells_x, uint grid_cells_y, uint grid_cells_z,
                      T h, T hash_grid_cell_size, T hash_grid_cell_size_inv,
                      T simulation_scale, T xmin, T ymin, T zmin) {

  int id = get_global_id(0);
  //  if (id == 0) {
  //    printf("grid_cell_count=%d\n", grid_cell_count);
  //  }
  if (p_start_id <= id < p_end_id)
    return;
  particle cur_part = particles[id];

  T h_squared = h * h;
  T closest_distances[NEIGHBOR_COUNT];
  int closest_indexes[NEIGHBOR_COUNT];
  int found_count = 0;
  for (int k = 0; k < NEIGHBOR_COUNT; k++) {
    closest_distances[k] = h_squared;
    closest_indexes[k] = -1;
  }
  // origin point
  T4 p0 = (T4)(xmin, ymin, zmin, 0.0f);
  // coordinates of the particle relatively to origin point
  T4 p = cur_part.pos - p0;
  // if (id == 0) {
  //  printf("cell_id=%d\n", cur_part.cell_id);
  //}
  // if (id == 0) {
  //  printf("cur_part.pos\n");
  //  disp(cur_part.pos);
  //}
  // if (id == 0) {
  //   printf("p\n");
  //   disp(p);
  // }

  // calculating cell starting point (point of cell with the lowest value of
  // coordinates)
  int4 cur_part_cell_factors =
      cellFactors(&cur_part, xmin, ymin, zmin, hash_grid_cell_size_inv);
  T4 cell_starting_point;
  cell_starting_point.x = cur_part_cell_factors.x * hash_grid_cell_size;
  cell_starting_point.y = cur_part_cell_factors.y * hash_grid_cell_size;
  cell_starting_point.z = cur_part_cell_factors.z * hash_grid_cell_size;

  T half_cell_size = hash_grid_cell_size / 2.0;

  T4 cell_centre_point;
  cell_centre_point.x = cell_starting_point.x + half_cell_size;
  cell_centre_point.y = cell_starting_point.y + half_cell_size;
  cell_centre_point.z = cell_starting_point.z + half_cell_size;

  int4 lo = ((p - cell_starting_point) < h);
  int4 search_direction = signum(cell_centre_point - p);

  int4 one = (int4)(1, 1, 1, 1);

  int4 delta;

  int cur_part_cell_id = cur_part.cell_id;

  // delta = one + 2 * lo;
  delta = validate_direction(cur_part_cell_id, search_direction, grid_cells_x,
                             grid_cells_y, grid_cells_z);

  //  if (id == 0) {
  //    printf("delta\n");
  //    printf("\tx=%d;\n\ty=%d;\n\tz=%d;\n", delta.x, delta.y, delta.z);
  //  }
  int search_cells[8];

  search_cells[0] = cur_part_cell_id;
  // determine surrounding cells 1..8
  search_cells[1] = search_cell(cur_part_cell_id, delta.x, 0, 0, grid_cells_x,
                                grid_cells_y, grid_cells_z, grid_cell_count);
  search_cells[2] = search_cell(cur_part_cell_id, 0, delta.y, 0, grid_cells_x,
                                grid_cells_y, grid_cells_z, grid_cell_count);
  search_cells[3] = search_cell(cur_part_cell_id, 0, 0, delta.z, grid_cells_x,
                                grid_cells_y, grid_cells_z, grid_cell_count);
  search_cells[4] =
      search_cell(cur_part_cell_id, delta.x, delta.y, 0, grid_cells_x,
                  grid_cells_y, grid_cells_z, grid_cell_count);
  search_cells[5] =
      search_cell(cur_part_cell_id, delta.x, 0, delta.z, grid_cells_x,
                  grid_cells_y, grid_cells_z, grid_cell_count);
  search_cells[6] =
      search_cell(cur_part_cell_id, 0, delta.y, delta.z, grid_cells_x,
                  grid_cells_y, grid_cells_z, grid_cell_count);
  search_cells[7] =
      search_cell(cur_part_cell_id, delta.x, delta.y, delta.z, grid_cells_x,
                  grid_cells_y, grid_cells_z, grid_cell_count);
  //
  // if (id == 0) {
  //   for (int i = 0; i < 8; i++) {
  //     printf("\t%d cell id = %d\n", i, search_cells[i]);
  //   }
  // }
  int last_farthest = 0;

  for (int i = 0; i < 8; i++) { // 0..7
                                //! check id!
    last_farthest = search_for_neighbors_in_cell(
        search_cells[i], grid_cell_index, cur_part, id, particles,
        ext_particles, PARTICLE_COUNT, closest_indexes, closest_distances,
        last_farthest, &found_count);
    // if (id == 0)
    //  printf("\t%d cell checked\n", i);
  } /*
   last_farthest = search_for_neighbors_in_cell(
       search_cells[0], grid_cell_index, cur_part, id, particles, ext_particles,
       closest_indexes, closest_distances, last_farthest, &found_count);
           */

  struct extendet_particle ext_part;
  //! check id!
  ext_part.p_id = id;
  // storing all found neighbors and their distances into neighborMap buffer
  for (int j = 0; j < NEIGHBOR_COUNT; j++) {
    ext_part.neighbour_list[j] = closest_indexes[j];
    if (closest_indexes[j] >= 0) {
      ext_part.neighbour_distances[j] =
          SQRT(closest_distances[j]) * simulation_scale;
    } else {
      ext_part.neighbour_distances[j] = -1.f;
    }
  }

  //! check id!
  ext_particles[id] = ext_part;
}

__kernel void hashParticles(__global struct
#ifdef _DOUBLE_PRECISION
                            particle_d
#else
                            particle_f
#endif
                                *particles,
                            uint gridCellsX, uint gridCellsY, uint gridCellsZ,
                            float hashGridCellSizeInv, float xmin, float ymin,
                            float zmin, __global uint2 *particleIndex,
                            uint PARTICLE_COUNT) {
  /*int id = get_global_id( 0 );
  if( id >= PARTICLE_COUNT ) return;
  float4 _position = position[ id ];
  int4 cellFactors_ = cellFactors( _position, xmin, ymin, zmin,
  hashGridCellSizeInv ); int cellId_ = cellId( cellFactors_, gridCellsX,
  gridCellsY, gridCellsZ ) & 0xffffff; // truncate to low 16 bits uint2 result;
  PI_CELL_ID( result ) = cellId_;
  PI_SERIAL_ID( result ) = id;
  particleIndex[ id ] = result;*/
}

double absolute(double value) {
  if (value < 0)
    return -value;
  if (value >= 0)
    return value;
}

double calc_distance(__global struct particle *particles, int id1, int id2) {
  particle part1 = particles[id1];
  particle part2 = particles[id2];
  double dist_x = absolute(part1.pos.x - part2.pos.x);
  dist_x = dist_x * dist_x;
  double dist_y = absolute(part1.pos.y - part2.pos.y);
  dist_y = dist_y * dist_y;
  double dist_z = absolute(part1.pos.z - part2.pos.z);
  dist_z = dist_z * dist_z;
  return SQRT(dist_x + dist_y + dist_z);
}

double power(double base, int grade) {
  for (int i = 0; i < grade; i++) {
    base *= base;
  }
  return base;
}

double tetration(double base, int grade) {
  for (int i = 0; i < grade; i++) {
    base = pow(base, base);
  }
}

__kernel void
_ker_calculations(__global struct extendet_particle *ext_particles,
                       __global struct particle *particles,
                       uint PARTICLE_COUNT) {
  double value = 0;
  int id = get_global_id(0);
  if (id >= PARTICLE_COUNT)
    return;

  const int MAX_SHIFT = 50;
  int shift = 0;

  while (id < PARTICLE_COUNT) {
    id += shift;
    particle cur_part = particles[id];
    struct extendet_particle cur_part_map = ext_particles[id];

    const int MAX_LOOP_VALUE = 1000;
    for (int loop = 0; loop < MAX_LOOP_VALUE; loop++) {
      for (int i = 0; i < NEIGHBOR_COUNT; i++) {
        for (int j = 0; j < NEIGHBOR_COUNT; j++) {
          value += tetration(
              SQRT(calc_distance(particles, cur_part_map.neighbour_list[i],
                                 cur_part_map.neighbour_list[j])) /
                  2.0f,
              10000);
        }
      }
    }
    shift++;
  }
}

__kernel void benchmarking() { return; }

void _ker_matrix_multiplication(__global float *A, __global float *B,
                                __global float *C, int widthA, int widthB) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  float value = 0;
  for (int k = 0; k < widthA; k++) {
    value = value + A[k + j * widthA] * B[k * widthB + i];
  }
  C[i + widthA * j] = value;
}

typedef struct {
  int width;
  int height;
  __global float *elements;
} Matrix;
// Thread block size
#define BLOCK_SIZE 16
// Matrix multiplication function called by MatMulKernel()
void matrixMul(Matrix A, Matrix B, Matrix C) {
  float Cvalue = 0;
  int row = get_global_id(1);
  int col = get_global_id(0);
  for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
  C.elements[row * C.width + col] = Cvalue;
}
// Matrix multiplication kernel called by MatMulHost()
__kernel void mat_mul_kernel(int Awidth, int Aheight, __global float *Aelements,
                             int Bwidth, int Bheight, __global float *Belements,
                             int Cwidth, int Cheight, __global float *Celements,
                             int factor) {
  Matrix A = {Awidth, Aheight, Aelements};
  Matrix B = {Bwidth, Bheight, Belements};
  Matrix C = {Cwidth, Cheight, Celements};
  for (int i = 0; i < factor; ++i) {
    matrixMul(A, B, C);
  }
}
