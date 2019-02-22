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
#ifndef X_OCLSOLVER
#define X_OCLSOLVER

#if defined(_WIN32) || defined(_WIN64)
#pragma comment(lib, "opencl.lib") // opencl.lib
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include "OpenCL/cl.hpp"
#else
#include <CL/cl.hpp>
#endif

#include "../X_ENGINE/measurement.hpp"
//#include "../spark/benchmarks.hpp"
#include "isolver.h"
#include "ocl_const.h"
#include "ocl_struct.h"
#include "particle.h"
#include "sph_model.hpp"
#include "util/x_error.h"
#include "x_device.h"
#include <assert.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>

namespace x_engine {
namespace solver {

using std::cout;
using std::endl;
using std::shared_ptr;
using x_engine::ocl_error;
using x_engine::model::particle;
using x_engine::model::partition;
using x_engine::model::sph_model;
// OCL constans block
#define QUEUE_EACH_KERNEL 1

template <class T = float> class ocl_solver : public i_solver {
public:
  typedef shared_ptr<sph_model<T>> model_ptr;

  ocl_solver(model_ptr &m, shared_ptr<device> d) : model(m), dev(d) {
    try {
      this->initialize_ocl();
      // LOCAL_NDRANGE_SIZE = dev->device_work_group_size;
      LOCAL_NDRANGE_SIZE = 512;
    } catch (ocl_error &ex) {
      throw;
    }
  }
  virtual void init_model(const partition &p) {
    this->p = p;
    init_vectors();
    init_buffers();
    init_kernels();
  }

  ~ocl_solver() {}
  virtual void run_neighbour_search() {
    run_grid_cell_indexing();
    run_init_ext_particles();
    run_neighbor_search_kernel();
    int a = 1;
  }
  virtual void run_physic() { run_calculations(); }
  virtual void run_tests() { /*neighbor_search_test();*/
  }
  virtual void synch_preparation() {
    run_calc_cell_id();
    run_single_thread_sort_by_cell_id();
  }

  virtual void show_info() {
    cout << endl
         << "=============INFO FOR SOLVER OF DEVICE " << dev->name
         << "=============" << endl;

    std::cout << "current solver's size :" << p.size() << std::endl;

      cout << "=============END INFO FOR SOLVER OF DEVICE " << dev->name
         << "=============" << endl;
  }

  virtual void set_ordinal_num(int i) { ord_num = i; }

  virtual int get_ordinal_num() { return ord_num; }

  shared_ptr<device> get_dev_ref() { return dev; }

  virtual int get_device_type() { return dev->t; }
  virtual double get_total_work_time() {
    double sum = 0;
    for (double i : work_results)
      sum += i;
    work_results.clear();
    return sum;
  }

private:
  model_ptr model;
  partition p;
  partition p_via_cell_id;
  // TODO Think about moving this feild into the partition struct
  shared_ptr<device> dev;
  std::string msg = dev->name + '\n';
  const std::string cl_program_file = "cl_code\\sph_cl_code.cl";
  std::vector<extendet_particle> neighbor_map;
  // Kernels
  cl::Kernel k_init_ext_particles;
  cl::Kernel k_grid_cell_index;
  cl::Kernel k_neighbour_search;
  cl::Kernel k_calc_cell_id;
  cl::Kernel k_calculations;
  // Buffers
  cl::Buffer b_particles;
  cl::Buffer b_all_particles;
  cl::Buffer b_ext_particles;
  cl::Buffer b_grid_cell_index;
  cl::Buffer b_all_grid_cell_index;
  std::vector<cl::Buffer> buffers_to_release;
  std::vector<double> work_results;
  // Queues
  cl::CommandQueue queue;
  // Programs
  cl::Program program;
  int ord_num;

  int LOCAL_NDRANGE_SIZE;

  // Initialization device-side, setting size and naming
  void init_buffers() {
    create_ocl_buffer("particles", b_particles, CL_MEM_READ_WRITE,
                      p.size() * sizeof(particle<T>));

    create_ocl_buffer("ext_particles", b_ext_particles, CL_MEM_READ_WRITE,
                      p.size() * sizeof(extendet_particle));

    copy_buffer_to_device((void *)&(model->get_particles()[p.start]),
                          b_particles, p.size() * sizeof(particle<T>));
  }

  void init_kernels() {
    create_ocl_kernel("_ker_grid_cell_indexing", k_grid_cell_index);
    create_ocl_kernel("_ker_init_ext_particles", k_init_ext_particles);
    create_ocl_kernel("_ker_neighbour_search", k_neighbour_search);
    create_ocl_kernel("_ker_calc_cell_id", k_calc_cell_id);
    create_ocl_kernel("_ker_calculations", k_calculations);
  }

  void init_vectors() {
    neighbor_map.resize(p.size());
  }

  void init_partition_via_cell_id() {
    particle<T> first_part = model->get_particles().at(p.start);
    particle<T> last_part = model->get_particles().at(p.end - 1);
    p_via_cell_id.start = first_part.cell_id;
    p_via_cell_id.end = last_part.cell_id + 1;
  }

  virtual void init_ext_particles() {}

  void initialize_ocl() {
    int err;
    queue = cl::CommandQueue(dev->context, dev->dev, 0, &err);
    if (err != CL_SUCCESS) {
      throw ocl_error(msg + "Failed to create command queue");
    }
    std::ifstream file(cl_program_file);
    if (!file.is_open()) {
      throw ocl_error(msg +
                      "Could not open file with OpenCL program check "
                      "input arguments oclsourcepath: " +
                      cl_program_file);
    }
    std::string programSource(std::istreambuf_iterator<char>(file),
                              (std::istreambuf_iterator<char>()));
    if (0) {
      programSource =
          "#define _DOUBLE_PRECISSION\n" +
          programSource; // not now it needs double extension check on device
    }
    cl::Program::Sources source(
        1, std::make_pair(programSource.c_str(), programSource.length() + 1));
    program = cl::Program(dev->context, source);
#if defined(__APPLE__)
    err = program.build("-g -cl-opt-disable -I .");
#else
#if INTEL_OPENCL_DEBUG
    err = program.build(OPENCL_DEBUG_PROGRAM_PATH + "-g -cl-opt-disable -I .");
#else
    err = program.build("");
#endif
#endif
    if (err != CL_SUCCESS) {
      std::string compilationErrors;
      compilationErrors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev->dev);
      msg += make_msg(msg, "Compilation failed: ", compilationErrors);
      throw ocl_error(msg);
    }
    std::cout
        << msg
        << "OPENCL program was successfully build. Program file oclsourcepath: "
        << cl_program_file << std::endl;
    return;
  }

  void create_ocl_buffer(const char *name, cl::Buffer &b,
                         const cl_mem_flags flags, const int size) {
    int err;
    b = cl::Buffer(dev->context, flags, size, NULL, &err);
    if (err != CL_SUCCESS) {
      std::string error_m = "Buffer creation failed: ";
      error_m.append(name);
      throw ocl_error(error_m);
    }
  }

  void create_ocl_kernel(const char *name, cl::Kernel &k) {
    int err;
    k = cl::Kernel(program, name, &err);
    if (err != CL_SUCCESS) {
      std::string error_m = "Kernel creation failed: ";
      error_m.append(name);
      throw ocl_error(error_m);
    }
  }

  void copy_buffer_to_device(const void *host_b, cl::Buffer &ocl_b,
                             const int size) {
    // Actualy we should check  size and type
    int err = queue.enqueueWriteBuffer(ocl_b, CL_TRUE, 0, size, host_b);
    if (err != CL_SUCCESS) {
      std::string error_m =
          "Could not enqueue read data from buffer  error code is: ";
      error_m.append(std::to_string(err));
      throw ocl_error(error_m);
    }
    queue.finish();
  }

  void copy_buffer_from_device(void *host_b, const cl::Buffer &ocl_b,
                               const int size) {
    // Actualy we should check  size and type
    int err = queue.enqueueReadBuffer(ocl_b, CL_TRUE, 0, size, host_b);
    if (err != CL_SUCCESS) {
      std::string error_m =
          "Could not enqueue read data from buffer  error code is: ";
      error_m.append(std::to_string(err));
      throw ocl_error(error_m);
    }
    queue.finish();
  }

  unsigned int round_up_with_localNDRange(const unsigned int value) {
    return ((value - 1) / LOCAL_NDRANGE_SIZE + 1) * LOCAL_NDRANGE_SIZE;
  }

  int calc_grid_id(const particle<T> &p) {

    int cell_num_x = model->get_cell_num_x();
    int cell_num_y = model->get_cell_num_y();
    int cell_num_z = model->get_cell_num_z();

    const float GRID_CELL_SIZE = model->get_grid_cell_size();
    const float GRID_CELL_SIZE_INV1 = 1.0f / GRID_CELL_SIZE;

    int A, B, C;
    A = static_cast<int>(p.pos[0] * GRID_CELL_SIZE_INV1);
    B = static_cast<int>(p.pos[1] * GRID_CELL_SIZE_INV1);
    C = static_cast<int>(p.pos[2] * GRID_CELL_SIZE_INV1);
    return A + B * cell_num_x + cell_num_x * cell_num_y * C;
  }

  template <typename T> void print_vector(std::vector<T> vec) {
    for (auto i : vec)
      cout << i << ' ';
    cout << endl;
  }

  unsigned int run_grid_cell_indexing() {
    init_partition_via_cell_id();
    cout << "[GCI]\tp_via_cell_id.start = " << p_via_cell_id.start << endl;
    cout << "[GCI]\tp_via_cell_id.end = " << p_via_cell_id.end << endl;
    cout << "[GCI]\tp_via_cell_id.size() = " << p_via_cell_id.size() << endl;

    create_ocl_buffer("[GCI]\tgrid_cell_index", b_grid_cell_index,
                      CL_MEM_READ_WRITE, p_via_cell_id.size() * sizeof(int));

    copy_buffer_to_device(
        (void *)&(model->get_grid_cell_index()[p_via_cell_id.start]),
        b_grid_cell_index, p_via_cell_id.size() * sizeof(int));

    int ind = 0;
    k_grid_cell_index.setArg(ind++, b_particles);
    k_grid_cell_index.setArg(ind++, p.size());
    k_grid_cell_index.setArg(ind++, p.start);
    k_grid_cell_index.setArg(ind++, p.end);
    k_grid_cell_index.setArg(ind++, b_grid_cell_index);
    k_grid_cell_index.setArg(ind++, model->get_total_cell_num());
    k_grid_cell_index.setArg(ind++, p_via_cell_id.start);
    k_grid_cell_index.setArg(ind++, p_via_cell_id.end);

    int err = queue.enqueueNDRangeKernel(
        k_grid_cell_index, cl::NullRange,
        cl::NDRange(round_up_with_localNDRange(p_via_cell_id.size())),
        cl::NDRange(LOCAL_NDRANGE_SIZE), NULL, NULL);
    queue.flush();
    queue.finish();

    if (err != CL_SUCCESS) {
      std::string error_m = "An ERROR appeared during work of kernel "
                            "_ker_grid_cell_indexing; error code is: ";
      error_m.append(std::to_string(err));
      throw ocl_error(error_m);
    }

    copy_buffer_from_device(
        (void *)&(model->get_grid_cell_index()[p_via_cell_id.start]),
        b_grid_cell_index, p_via_cell_id.size() * sizeof(int));

    int recent_non_empty_cell_value =
        model->get_grid_cell_index()[p_via_cell_id.end - 1];
    int st = p_via_cell_id.start;
    for (int i = p_via_cell_id.end - 1; i >= st; i--) {
      if (i == 0)
        int a = 1;
      if (model->get_grid_cell_index()[i] == NO_CELL_ID) {
        model->get_grid_cell_index()[i] = recent_non_empty_cell_value;
      } else {
        recent_non_empty_cell_value = model->get_grid_cell_index()[i];
      }
    }

    if (!model->is_grid_cell_index_checked()) {
      model->set_grid_cell_index_checked();
      if (model->get_grid_cell_index()[model->get_grid_cell_index().size() -
                                       1] == 0) {
        model->get_grid_cell_index()[model->get_grid_cell_index().size() - 1] =
            model->get_particles().size();
      }
    }
    copy_buffer_to_device(
        (void *)&(model->get_grid_cell_index()[p_via_cell_id.start]),
        b_grid_cell_index, p_via_cell_id.size() * sizeof(int));

    buffers_to_release.push_back(b_grid_cell_index);

    return err;
  }

  unsigned int run_init_ext_particles() {
    copy_buffer_to_device((void *)&neighbor_map[0], b_ext_particles,
                          neighbor_map.size());

    int i = 0;
    k_init_ext_particles.setArg(i++, b_ext_particles);
    k_init_ext_particles.setArg(i++, neighbor_map.size());

    int err = queue.enqueueNDRangeKernel(
        k_init_ext_particles, cl::NullRange,
        cl::NDRange(round_up_with_localNDRange(neighbor_map.size())),
        cl::NDRange(LOCAL_NDRANGE_SIZE), NULL, NULL);
    queue.flush();
    queue.finish();

    if (err != CL_SUCCESS) {
      std::string error_m = "An ERROR appeared during work of kernel "
                            "_ker_init_ext_particles; error code is: ";
      error_m.append(std::to_string(err));
      throw ocl_error(error_m);
    }
    return err;
  }

  unsigned int run_neighbor_search_kernel() {
    create_ocl_buffer("all_particles", b_all_particles, CL_MEM_READ_WRITE,
                      model->size() * sizeof(particle<T>));
    copy_buffer_to_device((void *)&(model->get_particles()[0]), b_all_particles,
                          model->size() * sizeof(particle<T>));

    create_ocl_buffer("all_grid_cell_index", b_all_grid_cell_index,
                      CL_MEM_READ_WRITE,
                      model->get_grid_cell_index().size() * sizeof(int));
    copy_buffer_to_device((void *)&(model->get_grid_cell_index()[0]),
                          b_all_grid_cell_index,
                          model->get_grid_cell_index().size() * sizeof(int));

    std::map<std::string, T> config = model->get_config();

    int i = 0;
    k_neighbour_search.setArg(i++, b_ext_particles);
    k_neighbour_search.setArg(i++, p.size());
    k_neighbour_search.setArg(i++, p.start);
    k_neighbour_search.setArg(i++, p.end);
    k_neighbour_search.setArg(i++, b_all_particles);
    k_neighbour_search.setArg(i++, model->size());
    k_neighbour_search.setArg(i++, b_grid_cell_index);
    k_neighbour_search.setArg(i++, model->get_total_cell_num()); //!!!!!!!!!!
    k_neighbour_search.setArg(i++, model->get_cell_num_x());
    k_neighbour_search.setArg(i++, model->get_cell_num_y());
    k_neighbour_search.setArg(i++, model->get_cell_num_z());
    k_neighbour_search.setArg(i++, model->get_h());
    k_neighbour_search.setArg(i++, model->get_grid_cell_size());
    k_neighbour_search.setArg(i++, model->get_grid_cell_size_inv());
    k_neighbour_search.setArg(i++, model->get_simulation_scale());
    k_neighbour_search.setArg(i++, config["x_min"]);
    k_neighbour_search.setArg(i++, config["y_min"]);
    k_neighbour_search.setArg(i++, config["z_min"]);

    int err = queue.enqueueNDRangeKernel(
        k_neighbour_search, cl::NullRange,
        cl::NDRange(round_up_with_localNDRange(neighbor_map.size())),
        cl::NDRange(LOCAL_NDRANGE_SIZE), NULL, NULL);
    queue.flush();
    queue.finish();

    if (err != CL_SUCCESS) {
      std::string error_m = "An ERROR appeared during work of kernel "
                            "_ker_neighbour_search; error code is: ";
      error_m.append(std::to_string(err));
      throw ocl_error(error_m);
    }

    return err;
  }

  unsigned int run_calc_cell_id() {
    int i = 0;
    k_calc_cell_id.setArg(i++, b_particles);
    k_calc_cell_id.setArg(i++, p.size());
    k_calc_cell_id.setArg(i++, model->get_grid_cell_size_inv());
    k_calc_cell_id.setArg(i++, model->get_cell_num_x());
    k_calc_cell_id.setArg(i++, model->get_cell_num_y());
    k_calc_cell_id.setArg(i++, model->get_cell_num_z());

    int err = queue.enqueueNDRangeKernel(
        k_calc_cell_id, cl::NullRange,
        cl::NDRange(round_up_with_localNDRange(p.size())),
        cl::NDRange(LOCAL_NDRANGE_SIZE), NULL, NULL);
    queue.flush();
    queue.finish();

    if (err != CL_SUCCESS) {
      std::string error_m = "An ERROR appeared during work of kernel "
                            "_ker_calc_cell_id; error code is: ";
      error_m.append(std::to_string(err));
      throw ocl_error(error_m);
    }
    return err;
  }

  unsigned int run_calculations() {
    int i = 0;
    k_calculations.setArg(i++, b_ext_particles);
    k_calculations.setArg(i++, b_particles);
    k_calculations.setArg(i++, p.size());

    int err = queue.enqueueNDRangeKernel(
        k_calculations, cl::NullRange,
        cl::NDRange(round_up_with_localNDRange(p.size())),
        cl::NDRange(LOCAL_NDRANGE_SIZE), NULL, NULL);
    queue.flush();
    queue.finish();

    if (err != CL_SUCCESS) {
      std::string error_m = "An ERROR appeared during work of kernel "
                            "_ker_calculations; error code is: ";
      error_m.append(std::to_string(err));
      throw ocl_error(error_m);
    }

    return err;
  }

  void run_single_thread_sort_by_cell_id() {
    model->arrange_particles_in_partition(p);
  }

  bool neighbor_search_test() {
    // preparing testing data
    std::vector<extendet_particle> expected_neighbor_map(p.size());

    std::vector<particle<T>> solver_particles = model->get_particles();
    solver_particles.erase(solver_particles.begin(),
                           solver_particles.begin() + p.start);
    solver_particles.erase(solver_particles.begin() + p.end,
                           solver_particles.end());

    expected_neighbor_map = single_thread_neighbor_search(solver_particles);

    copy_buffer_from_device((void *)(&neighbor_map[0]), b_ext_particles,
                            neighbor_map.size() * sizeof(extendet_particle));
    bool check = true;

    if (assert_equals(
            "Actual dimension of map doesn't correspond to expected value.",
            expected_neighbor_map.size(), neighbor_map.size())) {

      std::vector<std::vector<int>> check_map(p.size());
      for (auto v : check_map)
        v.resize(NEIGHBOR_COUNT);

      for (int i = 0; i < neighbor_map.size(); i++) {
        std::cout << "[TEST] Check of " << i << " element" << std::endl;
        bool check_neighbor_count;
        int expected_neighbor_count = 0;
        int actual_neighbor_count = 0;
        for (int j = 0; j < NEIGHBOR_COUNT; j++) {
          if (expected_neighbor_map[i].neighbour_list[j] > -1)
            expected_neighbor_count++;

          if (neighbor_map[i].neighbour_list[j] > -1)
            actual_neighbor_count++;

          if (assert_equals("\tActual number of neighbors for current element "
                            "doesn't correspond to expected value",
                            expected_neighbor_count, actual_neighbor_count)) {
            int element_to_find = expected_neighbor_map[i].neighbour_list[j];
            int match_index = get_neighbor_index_by_value(
                neighbor_map[i].neighbour_list, element_to_find);
            if (assert_not_equals("\tNeighbor list for current element doesn't "
                                  "contain expected element",
                                  match_index, -1)) {
              check_map[i][j] = match_index;
            } else {
              check = false;
            }
          } else {
            check = false;
          }
        }
      }
    } else {
      check = false;
    }
    return check;
  }

  // TODO Needs work (after removing grid_cell_index vector from solver)
  std::vector<int>
  single_thread_grid_cell_indexing(const std::vector<particle<T>> &particles) {
    int PARTICLE_COUNT = particles.size();

    particle<T> particle_start = model->get_particles().at(p.start);
    particle<T> particle_end = model->get_particles().at(p.end - 1);

    int grid_cell_count = particle_end.cell_id - particle_start.cell_id + 1;
    std::vector<int> grid_cell_index(grid_cell_count);

    for (int cell_id_to_find = 0; cell_id_to_find < grid_cell_count + 1;
         cell_id_to_find++) {
      if (PARTICLE_COUNT == 0) {
        /* array is empty */
        return;
      } else if (particles[0].cell_id > cell_id_to_find) {
        /* cell_id_to_find is lower than everyone else */
        return;
      } else if (particles[PARTICLE_COUNT - 1].cell_id < cell_id_to_find) {
        /* cell_id_to_find is greater than everyone else */
        return;
      }

      if (cell_id_to_find == 0) {
        grid_cell_index[cell_id_to_find] = 0;
        return;
      }
      if (cell_id_to_find == grid_cell_count) {
        grid_cell_index[cell_id_to_find] = PARTICLE_COUNT;
        return;
      }
      // end check

      // binary search
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

      // find lowest particle_id that gives cell_id_to_find and init
      // grid_cell_index value
      if (!converged && found &&
          particles[particle_id].cell_id == cell_id_to_find) {
        particle_id = low;

        // find lowest particle_id that gives cell_id_to_find
        while ((particle_id > 0) &&
               (particles[particle_id - 1].cell_id == cell_id_to_find)) {
          particle_id--;
        }
      } else {
        particle_id = NO_PARTICLE_ID;
      }
      grid_cell_index[cell_id_to_find] = particle_id;
    }
  }

  std::vector<extendet_particle>
  single_thread_neighbor_search(const std::vector<particle<T>> &particles) {
    const float H = 3.34f;

    // init neighbor map
    extendet_particle alloc_value;
    alloc_value.p_id = -1;
    for (int i = 0; i < NEIGHBOR_COUNT; ++i) {
      alloc_value.neighbour_list[i] = -1;
    }
    std::vector<extendet_particle> neighbor_map(particles.size(), alloc_value);

    // finding neighbors for each particle without checking cell_id
    for (int cur_part_id = 0; cur_part_id < particles.size(); cur_part_id++) {
      std::array<T, NEIGHBOR_COUNT> closest_distanses;
      closest_distanses.fill(pow(H, 2.0));
      // farthest neighbor is index of closest_distances and neigbour_list
      int farthest_neighbor = 0;
      T farthest_squared_distance = closest_distanses[farthest_neighbor];
      int found_count = 0;
      neighbor_map[cur_part_id].p_id = cur_part_id;
      for (int neighb_candid_id = 0; neighb_candid_id < particles.size();
           neighb_candid_id++) {
        if (cur_part_id == neighb_candid_id) {
          continue;
        }
        // calculating the squared distance for neighbor candidate
        T squared_distance = 0;
        for (int i = 0; i < 3; i++) {
          T p1 = particles[cur_part_id].pos[i];
          T p2 = particles[neighb_candid_id].pos[i];
          squared_distance += pow(p1 - p2, 2.0);
        }

        // comparing
        if (squared_distance < farthest_squared_distance) {
          // setting new neighbor and updating closest_distances
          neighbor_map[cur_part_id].neighbour_list[farthest_neighbor] =
              neighb_candid_id;
          closest_distanses[farthest_neighbor] = squared_distance;
          // updating farthest_squared_distance and farthest_neighbor (finding
          // new max distance and its index)
          T max_dist = 0;
          for (int i = 0; i < closest_distanses.size(); i++) {
            if (closest_distanses[i] > max_dist) {
              max_dist = closest_distanses[i];
              farthest_neighbor = i;
            }
          }
          farthest_squared_distance = max_dist;
          found_count++;
        }
      }
    }
    return neighbor_map;
  }

  // Needed for neighbor search test
  /**
   * Finds index of element in array which length is eq NEIGHBOR_COUNT
   * Returns index if such element exists
   * Returns -1 if there is no such element
   */
  int get_neighbor_index_by_value(int *neighbor_array, int value) {
    for (int i = 0; i < NEIGHBOR_COUNT; i++) {
      if (neighbor_array[i] = value) {
        return i;
      }
    }
    return -1;
  }

}; // namespace solver
} // namespace solver
} // namespace x_engine
#endif // X_OCLSOLVER
