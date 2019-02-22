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
#ifndef X_SPHMODEL
#define X_SPHMODEL

#include "particle.h"
#include "util/x_error.h"
#include <array>
#include <assert.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
namespace x_engine {
namespace model {
enum LOADMODE { NOMODE = -1, PARAMS, MODEL, POS, VEL };
const float H = 3.34f;
const float H_INV = 1.f / H;
const float GRID_CELL_SIZE = 2.0f * H;
const float GRID_CELL_SIZE_INV = 1.0f / GRID_CELL_SIZE;
const float R_0 = 0.5f * H;
/* const block end */
struct partition {
  /**each device has its own partition
   * in which we define where starts
   * and end particles for this device.
   */
  size_t start;
  size_t end;
  size_t size() { return end - start; }
  bool contains_particle(size_t id) { return start <= id && id < end; }
};
template <class T = float, class container = std::vector<particle<T>>>
class sph_model {
  typedef std::map<std::string, T> sph_config;

public:
  sph_model(const std::string &config_file) {
    config = {{"particles", T()}, {"x_max", T()}, {"x_min", T()},
              {"y_max", T()},     {"y_min", T()}, {"z_max", T()},
              {"z_min", T()},     {"mass", T()},  {"time_step", T()},
              {"rho0", T()}};
    read_model(config_file);
    default_particles = particles;
    arrange_particles();
    std::cout << "Model was loaded: " << particles.size() << " particles."
              << std::endl;
  }
  const sph_config &get_config() const { return config; }
  const container &get_particles() const { return particles; }
  const bool is_grid_cell_index_checked() const {
    return grid_cell_index_checked;
  }
  const int get_cell_num_x() const { return cell_num_x; }
  const int get_cell_num_y() const { return cell_num_y; }
  const int get_cell_num_z() const { return cell_num_z; }
  const int get_total_cell_num() const { return total_cell_num; }
  const float get_h() const { return H; }
  const float get_grid_cell_size() const { return GRID_CELL_SIZE; }
  const float get_grid_cell_size_inv() const { return GRID_CELL_SIZE_INV; }
  const T get_simulation_scale() const { return simulation_scale; }
  const std::vector<partition> get_partitions() const { return partitions; }
  int size() const { return particles.size(); }
  void set_grid_cell_index_checked() { grid_cell_index_checked = true; }
  container &set_particles() { return particles; }
  std::vector<int> &get_grid_cell_index() { return grid_cell_index; }
  void decrease_barrier() { barrier--; }
  void reset_barrier() { barrier = device_count; }

  void prepare_for_next_iteration() {
    grid_cell_index_checked = false;
    grid_cell_index.clear();
    grid_cell_index.shrink_to_fit();
    grid_cell_index.resize(total_cell_num + 1);
  }

  void reset_data() {
    if (first_iteration_finished)
      return;
    particles = default_particles;
    default_particles.clear();
    default_particles.shrink_to_fit();
    grid_cell_index.clear();
    grid_cell_index.shrink_to_fit();
    grid_cell_index.resize(total_cell_num + 1);
    first_iteration_finished = true;
  }

  /** Makes partition for each device
   */
  void make_partitions(size_t dev_count,
                       std::vector<double> bench_test_results) {
    barrier = dev_count;
    device_count = dev_count;
    next_partition = 0;
    if (dev_count == 1) {
      partitions.push_back(partition{0, static_cast<size_t>(size())});
      return;
    }

    double bench_sum = 0;
    for (size_t i = 0; i < dev_count; i++)
      bench_sum += bench_test_results[i];

    std::vector<size_t> part_sizes(dev_count);
    for (size_t i = 0; i < dev_count; i++)
      part_sizes[i] =
          static_cast<size_t>(bench_test_results[i] * size() / bench_sum);
    // Taking into account particle that missed during cast to integer
    part_sizes[dev_count - 1]++;

    size_t start = 0;

    for (size_t i = 0; i < dev_count; ++i) {
      size_t end = start + part_sizes[i];
      std::cout << i << "'th partition supposing size = " << part_sizes[i]
                << std::endl;
      if (i == dev_count - 1)
        partitions.push_back(partition{start, static_cast<size_t>(size())});
      else {
        if (particles[end - 1].cell_id != particles[end].cell_id) {
          partitions.push_back(partition{start, end});
          start = end;
        } else {
          for (; end < particles.size(); ++end) {
            if (particles[end - 1].cell_id != particles[end].cell_id) {
              break;
            }
          }
          partitions.push_back(partition{start, end});

          start = end;
        }
      }
    }
    default_partitions = partitions;
    for (int i = 0; i < partitions.size(); i++) {
      std::cout << i << "'th partition actual size = " << partitions[i].size()
                << std::endl;
      std::cout << i << "'th partition start = " << partitions[i].start
                << std::endl;
      std::cout << i << "'th partition end = " << partitions[i].end
                << std::endl;
    }
  }
  const partition &get_next_partition() {
    if (next_partition == partitions.size())
      next_partition = 0;
    ++next_partition;
    return partitions[next_partition - 1];
  }

  const partition get_general_partition() {
    return partition{0, particles.size()};
  }

  void add_cell_to_partition(int from, int to) {
    if (to < from) {
      int from_start = partitions[from].start;
      int from_start_cell_id = particles[from_start].cell_id;
      int i = from_start;
      while (from_start_cell_id == particles[i].cell_id) {
        i++;
      }
      partitions[to].end = i;
      partitions[from].start = i;
    }
    if (to > from) {
      int from_end = partitions[from].end;
      int from_end_cell_id = particles[from_end - 1].cell_id;
      int i = from_end - 1;
      while (from_end_cell_id == particles[i].cell_id)
        i--;
      partitions[to].start = i + 1;
      partitions[from].end = i + 1;
    }
  }

  /**Made for 2 devices
   */
  void rebalance_partitions(std::vector<double> work_results) {
    int gr_result_index = work_results[0] > work_results[1] ? 0 : 1;
    int l_result_index = 1 - gr_result_index;
    if (work_results[gr_result_index] - work_results[l_result_index] >
        work_results[l_result_index] / 2.0f)
      add_cell_to_partition(gr_result_index, l_result_index);
  }

  void arrange_particles_in_partition(partition p) {
    std::sort(particles.begin() + p.start, particles.begin() + p.end,
              [](const particle<T> &p1, const particle<T> &p2) {
                return p1.cell_id < p2.cell_id;
              });
  }

  void mess_up_cell_id() {
    srand(time(NULL));
    for (int item = 0; item < particles.size(); item++)
      particles[item].cell_id = rand() % total_cell_num;
  }

  /**Handles problems in joints of partitions after synch_preparation()
   */
  void synchronise_all_particles() {
    for (int i = 0; i < partitions.size() - 1; i++) {
      // Calculating cell_ids on the right edge of current partition
      int cur_end_cell_id = particles[partitions[i].end - 1].cell_id;
      int next_start_cell_id = particles[partitions[i].end].cell_id;
      // Case A: current partition's end particle has cell_id < than next
      // partition's start particle cell_id, it is correct situation
      if (cur_end_cell_id < next_start_cell_id)
        continue;
      // Case B:
      if (cur_end_cell_id == next_start_cell_id)
        expand_partition(i);
      // Case C:
      if (cur_end_cell_id > next_start_cell_id)
        handle_collision(i);
    }
  }

  /**Expands i'th partition to make sure that all particles in last cell of
   * current partition belong to the same partition Also changes next partiton
   * start id corresponding to new end of current partition
   * @param  partition_index - index of partition in the vector of partitions
   */
  void expand_partition(int partition_index) {
    std::cout << "&&&&&&&&&&&&&EXPANDING " << partition_index << " PARTITION";
    // Id of the last particle of current partition
    int particle_id = partitions[partition_index].end - 1;

    // Calculating cell_ids on the right edge of current partition
    int cur_end_cell_id = particles[particle_id].cell_id;
    int next_start_cell_id = particles[particle_id + 1].cell_id;
    // Trivial check
    if (cur_end_cell_id != next_start_cell_id)
      return;

    while (particles[particle_id].cell_id == cur_end_cell_id &&
           particle_id < particles.size())
      particle_id++;
    partitions[partition_index].end = particle_id;
    partitions[partition_index + 1].start = particle_id;
  }

  /**Handles collision on the right edge of current partition
   * Particles must be sorted by cell_id.
   * This method handles situations when particles in joint between current
   *partition and next partition are not sorted
   * @param  partition_index - index of current partition in the vector of
   *partitions
   * !!!This is naive approach for synchronisation. It works well in two cases:
   *		1. Particles are moving mainly in horizontal direction
   *		2. There are few devices (and few partitions so) used: one or
   *two ones
   */
  void handle_collision(int partition_index) {
    // Id of the last particle of current partition
    int cur_partition_last = partitions[partition_index].end - 1;
    // Id of the first particle of cuurent partition
    int next_partition_first = partitions[partition_index + 1].start;
    // Trivial check
    if (particles[cur_partition_last].cell_id <=
        particles[next_partition_first].cell_id)
      return;

    // Finding bounds of sorting
    typename container::iterator sort_start =
        particles.begin() + cur_partition_last;
    int reference_cell_id = particles[cur_partition_last].cell_id;
    typename container::iterator sort_end =
        std::find_if(particles.begin() + next_partition_first, particles.end(),
                     [reference_cell_id](const particle<T> &p) {
                       return p.cell_id == reference_cell_id;
                     });

    // If find_if points to particles.end() std::sort will still work correct
    std::sort(sort_start, sort_end,
              [](const particle<T> &p1, const particle<T> &p2) {
                return p1.cell_id < p2.cell_id;
              });
    // After sorting Case B can appear in a joint between current partiton and
    // next partiton
    if (particles[cur_partition_last].cell_id ==
        particles[next_partition_first].cell_id)
      expand_partition(partition_index);
  }

private:
  size_t next_partition;
  // vars block end
  bool first_iteration_finished = false;
  bool grid_cell_index_checked = false;
  int barrier;
  int device_count;
  int cell_num_x;
  int cell_num_y;
  int cell_num_z;
  long total_cell_num;
  T simulation_scale = 1.0;
  std::vector<int> grid_cell_index;
  container particles;
  container default_particles;

  sph_config config;
  std::map<std::string, T> phys_consts;
  // Vector of partitions that defined before the first iteration
  // Is needed to control partitions' size
  std::vector<partition> default_partitions;
  // Vector of partitions to work with
  std::vector<partition> partitions;
  /** Init variables for simulation
   */
  void init_vars() {
    cell_num_x =
        static_cast<int>((config["x_max"] - config["x_min"]) / GRID_CELL_SIZE);
    cell_num_y =
        static_cast<int>((config["y_max"] - config["y_min"]) / GRID_CELL_SIZE);
    cell_num_z =
        static_cast<int>((config["z_max"] - config["z_min"]) / GRID_CELL_SIZE);
    total_cell_num = cell_num_x * cell_num_y * cell_num_z;
    grid_cell_index.resize(total_cell_num + 1);
  }
  std::shared_ptr<std::array<T, 4>> get_vector(const std::string &line) {
    std::shared_ptr<std::array<T, 4>> v(new std::array<T, 4>());
    std::stringstream ss(line);
    ss >> (*v)[0] >> (*v)[1] >> (*v)[2] >> (*v)[3]; // TODO check here!!!
    return v;
  }
  /**Model reader
   * Read the model from file and load into memory
   */
  void read_model(const std::string &model_file) {
    std::ifstream file(model_file.c_str(), std::ios_base::binary);
    LOADMODE mode = NOMODE;
    bool is_model_mode = false;
    int index = 0;
    if (file.is_open()) {
      while (file.good()) {
        std::string cur_line;
        std::getline(file, cur_line);
        cur_line.erase(std::remove(cur_line.begin(), cur_line.end(), '\r'),
                       cur_line.end()); // crlf win fix
        auto i_space = cur_line.find_first_not_of(" ");
        auto i_tab = cur_line.find_first_not_of("\t");
        if (i_space) {
          cur_line.erase(cur_line.begin(), cur_line.begin() + i_space);
        }
        if (i_tab) {
          cur_line.erase(cur_line.begin(), cur_line.begin() + i_tab);
        }
        if (cur_line.compare("parametrs[") == 0) {
          mode = PARAMS;
          continue;
        } else if (cur_line.compare("model[") == 0) {
          mode = MODEL;
          is_model_mode = true;
          init_vars();
          continue;
        } else if (cur_line.compare("position[") == 0) {
          mode = POS;
          continue;
        } else if (cur_line.compare("velocity[") == 0) {
          mode = VEL;
          continue;
        } else if (cur_line.compare("]") == 0) {
          mode = NOMODE;
          continue;
        }
        if (mode == PARAMS) {
          std::regex rgx("^\\s*(\\w+)\\s*:\\s*(\\d+(\\.\\d*([eE]?[+-]?\\d+)?)?)"
                         "\\s*(//.*)?$");
          std::smatch matches;
          if (std::regex_search(cur_line, matches, rgx)) {
            if (matches.size() > 2) {
              if (config.find(matches[1]) != config.end()) {
                config[matches[1]] = static_cast<T>(stod(matches[2].str()));
                continue;
              }
            } else {
              std::string msg = x_engine::make_msg(
                  "Problem with parsing parametrs:", matches[0].str(),
                  "Please check parametrs.");
              throw parser_error(msg);
            }
          } else {
            throw parser_error(
                "Please check parametrs section there are no parametrs.");
          }
        }
        if (is_model_mode) {
          switch (mode) {
          case POS: {
            particle<T> p;
            p.pos = *get_vector(cur_line);
            calc_grid_id(p);
            particles.push_back(p);
            break;
          }
          case VEL: {
            if (index >= particles.size())
              throw parser_error(
                  "Config file problem. Velocities more than partiles is.");
            particles[index].vel = *get_vector(cur_line);
            ++index;
            break;
          }
          default: { break; }
          }
        }
      }
    } else {
      throw parser_error(
          "Check your file name or path there is no file with name " +
          model_file);
    }
    file.close();
  }
  /**Arrange particles according its cell id
   * it will need for future clustering
   * particles array on several devices.
   */
  void arrange_particles() {
    std::sort(particles.begin(), particles.end(),
              [](const particle<T> &p1, const particle<T> &p2) {
                return p1.cell_id < p2.cell_id;
              });
  }

  // Addition methods
  /** TODO Description here
   */
  void calc_grid_id(particle<T> &p) {
    int A, B, C;
    A = static_cast<int>(p.pos[0] * GRID_CELL_SIZE_INV);
    B = static_cast<int>(p.pos[1] * GRID_CELL_SIZE_INV);
    C = static_cast<int>(p.pos[2] * GRID_CELL_SIZE_INV);
    p.cell_id = A + B * cell_num_x + cell_num_x * cell_num_y * C; //
  }
};
} // namespace model
} // namespace x_engine
#endif // X_SPHMODEL
