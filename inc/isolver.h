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
#ifndef X_ISOLVER
#define X_ISOLVER
#include "sph_model.hpp"
#include <string>
namespace x_engine {
namespace solver {
enum SOLVER_TYPE { OCL = 1, CUDA, SINGLE, PARALLEL };
using x_engine::model::partition;
class i_solver {
public:
  virtual ~i_solver(){};
  virtual void run_neighbour_search() = 0;
  virtual void run_physic() = 0;
  virtual void run_tests() = 0;
  virtual void synch_preparation() = 0;
  virtual void init_model(const partition &) = 0;
  virtual void show_info() = 0;
  virtual void set_ordinal_num(int i) = 0;
  virtual int get_ordinal_num() = 0;
  virtual int get_device_type() = 0;
  virtual double get_total_work_time() = 0;

private:
  virtual void init_ext_particles() = 0;
};
} // namespace solver
} // namespace x_engine
#endif
