#ifndef X_SOLVER_CONTAINER
#define X_SOLVER_CONTAINER
#include "../X_ENGINE/measurement.hpp"
#include "isolver.h"
#include "ocl_const.h"
#include "ocl_solver.hpp"
#include "sph_model.hpp"
#include "util/ocl_helper.h"
#include "util/x_error.h"
#include <string>
#include <vector>
#include <conio.h>

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"

namespace x_engine {
namespace solver {
using model::sph_model;
using std::shared_ptr;
using x_engine::solver::ocl_solver;
template <class T = float> class solver_container {
  typedef shared_ptr<sph_model<T>> model_ptr;

public:
  solver_container(const solver_container &) = delete;
  solver_container &operator=(const solver_container &) = delete;
  /** Classic Maer's singleton
   */
  static solver_container &instance(model_ptr &model, size_t devices_number = 1,
                                    SOLVER_TYPE s_t = OCL) {
    static solver_container s(model, devices_number, s_t);
    return s;
  }

private:
  solver_container(model_ptr &model, size_t devices_number = 1,
                   SOLVER_TYPE s_type = OCL) {
    int iteration_number = 1;
    try {
      std::priority_queue<std::shared_ptr<device>> dev_q = get_dev_queue();
      int dev_q_size = dev_q.size();
      work_results.resize(dev_q_size);
      while (!dev_q.empty()) {
        try {
          std::shared_ptr<ocl_solver<T>> solver(
              new ocl_solver<T>(model, dev_q.top()));
          _solvers.push_back(solver);
        } catch (ocl_error &ex) {
          std::cout << ex.what() << std::endl;
        }
        dev_q.pop();
      }
      // BENCHMARKING START
      if (dev_q_size > 1) {
        for (int s = 0; s < _solvers.size(); s++) {
          _solvers[s]->set_ordinal_num(s);
          _solvers[s]->init_model(model->get_general_partition());
          _solvers[s]->show_info();
          int a = 1;

          mesure::refreshTime();
          _solvers[s]->run_neighbour_search();
          _solvers[s]->run_tests();
          _solvers[s]->run_physic();
          double bench_time =
              mesure::watch_report("\n||||||||||||||||| BENCH TIME: \t%9.3f ms "
                                   "|||||||||||||||||\n\n");
          double bench = 1000.0 / bench_time;
          bench_test_results.push_back(
              _solvers[s]->get_device_type() == CPU ? bench / 3.0 : bench);

        }
      }
      // BENCHMARKING END

      // reset particles here
      model->prepare_for_next_iteration();
      model->reset_data();

      std::cout << std::endl << std::endl;
      model->make_partitions(
          _solvers.size(),
          bench_test_results);
      char c = 'a';

      while (c != 'q') {
        std::cout << std::endl
                  << std::endl
                  << "=================================" << iteration_number++
                  << " iteration=================================" << std::endl;
        for (auto s : _solvers) {
          s->init_model(model->get_next_partition());
          s->show_info();

          mesure::refreshTime();
          s->run_neighbour_search();
          s->run_tests();
          s->run_physic();
          double work_time =
              mesure::watch_report("\n||||||||||||||||| WORK TIME: \t%9.3f ms "
                                   "|||||||||||||||||\n\n");
          work_results[s->get_ordinal_num()] = (1000.0 / work_time);
          s->synch_preparation();
        }
        model->synchronise_all_particles();
        model->check_partitions_consistency();
        model->prepare_for_next_iteration();
        c = _getch();
      }
    } catch (x_engine::ocl_error &err) {
      throw;
    } catch (x_engine::partitions_consistency_error &err) {
      throw;
    }
  }
  ~solver_container() {}
  std::vector<std::shared_ptr<i_solver>> _solvers;
  std::vector<double> bench_test_results;
  std::vector<double> work_results;
}; // namespace solver
} // namespace solver
} // namespace x_engine

#endif // X_SOLVER_CONTAINER
