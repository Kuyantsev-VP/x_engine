#ifndef OW_SOLVER_CONTAINER
#define OW_SOLVER_CONTAINER
#include "isolver.h"
#include "ocl_const.h"
#include "sph_model.h"
#include "util/x_error.h"
#include <string>
#include <vector>
namespace x_engine {
namespace solver {
using std::shared_ptr;
using model::sph_model;
enum SOLVER_TYPE { OCL = 1, CUDA, SINGLE, PARALLEL };
enum DEVICE { CPU = 0, GPU = 1, ALL = 2 };
struct device {
  DEVICE type;
  std::string name;
  bool is_buisy;
};
template <class T> class solver_container {
  typedef shared_ptr<sph_model<T>> model_ptr;

public:
  solver_container(const solver_container &) = delete;
  solver_container &operator=(const solver_container &) = delete;
  /** Classic Maer's singleton
   */
  static solver_container &instance(model_ptr model, size_t devices_number = 1,
                                    SOLVER_TYPE s_t = OCL) {
    static solver_container s(model, devices_number, s_t);
    return s;
  }

private:
  solver_container(model_ptr model, size_t devices_number = 1,
                   SOLVER_TYPE s_type = OCL) {
    _solvers.reserve(devices_number);
    try {
      for (int i = 0; i < devices_number; ++i) {
        std::shared_ptr<i_solver> s;
        std::shared_ptr<device> d(new device{CPU, "", false});
        switch (s_type) {
        case OCL: {
          // s = std::make_shared<ocl_solver>(d);
          devices.push_back(d);
          _solvers.push_back(s);
          break;
        }
        default:
          break;
        };
      }
    } catch (ocl_error &err) {
      throw;
    }
  }
  ~solver_container() {}
  std::vector<std::shared_ptr<i_solver>> _solvers;
  std::vector<std::shared_ptr<device>> devices;
};
}
}

#endif