#ifndef OW_SOLVER_CONTAINER
#define OW_SOLVER_CONTAINER
#include "ow_isolver.h"
#include "ow_cl_const.h"
#include <string>
#include <vector>
namespace x_engine
{
namespace solver
{
enum SOLVER_TYPE
{
  OCL = 1,
  CUDA,
  SINGLE,
  PARALLEL
};
enum DEVICE
{
  CPU = 0,
  GPU = 1,
  ALL = 2
};
struct device
{
  DEVICE type;
  std::string name;
  bool is_buisy;
};
class solver_container
{
public:
  solver_container(size_t devices_number = 1, SOLVER_TYPE s_t = OCL);
  ~solver_container() {}

private:
  std::vector<std::shared_ptr<i_solver>> _solvers;
  std::vector<std::shared_ptr<device>> devices;
};
}
}

#endif