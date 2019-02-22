#ifndef BENCHMARKS
#define BENCHMARKS

#if defined(_WIN32) || defined(_WIN64)
#pragma comment(lib, "opencl.lib") // opencl.lib
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include "OpenCL/cl.hpp"
#else
#include <CL/cl.hpp>
#endif

#include "../X_ENGINE/measurement.hpp"
#include "util/x_error.h"
#include "x_device.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <memory>
#include <time.h>

using std::shared_ptr;
using x_engine::ocl_error;
namespace bench {
const std::string cl_program_file = "cl_code\\sph_cl_code.cl";
shared_ptr<device> dev;
std::string msg;

cl::CommandQueue queue;
cl::Program program;
cl::Context context;
cl::Kernel k_matrix_multiplication;

typedef struct {
  int width;
  int height;
  float *elements;
} Matrix;

void matrix_multiply(Matrix A, Matrix B, Matrix C, uint32_t factor);
void initialize_ocl();
void create_ocl_kernel(const char *name, cl::Kernel &k);

float get_random_value(float min, float max) {
  return min + static_cast<float>(rand()) /
                   (static_cast<float>(RAND_MAX / (max - min)));
}

void benchmark_deviceS(std::shared_ptr<device> device_ptr) {
  dev = device_ptr;
  msg = dev->name + '\n';
  initialize_ocl();
  srand(time(NULL));
  Matrix A;
  Matrix B;
  Matrix C;
  const float min_val = 0.0;
  const float max_val = 500.0;
  const int height = 2048;
  const int width = 2048;

  A.elements = (float *)malloc(height * width * sizeof(float));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      A.elements[i * height + j] = get_random_value(min_val, max_val);
    }
  }
  // A.elements = elements;
  A.height = height;
  A.width = width;

  B.elements = (float *)malloc(height * width * sizeof(float));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      B.elements[i * height + j] = get_random_value(min_val, max_val);
    }
  }
  // B.elements = elements;
  B.height = height;
  B.width = width;

  C.elements = (float *)malloc(height * width * sizeof(float));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      C.elements[i * height + j] = 0;
    }
  }
  // C.elements = elements;
  C.height = height;
  C.width = width;

  create_ocl_kernel("_ker_matrix_multiplication", k_matrix_multiplication);
  matrix_multiply(A, B, C, 1);
}

void initialize_ocl() {
  int err;
  context = dev->context;
  queue = cl::CommandQueue(context, dev->dev, 0, &err);
  if (err != CL_SUCCESS) {
    throw ocl_error(msg + "Failed to create command queue for benchmark test");
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
    msg += "Benchmark test compilation failed: ";
    throw ocl_error(msg);
  }
  std::cout << msg
            << "OPENCL benchmark test program was successfully built. Program "
               "file oclsourcepath: "
            << cl_program_file << std::endl;
  return;
}

void matrix_multiply(Matrix A, Matrix B, Matrix C, uint32_t factor) {
  const int BLOCK_SIZE = 16;
  // Load A and B to device memory
  size_t size = A.width * A.height * sizeof(float);
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, size);
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, size, A.elements);
  size = B.width * B.height * sizeof(float);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, size);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, size, B.elements);
  // Allocate C in device memory
  size = C.width * C.height * sizeof(float);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, size);
  queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, size, C.elements);

  cl::Buffer resBuf(context, CL_MEM_READ_WRITE, sizeof(int));
  // Invoke kernel
  int i = 0;

  k_matrix_multiplication.setArg(i++, A.width);
  k_matrix_multiplication.setArg(i++, A.height);
  k_matrix_multiplication.setArg(i++, buffer_A);
  k_matrix_multiplication.setArg(i++, B.width);
  k_matrix_multiplication.setArg(i++, B.height);
  k_matrix_multiplication.setArg(i++, buffer_B);
  k_matrix_multiplication.setArg(i++, C.width);
  k_matrix_multiplication.setArg(i++, C.height);
  k_matrix_multiplication.setArg(i++, buffer_C);
  k_matrix_multiplication.setArg(i++, factor);

  cl::NDRange global(C.height, C.width);
  queue.enqueueNDRangeKernel(k_matrix_multiplication, cl::NullRange, global,
                             cl::NullRange);
  mesure::refreshTime();
  queue.finish();
  std::string naming("");
  naming += (*dev).name + " becnh results: \t%9.3f ms\n";
  mesure::watch_report(naming.c_str());

  float *res_1 = (float *)malloc(C.height * C.width * sizeof(float));
  ;
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, size, res_1);
  // queue.enqueueReadBuffer(resBuf, 1, 0, C.height * C.width, res_1);
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
} // namespace bench
#endif
