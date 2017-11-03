/*******************************************************************************
 * The MIT License (MIT)
 *
 * Copyright (c) 2011, 2013 OpenWorm.
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

#include "ow_oclsolver.h"
#include "util/x_error.h"
#include <iostream>
#include <fstream>

using x_engine::solver::ocl_solver;
using x_engine::ocl_error;
using std::cout;
using std::endl;
const std::string ocl_solver::program_name("cl_code//sph_cl_code.cl");

ocl_solver::ocl_solver(std::shared_ptr<device> d)
{
  try
  {
    this->initialize_ocl(d);
  }
  catch (ocl_error &ex)
  {
    throw;
  }
}

void ocl_solver::init_ext_particles() {}
void ocl_solver::run_neighbour_search() {}
void ocl_solver::run_physic() {}

void ocl_solver::initialize_ocl(std::shared_ptr<device> dev)
{
  cl_int err;
  std::vector<cl::Platform> platformList;
  std::vector<cl::Device> devices;
  err = cl::Platform::get(
      &platformList); // TODO make check that returned value isn't error
  if (platformList.size() < 1 || err != CL_SUCCESS)
  {
    throw ocl_error("No OpenCL platforms found");
  }
  char _name[1024];
  cl_platform_id cl_pl_id[10];
  cl_uint n_pl;
  clGetPlatformIDs(10, cl_pl_id, &n_pl);
  for (cl_uint i = 0; i < n_pl; i++)
  {
    // Get OpenCL platform name and version
    err = clGetPlatformInfo(cl_pl_id[i], CL_PLATFORM_VERSION,
                            sizeof(_name), _name, NULL);
    if (err == CL_SUCCESS)
    {
      cout << "CL_PLATFORM_VERSION [" << i << "]: \t" << _name << endl;
    }
    else
    {
      std::cerr << "Error " << err << " in clGetPlatformInfo Call \n\n"
                << endl;
    }
  }
  // 0-CPU, 1-GPU // depends on the time order of system OpenCL drivers
  // installation on your local machine
  // CL_DEVICE_TYPE
  cl_device_type type;
  unsigned int device_type[] = {CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU,
                                CL_DEVICE_TYPE_ALL};

  int pl_list = -1; // selected platform index in platformList array [choose CPU
                    // by default]
  // added autodetection of device number corresonding to preferrable device
  // type (CPU|GPU) | otherwise the choice will be made from list of existing
  // devices
  cl_uint cl_device_count = 0;
  cl_device_id *devices_t;
  bool b_passed = true, find_device = false;
  cl_int result;
  cl_uint device_coumpute_unit_num;
  cl_uint device_coumpute_unit_num_current = 0;
  unsigned int deviceNum = 0;
  // Selection of more appropriate device
  while (!find_device)
  {
    for (cl_uint id = 0; id < (int)n_pl;
         id++)
    {
      clGetDeviceIDs(cl_pl_id[id],
                     device_type[dev->type], 0, NULL,
                     &cl_device_count);
      if ((devices_t = static_cast<cl_device_id *>(
               malloc(sizeof(cl_device_id) * cl_device_count))) == NULL)
        b_passed = false;
      if (b_passed)
      {
        result = clGetDeviceIDs(cl_pl_id[id],
                                device_type[dev->type],
                                cl_device_count, devices_t, &cl_device_count);
        if (result == CL_SUCCESS)
        {
          for (cl_uint i = 0; i < cl_device_count; ++i)
          {
            clGetDeviceInfo(devices_t[i], CL_DEVICE_TYPE, sizeof(type), &type,
                            NULL);
            if (type & device_type[dev->type])
            {
              clGetDeviceInfo(devices_t[i], CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof(device_coumpute_unit_num),
                              &device_coumpute_unit_num, NULL);
              if (device_coumpute_unit_num_current <=
                  device_coumpute_unit_num)
              {
                pl_list = id;
                device_coumpute_unit_num_current = device_coumpute_unit_num;
                find_device = true;
                deviceNum = i;
              }
            }
          }
        }
        free(devices_t);
      }
    }
    if (!find_device)
    {
      deviceNum = 0;
      std::string deviceTypeName =
          (dev->type == ALL)
              ? "ALL"
              : (dev->type == CPU) ? "CPU" : "GPU";
      cout << "Unfortunately OpenCL couldn't find device "
           << deviceTypeName << endl;
      cout << "OpenCL try to init existing device " << endl;
      if (dev->type != ALL)
        dev->type = ALL;
      else
        throw ocl_error("Sibernetic can't find any OpenCL devices. "
                        "Please check you're environment "
                        "configuration.");
    }
  }
  cl_context_properties cprops[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[pl_list])(), 0};
  context = cl::Context(device_type[dev->type], cprops, NULL,
                        NULL, &err);
  devices = context.getInfo<CL_CONTEXT_DEVICES>();
  if (devices.size() < 1)
  {
    throw std::runtime_error("No OpenCL devices were found");
  }
  // Print some information about chosen platform
  size_t compUnintsCount, memoryInfo, workGroupSize;
  result = devices[deviceNum].getInfo(CL_DEVICE_NAME,
                                      &_name); // CL_INVALID_VALUE = -30;
  if (result == CL_SUCCESS)
  {
    cout << "CL_CONTEXT_PLATFORM [" << pl_list << "]: CL_DEVICE_NAME ["
         << deviceNum << "]:\t" << _name << "\n"
         << endl;
  }
  if (strlen(_name) < 1024)
  {
    dev->name = _name;
  }
  result = devices[deviceNum].getInfo(CL_DEVICE_TYPE, &_name);
  if (result == CL_SUCCESS)
  {
    cout << "CL_CONTEXT_PLATFORM [" << pl_list << "]: CL_DEVICE_TYPE ["
         << deviceNum << "]:\t"
         << ((_name[0] == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU")
         << endl;
  }
  result =
      devices[deviceNum].getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &workGroupSize);
  if (result == CL_SUCCESS)
  {
    cout << "CL_CONTEXT_PLATFORM [" << pl_list
         << "]: CL_DEVICE_MAX_WORK_GROUP_SIZE [" << deviceNum << "]: \t"
         << workGroupSize << endl;
  }
  result =
      devices[deviceNum].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compUnintsCount);
  if (result == CL_SUCCESS)
  {
    cout << "CL_CONTEXT_PLATFORM [" << pl_list
         << "]: CL_DEVICE_MAX_COMPUTE_UNITS [" << deviceNum << "]: \t"
         << compUnintsCount << endl;
  }
  result = devices[deviceNum].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &memoryInfo);
  if (result == CL_SUCCESS)
  {
    cout << "CL_CONTEXT_PLATFORM [" << pl_list
         << "]: CL_DEVICE_GLOBAL_MEM_SIZE [" << deviceNum << "]: \t"
         << deviceNum << endl;
  }
  result =
      devices[deviceNum].getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &memoryInfo);
  if (result == CL_SUCCESS)
  {
    cout << "CL_CONTEXT_PLATFORM [" << pl_list
         << "]: CL_DEVICE_GLOBAL_MEM_CACHE_SIZE [" << deviceNum << "]:\t"
         << memoryInfo << endl;
  }
  result = devices[deviceNum].getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &memoryInfo);
  if (result == CL_SUCCESS)
  {
    cout << "CL_CONTEXT_PLATFORM " << pl_list
         << ": CL_DEVICE_LOCAL_MEM_SIZE [" << deviceNum << "]:\t"
         << memoryInfo << endl;
  }
  queue = cl::CommandQueue(context, devices[deviceNum], 0, &err);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Failed to create command queue");
  }
  std::ifstream file(ocl_solver::program_name.c_str());
  if (!file.is_open())
  {
    throw ocl_error("Could not open file with OpenCL program check "
                    "input arguments oclsourcepath: ");
  }
  std::string programSource(std::istreambuf_iterator<char>(file),
                            (std::istreambuf_iterator<char>()));
  cl::Program::Sources source(
      1, std::make_pair(programSource.c_str(), programSource.length() + 1));
  program = cl::Program(context, source);
#if defined(__APPLE__)
  err = program.build(devices, "-g -cl-opt-disable");
#else
#if INTEL_OPENCL_DEBUG
  err =
      program.build(devices, OPENCL_DEBUG_PROGRAM_PATH + "-g -cl-opt-disable");
#else
  err = program.build(devices, "");
#endif
#endif
  if (err != CL_SUCCESS)
  {
    std::string compilationErrors;
    compilationErrors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
    std::cerr << "Compilation failed: " << endl
              << compilationErrors << endl;
    throw ocl_error("Failed to build program");
  }
  cout
      << "OPENCL program was successfully build. Program file oclsourcepath: "
      << endl;
}