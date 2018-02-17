#include "util/ocl_helper.h"
#include "util/x_error.h"
#include <algorithm>
#include <iostream>
#include <functional>
#include <fstream>

using x_engine::ocl_error;

void show_platform_info(const cl::Platform &, int counter);
void show_platform_info(const cl::Platform &);
void init_cl_devices(std::priority_queue<std::shared_ptr<device>> &);

std::priority_queue<std::shared_ptr<device>> get_dev_queue() {
	std::priority_queue<std::shared_ptr<device>> q;
	init_cl_devices(q);
	if (q.size() < 1) {
		throw ocl_error("No OpenCL devices were found");
	}
	return q;
}

size_t get_device_count(const cl::Platform &p) {
	std::vector<cl::Device> devices;
	p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	return devices.size();
}

void init_devices(cl::Platform &p,
	std::priority_queue<std::shared_ptr<device>> &q) {
	cl_int err;
	cl::Context context;
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM,
		(cl_context_properties)(p)(), 0 };
	context = cl::Context(CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &err);
	std::vector<cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	// std::distance(platform_list, it);
	for (size_t i = 0; i < devices.size(); ++i) {
		std::shared_ptr<device> d(new device(devices[i], 0, i));
		q.push(d);
		std::cout << "Init device " << d->name << " device id: " << d->dev_id << std::endl << "-----------------------------------------------" << std::endl;

	}
}

void init_cl_devices(std::priority_queue<std::shared_ptr<device>> &q) {
	cl_int err;
	std::vector<cl::Platform> platform_list;
	err = cl::Platform::get(
		&platform_list); // TODO make check that returned value isn't error
	if (platform_list.size() < 1 || err != CL_SUCCESS) {
		throw ocl_error("No OpenCL platforms were found");
	}
	int counter = 0;

	std::vector<int> platform_number;

	for (auto it = platform_list.begin(); it != platform_list.end(); ++it) {
		show_platform_info(*it, counter);
		counter++;
	}

	while (true) {
		int num;
		std::cin >> num;
		if (num == -1) {
			break;
		}
		platform_number.push_back(num);
	}

	std::vector<cl::Platform> selected_platforms;

	counter = 0;
	for (auto it = platform_number.begin(); it != platform_number.end(); ++it) {
		selected_platforms.push_back(platform_list[*it]);
	}

	std::cout << "Selected platforms: " << std::endl;
	for_each(selected_platforms.begin(), selected_platforms.end(), [&](cl::Platform &p) {
		show_platform_info(p);
		init_devices(p, q);
	});


	//auto it =
	//	std::max_element(platform_list.begin(), platform_list.end(),
	//		[](const cl::Platform &p1, const cl::Platform &p2) {
	//	return get_device_count(p1) < get_device_count(p2);
	//});
	//if (get_device_count(*it) == 1 && platform_list.size() > 1) {
	//	std::cout << "Use all available platforms" << std::endl;
	//	for_each(platform_list.begin(), platform_list.end(), [&](cl::Platform &p) {
	//		show_platform_info(p, -1);
	//		init_devices(p, q);
	//	});
	//}
	//else {
	//	std::cout << "Use platform" << std::endl;
	//	show_platform_info(*it);
	//	init_devices(*it, q);
	//}
}

//void initialize_device(cl::Device& cl_device) {
//	int err;
//	const std::string cl_program_file = "???CLCODE";
//	std::shared_ptr<device> dev(new device(cl_device, 0, 0));
//	auto queue = cl::CommandQueue(dev->context, dev->dev, 0, &err);
//	if (err != CL_SUCCESS) {
//		throw ocl_error("Failed to create command queue");
//	}
//	std::ifstream file(cl_program_file);
//	if (!file.is_open()) {
//		throw ocl_error("Could not open file with OpenCL program check "
//			"input arguments oclsourcepath: " +
//			cl_program_file);
//	}
//	std::string programSource(std::istreambuf_iterator<char>(file),
//		(std::istreambuf_iterator<char>()));
//	if (0) {
//		programSource =
//			"#define _DOUBLE_PRECISSION\n" +
//			programSource; // not now it needs double extension check on device
//	}
//	cl::Program::Sources source(
//		1, std::make_pair(programSource.c_str(), programSource.length() + 1));
//	auto program = cl::Program(dev->context, source);
//#if defined(__APPLE__)
//	err = program.build("-g -cl-opt-disable -I .");
//#else
//#if INTEL_OPENCL_DEBUG
//	err = program.build(OPENCL_DEBUG_PROGRAM_PATH + "-g -cl-opt-disable -I .");
//#else
//	err = program.build("");
//#endif
//#endif
//	if (err != CL_SUCCESS) {
//		std::string compilationErrors;
//		compilationErrors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev->dev);
//		throw ocl_error(compilationErrors);
//	}
//	std::cout
//		<< "OPENCL program was successfully build. Program file oclsourcepath: "
//		<< cl_program_file << std::endl;
//	return;
//}

void show_platform_info(const cl::Platform &platform, int counter) {
	// Get OpenCL platform name and version
	std::cout << "Platform Number: " << counter << std::endl;
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>()
		<< std::endl;
	std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>()
		<< std::endl;
	std::cout << "Platform Version: " << platform.getInfo<CL_PLATFORM_VERSION>()
		<< std::endl;
	std::cout << "Devices: " << get_device_count(platform) << std::endl;
	std::cout << "===============================================" << std::endl;
}

void show_platform_info(const cl::Platform &platform) {
	std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>()
		<< std::endl;
	std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>()
		<< std::endl;
	std::cout << "Platform Version: " << platform.getInfo<CL_PLATFORM_VERSION>()
		<< std::endl;
	std::cout << "Devices: " << get_device_count(platform) << std::endl;
	std::cout << "===============================================" << std::endl;
}