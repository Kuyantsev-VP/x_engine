#include <cstdio>
#include <iostream>



#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#elif defined(__linux__)
#include <time.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <stddef.h>
#endif
#if defined(_WIN32) || defined(_WIN64)
LARGE_INTEGER frequency;  // ticks per second
LARGE_INTEGER t0, t1, t2; // ticks
LARGE_INTEGER t3, t4;
#elif defined(__linux__)
timespec t0, t1, t2;
timespec t3, t4;
double us;
#elif defined(__APPLE__)
unsigned long t0, t1, t2;
#endif
double elapsedTime;
void refreshTime() {
#if defined(_WIN32) || defined(_WIN64)
	QueryPerformanceFrequency(&frequency); // get ticks per second
	QueryPerformanceCounter(&t1);          // start timer
	t0 = t1;
#elif defined(__linux__)
	clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
	t0 = t1;
#elif defined(__APPLE__)
	t1 = mach_absolute_time();
	t0 = t1;
#endif
}

void watch_report(const char *str) {
#if defined(_WIN32) || defined(_WIN64)
	QueryPerformanceCounter(&t2);
	printf(str, (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart);
	t1 = t2;
	elapsedTime = (t2.QuadPart - t0.QuadPart) * 1000.0 / frequency.QuadPart;
#elif defined(__linux__)
	clock_gettime(CLOCK_MONOTONIC_RAW, &t2);
	time_t sec = t2.tv_sec - t1.tv_sec;
	long nsec;
	if (t2.tv_nsec >= t1.tv_nsec) {
		nsec = t2.tv_nsec - t1.tv_nsec;
	}
	else {
		nsec = 1000000000 - (t1.tv_nsec - t2.tv_nsec);
		sec -= 1;
	}
	printf(str, (float)sec * 1000.f + (float)nsec / 1000000.f);
	t1 = t2;
	elapsedTime = (float)(t2.tv_sec - t0.tv_sec) * 1000.f +
		(float)(t2.tv_nsec - t0.tv_nsec) / 1000000.f;
#elif defined(__APPLE__)
	uint64_t elapsedNano;
	static mach_timebase_info_data_t sTimebaseInfo;

	if (sTimebaseInfo.denom == 0) {
		(void)mach_timebase_info(&sTimebaseInfo);
	}

	t2 = mach_absolute_time();
	elapsedNano = (t2 - t1) * sTimebaseInfo.numer / sTimebaseInfo.denom;
	printf(str, (float)elapsedNano / 1000000.f);
	t1 = t2;
	elapsedNano = (t2 - t0) * sTimebaseInfo.numer / sTimebaseInfo.denom;
	elapsedTime = (float)elapsedNano / 1000000.f;
#endif
}