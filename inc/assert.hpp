#ifndef ASSERT_HPP
#define ASSERT_HPP

#include <iostream>
#include <fstream>
#include <string>


bool assert_equals(std::string message, int expected_value,
	int actual_value) {
	if (expected_value == actual_value) {
		return true;
	}
	else {
		std::string test_mark = "[TEST] ";
		std::string error_m = test_mark.append(message);
		error_m.append("| Expected: ");
		error_m.append(std::to_string(expected_value));
		error_m.append("; Actual: ");
		error_m.append(std::to_string(actual_value));
		std::cout << error_m;
	}
	return false;
}

bool assert_not_equals(std::string message, int expected_value,
	int actual_value) {
	if (expected_value != actual_value) {
		return true;
	}
	else {
		std::string test_mark = "[TEST] ";
		std::string error_m = test_mark.append(message);
		error_m.append("| Expected: ");
		error_m.append(std::to_string(expected_value));
		error_m.append("; Actual: ");
		error_m.append(std::to_string(actual_value));
		std::cout << error_m;
	}
	return false;
}


#endif
