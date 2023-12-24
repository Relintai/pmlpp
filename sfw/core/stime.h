#ifndef STIME_H
#define STIME_H

// -----------------------------------------------------------------------------
// time framework utils
// - originally by rlyeh, public domain.

#include "int_types.h"

class STime {
public:
	static uint64_t date(); // YYYYMMDDhhmmss
	static uint64_t date_epoch(); // linux epoch
	static char *date_string(); // "YYYY-MM-DD hh:mm:ss"
	static double time_hh();
	static double time_mm();
	static double time_ss();
	static uint64_t time_ms();
	static uint64_t time_us();
	static uint64_t time_ns();
	static void sleep_ss(double ss);
	static void sleep_ms(double ms);
	static void sleep_us(double us);
	static void sleep_ns(double us);
};

#endif
