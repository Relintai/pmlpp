#ifndef SFWL_H
#define SFWL_H

#define SFWL_VERSION 1

// SFW - Simple Framework - Lite Version - https://github.com/Relintai/sfw

// Simple c++ app / game framework inspired by the single file c game engines
// and libraries, especially [FWK](https://github.com/r-lyeh/FWK).

// MODULES:
// CORE + OBJECT

// LICENSES:

/*
SFW - (MIT)

Copyright (c) 2023-present Péter Magyar.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// This Library uses code from the following projects:

// Pandemonium Engine (MIT)

/*************************************************************************/
/*                     PANDEMONIUM ENGINE LICENSE                        */
/*             https://github.com/Relintai/pandemonium_engine            */
/*************************************************************************/
/* Copyright (c) 2022-present Péter Magyar.                              */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

// FWK

//https://github.com/r-lyeh/FWK

/*

Unlicense

This software is released into the [public domain](https://unlicense.org/).
Also dual-licensed as [0-BSD](https://opensource.org/licenses/0BSD) or [MIT (No Attribution)](https://github.com/aws/mit-0)
for those countries where public domain is a concern (sigh).
Any contribution to this repository is implicitly subjected to the same release conditions aforementioned.

*/

// Trantor

//inet_address.h, .cpp, socket.h and .cpp originates from trantor (they were heavily modified):

// Trantor - A non-blocking I/O based TCP network library, using C++14/17,
// Copyright (c) 2016-2021, Tao An.  All rights reserved.
// https://github.com/an-tao/trantor
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//   * Neither the name of Tao An nor the names of other contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Muduo - A reactor-based C++ network library for Linux
// Copyright (c) 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//   * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//   * Neither the name of Shuo Chen nor the names of other contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// PCG

// pcg.h's License:

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

#ifndef _REENTRANT
#define _REENTRANT
#endif

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

//===================  CORE SECTION  ===================

#line 1 "sfwl/core/int_types.h"
/*************************************************************************/
/*  int_types.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#ifdef _MSC_VER

typedef signed __int8 int8_t;
typedef unsigned __int8 uint8_t;
typedef signed __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef signed __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;

#else

#ifdef NO_STDINT_H
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef signed short int16_t;
typedef unsigned int uint32_t;
typedef signed int int32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

#endif
#line 0

#line 1 "sfwl/core/math_defs.h"

/*************************************************************************/
/*  math_defs.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#define CMP_EPSILON 0.00001
#define CMP_EPSILON2 (CMP_EPSILON * CMP_EPSILON)

#define CMP_NORMALIZE_TOLERANCE 0.000001
#define CMP_POINT_IN_PLANE_EPSILON 0.00001

#define Math_SQRT12 0.7071067811865475244008443621048490
#define Math_SQRT2 1.4142135623730950488016887242
#define Math_LN2 0.6931471805599453094172321215
#define Math_TAU 6.2831853071795864769252867666
#define Math_PI 3.1415926535897932384626433833
#define Math_E 2.7182818284590452353602874714
#define Math_INF INFINITY
#define Math_NAN NAN

#ifdef DEBUG_ENABLED
#define MATH_CHECKS
#endif

//this epsilon is for values related to a unit size (scalar or vector len)
#ifdef PRECISE_MATH_CHECKS
#define UNIT_EPSILON 0.00001
#else
//tolerate some more floating point error normally
#define UNIT_EPSILON 0.001
#endif

#define USEC_TO_SEC(m_usec) ((m_usec) / 1000000.0)
#define SEC_TO_USEC(m_usec) ((m_usec)*1000000.0)

enum ClockDirection {
	CLOCKWISE,
	COUNTERCLOCKWISE
};

enum Orientation {

	HORIZONTAL,
	VERTICAL
};

enum HAlign {

	HALIGN_LEFT,
	HALIGN_CENTER,
	HALIGN_RIGHT
};

enum VAlign {

	VALIGN_TOP,
	VALIGN_CENTER,
	VALIGN_BOTTOM
};

enum Margin {

	MARGIN_LEFT,
	MARGIN_TOP,
	MARGIN_RIGHT,
	MARGIN_BOTTOM
};

enum Side {
	SIDE_LEFT,
	SIDE_TOP,
	SIDE_RIGHT,
	SIDE_BOTTOM
};

enum Corner {

	CORNER_TOP_LEFT,
	CORNER_TOP_RIGHT,
	CORNER_BOTTOM_RIGHT,
	CORNER_BOTTOM_LEFT
};

/**
 * The "Real" type is an abstract type used for real numbers, such as 1.5,
 * in contrast to integer numbers. Precision can be controlled with the
 * presence or absence of the REAL_T_IS_DOUBLE define.
 */
#ifdef REAL_T_IS_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#line 0

#line 1 "sfwl/core/error_list.h"
/*************************************************************************/
/*  error_list.h                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/** Error List. Please never compare an error against FAILED
 * Either do result != OK , or !result. This way, Error fail
 * values can be more detailed in the future.
 *
 * This is a generic error list, mainly for organizing a language of returning errors.
 */

enum Error {
	OK, // (0)
	FAILED, ///< Generic fail error
	ERR_UNAVAILABLE, ///< What is requested is unsupported/unavailable
	ERR_UNCONFIGURED, ///< The object being used hasn't been properly set up yet
	ERR_UNAUTHORIZED, ///< Missing credentials for requested resource
	ERR_PARAMETER_RANGE_ERROR, ///< Parameter given out of range (5)
	ERR_OUT_OF_MEMORY, ///< Out of memory
	ERR_FILE_NOT_FOUND,
	ERR_FILE_BAD_DRIVE,
	ERR_FILE_BAD_PATH,
	ERR_FILE_NO_PERMISSION, // (10)
	ERR_FILE_ALREADY_IN_USE,
	ERR_FILE_CANT_OPEN,
	ERR_FILE_CANT_WRITE,
	ERR_FILE_CANT_READ,
	ERR_FILE_UNRECOGNIZED, // (15)
	ERR_FILE_CORRUPT,
	ERR_FILE_MISSING_DEPENDENCIES,
	ERR_FILE_EOF,
	ERR_CANT_OPEN, ///< Can't open a resource/socket/file
	ERR_CANT_CREATE, // (20)
	ERR_QUERY_FAILED,
	ERR_ALREADY_IN_USE,
	ERR_LOCKED, ///< resource is locked
	ERR_TIMEOUT,
	ERR_CANT_CONNECT, // (25)
	ERR_CANT_RESOLVE,
	ERR_CONNECTION_ERROR,
	ERR_CANT_ACQUIRE_RESOURCE,
	ERR_CANT_FORK,
	ERR_INVALID_DATA, ///< Data passed is invalid (30)
	ERR_INVALID_PARAMETER, ///< Parameter passed is invalid
	ERR_ALREADY_EXISTS, ///< When adding, item already exists
	ERR_DOES_NOT_EXIST, ///< When retrieving/erasing, if item does not exist
	ERR_DATABASE_CANT_READ, ///< database is full
	ERR_DATABASE_CANT_WRITE, ///< database is full (35)
	ERR_COMPILATION_FAILED,
	ERR_METHOD_NOT_FOUND,
	ERR_LINK_FAILED,
	ERR_SCRIPT_FAILED,
	ERR_CYCLIC_LINK, // (40)
	ERR_INVALID_DECLARATION,
	ERR_DUPLICATE_SYMBOL,
	ERR_PARSE_ERROR,
	ERR_BUSY,
	ERR_SKIP, // (45)
	ERR_HELP, ///< user requested help!!
	ERR_BUG, ///< a bug in the software certainly happened, due to a double check failing or unexpected behavior.
	ERR_PRINTER_ON_FIRE, /// the parallel port printer is engulfed in flames
};

#line 0

#line 1 "sfwl/core/logger.h"

class String;

class RLogger {
public:
	static void print_trace(const String &str);
	static void print_trace(const char *str);
	static void print_trace(const char *p_function, const char *p_file, int p_line, const String &str);
	static void print_trace(const char *p_function, const char *p_file, int p_line, const char *str);

	static void print_message(const String &str);
	static void print_message(const char *str);
	static void print_message(const char *p_function, const char *p_file, int p_line, const String &str);
	static void print_message(const char *p_function, const char *p_file, int p_line, const char *str);

	static void print_warning(const String &str);
	static void print_warning(const char *str);
	static void print_warning(const char *p_function, const char *p_file, int p_line, const String &str);
	static void print_warning(const char *p_function, const char *p_file, int p_line, const char *str);

	static void print_error(const String &str);
	static void print_error(const char *str);
	static void print_error(const char *p_function, const char *p_file, int p_line, const char *str);
	static void print_error(const char *p_function, const char *p_file, int p_line, const String &str);
	static void print_msg_error(const char *p_function, const char *p_file, int p_line, const char *p_msg, const char *str);
	static void print_index_error(const char *p_function, const char *p_file, int p_line, const int index, const int size, const char *str);

	static void log_trace(const String &str);
	static void log_trace(const char *str);
	static void log_trace(const char *p_function, const char *p_file, int p_line, const String &str);
	static void log_trace(const char *p_function, const char *p_file, int p_line, const char *str);

	static void log_message(const String &str);
	static void log_message(const char *str);
	static void log_message(const char *p_function, const char *p_file, int p_line, const String &str);
	static void log_message(const char *p_function, const char *p_file, int p_line, const char *str);

	static void log_warning(const String &str);
	static void log_warning(const char *str);
	static void log_warning(const char *p_function, const char *p_file, int p_line, const String &str);
	static void log_warning(const char *p_function, const char *p_file, int p_line, const char *str);

	static void log_error(const String &str);
	static void log_error(const char *str);
	static void log_error(const char *p_function, const char *p_file, int p_line, const char *str);
	static void log_error(const char *p_function, const char *p_file, int p_line, const String &str);
	static void log_msg_error(const char *p_function, const char *p_file, int p_line, const char *p_msg, const char *str);
	static void log_index_error(const char *p_function, const char *p_file, int p_line, const int index, const int size, const char *str);
	static void log_index_error(const char *p_function, const char *p_file, int p_line, const int index, const int size, const String &str);

	static String *get_string_ptr(const int p_default_size = 100);
	static String *get_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size = 300);
	static String *get_string_ptr(const char *p_prefix, const char *p_function, const char *p_file, int p_line, const int p_default_size = 300);
	static void return_string_ptr(String *str);

	static String *get_trace_string_ptr(const int p_default_size = 100);
	static String *get_message_string_ptr(const int p_default_size = 100);
	static String *get_warning_string_ptr(const int p_default_size = 100);
	static String *get_error_string_ptr(const int p_default_size = 100);

	static String *get_trace_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size = 300);
	static String *get_message_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size = 300);
	static String *get_warning_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size = 300);
	static String *get_error_string_ptr(const char *p_function, const char *p_file, int p_line, const int p_default_size = 300);

	static void log_ptr(String *str);
	static void log_ret_ptr(String *str);
};

#line 0

#line 1 "sfwl/core/typedefs.h"

/*************************************************************************/
/*  typedefs.h                                                           */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * Basic definitions and simple functions to be used everywhere.
 */

//#include "platform_config.h"

#ifndef _STR
#define _STR(m_x) #m_x
#define _MKSTR(m_x) _STR(m_x)
#endif

//should always inline no matter what
#ifndef _ALWAYS_INLINE_

#if defined(__GNUC__) && (__GNUC__ >= 4)
#define _ALWAYS_INLINE_ __attribute__((always_inline)) inline
#elif defined(__llvm__)
#define _ALWAYS_INLINE_ __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define _ALWAYS_INLINE_ __forceinline
#else
#define _ALWAYS_INLINE_ inline
#endif

#endif

// Should always inline, except in dev builds because it makes debugging harder.
#ifndef _FORCE_INLINE_
#ifdef DEV_ENABLED
#define _FORCE_INLINE_ inline
#else
#define _FORCE_INLINE_ _ALWAYS_INLINE_
#endif

#endif

// No discard allows the compiler to flag warnings if we don't use the return value of functions / classes
#ifndef _NO_DISCARD_
// c++ 17 onwards
#if __cplusplus >= 201703L
#define _NO_DISCARD_ [[nodiscard]]
#else
// __warn_unused_result__ supported on clang and GCC
#if (defined(__clang__) || defined(__GNUC__)) && defined(__has_attribute)
#if __has_attribute(__warn_unused_result__)
#define _NO_DISCARD_ __attribute__((__warn_unused_result__))
#endif
#endif

// Visual Studio 2012 onwards
#if _MSC_VER >= 1700
#define _NO_DISCARD_ _Check_return_
#endif

// If nothing supported, just noop the macro
#ifndef _NO_DISCARD_
#define _NO_DISCARD_
#endif
#endif // not c++ 17
#endif // not defined _NO_DISCARD_

// In some cases _NO_DISCARD_ will get false positives,
// we can prevent the warning in specific cases by preceding the call with a cast.
#ifndef _ALLOW_DISCARD_
#define _ALLOW_DISCARD_ (void)
#endif

// GCC (prior to c++ 17) Does not seem to support no discard with classes, only functions.
// So we will use a specific macro for classes.
#ifndef _NO_DISCARD_CLASS_
#if (defined(__clang__) || defined(_MSC_VER))
#define _NO_DISCARD_CLASS_ _NO_DISCARD_
#else
#define _NO_DISCARD_CLASS_
#endif
#endif

//custom, gcc-safe offsetof, because gcc complains a lot.
template <class T>
T *_nullptr() {
	T *t = NULL;
	return t;
}

#define OFFSET_OF(st, m) \
	((size_t)((char *)&(_nullptr<st>()->m) - (char *)0))
/**
 * Some platforms (devices) don't define NULL
 */

#ifndef NULL
#define NULL 0
#endif

/**
 * Windows badly defines a lot of stuff we'll never use. Undefine it.
 */

#if defined(_WIN64) || defined(_WIN32)
#undef min // override standard definition
#undef max // override standard definition
#undef ERROR // override (really stupid) wingdi.h standard definition
#undef DELETE // override (another really stupid) winnt.h standard definition
#undef MessageBox // override winuser.h standard definition
#undef MIN // override standard definition
#undef MAX // override standard definition
#undef CLAMP // override standard definition
#undef Error
#undef OK
#undef CONNECT_DEFERRED // override from Windows SDK, clashes with Object enum
#endif

/** Generic ABS function, for math uses please use Math::abs */

#ifndef ABS
#define ABS(m_v) (((m_v) < 0) ? (-(m_v)) : (m_v))
#endif

#define ABSDIFF(x, y) (((x) < (y)) ? ((y) - (x)) : ((x) - (y)))

#ifndef SGN
#define SGN(m_v) (((m_v) < 0) ? (-1.0f) : (+1.0f))
#endif

#ifndef MIN
#define MIN(m_a, m_b) (((m_a) < (m_b)) ? (m_a) : (m_b))
#endif

#ifndef MAX
#define MAX(m_a, m_b) (((m_a) > (m_b)) ? (m_a) : (m_b))
#endif

#ifndef CLAMP
#define CLAMP(m_a, m_min, m_max) (((m_a) < (m_min)) ? (m_min) : (((m_a) > (m_max)) ? m_max : m_a))
#endif

/** Generic swap template */
#ifndef SWAP

#define SWAP(m_x, m_y) __swap_tmpl((m_x), (m_y))
template <class T>
inline void __swap_tmpl(T &x, T &y) {
	T aux = x;
	x = y;
	y = aux;
}

#endif //swap

/* clang-format off */
#define HEX2CHR(m_hex) \
	((m_hex >= '0' && m_hex <= '9') ? (m_hex - '0') : \
	((m_hex >= 'A' && m_hex <= 'F') ? (10 + m_hex - 'A') : \
	((m_hex >= 'a' && m_hex <= 'f') ? (10 + m_hex - 'a') : 0)))
/* clang-format on */

// Macro to check whether we are compiled by clang
// and we have a specific builtin
#if defined(__llvm__) && defined(__has_builtin)
#define _llvm_has_builtin(x) __has_builtin(x)
#else
#define _llvm_has_builtin(x) 0
#endif

#if (defined(__GNUC__) && (__GNUC__ >= 5)) || _llvm_has_builtin(__builtin_mul_overflow)
#define _mul_overflow __builtin_mul_overflow
#endif

#if (defined(__GNUC__) && (__GNUC__ >= 5)) || _llvm_has_builtin(__builtin_add_overflow)
#define _add_overflow __builtin_add_overflow
#endif

/** Function to find the next power of 2 to an integer */

static _FORCE_INLINE_ unsigned int next_power_of_2(unsigned int x) {
	if (x == 0) {
		return 0;
	}

	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;

	return ++x;
}

static _FORCE_INLINE_ unsigned int previous_power_of_2(unsigned int x) {
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x - (x >> 1);
}

static _FORCE_INLINE_ unsigned int closest_power_of_2(unsigned int x) {
	unsigned int nx = next_power_of_2(x);
	unsigned int px = previous_power_of_2(x);
	return (nx - x) > (x - px) ? px : nx;
}

// We need this definition inside the function below.
static inline int get_shift_from_power_of_2(unsigned int p_pixel);

template <class T>
static _FORCE_INLINE_ T nearest_power_of_2_templated(T x) {
	--x;

	// The number of operations on x is the base two logarithm
	// of the p_number of bits in the type. Add three to account
	// for sizeof(T) being in bytes.
	size_t num = get_shift_from_power_of_2(sizeof(T)) + 3;

	// If the compiler is smart, it unrolls this loop
	// If its dumb, this is a bit slow.
	for (size_t i = 0; i < num; i++) {
		x |= x >> (1 << i);
	}

	return ++x;
}

/** Function to find the nearest (bigger) power of 2 to an integer */

static inline unsigned int nearest_shift(unsigned int p_number) {
	for (int i = 30; i >= 0; i--) {
		if (p_number & (1 << i)) {
			return i + 1;
		}
	}

	return 0;
}

/** get a shift value from a power of 2 */
static inline int get_shift_from_power_of_2(unsigned int p_pixel) {
	// return a GL_TEXTURE_SIZE_ENUM

	for (unsigned int i = 0; i < 32; i++) {
		if (p_pixel == (unsigned int)(1 << i)) {
			return i;
		}
	}

	return -1;
}

/** Swap 16 bits value for endianness */
#if defined(__GNUC__) || _llvm_has_builtin(__builtin_bswap16)
#define BSWAP16(x) __builtin_bswap16(x)
#else
static inline uint16_t BSWAP16(uint16_t x) {
	return (x >> 8) | (x << 8);
}
#endif

/** Swap 32 bits value for endianness */
#if defined(__GNUC__) || _llvm_has_builtin(__builtin_bswap32)
#define BSWAP32(x) __builtin_bswap32(x)
#else
static inline uint32_t BSWAP32(uint32_t x) {
	return ((x << 24) | ((x << 8) & 0x00FF0000) | ((x >> 8) & 0x0000FF00) | (x >> 24));
}
#endif

/** Swap 64 bits value for endianness */
#if defined(__GNUC__) || _llvm_has_builtin(__builtin_bswap64)
#define BSWAP64(x) __builtin_bswap64(x)
#else
static inline uint64_t BSWAP64(uint64_t x) {
	x = (x & 0x00000000FFFFFFFF) << 32 | (x & 0xFFFFFFFF00000000) >> 32;
	x = (x & 0x0000FFFF0000FFFF) << 16 | (x & 0xFFFF0000FFFF0000) >> 16;
	x = (x & 0x00FF00FF00FF00FF) << 8 | (x & 0xFF00FF00FF00FF00) >> 8;
	return x;
}
#endif

/** When compiling with RTTI, we can add an "extra"
 * layer of safeness in many operations, so dynamic_cast
 * is used besides casting by enum.
 */

template <class T>
struct Comparator {
	_ALWAYS_INLINE_ bool operator()(const T &p_a, const T &p_b) const { return (p_a < p_b); }
};

void _global_lock();
void _global_unlock();

struct _GlobalLock {
	_GlobalLock() { _global_lock(); }
	~_GlobalLock() { _global_unlock(); }
};

#define GLOBAL_LOCK_FUNCTION _GlobalLock _global_lock_;

#ifdef NO_SAFE_CAST
#define SAFE_CAST static_cast
#else
#define SAFE_CAST dynamic_cast
#endif

#define MT_SAFE

#define __STRX(m_index) #m_index
#define __STR(m_index) __STRX(m_index)

#ifdef __GNUC__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) x
#define unlikely(x) x
#endif

#if defined(__GNUC__)
#define _PRINTF_FORMAT_ATTRIBUTE_2_0 __attribute__((format(printf, 2, 0)))
#define _PRINTF_FORMAT_ATTRIBUTE_2_3 __attribute__((format(printf, 2, 3)))
#else
#define _PRINTF_FORMAT_ATTRIBUTE_2_0
#define _PRINTF_FORMAT_ATTRIBUTE_2_3
#endif

/** This is needed due to a strange OpenGL API that expects a pointer
 *  type for an argument that is actually an offset.
 */
#define CAST_INT_TO_UCHAR_PTR(ptr) ((uint8_t *)(uintptr_t)(ptr))

/** Hint for compilers that this fallthrough in a switch is intentional.
 *  Can be replaced by [[fallthrough]] annotation if we move to C++17.
 *  Including conditional support for it for people who set -std=c++17
 *  themselves.
 *  Requires a trailing semicolon when used.
 */
#if __cplusplus >= 201703L
#define FALLTHROUGH [[fallthrough]]
#elif defined(__GNUC__) && __GNUC__ >= 7
#define FALLTHROUGH __attribute__((fallthrough))
#elif defined(__llvm__) && __cplusplus >= 201103L && defined(__has_feature)
#if __has_feature(cxx_attributes) && defined(__has_warning)
#if __has_warning("-Wimplicit-fallthrough")
#define FALLTHROUGH [[clang::fallthrough]]
#endif
#endif
#endif

#ifndef FALLTHROUGH
#define FALLTHROUGH
#endif

// Limit the depth of recursive algorithms when dealing with Array/Dictionary
#define MAX_RECURSION 100

//HAS_TRIVIAL_CONSTRUCTOR

#if defined(__llvm__) && _llvm_has_builtin(__is_trivially_constructible)
#define HAS_TRIVIAL_CONSTRUCTOR(T) __is_trivially_constructible(T)
#endif

#ifndef HAS_TRIVIAL_CONSTRUCTOR
#define HAS_TRIVIAL_CONSTRUCTOR(T) __has_trivial_constructor(T)
#endif

//HAS_TRIVIAL_DESTRUCTOR

#if defined(__llvm__) && _llvm_has_builtin(__is_trivially_destructible)
#define HAS_TRIVIAL_DESTRUCTOR(T) __is_trivially_destructible(T)
#endif

#ifndef HAS_TRIVIAL_DESTRUCTOR
#define HAS_TRIVIAL_DESTRUCTOR(T) __has_trivial_destructor(T)
#endif

//HAS_TRIVIAL_COPY

#if defined(__llvm__) && _llvm_has_builtin(__is_trivially_copyable)
#define HAS_TRIVIAL_COPY(T) __is_trivially_copyable(T)
#endif

#ifndef HAS_TRIVIAL_COPY
#define HAS_TRIVIAL_COPY(T) __has_trivial_copy(T)
#endif

#line 0

#line 1 "sfwl/core/marshalls.h"

/*************************************************************************/
/*  marshalls.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * Miscellaneous helpers for marshalling data types, and encoding
 * in an endian independent way
 */

union MarshallFloat {
	uint32_t i; ///< int
	float f; ///< float
};

union MarshallDouble {
	uint64_t l; ///< long long
	double d; ///< double
};

static inline unsigned int encode_uint16(uint16_t p_uint, uint8_t *p_arr) {
	for (int i = 0; i < 2; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}

	return sizeof(uint16_t);
}

static inline unsigned int encode_uint32(uint32_t p_uint, uint8_t *p_arr) {
	for (int i = 0; i < 4; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}

	return sizeof(uint32_t);
}

static inline unsigned int encode_float(float p_float, uint8_t *p_arr) {
	MarshallFloat mf;
	mf.f = p_float;
	encode_uint32(mf.i, p_arr);

	return sizeof(uint32_t);
}

static inline unsigned int encode_uint64(uint64_t p_uint, uint8_t *p_arr) {
	for (int i = 0; i < 8; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}

	return sizeof(uint64_t);
}

static inline unsigned int encode_double(double p_double, uint8_t *p_arr) {
	MarshallDouble md;
	md.d = p_double;
	encode_uint64(md.l, p_arr);

	return sizeof(uint64_t);
}

static inline int encode_cstring(const char *p_string, uint8_t *p_data) {
	int len = 0;

	while (*p_string) {
		if (p_data) {
			*p_data = (uint8_t)*p_string;
			p_data++;
		}
		p_string++;
		len++;
	};

	if (p_data) {
		*p_data = 0;
	}
	return len + 1;
}

static inline uint16_t decode_uint16(const uint8_t *p_arr) {
	uint16_t u = 0;

	for (int i = 0; i < 2; i++) {
		uint16_t b = *p_arr;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}

	return u;
}

static inline uint32_t decode_uint32(const uint8_t *p_arr) {
	uint32_t u = 0;

	for (int i = 0; i < 4; i++) {
		uint32_t b = *p_arr;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}

	return u;
}

static inline float decode_float(const uint8_t *p_arr) {
	MarshallFloat mf;
	mf.i = decode_uint32(p_arr);
	return mf.f;
}

static inline uint64_t decode_uint64(const uint8_t *p_arr) {
	uint64_t u = 0;

	for (int i = 0; i < 8; i++) {
		uint64_t b = (*p_arr) & 0xFF;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}

	return u;
}

static inline double decode_double(const uint8_t *p_arr) {
	MarshallDouble md;
	md.l = decode_uint64(p_arr);
	return md.d;
}

#line 0

#line 1 "sfwl/core/sfw_time.h"

// -----------------------------------------------------------------------------
// time framework utils
// - originally by rlyeh, public domain.

class SFWTime {
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

#line 0

#line 1 "sfwl/core/safe_refcount.h"

/*************************************************************************/
/*  safe_refcount.h                                                      */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if !defined(NO_THREADS)

#include <atomic>
#include <type_traits>

// Design goals for these classes:
// - No automatic conversions or arithmetic operators,
//   to keep explicit the use of atomics everywhere.
// - Using acquire-release semantics, even to set the first value.
//   The first value may be set relaxedly in many cases, but adding the distinction
//   between relaxed and unrelaxed operation to the interface would make it needlessly
//   flexible. There's negligible waste in having release semantics for the initial
//   value and, as an important benefit, you can be sure the value is properly synchronized
//   even with threads that are already running.

// This is used in very specific areas of the engine where it's critical that these guarantees are held
#define SAFE_NUMERIC_TYPE_PUN_GUARANTEES(m_type)                        \
	static_assert(sizeof(SafeNumeric<m_type>) == sizeof(m_type), "");   \
	static_assert(alignof(SafeNumeric<m_type>) == alignof(m_type), ""); \
	static_assert(std::is_trivially_destructible<std::atomic<m_type>>::value, "");

#if defined(DEBUG_ENABLED)
void check_lockless_atomics();
#endif

template <class T>
class SafeNumeric {
	std::atomic<T> value;

public:
	_ALWAYS_INLINE_ void set(T p_value) {
		value.store(p_value, std::memory_order_release);
	}

	_ALWAYS_INLINE_ T get() const {
		return value.load(std::memory_order_acquire);
	}

	_ALWAYS_INLINE_ T increment() {
		return value.fetch_add(1, std::memory_order_acq_rel) + 1;
	}

	// Returns the original value instead of the new one
	_ALWAYS_INLINE_ T postincrement() {
		return value.fetch_add(1, std::memory_order_acq_rel);
	}

	_ALWAYS_INLINE_ T decrement() {
		return value.fetch_sub(1, std::memory_order_acq_rel) - 1;
	}

	// Returns the original value instead of the new one
	_ALWAYS_INLINE_ T postdecrement() {
		return value.fetch_sub(1, std::memory_order_acq_rel);
	}

	_ALWAYS_INLINE_ T add(T p_value) {
		return value.fetch_add(p_value, std::memory_order_acq_rel) + p_value;
	}

	// Returns the original value instead of the new one
	_ALWAYS_INLINE_ T postadd(T p_value) {
		return value.fetch_add(p_value, std::memory_order_acq_rel);
	}

	_ALWAYS_INLINE_ T sub(T p_value) {
		return value.fetch_sub(p_value, std::memory_order_acq_rel) - p_value;
	}

	// Returns the original value instead of the new one
	_ALWAYS_INLINE_ T postsub(T p_value) {
		return value.fetch_sub(p_value, std::memory_order_acq_rel);
	}

	_ALWAYS_INLINE_ T exchange_if_greater(T p_value) {
		while (true) {
			T tmp = value.load(std::memory_order_acquire);
			if (tmp >= p_value) {
				return tmp; // already greater, or equal
			}
			if (value.compare_exchange_weak(tmp, p_value, std::memory_order_acq_rel)) {
				return p_value;
			}
		}
	}

	_ALWAYS_INLINE_ T conditional_increment() {
		while (true) {
			T c = value.load(std::memory_order_acquire);
			if (c == 0) {
				return 0;
			}
			if (value.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel)) {
				return c + 1;
			}
		}
	}

	_ALWAYS_INLINE_ explicit SafeNumeric<T>(T p_value = static_cast<T>(0)) {
		set(p_value);
	}
};

class SafeFlag {
	std::atomic_bool flag;

public:
	_ALWAYS_INLINE_ bool is_set() const {
		return flag.load(std::memory_order_acquire);
	}

	_ALWAYS_INLINE_ void set() {
		flag.store(true, std::memory_order_release);
	}

	_ALWAYS_INLINE_ void clear() {
		flag.store(false, std::memory_order_release);
	}

	_ALWAYS_INLINE_ void set_to(bool p_value) {
		flag.store(p_value, std::memory_order_release);
	}

	_ALWAYS_INLINE_ explicit SafeFlag(bool p_value = false) {
		set_to(p_value);
	}
};

class SafeRefCount {
	SafeNumeric<uint32_t> count;

public:
	_ALWAYS_INLINE_ bool ref() { // true on success
		return count.conditional_increment() != 0;
	}

	_ALWAYS_INLINE_ uint32_t refval() { // none-zero on success
		return count.conditional_increment();
	}

	_ALWAYS_INLINE_ bool unref() { // true if must be disposed of
		return count.decrement() == 0;
	}

	_ALWAYS_INLINE_ uint32_t unrefval() { // 0 if must be disposed of
		return count.decrement();
	}

	_ALWAYS_INLINE_ uint32_t get() const {
		return count.get();
	}

	_ALWAYS_INLINE_ void init(uint32_t p_value = 1) {
		count.set(p_value);
	}
};

#else

template <class T>
class SafeNumeric {
protected:
	T value;

public:
	_ALWAYS_INLINE_ void set(T p_value) {
		value = p_value;
	}

	_ALWAYS_INLINE_ T get() const {
		return value;
	}

	_ALWAYS_INLINE_ T increment() {
		return ++value;
	}

	_ALWAYS_INLINE_ T postincrement() {
		return value++;
	}

	_ALWAYS_INLINE_ T decrement() {
		return --value;
	}

	_ALWAYS_INLINE_ T postdecrement() {
		return value--;
	}

	_ALWAYS_INLINE_ T add(T p_value) {
		return value += p_value;
	}

	_ALWAYS_INLINE_ T postadd(T p_value) {
		T old = value;
		value += p_value;
		return old;
	}

	_ALWAYS_INLINE_ T sub(T p_value) {
		return value -= p_value;
	}

	_ALWAYS_INLINE_ T postsub(T p_value) {
		T old = value;
		value -= p_value;
		return old;
	}

	_ALWAYS_INLINE_ T exchange_if_greater(T p_value) {
		if (value < p_value) {
			value = p_value;
		}
		return value;
	}

	_ALWAYS_INLINE_ T conditional_increment() {
		if (value == 0) {
			return 0;
		} else {
			return ++value;
		}
	}

	_ALWAYS_INLINE_ explicit SafeNumeric<T>(T p_value = static_cast<T>(0)) :
			value(p_value) {
	}
};

class SafeFlag {
protected:
	bool flag;

public:
	_ALWAYS_INLINE_ bool is_set() const {
		return flag;
	}

	_ALWAYS_INLINE_ void set() {
		flag = true;
	}

	_ALWAYS_INLINE_ void clear() {
		flag = false;
	}

	_ALWAYS_INLINE_ void set_to(bool p_value) {
		flag = p_value;
	}

	_ALWAYS_INLINE_ explicit SafeFlag(bool p_value = false) :
			flag(p_value) {}
};

class SafeRefCount {
	uint32_t count;

public:
	_ALWAYS_INLINE_ bool ref() { // true on success
		if (count != 0) {
			++count;
			return true;
		} else {
			return false;
		}
	}

	_ALWAYS_INLINE_ uint32_t refval() { // none-zero on success
		if (count != 0) {
			return ++count;
		} else {
			return 0;
		}
	}

	_ALWAYS_INLINE_ bool unref() { // true if must be disposed of
		return --count == 0;
	}

	_ALWAYS_INLINE_ uint32_t unrefval() { // 0 if must be disposed of
		return --count;
	}

	_ALWAYS_INLINE_ uint32_t get() const {
		return count;
	}

	_ALWAYS_INLINE_ void init(uint32_t p_value = 1) {
		count = p_value;
	}

	SafeRefCount() :
			count(0) {}
};

#endif

#line 0

#line 1 "sfwl/core/thread.h"

/*************************************************************************/
/*  thread.h                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if !defined(NO_THREADS)
#include <thread>
#endif

class String;

class Thread {
public:
	typedef void (*Callback)(void *p_userdata);

	typedef uint64_t ID;

	enum Priority {
		PRIORITY_LOW,
		PRIORITY_NORMAL,
		PRIORITY_HIGH
	};

	struct Settings {
		Priority priority;
		Settings() { priority = PRIORITY_NORMAL; }
	};

private:
#if !defined(NO_THREADS)
	friend class Main;

	static ID main_thread_id;

	static uint64_t _thread_id_hash(const std::thread::id &p_t);

	ID id = _thread_id_hash(std::thread::id());
	std::thread thread;

	static void callback(Thread *p_self, const Settings &p_settings, Thread::Callback p_callback, void *p_userdata);

	static Error (*set_name_func)(const String &);
	static void (*set_priority_func)(Thread::Priority);
	static void (*init_func)();
	static void (*term_func)();
#endif

public:
	static void _set_platform_funcs(
			Error (*p_set_name_func)(const String &),
			void (*p_set_priority_func)(Thread::Priority),
			void (*p_init_func)() = nullptr,
			void (*p_term_func)() = nullptr);

#if !defined(NO_THREADS)
	_FORCE_INLINE_ ID get_id() const { return id; }
	// get the ID of the caller thread
	static ID get_caller_id();
	// get the ID of the main thread
	_FORCE_INLINE_ static ID get_main_id() { return main_thread_id; }

	_FORCE_INLINE_ static bool is_main_thread() { return get_caller_id() == main_thread_id; }

	static Error set_name(const String &p_name);

	void start(Thread::Callback p_callback, void *p_user, const Settings &p_settings = Settings());
	bool is_started() const;
	///< waits until thread is finished, and deallocates it.
	void wait_to_finish();

	~Thread();
#else
	_FORCE_INLINE_ ID get_id() const { return 0; }
	// get the ID of the caller thread
	_FORCE_INLINE_ static ID get_caller_id() { return 0; }
	// get the ID of the main thread
	_FORCE_INLINE_ static ID get_main_id() { return 0; }

	_FORCE_INLINE_ static bool is_main_thread() { return true; }

	static Error set_name(const String &p_name) { return ERR_UNAVAILABLE; }

	void start(Thread::Callback p_callback, void *p_user, const Settings &p_settings = Settings()) {}
	bool is_started() const { return false; }
	void wait_to_finish() {}
#endif
};

#line 0

#line 1 "sfwl/core/error_macros.h"

// Based on Godot Engine's error_macros.h
// MIT License
// Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.
// Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).

#ifdef _MSC_VER
#define GENERATE_TRAP                       \
	__debugbreak();                         \
	/* Avoid warning about control paths */ \
	for (;;) {                              \
	}
#else
#define GENERATE_TRAP __builtin_trap();
#endif

// template methods for the variadic log macros. Add more as needed.
template <class STR, class A>
_FORCE_INLINE_ void _RLOG_MACRO_TEMPLATE_FUNC(STR str, A p0) {
	str->operator+=(p0);
}

template <class STR, class A, class B>
_FORCE_INLINE_ void _RLOG_MACRO_TEMPLATE_FUNC(STR str, A p0, B p1) {
	str->operator+=(p0);
	str->operator+=(' ');
	str->operator+=(p1);
}

template <class STR, class A, class B, class C>
_FORCE_INLINE_ void _RLOG_MACRO_TEMPLATE_FUNC(STR str, A p0, B p1, C p2) {
	str->operator+=(p0);
	str->operator+=(' ');
	str->operator+=(p1);
	str->operator+=(' ');
	str->operator+=(p2);
}

template <class STR, class A, class B, class C, class D>
_FORCE_INLINE_ void _RLOG_MACRO_TEMPLATE_FUNC(STR str, A p0, B p1, C p2, D p3) {
	str->operator+=(p0);
	str->operator+=(' ');
	str->operator+=(p1);
	str->operator+=(' ');
	str->operator+=(p2);
	str->operator+=(' ');
	str->operator+=(p3);
}

template <class STR, class A, class B, class C, class D, class E>
_FORCE_INLINE_ void _RLOG_MACRO_TEMPLATE_FUNC(STR str, A p0, B p1, C p2, D p3, E p4) {
	str->operator+=(p0);
	str->operator+=(' ');
	str->operator+=(p1);
	str->operator+=(' ');
	str->operator+=(p2);
	str->operator+=(' ');
	str->operator+=(p3);
	str->operator+=(' ');
	str->operator+=(p4);
}

#define PRINT_TRACE(str) \
	RLogger::print_trace(__FUNCTION__, __FILE__, __LINE__, str);

#define LOG_TRACE(...)                                                                                 \
	{                                                                                                  \
		String *_rlogger_string_ptr = RLogger::get_trace_string_ptr(__FUNCTION__, __FILE__, __LINE__); \
		_RLOG_MACRO_TEMPLATE_FUNC(_rlogger_string_ptr, __VA_ARGS__);                                   \
		RLogger::log_ret_ptr(_rlogger_string_ptr);                                                     \
	}

#define PRINT_MSG(str) \
	RLogger::print_message(__FUNCTION__, __FILE__, __LINE__, str);

#define LOG_MSG(...)                                                                                     \
	{                                                                                                    \
		String *_rlogger_string_ptr = RLogger::get_message_string_ptr(__FUNCTION__, __FILE__, __LINE__); \
		_RLOG_MACRO_TEMPLATE_FUNC(_rlogger_string_ptr, __VA_ARGS__);                                     \
		RLogger::log_ret_ptr(_rlogger_string_ptr);                                                       \
	}

#define PRINT_WARN(str) \
	RLogger::print_warning(__FUNCTION__, __FILE__, __LINE__, str);

#define LOG_WARN(...)                                                                                    \
	{                                                                                                    \
		String *_rlogger_string_ptr = RLogger::get_warning_string_ptr(__FUNCTION__, __FILE__, __LINE__); \
		_RLOG_MACRO_TEMPLATE_FUNC(_rlogger_string_ptr, __VA_ARGS__);                                     \
		RLogger::log_ret_ptr(_rlogger_string_ptr);                                                       \
	}

#define PRINT_ERR(str) \
	RLogger::print_error(__FUNCTION__, __FILE__, __LINE__, str);

#define ERR_PRINT(str) \
	RLogger::print_error(__FUNCTION__, __FILE__, __LINE__, str);

#define LOG_ERR(...)                                                                                   \
	{                                                                                                  \
		String *_rlogger_string_ptr = RLogger::get_error_string_ptr(__FUNCTION__, __FILE__, __LINE__); \
		_RLOG_MACRO_TEMPLATE_FUNC(_rlogger_string_ptr, __VA_ARGS__);                                   \
		RLogger::log_ret_ptr(_rlogger_string_ptr);                                                     \
	}

#define ERR_FAIL_MSG(msg)                                      \
	RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
	return;

#define ERR_FAIL_V(val)                                       \
	RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, ""); \
	return val;

#define ERR_FAIL_V_MSG(val, msg)                               \
	RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
	return val;

#define ERR_FAIL_INDEX(index, size)                                                  \
	if ((index < 0) || (index >= size)) {                                            \
		RLogger::log_index_error(__FUNCTION__, __FILE__, __LINE__, index, size, ""); \
		return;                                                                      \
	} else                                                                           \
		((void)0)

#define ERR_FAIL_INDEX_MSG(index, size, msg)                                          \
	if ((index < 0) || (index >= size)) {                                             \
		RLogger::log_index_error(__FUNCTION__, __FILE__, __LINE__, index, size, msg); \
		return;                                                                       \
	} else                                                                            \
		((void)0)

#define ERR_FAIL_INDEX_V(index, size, val)                                           \
	if ((index < 0) || (index >= size)) {                                            \
		RLogger::log_index_error(__FUNCTION__, __FILE__, __LINE__, index, size, ""); \
		return val;                                                                  \
	} else                                                                           \
		((void)0)

#define ERR_FAIL_INDEX_V_MSG(index, size, val, msg)                                   \
	if ((index < 0) || (index >= size)) {                                             \
		RLogger::log_index_error(__FUNCTION__, __FILE__, __LINE__, index, size, msg); \
		return val;                                                                   \
	} else                                                                            \
		((void)0)

#define ERR_FAIL_COND(cond)                                                                            \
	if (cond) {                                                                                        \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, "ERR_FAIL_COND: \"" #cond "\" is true!"); \
		return;                                                                                        \
	} else                                                                                             \
		((void)0)

#define ERR_FAIL_COND_MSG(cond, msg)                               \
	if (cond) {                                                    \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
		return;                                                    \
	} else                                                         \
		((void)0)

#define ERR_FAIL_COND_V(cond, val)                                                                     \
	if (cond) {                                                                                        \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, "ERR_FAIL_COND: \"" #cond "\" is true!"); \
		return val;                                                                                    \
	} else                                                                                             \
		((void)0)

#define ERR_FAIL_COND_V_MSG(cond, val, msg)                        \
	if (cond) {                                                    \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
		return val;                                                \
	} else                                                         \
		((void)0)

#define ERR_CONTINUE(cond)                                                                            \
	if (cond) {                                                                                       \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, "ERR_CONTINUE: \"" #cond "\" is true!"); \
		continue;                                                                                     \
	} else                                                                                            \
		((void)0)

#define ERR_CONTINUE_MSG(cond, msg)                                \
	if (cond) {                                                    \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
		continue;                                                  \
	} else                                                         \
		((void)0)

#define ERR_CONTINUE_ACTION(cond, action)                                                             \
	if (cond) {                                                                                       \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, "ERR_CONTINUE: \"" #cond "\" is true!"); \
		action;                                                                                       \
		continue;                                                                                     \
	} else                                                                                            \
		((void)0)

#define ERR_CONTINUE_ACTION_MSG(cond, action, msg)                 \
	if (cond) {                                                    \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
		action;                                                    \
		continue;                                                  \
	} else                                                         \
		((void)0)

#define CRASH_INDEX(index, size)                                                           \
	if ((index < 0) || (index >= size)) {                                                  \
		RLogger::log_index_error(__FUNCTION__, __FILE__, __LINE__, index, size, "CRASH!"); \
		GENERATE_TRAP                                                                      \
	} else                                                                                 \
		((void)0)

#define CRASH_BAD_INDEX(index, size)                                                       \
	if ((index < 0) || (index >= size)) {                                                  \
		RLogger::log_index_error(__FUNCTION__, __FILE__, __LINE__, index, size, "CRASH!"); \
		GENERATE_TRAP                                                                      \
	} else                                                                                 \
		((void)0)

#define CRASH_BAD_UNSIGNED_INDEX(index, size)                                              \
	if ((index < 0) || (index >= size)) {                                                  \
		RLogger::log_index_error(__FUNCTION__, __FILE__, __LINE__, index, size, "CRASH!"); \
		GENERATE_TRAP                                                                      \
	} else                                                                                 \
		((void)0)

#define CRASH_COND(cond)                                                                            \
	if (cond) {                                                                                     \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, "CRASH_COND: \"" #cond "\" is true!"); \
		GENERATE_TRAP                                                                               \
	} else                                                                                          \
		((void)0)

#define CRASH_COND_MSG(cond, msg)                                  \
	if (cond) {                                                    \
		RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
		GENERATE_TRAP                                              \
	} else                                                         \
		((void)0)

#define CRASH_MSG(msg)                                         \
	RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
	GENERATE_TRAP

#define CRASH_NOW()                                                 \
	RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, "CRASH!"); \
	GENERATE_TRAP

/**
 * This should be a 'free' assert for program flow and should not be needed in any releases,
 *  only used in dev builds.
 */
#ifdef DEV_ENABLED
#define DEV_ASSERT(m_cond)                                                                                              \
	if (unlikely(!(m_cond))) {                                                                                          \
		_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "FATAL: DEV_ASSERT failed  \"" _STR(m_cond) "\" is false."); \
		_err_flush_stdout();                                                                                            \
		GENERATE_TRAP                                                                                                   \
	} else                                                                                                              \
		((void)0)
#else
#define DEV_ASSERT(m_cond)
#endif

/**
 * These should be 'free' checks for program flow and should not be needed in any releases,
 *  only used in dev builds.
 */
#ifdef DEV_ENABLED
#define DEV_CHECK(m_cond)                                              \
	if (unlikely(!(m_cond))) {                                         \
		ERR_PRINT("DEV_CHECK failed  \"" _STR(m_cond) "\" is false."); \
	} else                                                             \
		((void)0)
#else
#define DEV_CHECK(m_cond)
#endif

#ifdef DEV_ENABLED
#define DEV_CHECK_ONCE(m_cond)                                                   \
	if (unlikely(!(m_cond))) {                                                   \
		ERR_PRINT_ONCE("DEV_CHECK_ONCE failed  \"" _STR(m_cond) "\" is false."); \
	} else                                                                       \
		((void)0)
#else
#define DEV_CHECK_ONCE(m_cond)
#endif

#line 0

#line 1 "sfwl/core/memory.h"

/*************************************************************************/
/*  memory.h                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#ifndef PAD_ALIGN
#define PAD_ALIGN 16 //must always be greater than this at much
#endif

class Memory {
#ifdef DEBUG_ENABLED
	static SafeNumeric<uint64_t> mem_usage;
	static SafeNumeric<uint64_t> max_usage;
#endif

	static SafeNumeric<uint64_t> alloc_count;

public:
	static void *alloc_static(size_t p_bytes, bool p_pad_align = false);
	static void *realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align = false);
	static void free_static(void *p_ptr, bool p_pad_align = false);

	static uint64_t get_mem_available();
	static uint64_t get_mem_usage();
	static uint64_t get_mem_max_usage();
};

class DefaultAllocator {
public:
	_FORCE_INLINE_ static void *alloc(size_t p_memory) { return Memory::alloc_static(p_memory, false); }
	_FORCE_INLINE_ static void free(void *p_ptr) { Memory::free_static(p_ptr, false); }
};

void *operator new(size_t p_size, const char *p_description); ///< operator new that takes a description and uses MemoryStaticPool
void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size)); ///< operator new that takes a description and uses MemoryStaticPool

void *operator new(size_t p_size, void *p_pointer, size_t check, const char *p_description); ///< operator new that takes a description and uses a pointer to the preallocated memory

#ifdef _MSC_VER
// When compiling with VC++ 2017, the above declarations of placement new generate many irrelevant warnings (C4291).
// The purpose of the following definitions is to muffle these warnings, not to provide a usable implementation of placement delete.
void operator delete(void *p_mem, const char *p_description);
void operator delete(void *p_mem, void *(*p_allocfunc)(size_t p_size));
void operator delete(void *p_mem, void *p_pointer, size_t check, const char *p_description);
#endif

#define memalloc(m_size) Memory::alloc_static(m_size)
#define memrealloc(m_mem, m_size) Memory::realloc_static(m_mem, m_size)
#define memfree(m_mem) Memory::free_static(m_mem)

_ALWAYS_INLINE_ void postinitialize_handler(void *) {}

template <class T>
_ALWAYS_INLINE_ T *_post_initialize(T *p_obj) {
	postinitialize_handler(p_obj);
	return p_obj;
}

#define memnew(m_class) _post_initialize(new ("") m_class)

_ALWAYS_INLINE_ void *operator new(size_t p_size, void *p_pointer, size_t check, const char *p_description) {
	//void *failptr=0;
	//ERR_FAIL_COND_V( check < p_size , failptr); /** bug, or strange compiler, most likely */

	return p_pointer;
}

#define memnew_allocator(m_class, m_allocator) _post_initialize(new (m_allocator::alloc) m_class)
#define memnew_placement(m_placement, m_class) _post_initialize(new (m_placement, sizeof(m_class), "") m_class)

_ALWAYS_INLINE_ bool predelete_handler(void *) {
	return true;
}

template <class T>
void memdelete(T *p_class) {
	if (!predelete_handler(p_class)) {
		return; // doesn't want to be deleted
	}
	if (!HAS_TRIVIAL_DESTRUCTOR(T)) {
		p_class->~T();
	}

	Memory::free_static(p_class, false);
}

template <class T, class A>
void memdelete_allocator(T *p_class) {
	if (!predelete_handler(p_class)) {
		return; // doesn't want to be deleted
	}
	if (!HAS_TRIVIAL_DESTRUCTOR(T)) {
		p_class->~T();
	}

	A::free(p_class);
}

#define memdelete_notnull(m_v) \
	{                          \
		if (m_v)               \
			memdelete(m_v);    \
	}

#define memnew_arr(m_class, m_count) memnew_arr_template<m_class>(m_count)

template <typename T>
T *memnew_arr_template(size_t p_elements, const char *p_descr = "") {
	if (p_elements == 0) {
		return nullptr;
	}
	/** overloading operator new[] cannot be done , because it may not return the real allocated address (it may pad the 'element count' before the actual array). Because of that, it must be done by hand. This is the
	same strategy used by std::vector, and the PoolVector class, so it should be safe.*/

	size_t len = sizeof(T) * p_elements;
	uint64_t *mem = (uint64_t *)Memory::alloc_static(len, true);
	T *failptr = nullptr; //get rid of a warning
	ERR_FAIL_COND_V(!mem, failptr);
	*(mem - 1) = p_elements;

	if (!HAS_TRIVIAL_CONSTRUCTOR(T)) {
		T *elems = (T *)mem;

		/* call operator new */
		for (size_t i = 0; i < p_elements; i++) {
			new (&elems[i], sizeof(T), p_descr) T;
		}
	}

	return (T *)mem;
}

/**
 * Wonders of having own array functions, you can actually check the length of
 * an allocated-with memnew_arr() array
 */

template <typename T>
size_t memarr_len(const T *p_class) {
	uint64_t *ptr = (uint64_t *)p_class;
	return *(ptr - 1);
}

template <typename T>
void memdelete_arr(T *p_class) {
	uint64_t *ptr = (uint64_t *)p_class;

	if (!HAS_TRIVIAL_DESTRUCTOR(T)) {
		uint64_t elem_count = *(ptr - 1);

		for (uint64_t i = 0; i < elem_count; i++) {
			p_class[i].~T();
		}
	}

	Memory::free_static(ptr, true);
}

struct _GlobalNil {
	int color;
	_GlobalNil *right;
	_GlobalNil *left;
	_GlobalNil *parent;

	_GlobalNil();
};

struct _GlobalNilClass {
	static _GlobalNil _nil;
};

#line 0

#line 1 "sfwl/core/mutex.h"

/*************************************************************************/
/*  mutex.h                                                              */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if !defined(NO_THREADS)

#include <mutex>

template <class StdMutexT>
class MutexImpl {
	mutable StdMutexT mutex;
	friend class MutexLock;

public:
	_ALWAYS_INLINE_ void lock() const {
		mutex.lock();
	}

	_ALWAYS_INLINE_ void unlock() const {
		mutex.unlock();
	}

	_ALWAYS_INLINE_ Error try_lock() const {
		return mutex.try_lock() ? OK : ERR_BUSY;
	}
};

// This is written this way instead of being a template to overcome a limitation of C++ pre-17
// that would require MutexLock to be used like this: MutexLock<Mutex> lock;
class MutexLock {
	union {
		std::recursive_mutex *recursive_mutex;
		std::mutex *mutex;
	};
	bool recursive;

public:
	_ALWAYS_INLINE_ explicit MutexLock(const MutexImpl<std::recursive_mutex> &p_mutex) :
			recursive_mutex(&p_mutex.mutex),
			recursive(true) {
		recursive_mutex->lock();
	}
	_ALWAYS_INLINE_ explicit MutexLock(const MutexImpl<std::mutex> &p_mutex) :
			mutex(&p_mutex.mutex),
			recursive(false) {
		mutex->lock();
	}

	_ALWAYS_INLINE_ ~MutexLock() {
		if (recursive) {
			recursive_mutex->unlock();
		} else {
			mutex->unlock();
		}
	}
};

using Mutex = MutexImpl<std::recursive_mutex>; // Recursive, for general use
using BinaryMutex = MutexImpl<std::mutex>; // Non-recursive, handle with care

extern template class MutexImpl<std::recursive_mutex>;
extern template class MutexImpl<std::mutex>;

#else

class FakeMutex {
	FakeMutex() {}
};

template <class MutexT>
class MutexImpl {
public:
	_ALWAYS_INLINE_ void lock() const {}
	_ALWAYS_INLINE_ void unlock() const {}
	_ALWAYS_INLINE_ Error try_lock() const { return OK; }
};

class MutexLock {
public:
	explicit MutexLock(const MutexImpl<FakeMutex> &p_mutex) {}
};

using Mutex = MutexImpl<FakeMutex>;
using BinaryMutex = MutexImpl<FakeMutex>; // Non-recursive, handle with care

#endif // !NO_THREADS

#line 0

#line 1 "sfwl/core/rw_lock.h"

/*************************************************************************/
/*  rw_lock.h                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if !defined(NO_THREADS)

#include <shared_mutex>

class RWLock {
	mutable std::shared_timed_mutex mutex;

public:
	// Lock the rwlock, block if locked by someone else
	void read_lock() const {
		mutex.lock_shared();
	}

	// Unlock the rwlock, let other threads continue
	void read_unlock() const {
		mutex.unlock_shared();
	}

	// Attempt to lock the rwlock, OK on success, ERR_BUSY means it can't lock.
	Error read_try_lock() const {
		return mutex.try_lock_shared() ? OK : ERR_BUSY;
	}

	// Lock the rwlock, block if locked by someone else
	void write_lock() {
		mutex.lock();
	}

	// Unlock the rwlock, let other thwrites continue
	void write_unlock() {
		mutex.unlock();
	}

	// Attempt to lock the rwlock, OK on success, ERR_BUSY means it can't lock.
	Error write_try_lock() {
		return mutex.try_lock() ? OK : ERR_BUSY;
	}
};

#else

class RWLock {
public:
	void read_lock() const {}
	void read_unlock() const {}
	Error read_try_lock() const { return OK; }

	void write_lock() {}
	void write_unlock() {}
	Error write_try_lock() { return OK; }
};

#endif

class RWLockRead {
	const RWLock &lock;

public:
	RWLockRead(const RWLock &p_lock) :
			lock(p_lock) {
		lock.read_lock();
	}
	~RWLockRead() {
		lock.read_unlock();
	}
};

class RWLockWrite {
	RWLock &lock;

public:
	RWLockWrite(RWLock &p_lock) :
			lock(p_lock) {
		lock.write_lock();
	}
	~RWLockWrite() {
		lock.write_unlock();
	}
};

#line 0

#line 1 "sfwl/core/spin_lock.h"

/*************************************************************************/
/*  spin_lock.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class SpinLock {
	std::atomic_flag locked = ATOMIC_FLAG_INIT;

public:
	_ALWAYS_INLINE_ void lock() {
		while (locked.test_and_set(std::memory_order_acquire)) {
			;
		}
	}
	_ALWAYS_INLINE_ void unlock() {
		locked.clear(std::memory_order_release);
	}
};

#line 0

#line 1 "sfwl/core/thread_safe.h"

/*************************************************************************/
/*  thread_safe.h                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#define _THREAD_SAFE_CLASS_ mutable Mutex _thread_safe_;
#define _THREAD_SAFE_METHOD_ MutexLock _thread_safe_method_(_thread_safe_);
#define _THREAD_SAFE_LOCK_ _thread_safe_.lock();
#define _THREAD_SAFE_UNLOCK_ _thread_safe_.unlock();

#line 0

#line 1 "sfwl/core/pcg.h"
// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

#define PCG_DEFAULT_INC_64 1442695040888963407ULL

typedef struct {
	uint64_t state;
	uint64_t inc;
} pcg32_random_t;
uint32_t pcg32_random_r(pcg32_random_t *rng);
void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate, uint64_t initseq);
uint32_t pcg32_boundedrand_r(pcg32_random_t *rng, uint32_t bound);

#line 0

#line 1 "sfwl/core/random_pcg.h"

/*************************************************************************/
/*  random_pcg.h                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(__GNUC__) || (_llvm_has_builtin(__builtin_clz))
#define CLZ32(x) __builtin_clz(x)
#elif defined(_MSC_VER)
#include "intrin.h"
static int __bsr_clz32(uint32_t x) {
	unsigned long index;
	_BitScanReverse(&index, x);
	return 31 - index;
}
#define CLZ32(x) __bsr_clz32(x)
#else
#endif

#if defined(__GNUC__) || (_llvm_has_builtin(__builtin_ldexp) && _llvm_has_builtin(__builtin_ldexpf))
#define LDEXP(s, e) __builtin_ldexp(s, e)
#define LDEXPF(s, e) __builtin_ldexpf(s, e)
#else
#include "math.h"
#define LDEXP(s, e) ldexp(s, e)
#define LDEXPF(s, e) ldexp(s, e)
#endif

class RandomPCG {
	pcg32_random_t pcg;
	uint64_t current_seed; // The seed the current generator state started from.
	uint64_t current_inc;

public:
	static const uint64_t DEFAULT_SEED = 12047754176567800795U;
	static const uint64_t DEFAULT_INC = PCG_DEFAULT_INC_64;

	RandomPCG(uint64_t p_seed = DEFAULT_SEED, uint64_t p_inc = DEFAULT_INC);

	_FORCE_INLINE_ void seed(uint64_t p_seed) {
		current_seed = p_seed;
		pcg32_srandom_r(&pcg, current_seed, current_inc);
	}
	_FORCE_INLINE_ uint64_t get_seed() { return current_seed; }

	_FORCE_INLINE_ void set_state(uint64_t p_state) { pcg.state = p_state; }
	_FORCE_INLINE_ uint64_t get_state() const { return pcg.state; }

	void randomize();
	_FORCE_INLINE_ uint32_t rand() {
		return pcg32_random_r(&pcg);
	}
	_FORCE_INLINE_ uint32_t rand(uint32_t bounds) {
		return pcg32_boundedrand_r(&pcg, bounds);
	}

	// Obtaining floating point numbers in [0, 1] range with "good enough" uniformity.
	// These functions sample the output of rand() as the fraction part of an infinite binary number,
	// with some tricks applied to reduce ops and branching:
	// 1. Instead of shifting to the first 1 and connecting random bits, we simply set the MSB and LSB to 1.
	//    Provided that the RNG is actually uniform bit by bit, this should have the exact same effect.
	// 2. In order to compensate for exponent info loss, we count zeros from another random number,
	//    and just add that to the initial offset.
	//    This has the same probability as counting and shifting an actual bit stream: 2^-n for n zeroes.
	// For all numbers above 2^-96 (2^-64 for floats), the functions should be uniform.
	// However, all numbers below that threshold are floored to 0.
	// The thresholds are chosen to minimize rand() calls while keeping the numbers within a totally subjective quality standard.
	// If clz or ldexp isn't available, fall back to bit truncation for performance, sacrificing uniformity.
	_FORCE_INLINE_ double randd() {
#if defined(CLZ32)
		uint32_t proto_exp_offset = rand();
		if (unlikely(proto_exp_offset == 0)) {
			return 0;
		}
		uint64_t significand = (((uint64_t)rand()) << 32) | rand() | 0x8000000000000001U;
		return LDEXP((double)significand, -64 - CLZ32(proto_exp_offset));
#else
#pragma message("RandomPCG::randd - intrinsic clz is not available, falling back to bit truncation")
		return (double)(((((uint64_t)rand()) << 32) | rand()) & 0x1FFFFFFFFFFFFFU) / (double)0x1FFFFFFFFFFFFFU;
#endif
	}
	_FORCE_INLINE_ float randf() {
#if defined(CLZ32)
		uint32_t proto_exp_offset = rand();
		if (unlikely(proto_exp_offset == 0)) {
			return 0;
		}
		return LDEXPF((float)(rand() | 0x80000001), -32 - CLZ32(proto_exp_offset));
#else
#pragma message("RandomPCG::randf - intrinsic clz is not available, falling back to bit truncation")
		return (float)(rand() & 0xFFFFFF) / (float)0xFFFFFF;
#endif
	}

	_FORCE_INLINE_ double randfn(double p_mean, double p_deviation) {
		return p_mean + p_deviation * (cos(Math_TAU * randd()) * sqrt(-2.0 * log(randd()))); // Box-Muller transform
	}
	_FORCE_INLINE_ float randfn(float p_mean, float p_deviation) {
		return p_mean + p_deviation * (cos(Math_TAU * randf()) * sqrt(-2.0 * log(randf()))); // Box-Muller transform
	}

	double random(double p_from, double p_to);
	float random(float p_from, float p_to);
	real_t randomr(real_t p_from, real_t p_to) { return random(p_from, p_to); }
	int random(int p_from, int p_to);
};

#line 0

#line 1 "sfwl/core/math_funcs.h"

/*************************************************************************/
/*  math_funcs.h                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class Math {
	static RandomPCG default_rand;

public:
	Math() {} // useless to instance

	// Not using 'RANDOM_MAX' to avoid conflict with system headers on some OSes (at least NetBSD).
	static const uint64_t RANDOM_32BIT_MAX = 0xFFFFFFFF;

	static _ALWAYS_INLINE_ double sin(double p_x) { return ::sin(p_x); }
	static _ALWAYS_INLINE_ float sin(float p_x) { return ::sinf(p_x); }

	static _ALWAYS_INLINE_ double cos(double p_x) { return ::cos(p_x); }
	static _ALWAYS_INLINE_ float cos(float p_x) { return ::cosf(p_x); }

	static _ALWAYS_INLINE_ double tan(double p_x) { return ::tan(p_x); }
	static _ALWAYS_INLINE_ float tan(float p_x) { return ::tanf(p_x); }

	static _ALWAYS_INLINE_ double sinh(double p_x) { return ::sinh(p_x); }
	static _ALWAYS_INLINE_ float sinh(float p_x) { return ::sinhf(p_x); }

	static _ALWAYS_INLINE_ float sinc(float p_x) { return p_x == 0 ? 1 : ::sin(p_x) / p_x; }
	static _ALWAYS_INLINE_ double sinc(double p_x) { return p_x == 0 ? 1 : ::sin(p_x) / p_x; }

	static _ALWAYS_INLINE_ float sincn(float p_x) { return sinc((float)Math_PI * p_x); }
	static _ALWAYS_INLINE_ double sincn(double p_x) { return sinc(Math_PI * p_x); }

	static _ALWAYS_INLINE_ double cosh(double p_x) { return ::cosh(p_x); }
	static _ALWAYS_INLINE_ float cosh(float p_x) { return ::coshf(p_x); }

	static _ALWAYS_INLINE_ double tanh(double p_x) { return ::tanh(p_x); }
	static _ALWAYS_INLINE_ float tanh(float p_x) { return ::tanhf(p_x); }

	// Always does clamping so always safe to use.
	static _ALWAYS_INLINE_ double asin(double p_x) { return p_x < -1 ? (-Math_PI / 2) : (p_x > 1 ? (Math_PI / 2) : ::asin(p_x)); }
	static _ALWAYS_INLINE_ float asin(float p_x) { return p_x < -1 ? (-Math_PI / 2) : (p_x > 1 ? (Math_PI / 2) : ::asinf(p_x)); }

	// Always does clamping so always safe to use.
	static _ALWAYS_INLINE_ double acos(double p_x) { return p_x < -1 ? Math_PI : (p_x > 1 ? 0 : ::acos(p_x)); }
	static _ALWAYS_INLINE_ float acos(float p_x) { return p_x < -1 ? Math_PI : (p_x > 1 ? 0 : ::acosf(p_x)); }

	static _ALWAYS_INLINE_ double asin_unsafe(double p_x) { return ::asin(p_x); }
	static _ALWAYS_INLINE_ float asin_unsafe(float p_x) { return ::asinf(p_x); }

	static _ALWAYS_INLINE_ double acos_unsafe(double p_x) { return ::acos(p_x); }
	static _ALWAYS_INLINE_ float acos_unsafe(float p_x) { return ::acosf(p_x); }

	static _ALWAYS_INLINE_ double atan(double p_x) { return ::atan(p_x); }
	static _ALWAYS_INLINE_ float atan(float p_x) { return ::atanf(p_x); }

	static _ALWAYS_INLINE_ double atan2(double p_y, double p_x) { return ::atan2(p_y, p_x); }
	static _ALWAYS_INLINE_ float atan2(float p_y, float p_x) { return ::atan2f(p_y, p_x); }

	static _ALWAYS_INLINE_ double sqrt(double p_x) { return ::sqrt(p_x); }
	static _ALWAYS_INLINE_ float sqrt(float p_x) { return ::sqrtf(p_x); }

	static _ALWAYS_INLINE_ double fmod(double p_x, double p_y) { return ::fmod(p_x, p_y); }
	static _ALWAYS_INLINE_ float fmod(float p_x, float p_y) { return ::fmodf(p_x, p_y); }

	static _ALWAYS_INLINE_ double floor(double p_x) { return ::floor(p_x); }
	static _ALWAYS_INLINE_ float floor(float p_x) { return ::floorf(p_x); }
	// x + 0.5 -> so f.e. 0.9999999 will become 1
	static _ALWAYS_INLINE_ int floorf_int(const float x) { return static_cast<int>(x + 0.5); }

	static _ALWAYS_INLINE_ double ceil(double p_x) { return ::ceil(p_x); }
	static _ALWAYS_INLINE_ float ceil(float p_x) { return ::ceilf(p_x); }

	static _ALWAYS_INLINE_ double pow(double p_x, double p_y) { return ::pow(p_x, p_y); }
	static _ALWAYS_INLINE_ float pow(float p_x, float p_y) { return ::powf(p_x, p_y); }

	static _ALWAYS_INLINE_ double log(double p_x) { return ::log(p_x); }
	static _ALWAYS_INLINE_ float log(float p_x) { return ::logf(p_x); }

	static _ALWAYS_INLINE_ double log1p(double p_x) { return ::log1p(p_x); }
	static _ALWAYS_INLINE_ float log1p(float p_x) { return ::log1pf(p_x); }

	static _ALWAYS_INLINE_ double log10(double p_x) { return ::log10f(p_x); }
	static _ALWAYS_INLINE_ float log10(float p_x) { return ::log10(p_x); }

	static _ALWAYS_INLINE_ double log2(double p_x) { return ::log2(p_x); }
	static _ALWAYS_INLINE_ float log2(float p_x) { return ::log2f(p_x); }

	static _ALWAYS_INLINE_ double exp(double p_x) { return ::exp(p_x); }
	static _ALWAYS_INLINE_ float exp(float p_x) { return ::expf(p_x); }

	static _ALWAYS_INLINE_ double erf(double p_x) { return ::erf(p_x); }
	static _ALWAYS_INLINE_ float erf(float p_x) { return ::erff(p_x); }

	// can save typing static_cast<float>
	inline static float divf(const float a, const float b) { return a / b; }

	static _ALWAYS_INLINE_ bool is_nan(double p_val) {
#ifdef _MSC_VER
		return _isnan(p_val);
#elif defined(__GNUC__) && __GNUC__ < 6
		union {
			uint64_t u;
			double f;
		} ieee754;
		ieee754.f = p_val;
		// (unsigned)(0x7ff0000000000001 >> 32) : 0x7ff00000
		return ((((unsigned)(ieee754.u >> 32) & 0x7fffffff) + ((unsigned)ieee754.u != 0)) > 0x7ff00000);
#else
		return isnan(p_val);
#endif
	}

	static _ALWAYS_INLINE_ bool is_nan(float p_val) {
#ifdef _MSC_VER
		return _isnan(p_val);
#elif defined(__GNUC__) && __GNUC__ < 6
		union {
			uint32_t u;
			float f;
		} ieee754;
		ieee754.f = p_val;
		// -----------------------------------
		// (single-precision floating-point)
		// NaN : s111 1111 1xxx xxxx xxxx xxxx xxxx xxxx
		//     : (> 0x7f800000)
		// where,
		//   s : sign
		//   x : non-zero number
		// -----------------------------------
		return ((ieee754.u & 0x7fffffff) > 0x7f800000);
#else
		return isnan(p_val);
#endif
	}

	static _ALWAYS_INLINE_ bool is_inf(double p_val) {
#ifdef _MSC_VER
		return !_finite(p_val);
// use an inline implementation of isinf as a workaround for problematic libstdc++ versions from gcc 5.x era
#elif defined(__GNUC__) && __GNUC__ < 6
		union {
			uint64_t u;
			double f;
		} ieee754;
		ieee754.f = p_val;
		return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) == 0x7ff00000 &&
				((unsigned)ieee754.u == 0);
#else
		return isinf(p_val);
#endif
	}

	static _ALWAYS_INLINE_ bool is_inf(float p_val) {
#ifdef _MSC_VER
		return !_finite(p_val);
// use an inline implementation of isinf as a workaround for problematic libstdc++ versions from gcc 5.x era
#elif defined(__GNUC__) && __GNUC__ < 6
		union {
			uint32_t u;
			float f;
		} ieee754;
		ieee754.f = p_val;
		return (ieee754.u & 0x7fffffff) == 0x7f800000;
#else
		return isinf(p_val);
#endif
	}

	static _ALWAYS_INLINE_ double abs(double g) {
		return absd(g);
	}
	static _ALWAYS_INLINE_ float abs(float g) {
		return absf(g);
	}
	static _ALWAYS_INLINE_ int abs(int g) {
		return g > 0 ? g : -g;
	}
	static _ALWAYS_INLINE_ int64_t abs(int64_t g) {
		return g > 0 ? g : -g;
	}
	static _ALWAYS_INLINE_ int absi(int g) {
		return g > 0 ? g : -g;
	}
	static _ALWAYS_INLINE_ int64_t absi(int64_t g) {
		return g > 0 ? g : -g;
	}

	static _ALWAYS_INLINE_ double fposmod(double p_x, double p_y) {
		double value = Math::fmod(p_x, p_y);
		if (((value < 0) && (p_y > 0)) || ((value > 0) && (p_y < 0))) {
			value += p_y;
		}
		value += 0.0;
		return value;
	}
	static _ALWAYS_INLINE_ float fposmod(float p_x, float p_y) {
		float value = Math::fmod(p_x, p_y);
		if (((value < 0) && (p_y > 0)) || ((value > 0) && (p_y < 0))) {
			value += p_y;
		}
		value += 0.0f;
		return value;
	}

	static _ALWAYS_INLINE_ float fposmodp(float p_x, float p_y) {
		float value = Math::fmod(p_x, p_y);
		if (value < 0) {
			value += p_y;
		}
		value += 0.0f;
		return value;
	}
	static _ALWAYS_INLINE_ double fposmodp(double p_x, double p_y) {
		double value = Math::fmod(p_x, p_y);
		if (value < 0) {
			value += p_y;
		}
		value += 0.0;
		return value;
	}

	static _ALWAYS_INLINE_ int64_t posmod(int64_t p_x, int64_t p_y) {
		ERR_FAIL_COND_V_MSG(p_y == 0, 0, "Division by zero in posmod is undefined. Returning 0 as fallback.");
		int64_t value = p_x % p_y;
		if (((value < 0) && (p_y > 0)) || ((value > 0) && (p_y < 0))) {
			value += p_y;
		}
		return value;
	}

	static _ALWAYS_INLINE_ double deg2rad(double p_y) {
		return p_y * Math_PI / 180.0;
	}
	static _ALWAYS_INLINE_ float deg2rad(float p_y) {
		return p_y * (float)(Math_PI / 180.0);
	}

	static _ALWAYS_INLINE_ double rad2deg(double p_y) {
		return p_y * 180.0 / Math_PI;
	}
	static _ALWAYS_INLINE_ float rad2deg(float p_y) {
		return p_y * (float)(180.0 / Math_PI);
	}

	static _ALWAYS_INLINE_ double lerp(double p_from, double p_to, double p_weight) {
		return p_from + (p_to - p_from) * p_weight;
	}
	static _ALWAYS_INLINE_ float lerp(float p_from, float p_to, float p_weight) {
		return p_from + (p_to - p_from) * p_weight;
	}

	static _ALWAYS_INLINE_ double lerp_angle(double p_from, double p_to, double p_weight) {
		double difference = fmod(p_to - p_from, Math_TAU);
		double distance = fmod(2.0 * difference, Math_TAU) - difference;
		return p_from + distance * p_weight;
	}
	static _ALWAYS_INLINE_ float lerp_angle(float p_from, float p_to, float p_weight) {
		float difference = fmod(p_to - p_from, (float)Math_TAU);
		float distance = fmod(2.0f * difference, (float)Math_TAU) - difference;
		return p_from + distance * p_weight;
	}

	static _ALWAYS_INLINE_ double inverse_lerp(double p_from, double p_to, double p_value) {
		return (p_value - p_from) / (p_to - p_from);
	}
	static _ALWAYS_INLINE_ float inverse_lerp(float p_from, float p_to, float p_value) {
		return (p_value - p_from) / (p_to - p_from);
	}

	static _ALWAYS_INLINE_ double range_lerp(double p_value, double p_istart, double p_istop, double p_ostart, double p_ostop) {
		return Math::lerp(p_ostart, p_ostop, Math::inverse_lerp(p_istart, p_istop, p_value));
	}
	static _ALWAYS_INLINE_ float range_lerp(float p_value, float p_istart, float p_istop, float p_ostart, float p_ostop) {
		return Math::lerp(p_ostart, p_ostop, Math::inverse_lerp(p_istart, p_istop, p_value));
	}

	static _ALWAYS_INLINE_ double cubic_interpolate(double p_from, double p_to, double p_pre, double p_post, double p_weight) {
		return 0.5 *
				((p_from * 2.0) +
						(-p_pre + p_to) * p_weight +
						(2.0 * p_pre - 5.0 * p_from + 4.0 * p_to - p_post) * (p_weight * p_weight) +
						(-p_pre + 3.0 * p_from - 3.0 * p_to + p_post) * (p_weight * p_weight * p_weight));
	}
	static _ALWAYS_INLINE_ float cubic_interpolate(float p_from, float p_to, float p_pre, float p_post, float p_weight) {
		return 0.5f *
				((p_from * 2.0f) +
						(-p_pre + p_to) * p_weight +
						(2.0f * p_pre - 5.0f * p_from + 4.0f * p_to - p_post) * (p_weight * p_weight) +
						(-p_pre + 3.0f * p_from - 3.0f * p_to + p_post) * (p_weight * p_weight * p_weight));
	}

	static _ALWAYS_INLINE_ double bezier_interpolate(double p_start, double p_control_1, double p_control_2, double p_end, double p_t) {
		/* Formula from Wikipedia article on Bezier curves. */
		double omt = (1.0 - p_t);
		double omt2 = omt * omt;
		double omt3 = omt2 * omt;
		double t2 = p_t * p_t;
		double t3 = t2 * p_t;

		return p_start * omt3 + p_control_1 * omt2 * p_t * 3.0 + p_control_2 * omt * t2 * 3.0 + p_end * t3;
	}
	static _ALWAYS_INLINE_ float bezier_interpolate(float p_start, float p_control_1, float p_control_2, float p_end, float p_t) {
		/* Formula from Wikipedia article on Bezier curves. */
		float omt = (1.0f - p_t);
		float omt2 = omt * omt;
		float omt3 = omt2 * omt;
		float t2 = p_t * p_t;
		float t3 = t2 * p_t;

		return p_start * omt3 + p_control_1 * omt2 * p_t * 3.0f + p_control_2 * omt * t2 * 3.0f + p_end * t3;
	}

	static _ALWAYS_INLINE_ double smoothstep(double p_from, double p_to, double p_s) {
		if (is_equal_approx(p_from, p_to)) {
			return p_from;
		}
		double s = CLAMP((p_s - p_from) / (p_to - p_from), 0.0, 1.0);
		return s * s * (3.0 - 2.0 * s);
	}
	static _ALWAYS_INLINE_ float smoothstep(float p_from, float p_to, float p_s) {
		if (is_equal_approx(p_from, p_to)) {
			return p_from;
		}
		float s = CLAMP((p_s - p_from) / (p_to - p_from), 0.0f, 1.0f);
		return s * s * (3.0f - 2.0f * s);
	}
	static _ALWAYS_INLINE_ double move_toward(double p_from, double p_to, double p_delta) {
		return abs(p_to - p_from) <= p_delta ? p_to : p_from + SGN(p_to - p_from) * p_delta;
	}
	static _ALWAYS_INLINE_ float move_toward(float p_from, float p_to, float p_delta) {
		return abs(p_to - p_from) <= p_delta ? p_to : p_from + SGN(p_to - p_from) * p_delta;
	}

	static _ALWAYS_INLINE_ double linear2db(double p_linear) {
		return Math::log(p_linear) * 8.6858896380650365530225783783321;
	}
	static _ALWAYS_INLINE_ float linear2db(float p_linear) {
		return Math::log(p_linear) * (float)8.6858896380650365530225783783321;
	}

	static _ALWAYS_INLINE_ double db2linear(double p_db) {
		return Math::exp(p_db * 0.11512925464970228420089957273422);
	}
	static _ALWAYS_INLINE_ float db2linear(float p_db) {
		return Math::exp(p_db * (float)0.11512925464970228420089957273422);
	}

	static _ALWAYS_INLINE_ double round(double p_val) {
		return ::round(p_val);
	}
	static _ALWAYS_INLINE_ float round(float p_val) {
		return ::roundf(p_val);
	}

	static _ALWAYS_INLINE_ int64_t wrapi(int64_t value, int64_t min, int64_t max) {
		int64_t range = max - min;
		return range == 0 ? min : min + ((((value - min) % range) + range) % range);
	}
	static _ALWAYS_INLINE_ double wrapf(double value, double min, double max) {
		double range = max - min;
		double result = is_zero_approx(range) ? min : value - (range * Math::floor((value - min) / range));
		if (is_equal_approx(result, max)) {
			return min;
		}
		return result;
	}
	static _ALWAYS_INLINE_ float wrapf(float value, float min, float max) {
		float range = max - min;
		float result = is_zero_approx(range) ? min : value - (range * Math::floor((value - min) / range));
		if (is_equal_approx(result, max)) {
			return min;
		}
		return result;
	}

	static _ALWAYS_INLINE_ float fract(float value) {
		return value - floor(value);
	}
	static _ALWAYS_INLINE_ double fract(double value) {
		return value - floor(value);
	}
	static _ALWAYS_INLINE_ float pingpong(float value, float length) {
		return (length != 0.0f) ? abs(fract((value - length) / (length * 2.0f)) * length * 2.0f - length) : 0.0f;
	}
	static _ALWAYS_INLINE_ double pingpong(double value, double length) {
		return (length != 0.0) ? abs(fract((value - length) / (length * 2.0)) * length * 2.0 - length) : 0.0;
	}

	// double only, as these functions are mainly used by the editor and not performance-critical,
	static double ease(double p_x, double p_c);
	static int step_decimals(double p_step);
	static int range_step_decimals(double p_step);
	static double stepify(double p_value, double p_step);
	static double dectime(double p_value, double p_amount, double p_step);

	static uint32_t larger_prime(uint32_t p_val);

	static void seed(uint64_t x);
	static void randomize();
	static uint32_t rand_from_seed(uint64_t *seed);
	static uint32_t rand();
	static _ALWAYS_INLINE_ double randd() {
		return (double)rand() / (double)Math::RANDOM_32BIT_MAX;
	}
	static _ALWAYS_INLINE_ float randf() {
		return (float)rand() / (float)Math::RANDOM_32BIT_MAX;
	}
	static double randfn(double mean, double deviation);

	static double random(double from, double to);
	static float random(float from, float to);
	static real_t randomr(real_t from, real_t to);
	static int random(int from, int to);
	static _ALWAYS_INLINE_ int randomi(int from, int to) {
		return random(from, to);
	}

	static _ALWAYS_INLINE_ bool is_equal_approx_ratio(real_t a, real_t b, real_t epsilon = CMP_EPSILON, real_t min_epsilon = CMP_EPSILON) {
		// this is an approximate way to check that numbers are close, as a ratio of their average size
		// helps compare approximate numbers that may be very big or very small
		real_t diff = abs(a - b);
		if (diff == 0 || diff < min_epsilon) {
			return true;
		}
		real_t avg_size = (abs(a) + abs(b)) / 2;
		diff /= avg_size;
		return diff < epsilon;
	}

	static _ALWAYS_INLINE_ bool is_equal_approx(float a, float b) {
		// Check for exact equality first, required to handle "infinity" values.
		if (a == b) {
			return true;
		}
		// Then check for approximate equality.
		float tolerance = (float)CMP_EPSILON * abs(a);
		if (tolerance < (float)CMP_EPSILON) {
			tolerance = (float)CMP_EPSILON;
		}
		return abs(a - b) < tolerance;
	}

	static _ALWAYS_INLINE_ bool is_equal_approx(float a, float b, float tolerance) {
		// Check for exact equality first, required to handle "infinity" values.
		if (a == b) {
			return true;
		}
		// Then check for approximate equality.
		return abs(a - b) < tolerance;
	}

	static _ALWAYS_INLINE_ bool is_equal_approxt(float a, float b, float tolerance) {
		// Check for exact equality first, required to handle "infinity" values.
		if (a == b) {
			return true;
		}
		// Then check for approximate equality.
		return abs(a - b) < tolerance;
	}

	static _ALWAYS_INLINE_ bool is_zero_approx(float s) {
		return abs(s) < (float)CMP_EPSILON;
	}

	static _ALWAYS_INLINE_ bool is_equal_approx(double a, double b) {
		// Check for exact equality first, required to handle "infinity" values.
		if (a == b) {
			return true;
		}
		// Then check for approximate equality.
		double tolerance = CMP_EPSILON * abs(a);
		if (tolerance < CMP_EPSILON) {
			tolerance = CMP_EPSILON;
		}
		return abs(a - b) < tolerance;
	}

	static _ALWAYS_INLINE_ bool is_equal_approx(double a, double b, double tolerance) {
		// Check for exact equality first, required to handle "infinity" values.
		if (a == b) {
			return true;
		}
		// Then check for approximate equality.
		return abs(a - b) < tolerance;
	}

	static _ALWAYS_INLINE_ bool is_zero_approx(double s) {
		return abs(s) < CMP_EPSILON;
	}

	static _ALWAYS_INLINE_ float absf(float g) {
		union {
			float f;
			uint32_t i;
		} u;

		u.f = g;
		u.i &= 2147483647u;
		return u.f;
	}

	static _ALWAYS_INLINE_ double absd(double g) {
		union {
			double d;
			uint64_t i;
		} u;
		u.d = g;
		u.i &= (uint64_t)9223372036854775807ll;
		return u.d;
	}

	// This function should be as fast as possible and rounding mode should not matter.
	static _ALWAYS_INLINE_ int fast_ftoi(float a) {
		// Assuming every supported compiler has `lrint()`.
		return lrintf(a);
	}

	static _ALWAYS_INLINE_ uint32_t halfbits_to_floatbits(uint16_t h) {
		uint16_t h_exp, h_sig;
		uint32_t f_sgn, f_exp, f_sig;

		h_exp = (h & 0x7c00u);
		f_sgn = ((uint32_t)h & 0x8000u) << 16;
		switch (h_exp) {
			case 0x0000u: /* 0 or subnormal */
				h_sig = (h & 0x03ffu);
				/* Signed zero */
				if (h_sig == 0) {
					return f_sgn;
				}
				/* Subnormal */
				h_sig <<= 1;
				while ((h_sig & 0x0400u) == 0) {
					h_sig <<= 1;
					h_exp++;
				}
				f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
				f_sig = ((uint32_t)(h_sig & 0x03ffu)) << 13;
				return f_sgn + f_exp + f_sig;
			case 0x7c00u: /* inf or NaN */
				/* All-ones exponent and a copy of the significand */
				return f_sgn + 0x7f800000u + (((uint32_t)(h & 0x03ffu)) << 13);
			default: /* normalized */
				/* Just need to adjust the exponent and shift */
				return f_sgn + (((uint32_t)(h & 0x7fffu) + 0x1c000u) << 13);
		}
	}

	static _ALWAYS_INLINE_ float halfptr_to_float(const uint16_t *h) {
		union {
			uint32_t u32;
			float f32;
		} u;

		u.u32 = halfbits_to_floatbits(*h);
		return u.f32;
	}

	static _ALWAYS_INLINE_ float half_to_float(const uint16_t h) {
		return halfptr_to_float(&h);
	}

	static _ALWAYS_INLINE_ uint16_t make_half_float(float f) {
		union {
			float fv;
			uint32_t ui;
		} ci;
		ci.fv = f;

		uint32_t x = ci.ui;
		uint32_t sign = (unsigned short)(x >> 31);
		uint32_t mantissa;
		uint32_t exp;
		uint16_t hf;

		// get mantissa
		mantissa = x & ((1 << 23) - 1);
		// get exponent bits
		exp = x & (0xFF << 23);
		if (exp >= 0x47800000) {
			// check if the original single precision float number is a NaN
			if (mantissa && (exp == (0xFF << 23))) {
				// we have a single precision NaN
				mantissa = (1 << 23) - 1;
			} else {
				// 16-bit half-float representation stores number as Inf
				mantissa = 0;
			}
			hf = (((uint16_t)sign) << 15) | (uint16_t)((0x1F << 10)) |
					(uint16_t)(mantissa >> 13);
		}
		// check if exponent is <= -15
		else if (exp <= 0x38000000) {
			/*// store a denorm half-float value or zero
		exp = (0x38000000 - exp) >> 23;
		mantissa >>= (14 + exp);

		hf = (((uint16_t)sign) << 15) | (uint16_t)(mantissa);
		*/
			hf = 0; //denormals do not work for 3D, convert to zero
		} else {
			hf = (((uint16_t)sign) << 15) |
					(uint16_t)((exp - 0x38000000) >> 13) |
					(uint16_t)(mantissa >> 13);
		}

		return hf;
	}

	static _ALWAYS_INLINE_ float snap_scalar(float p_offset, float p_step, float p_target) {
		return p_step != 0 ? Math::stepify(p_target - p_offset, p_step) + p_offset : p_target;
	}

	static _ALWAYS_INLINE_ float snap_scalar_separation(float p_offset, float p_step, float p_target, float p_separation) {
		if (p_step != 0) {
			float a = Math::stepify(p_target - p_offset, p_step + p_separation) + p_offset;
			float b = a;
			if (p_target >= 0) {
				b -= p_separation;
			} else {
				b += p_step;
			}
			return (Math::abs(p_target - a) < Math::abs(p_target - b)) ? a : b;
		}
		return p_target;
	}
};

#line 0

#line 1 "sfwl/core/cowdata.h"

/*************************************************************************/
/*  cowdata.h                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

template <class T>
class Vector;
class String;
class Char16String;
class CharString;
template <class T, class V>
class VMap;

#if !defined(NO_THREADS)
SAFE_NUMERIC_TYPE_PUN_GUARANTEES(uint32_t)
#endif

template <class T>
class CowData {
	template <class TV>
	friend class Vector;
	friend class String;
	friend class Char16String;
	friend class CharString;
	template <class TV, class VV>
	friend class VMap;

private:
	mutable T *_ptr;

	// internal helpers

	_FORCE_INLINE_ SafeNumeric<uint32_t> *_get_refcount() const {
		if (!_ptr) {
			return nullptr;
		}

		return reinterpret_cast<SafeNumeric<uint32_t> *>(_ptr) - 2;
	}

	_FORCE_INLINE_ uint32_t *_get_size() const {
		if (!_ptr) {
			return nullptr;
		}

		return reinterpret_cast<uint32_t *>(_ptr) - 1;
	}

	_FORCE_INLINE_ size_t _get_alloc_size(size_t p_elements) const {
		//return nearest_power_of_2_templated(p_elements*sizeof(T)+sizeof(SafeRefCount)+sizeof(int));
		return next_power_of_2(p_elements * sizeof(T));
	}

	_FORCE_INLINE_ bool _get_alloc_size_checked(size_t p_elements, size_t *out) const {
#if defined(_add_overflow) && defined(_mul_overflow)
		size_t o;
		size_t p;
		if (_mul_overflow(p_elements, sizeof(T), &o)) {
			*out = 0;
			return false;
		}
		*out = next_power_of_2(o);
		if (_add_overflow(o, static_cast<size_t>(32), &p)) {
			return false; //no longer allocated here
		}
		return true;
#else
		// Speed is more important than correctness here, do the operations unchecked
		// and hope the best
		*out = _get_alloc_size(p_elements);
		return true;
#endif
	}

	void _unref(void *p_data);
	void _ref(const CowData *p_from);
	void _ref(const CowData &p_from);
	uint32_t _copy_on_write();

public:
	void operator=(const CowData<T> &p_from) { _ref(p_from); }

	_FORCE_INLINE_ T *ptrw() {
		_copy_on_write();
		return _ptr;
	}

	_FORCE_INLINE_ const T *ptr() const {
		return _ptr;
	}

	_FORCE_INLINE_ int size() const {
		uint32_t *size = (uint32_t *)_get_size();
		if (size) {
			return *size;
		} else {
			return 0;
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ bool empty() const { return _ptr == nullptr; }

	_FORCE_INLINE_ void set(int p_index, const T &p_elem) {
		CRASH_BAD_INDEX(p_index, size());
		_copy_on_write();
		_ptr[p_index] = p_elem;
	}

	_FORCE_INLINE_ T &get_m(int p_index) {
		CRASH_BAD_INDEX(p_index, size());
		_copy_on_write();
		return _ptr[p_index];
	}

	_FORCE_INLINE_ const T &get(int p_index) const {
		CRASH_BAD_INDEX(p_index, size());

		return _ptr[p_index];
	}

	Error resize(int p_size);

	_FORCE_INLINE_ void remove(int p_index) {
		ERR_FAIL_INDEX(p_index, size());
		T *p = ptrw();
		int len = size();
		for (int i = p_index; i < len - 1; i++) {
			p[i] = p[i + 1];
		};

		resize(len - 1);
	}

	Error insert(int p_pos, const T &p_val) {
		ERR_FAIL_INDEX_V(p_pos, size() + 1, ERR_INVALID_PARAMETER);
		resize(size() + 1);
		for (int i = (size() - 1); i > p_pos; i--) {
			set(i, get(i - 1));
		}
		set(p_pos, p_val);

		return OK;
	}

	void fill(const T &p_val) {
		int len = size();

		if (len == 0) {
			return;
		}

		T *p = ptrw();
		for (int i = 0; i < len; ++i) {
			p[i] = p_val;
		}
	}

	int find(const T &p_val, int p_from = 0) const;

	_FORCE_INLINE_ CowData();
	_FORCE_INLINE_ ~CowData();
	_FORCE_INLINE_ CowData(CowData<T> &p_from) { _ref(p_from); };
};

template <class T>
void CowData<T>::_unref(void *p_data) {
	if (!p_data) {
		return;
	}

	SafeNumeric<uint32_t> *refc = _get_refcount();

	if (refc->decrement() > 0) {
		return; // still in use
	}
	// clean up

	if (!HAS_TRIVIAL_DESTRUCTOR(T)) {
		uint32_t *count = _get_size();
		T *data = (T *)(count + 1);

		for (uint32_t i = 0; i < *count; ++i) {
			// call destructors
			data[i].~T();
		}
	}

	// free mem
	Memory::free_static((uint8_t *)p_data, true);
}

template <class T>
uint32_t CowData<T>::_copy_on_write() {
	if (!_ptr) {
		return 0;
	}

	SafeNumeric<uint32_t> *refc = _get_refcount();

	uint32_t rc = refc->get();
	if (likely(rc > 1)) {
		/* in use by more than me */
		uint32_t current_size = *_get_size();

		uint32_t *mem_new = (uint32_t *)Memory::alloc_static(_get_alloc_size(current_size), true);

		new (mem_new - 2, sizeof(uint32_t), "") SafeNumeric<uint32_t>(1); //refcount
		*(mem_new - 1) = current_size; //size

		T *_data = (T *)(mem_new);

		// initialize new elements
		if (HAS_TRIVIAL_COPY(T)) {
			memcpy(mem_new, _ptr, current_size * sizeof(T));

		} else {
			for (uint32_t i = 0; i < current_size; i++) {
				memnew_placement(&_data[i], T(_ptr[i]));
			}
		}

		_unref(_ptr);
		_ptr = _data;

		rc = 1;
	}
	return rc;
}

template <class T>
Error CowData<T>::resize(int p_size) {
	ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);

	int current_size = size();

	if (p_size == current_size) {
		return OK;
	}

	if (p_size == 0) {
		// wants to clean up
		_unref(_ptr);
		_ptr = nullptr;
		return OK;
	}

	// possibly changing size, copy on write
	uint32_t rc = _copy_on_write();

	size_t current_alloc_size = _get_alloc_size(current_size);
	size_t alloc_size;
	ERR_FAIL_COND_V(!_get_alloc_size_checked(p_size, &alloc_size), ERR_OUT_OF_MEMORY);

	if (p_size > current_size) {
		if (alloc_size != current_alloc_size) {
			if (current_size == 0) {
				// alloc from scratch
				uint32_t *ptr = (uint32_t *)Memory::alloc_static(alloc_size, true);
				ERR_FAIL_COND_V(!ptr, ERR_OUT_OF_MEMORY);
				*(ptr - 1) = 0; //size, currently none
				new (ptr - 2, sizeof(uint32_t), "") SafeNumeric<uint32_t>(1); //refcount

				_ptr = (T *)ptr;

			} else {
				uint32_t *_ptrnew = (uint32_t *)Memory::realloc_static(_ptr, alloc_size, true);
				ERR_FAIL_COND_V(!_ptrnew, ERR_OUT_OF_MEMORY);
				new (_ptrnew - 2, sizeof(uint32_t), "") SafeNumeric<uint32_t>(rc); //refcount

				_ptr = (T *)(_ptrnew);
			}
		}

		// construct the newly created elements

		if (!HAS_TRIVIAL_CONSTRUCTOR(T)) {
			for (int i = *_get_size(); i < p_size; i++) {
				memnew_placement(&_ptr[i], T);
			}
		}

		*_get_size() = p_size;

	} else if (p_size < current_size) {
		if (!HAS_TRIVIAL_DESTRUCTOR(T)) {
			// deinitialize no longer needed elements
			for (uint32_t i = p_size; i < *_get_size(); i++) {
				T *t = &_ptr[i];
				t->~T();
			}
		}

		if (alloc_size != current_alloc_size) {
			uint32_t *_ptrnew = (uint32_t *)Memory::realloc_static(_ptr, alloc_size, true);
			ERR_FAIL_COND_V(!_ptrnew, ERR_OUT_OF_MEMORY);
			new (_ptrnew - 2, sizeof(uint32_t), "") SafeNumeric<uint32_t>(rc); //refcount

			_ptr = (T *)(_ptrnew);
		}

		*_get_size() = p_size;
	}

	return OK;
}

template <class T>
int CowData<T>::find(const T &p_val, int p_from) const {
	int ret = -1;

	if (p_from < 0 || size() == 0) {
		return ret;
	}

	for (int i = p_from; i < size(); i++) {
		if (get(i) == p_val) {
			ret = i;
			break;
		}
	}

	return ret;
}

template <class T>
void CowData<T>::_ref(const CowData *p_from) {
	_ref(*p_from);
}

template <class T>
void CowData<T>::_ref(const CowData &p_from) {
	if (_ptr == p_from._ptr) {
		return; // self assign, do nothing.
	}

	_unref(_ptr);
	_ptr = nullptr;

	if (!p_from._ptr) {
		return; //nothing to do
	}

	if (p_from._get_refcount()->conditional_increment() > 0) { // could reference
		_ptr = p_from._ptr;
	}
}

template <class T>
CowData<T>::CowData() {
	_ptr = nullptr;
}

template <class T>
CowData<T>::~CowData() {
	_unref(_ptr);
}

#line 0

#line 1 "sfwl/core/sort_array.h"

/*************************************************************************/
/*  sort_array.h                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#define ERR_BAD_COMPARE(cond)                                         \
	if (unlikely(cond)) {                                             \
		ERR_PRINT("bad comparison function; sorting will be broken"); \
		break;                                                        \
	}

template <class T>
struct _DefaultComparator {
	_FORCE_INLINE_ bool operator()(const T &a, const T &b) const { return (a < b); }
};

#ifdef DEBUG_ENABLED
#define SORT_ARRAY_VALIDATE_ENABLED true
#else
#define SORT_ARRAY_VALIDATE_ENABLED false
#endif

template <class T, class Comparator = _DefaultComparator<T>, bool Validate = SORT_ARRAY_VALIDATE_ENABLED>
class SortArray {
	enum {

		INTROSORT_THRESHOLD = 16
	};

public:
	Comparator compare;

	inline const T &median_of_3(const T &a, const T &b, const T &c) const {
		if (compare(a, b)) {
			if (compare(b, c)) {
				return b;
			} else if (compare(a, c)) {
				return c;
			} else {
				return a;
			}
		} else if (compare(a, c)) {
			return a;
		} else if (compare(b, c)) {
			return c;
		} else {
			return b;
		}
	}

	inline int bitlog(int n) const {
		int k;
		for (k = 0; n != 1; n >>= 1) {
			++k;
		}
		return k;
	}

	/* Heap / Heapsort functions */

	inline void push_heap(int p_first, int p_hole_idx, int p_top_index, T p_value, T *p_array) const {
		int parent = (p_hole_idx - 1) / 2;
		while (p_hole_idx > p_top_index && compare(p_array[p_first + parent], p_value)) {
			p_array[p_first + p_hole_idx] = p_array[p_first + parent];
			p_hole_idx = parent;
			parent = (p_hole_idx - 1) / 2;
		}
		p_array[p_first + p_hole_idx] = p_value;
	}

	inline void pop_heap(int p_first, int p_last, int p_result, T p_value, T *p_array) const {
		p_array[p_result] = p_array[p_first];
		adjust_heap(p_first, 0, p_last - p_first, p_value, p_array);
	}
	inline void pop_heap(int p_first, int p_last, T *p_array) const {
		pop_heap(p_first, p_last - 1, p_last - 1, p_array[p_last - 1], p_array);
	}

	inline void adjust_heap(int p_first, int p_hole_idx, int p_len, T p_value, T *p_array) const {
		int top_index = p_hole_idx;
		int second_child = 2 * p_hole_idx + 2;

		while (second_child < p_len) {
			if (compare(p_array[p_first + second_child], p_array[p_first + (second_child - 1)])) {
				second_child--;
			}

			p_array[p_first + p_hole_idx] = p_array[p_first + second_child];
			p_hole_idx = second_child;
			second_child = 2 * (second_child + 1);
		}

		if (second_child == p_len) {
			p_array[p_first + p_hole_idx] = p_array[p_first + (second_child - 1)];
			p_hole_idx = second_child - 1;
		}
		push_heap(p_first, p_hole_idx, top_index, p_value, p_array);
	}

	inline void sort_heap(int p_first, int p_last, T *p_array) const {
		while (p_last - p_first > 1) {
			pop_heap(p_first, p_last--, p_array);
		}
	}

	inline void make_heap(int p_first, int p_last, T *p_array) const {
		if (p_last - p_first < 2) {
			return;
		}
		int len = p_last - p_first;
		int parent = (len - 2) / 2;

		while (true) {
			adjust_heap(p_first, parent, len, p_array[p_first + parent], p_array);
			if (parent == 0) {
				return;
			}
			parent--;
		}
	}

	inline void partial_sort(int p_first, int p_last, int p_middle, T *p_array) const {
		make_heap(p_first, p_middle, p_array);
		for (int i = p_middle; i < p_last; i++) {
			if (compare(p_array[i], p_array[p_first])) {
				pop_heap(p_first, p_middle, i, p_array[i], p_array);
			}
		}
		sort_heap(p_first, p_middle, p_array);
	}

	inline void partial_select(int p_first, int p_last, int p_middle, T *p_array) const {
		make_heap(p_first, p_middle, p_array);
		for (int i = p_middle; i < p_last; i++) {
			if (compare(p_array[i], p_array[p_first])) {
				pop_heap(p_first, p_middle, i, p_array[i], p_array);
			}
		}
	}

	inline int partitioner(int p_first, int p_last, T p_pivot, T *p_array) const {
		const int unmodified_first = p_first;
		const int unmodified_last = p_last;

		while (true) {
			while (compare(p_array[p_first], p_pivot)) {
				if (Validate) {
					ERR_BAD_COMPARE(p_first == unmodified_last - 1);
				}
				p_first++;
			}
			p_last--;
			while (compare(p_pivot, p_array[p_last])) {
				if (Validate) {
					ERR_BAD_COMPARE(p_last == unmodified_first);
				}
				p_last--;
			}

			if (!(p_first < p_last)) {
				return p_first;
			}

			SWAP(p_array[p_first], p_array[p_last]);
			p_first++;
		}
	}

	inline void introsort(int p_first, int p_last, T *p_array, int p_max_depth) const {
		while (p_last - p_first > INTROSORT_THRESHOLD) {
			if (p_max_depth == 0) {
				partial_sort(p_first, p_last, p_last, p_array);
				return;
			}

			p_max_depth--;

			int cut = partitioner(
					p_first,
					p_last,
					median_of_3(
							p_array[p_first],
							p_array[p_first + (p_last - p_first) / 2],
							p_array[p_last - 1]),
					p_array);

			introsort(cut, p_last, p_array, p_max_depth);
			p_last = cut;
		}
	}

	inline void introselect(int p_first, int p_nth, int p_last, T *p_array, int p_max_depth) const {
		while (p_last - p_first > 3) {
			if (p_max_depth == 0) {
				partial_select(p_first, p_nth + 1, p_last, p_array);
				SWAP(p_first, p_nth);
				return;
			}

			p_max_depth--;

			int cut = partitioner(
					p_first,
					p_last,
					median_of_3(
							p_array[p_first],
							p_array[p_first + (p_last - p_first) / 2],
							p_array[p_last - 1]),
					p_array);

			if (cut <= p_nth) {
				p_first = cut;
			} else {
				p_last = cut;
			}
		}

		insertion_sort(p_first, p_last, p_array);
	}

	inline void unguarded_linear_insert(int p_last, T p_value, T *p_array) const {
		int next = p_last - 1;
		while (compare(p_value, p_array[next])) {
			if (Validate) {
				ERR_BAD_COMPARE(next == 0);
			}
			p_array[p_last] = p_array[next];
			p_last = next;
			next--;
		}
		p_array[p_last] = p_value;
	}

	inline void linear_insert(int p_first, int p_last, T *p_array) const {
		T val = p_array[p_last];
		if (compare(val, p_array[p_first])) {
			for (int i = p_last; i > p_first; i--) {
				p_array[i] = p_array[i - 1];
			}

			p_array[p_first] = val;
		} else {
			unguarded_linear_insert(p_last, val, p_array);
		}
	}

	inline void insertion_sort(int p_first, int p_last, T *p_array) const {
		if (p_first == p_last) {
			return;
		}
		for (int i = p_first + 1; i != p_last; i++) {
			linear_insert(p_first, i, p_array);
		}
	}

	inline void unguarded_insertion_sort(int p_first, int p_last, T *p_array) const {
		for (int i = p_first; i != p_last; i++) {
			unguarded_linear_insert(i, p_array[i], p_array);
		}
	}

	inline void final_insertion_sort(int p_first, int p_last, T *p_array) const {
		if (p_last - p_first > INTROSORT_THRESHOLD) {
			insertion_sort(p_first, p_first + INTROSORT_THRESHOLD, p_array);
			unguarded_insertion_sort(p_first + INTROSORT_THRESHOLD, p_last, p_array);
		} else {
			insertion_sort(p_first, p_last, p_array);
		}
	}

	inline void sort_range(int p_first, int p_last, T *p_array) const {
		if (p_first != p_last) {
			introsort(p_first, p_last, p_array, bitlog(p_last - p_first) * 2);
			final_insertion_sort(p_first, p_last, p_array);
		}
	}

	inline void sort(T *p_array, int p_len) const {
		sort_range(0, p_len, p_array);
	}

	inline void nth_element(int p_first, int p_last, int p_nth, T *p_array) const {
		if (p_first == p_last || p_nth == p_last) {
			return;
		}
		introselect(p_first, p_nth, p_last, p_array, bitlog(p_last - p_first) * 2);
	}
};

#undef ERR_BAD_COMPARE

#line 0

#line 1 "sfwl/core/rb_map.h"

/*************************************************************************/
/*  rb_map.h                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

// based on the very nice implementation of rb-trees by:
// https://web.archive.org/web/20120507164830/http://web.mit.edu/~emin/www/source_code/red_black_tree/index.html

template <class K, class V, class C = Comparator<K>, class A = DefaultAllocator>
class RBMap {
	enum Color {
		RED,
		BLACK
	};
	struct _Data;

public:
	class Element {
	private:
		friend class RBMap<K, V, C, A>;
		int color;
		Element *right;
		Element *left;
		Element *parent;
		Element *_next;
		Element *_prev;
		K _key;
		V _value;
		//_Data *data;

	public:
		const Element *next() const {
			return _next;
		}
		Element *next() {
			return _next;
		}
		const Element *prev() const {
			return _prev;
		}
		Element *prev() {
			return _prev;
		}
		const K &key() const {
			return _key;
		};
		V &value() {
			return _value;
		};
		const V &value() const {
			return _value;
		};
		V &get() {
			return _value;
		};
		const V &get() const {
			return _value;
		};
		Element() {
			color = RED;
			right = nullptr;
			left = nullptr;
			parent = nullptr;
			_next = nullptr;
			_prev = nullptr;
		};
	};

private:
	struct _Data {
		Element *_root;
		Element *_nil;
		int size_cache;

		_FORCE_INLINE_ _Data() {
#ifdef GLOBALNIL_DISABLED
			_nil = memnew_allocator(Element, A);
			_nil->parent = _nil->left = _nil->right = _nil;
			_nil->color = BLACK;
#else
			_nil = (Element *)&_GlobalNilClass::_nil;
#endif
			_root = nullptr;
			size_cache = 0;
		}

		void _create_root() {
			_root = memnew_allocator(Element, A);
			_root->parent = _root->left = _root->right = _nil;
			_root->color = BLACK;
		}

		void _free_root() {
			if (_root) {
				memdelete_allocator<Element, A>(_root);
				_root = nullptr;
			}
		}

		~_Data() {
			_free_root();

#ifdef GLOBALNIL_DISABLED
			memdelete_allocator<Element, A>(_nil);
#endif
		}
	};

	_Data _data;

	inline void _set_color(Element *p_node, int p_color) {
		ERR_FAIL_COND(p_node == _data._nil && p_color == RED);
		p_node->color = p_color;
	}

	inline void _rotate_left(Element *p_node) {
		Element *r = p_node->right;
		p_node->right = r->left;
		if (r->left != _data._nil) {
			r->left->parent = p_node;
		}
		r->parent = p_node->parent;
		if (p_node == p_node->parent->left) {
			p_node->parent->left = r;
		} else {
			p_node->parent->right = r;
		}

		r->left = p_node;
		p_node->parent = r;
	}

	inline void _rotate_right(Element *p_node) {
		Element *l = p_node->left;
		p_node->left = l->right;
		if (l->right != _data._nil) {
			l->right->parent = p_node;
		}
		l->parent = p_node->parent;
		if (p_node == p_node->parent->right) {
			p_node->parent->right = l;
		} else {
			p_node->parent->left = l;
		}

		l->right = p_node;
		p_node->parent = l;
	}

	inline Element *_successor(Element *p_node) const {
		Element *node = p_node;

		if (node->right != _data._nil) {
			node = node->right;
			while (node->left != _data._nil) { /* returns the minimum of the right subtree of node */
				node = node->left;
			}
			return node;
		} else {
			while (node == node->parent->right) {
				node = node->parent;
			}

			if (node->parent == _data._root) {
				return nullptr; // No successor, as p_node = last node
			}
			return node->parent;
		}
	}

	inline Element *_predecessor(Element *p_node) const {
		Element *node = p_node;

		if (node->left != _data._nil) {
			node = node->left;
			while (node->right != _data._nil) { /* returns the minimum of the left subtree of node */
				node = node->right;
			}
			return node;
		} else {
			while (node == node->parent->left) {
				node = node->parent;
			}

			if (node == _data._root) {
				return nullptr; // No predecessor, as p_node = first node
			}
			return node->parent;
		}
	}

	Element *_find(const K &p_key) const {
		Element *node = _data._root->left;
		C less;

		while (node != _data._nil) {
			if (less(p_key, node->_key)) {
				node = node->left;
			} else if (less(node->_key, p_key)) {
				node = node->right;
			} else {
				return node; // found
			}
		}

		return nullptr;
	}

	Element *_find_closest(const K &p_key) const {
		Element *node = _data._root->left;
		Element *prev = nullptr;
		C less;

		while (node != _data._nil) {
			prev = node;

			if (less(p_key, node->_key)) {
				node = node->left;
			} else if (less(node->_key, p_key)) {
				node = node->right;
			} else {
				return node; // found
			}
		}

		if (prev == nullptr) {
			return nullptr; // tree empty
		}

		if (less(p_key, prev->_key)) {
			prev = prev->_prev;
		}

		return prev;
	}

	void _insert_rb_fix(Element *p_new_node) {
		Element *node = p_new_node;
		Element *nparent = node->parent;
		Element *ngrand_parent;

		while (nparent->color == RED) {
			ngrand_parent = nparent->parent;

			if (nparent == ngrand_parent->left) {
				if (ngrand_parent->right->color == RED) {
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent->right, BLACK);
					_set_color(ngrand_parent, RED);
					node = ngrand_parent;
					nparent = node->parent;
				} else {
					if (node == nparent->right) {
						_rotate_left(nparent);
						node = nparent;
						nparent = node->parent;
					}
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent, RED);
					_rotate_right(ngrand_parent);
				}
			} else {
				if (ngrand_parent->left->color == RED) {
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent->left, BLACK);
					_set_color(ngrand_parent, RED);
					node = ngrand_parent;
					nparent = node->parent;
				} else {
					if (node == nparent->left) {
						_rotate_right(nparent);
						node = nparent;
						nparent = node->parent;
					}
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent, RED);
					_rotate_left(ngrand_parent);
				}
			}
		}

		_set_color(_data._root->left, BLACK);
	}

	Element *_insert(const K &p_key, const V &p_value) {
		Element *new_parent = _data._root;
		Element *node = _data._root->left;
		C less;

		while (node != _data._nil) {
			new_parent = node;

			if (less(p_key, node->_key)) {
				node = node->left;
			} else if (less(node->_key, p_key)) {
				node = node->right;
			} else {
				node->_value = p_value;
				return node; // Return existing node with new value
			}
		}

		Element *new_node = memnew_allocator(Element, A);
		new_node->parent = new_parent;
		new_node->right = _data._nil;
		new_node->left = _data._nil;
		new_node->_key = p_key;
		new_node->_value = p_value;
		//new_node->data=_data;

		if (new_parent == _data._root || less(p_key, new_parent->_key)) {
			new_parent->left = new_node;
		} else {
			new_parent->right = new_node;
		}

		new_node->_next = _successor(new_node);
		new_node->_prev = _predecessor(new_node);
		if (new_node->_next) {
			new_node->_next->_prev = new_node;
		}
		if (new_node->_prev) {
			new_node->_prev->_next = new_node;
		}

		_data.size_cache++;
		_insert_rb_fix(new_node);
		return new_node;
	}

	void _erase_fix_rb(Element *p_node) {
		Element *root = _data._root->left;
		Element *node = _data._nil;
		Element *sibling = p_node;
		Element *parent = sibling->parent;

		while (node != root) { // If red node found, will exit at a break
			if (sibling->color == RED) {
				_set_color(sibling, BLACK);
				_set_color(parent, RED);
				if (sibling == parent->right) {
					sibling = sibling->left;
					_rotate_left(parent);
				} else {
					sibling = sibling->right;
					_rotate_right(parent);
				}
			}
			if ((sibling->left->color == BLACK) && (sibling->right->color == BLACK)) {
				_set_color(sibling, RED);
				if (parent->color == RED) {
					_set_color(parent, BLACK);
					break;
				} else { // loop: haven't found any red nodes yet
					node = parent;
					parent = node->parent;
					sibling = (node == parent->left) ? parent->right : parent->left;
				}
			} else {
				if (sibling == parent->right) {
					if (sibling->right->color == BLACK) {
						_set_color(sibling->left, BLACK);
						_set_color(sibling, RED);
						_rotate_right(sibling);
						sibling = sibling->parent;
					}
					_set_color(sibling, parent->color);
					_set_color(parent, BLACK);
					_set_color(sibling->right, BLACK);
					_rotate_left(parent);
					break;
				} else {
					if (sibling->left->color == BLACK) {
						_set_color(sibling->right, BLACK);
						_set_color(sibling, RED);
						_rotate_left(sibling);
						sibling = sibling->parent;
					}

					_set_color(sibling, parent->color);
					_set_color(parent, BLACK);
					_set_color(sibling->left, BLACK);
					_rotate_right(parent);
					break;
				}
			}
		}

		ERR_FAIL_COND(_data._nil->color != BLACK);
	}

	void _erase(Element *p_node) {
		Element *rp = ((p_node->left == _data._nil) || (p_node->right == _data._nil)) ? p_node : p_node->_next;
		Element *node = (rp->left == _data._nil) ? rp->right : rp->left;

		Element *sibling;
		if (rp == rp->parent->left) {
			rp->parent->left = node;
			sibling = rp->parent->right;
		} else {
			rp->parent->right = node;
			sibling = rp->parent->left;
		}

		if (node->color == RED) {
			node->parent = rp->parent;
			_set_color(node, BLACK);
		} else if (rp->color == BLACK && rp->parent != _data._root) {
			_erase_fix_rb(sibling);
		}

		if (rp != p_node) {
			ERR_FAIL_COND(rp == _data._nil);

			rp->left = p_node->left;
			rp->right = p_node->right;
			rp->parent = p_node->parent;
			rp->color = p_node->color;
			if (p_node->left != _data._nil) {
				p_node->left->parent = rp;
			}
			if (p_node->right != _data._nil) {
				p_node->right->parent = rp;
			}

			if (p_node == p_node->parent->left) {
				p_node->parent->left = rp;
			} else {
				p_node->parent->right = rp;
			}
		}

		if (p_node->_next) {
			p_node->_next->_prev = p_node->_prev;
		}
		if (p_node->_prev) {
			p_node->_prev->_next = p_node->_next;
		}

		memdelete_allocator<Element, A>(p_node);
		_data.size_cache--;
		ERR_FAIL_COND(_data._nil->color == RED);
	}

	void _calculate_depth(Element *p_element, int &max_d, int d) const {
		if (p_element == _data._nil) {
			return;
		}

		_calculate_depth(p_element->left, max_d, d + 1);
		_calculate_depth(p_element->right, max_d, d + 1);

		if (d > max_d) {
			max_d = d;
		}
	}

	void _cleanup_tree(Element *p_element) {
		if (p_element == _data._nil) {
			return;
		}

		_cleanup_tree(p_element->left);
		_cleanup_tree(p_element->right);
		memdelete_allocator<Element, A>(p_element);
	}

	void _copy_from(const RBMap &p_map) {
		clear();
		// not the fastest way, but safeset to write.
		for (Element *I = p_map.front(); I; I = I->next()) {
			insert(I->key(), I->value());
		}
	}

public:
	const Element *find(const K &p_key) const {
		if (!_data._root) {
			return nullptr;
		}

		const Element *res = _find(p_key);
		return res;
	}

	Element *find(const K &p_key) {
		if (!_data._root) {
			return nullptr;
		}

		Element *res = _find(p_key);
		return res;
	}

	const Element *find_closest(const K &p_key) const {
		if (!_data._root) {
			return NULL;
		}

		const Element *res = _find_closest(p_key);
		return res;
	}

	Element *find_closest(const K &p_key) {
		if (!_data._root) {
			return nullptr;
		}

		Element *res = _find_closest(p_key);
		return res;
	}

	bool has(const K &p_key) const {
		return find(p_key) != nullptr;
	}

	Element *insert(const K &p_key, const V &p_value) {
		if (!_data._root) {
			_data._create_root();
		}
		return _insert(p_key, p_value);
	}

	void erase(Element *p_element) {
		if (!_data._root || !p_element) {
			return;
		}

		_erase(p_element);
		if (_data.size_cache == 0 && _data._root) {
			_data._free_root();
		}
	}

	bool erase(const K &p_key) {
		if (!_data._root) {
			return false;
		}

		Element *e = find(p_key);
		if (!e) {
			return false;
		}

		_erase(e);
		if (_data.size_cache == 0 && _data._root) {
			_data._free_root();
		}
		return true;
	}

	const V &operator[](const K &p_key) const {
		CRASH_COND(!_data._root);
		const Element *e = find(p_key);
		CRASH_COND(!e);
		return e->_value;
	}

	V &operator[](const K &p_key) {
		if (!_data._root) {
			_data._create_root();
		}

		Element *e = find(p_key);
		if (!e) {
			e = insert(p_key, V());
		}

		return e->_value;
	}

	Element *front() const {
		if (!_data._root) {
			return nullptr;
		}

		Element *e = _data._root->left;
		if (e == _data._nil) {
			return nullptr;
		}

		while (e->left != _data._nil) {
			e = e->left;
		}

		return e;
	}

	Element *back() const {
		if (!_data._root) {
			return nullptr;
		}

		Element *e = _data._root->left;
		if (e == _data._nil) {
			return nullptr;
		}

		while (e->right != _data._nil) {
			e = e->right;
		}

		return e;
	}

	inline bool empty() const { return _data.size_cache == 0; }
	inline int size() const { return _data.size_cache; }

	int calculate_depth() const {
		// used for debug mostly
		if (!_data._root) {
			return 0;
		}

		int max_d = 0;
		_calculate_depth(_data._root->left, max_d, 0);
		return max_d;
	}

	void clear() {
		if (!_data._root) {
			return;
		}

		_cleanup_tree(_data._root->left);
		_data._root->left = _data._nil;
		_data.size_cache = 0;
		_data._free_root();
	}

	void operator=(const RBMap &p_map) {
		_copy_from(p_map);
	}

	RBMap(const RBMap &p_map) {
		_copy_from(p_map);
	}

	_FORCE_INLINE_ RBMap() {
	}

	~RBMap() {
		clear();
	}
};

#line 0

#line 1 "sfwl/core/rb_set.h"

/*************************************************************************/
/*  rb_set.h                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

// based on the very nice implementation of rb-trees by:
// https://web.archive.org/web/20120507164830/http://web.mit.edu/~emin/www/source_code/red_black_tree/index.html

template <class T, class C = Comparator<T>, class A = DefaultAllocator>
class RBSet {
	enum Color {
		RED,
		BLACK
	};
	struct _Data;

public:
	class Element {
	private:
		friend class RBSet<T, C, A>;
		int color;
		Element *right;
		Element *left;
		Element *parent;
		Element *_next;
		Element *_prev;
		T value;
		//_Data *data;

	public:
		const Element *next() const {
			return _next;
		}
		Element *next() {
			return _next;
		}
		const Element *prev() const {
			return _prev;
		}
		Element *prev() {
			return _prev;
		}
		const T &get() const {
			return value;
		};
		Element() {
			color = RED;
			right = nullptr;
			left = nullptr;
			parent = nullptr;
			_next = nullptr;
			_prev = nullptr;
		};
	};

private:
	struct _Data {
		Element *_root;
		Element *_nil;
		int size_cache;

		_FORCE_INLINE_ _Data() {
#ifdef GLOBALNIL_DISABLED
			_nil = memnew_allocator(Element, A);
			_nil->parent = _nil->left = _nil->right = _nil;
			_nil->color = BLACK;
#else
			_nil = (Element *)&_GlobalNilClass::_nil;
#endif
			_root = nullptr;
			size_cache = 0;
		}

		void _create_root() {
			_root = memnew_allocator(Element, A);
			_root->parent = _root->left = _root->right = _nil;
			_root->color = BLACK;
		}

		void _free_root() {
			if (_root) {
				memdelete_allocator<Element, A>(_root);
				_root = nullptr;
			}
		}

		~_Data() {
			_free_root();

#ifdef GLOBALNIL_DISABLED
			memdelete_allocator<Element, A>(_nil);
#endif
		}
	};

	_Data _data;

	inline void _set_color(Element *p_node, int p_color) {
		ERR_FAIL_COND(p_node == _data._nil && p_color == RED);
		p_node->color = p_color;
	}

	inline void _rotate_left(Element *p_node) {
		Element *r = p_node->right;
		p_node->right = r->left;
		if (r->left != _data._nil) {
			r->left->parent = p_node;
		}
		r->parent = p_node->parent;
		if (p_node == p_node->parent->left) {
			p_node->parent->left = r;
		} else {
			p_node->parent->right = r;
		}

		r->left = p_node;
		p_node->parent = r;
	}

	inline void _rotate_right(Element *p_node) {
		Element *l = p_node->left;
		p_node->left = l->right;
		if (l->right != _data._nil) {
			l->right->parent = p_node;
		}
		l->parent = p_node->parent;
		if (p_node == p_node->parent->right) {
			p_node->parent->right = l;
		} else {
			p_node->parent->left = l;
		}

		l->right = p_node;
		p_node->parent = l;
	}

	inline Element *_successor(Element *p_node) const {
		Element *node = p_node;

		if (node->right != _data._nil) {
			node = node->right;
			while (node->left != _data._nil) { /* returns the minimum of the right subtree of node */
				node = node->left;
			}
			return node;
		} else {
			while (node == node->parent->right) {
				node = node->parent;
			}

			if (node->parent == _data._root) {
				return nullptr; // No successor, as p_node = last node
			}
			return node->parent;
		}
	}

	inline Element *_predecessor(Element *p_node) const {
		Element *node = p_node;

		if (node->left != _data._nil) {
			node = node->left;
			while (node->right != _data._nil) { /* returns the minimum of the left subtree of node */
				node = node->right;
			}
			return node;
		} else {
			while (node == node->parent->left) {
				node = node->parent;
			}

			if (node == _data._root) {
				return nullptr; // No predecessor, as p_node = first node.
			}
			return node->parent;
		}
	}

	Element *_find(const T &p_value) const {
		Element *node = _data._root->left;
		C less;

		while (node != _data._nil) {
			if (less(p_value, node->value)) {
				node = node->left;
			} else if (less(node->value, p_value)) {
				node = node->right;
			} else {
				return node; // found
			}
		}

		return nullptr;
	}

	Element *_lower_bound(const T &p_value) const {
		Element *node = _data._root->left;
		Element *prev = nullptr;
		C less;

		while (node != _data._nil) {
			prev = node;

			if (less(p_value, node->value)) {
				node = node->left;
			} else if (less(node->value, p_value)) {
				node = node->right;
			} else {
				return node; // found
			}
		}

		if (prev == nullptr) {
			return nullptr; // tree empty
		}

		if (less(prev->value, p_value)) {
			prev = prev->_next;
		}

		return prev;
	}

	void _insert_rb_fix(Element *p_new_node) {
		Element *node = p_new_node;
		Element *nparent = node->parent;
		Element *ngrand_parent;

		while (nparent->color == RED) {
			ngrand_parent = nparent->parent;

			if (nparent == ngrand_parent->left) {
				if (ngrand_parent->right->color == RED) {
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent->right, BLACK);
					_set_color(ngrand_parent, RED);
					node = ngrand_parent;
					nparent = node->parent;
				} else {
					if (node == nparent->right) {
						_rotate_left(nparent);
						node = nparent;
						nparent = node->parent;
					}
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent, RED);
					_rotate_right(ngrand_parent);
				}
			} else {
				if (ngrand_parent->left->color == RED) {
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent->left, BLACK);
					_set_color(ngrand_parent, RED);
					node = ngrand_parent;
					nparent = node->parent;
				} else {
					if (node == nparent->left) {
						_rotate_right(nparent);
						node = nparent;
						nparent = node->parent;
					}
					_set_color(nparent, BLACK);
					_set_color(ngrand_parent, RED);
					_rotate_left(ngrand_parent);
				}
			}
		}

		_set_color(_data._root->left, BLACK);
	}

	Element *_insert(const T &p_value) {
		Element *new_parent = _data._root;
		Element *node = _data._root->left;
		C less;

		while (node != _data._nil) {
			new_parent = node;

			if (less(p_value, node->value)) {
				node = node->left;
			} else if (less(node->value, p_value)) {
				node = node->right;
			} else {
				return node; // Return existing node
			}
		}

		Element *new_node = memnew_allocator(Element, A);
		new_node->parent = new_parent;
		new_node->right = _data._nil;
		new_node->left = _data._nil;
		new_node->value = p_value;
		//new_node->data=_data;

		if (new_parent == _data._root || less(p_value, new_parent->value)) {
			new_parent->left = new_node;
		} else {
			new_parent->right = new_node;
		}

		new_node->_next = _successor(new_node);
		new_node->_prev = _predecessor(new_node);
		if (new_node->_next) {
			new_node->_next->_prev = new_node;
		}
		if (new_node->_prev) {
			new_node->_prev->_next = new_node;
		}

		_data.size_cache++;
		_insert_rb_fix(new_node);
		return new_node;
	}

	void _erase_fix_rb(Element *p_node) {
		Element *root = _data._root->left;
		Element *node = _data._nil;
		Element *sibling = p_node;
		Element *parent = sibling->parent;

		while (node != root) { // If red node found, will exit at a break
			if (sibling->color == RED) {
				_set_color(sibling, BLACK);
				_set_color(parent, RED);
				if (sibling == parent->right) {
					sibling = sibling->left;
					_rotate_left(parent);
				} else {
					sibling = sibling->right;
					_rotate_right(parent);
				}
			}
			if ((sibling->left->color == BLACK) && (sibling->right->color == BLACK)) {
				_set_color(sibling, RED);
				if (parent->color == RED) {
					_set_color(parent, BLACK);
					break;
				} else { // loop: haven't found any red nodes yet
					node = parent;
					parent = node->parent;
					sibling = (node == parent->left) ? parent->right : parent->left;
				}
			} else {
				if (sibling == parent->right) {
					if (sibling->right->color == BLACK) {
						_set_color(sibling->left, BLACK);
						_set_color(sibling, RED);
						_rotate_right(sibling);
						sibling = sibling->parent;
					}
					_set_color(sibling, parent->color);
					_set_color(parent, BLACK);
					_set_color(sibling->right, BLACK);
					_rotate_left(parent);
					break;
				} else {
					if (sibling->left->color == BLACK) {
						_set_color(sibling->right, BLACK);
						_set_color(sibling, RED);
						_rotate_left(sibling);
						sibling = sibling->parent;
					}

					_set_color(sibling, parent->color);
					_set_color(parent, BLACK);
					_set_color(sibling->left, BLACK);
					_rotate_right(parent);
					break;
				}
			}
		}

		ERR_FAIL_COND(_data._nil->color != BLACK);
	}

	void _erase(Element *p_node) {
		Element *rp = ((p_node->left == _data._nil) || (p_node->right == _data._nil)) ? p_node : p_node->_next;
		Element *node = (rp->left == _data._nil) ? rp->right : rp->left;

		Element *sibling;
		if (rp == rp->parent->left) {
			rp->parent->left = node;
			sibling = rp->parent->right;
		} else {
			rp->parent->right = node;
			sibling = rp->parent->left;
		}

		if (node->color == RED) {
			node->parent = rp->parent;
			_set_color(node, BLACK);
		} else if (rp->color == BLACK && rp->parent != _data._root) {
			_erase_fix_rb(sibling);
		}

		if (rp != p_node) {
			ERR_FAIL_COND(rp == _data._nil);

			rp->left = p_node->left;
			rp->right = p_node->right;
			rp->parent = p_node->parent;
			rp->color = p_node->color;
			if (p_node->left != _data._nil) {
				p_node->left->parent = rp;
			}
			if (p_node->right != _data._nil) {
				p_node->right->parent = rp;
			}

			if (p_node == p_node->parent->left) {
				p_node->parent->left = rp;
			} else {
				p_node->parent->right = rp;
			}
		}

		if (p_node->_next) {
			p_node->_next->_prev = p_node->_prev;
		}
		if (p_node->_prev) {
			p_node->_prev->_next = p_node->_next;
		}

		memdelete_allocator<Element, A>(p_node);
		_data.size_cache--;
		ERR_FAIL_COND(_data._nil->color == RED);
	}

	void _calculate_depth(Element *p_element, int &max_d, int d) const {
		if (p_element == _data._nil) {
			return;
		}

		_calculate_depth(p_element->left, max_d, d + 1);
		_calculate_depth(p_element->right, max_d, d + 1);

		if (d > max_d) {
			max_d = d;
		}
	}

	void _cleanup_tree(Element *p_element) {
		if (p_element == _data._nil) {
			return;
		}

		_cleanup_tree(p_element->left);
		_cleanup_tree(p_element->right);
		memdelete_allocator<Element, A>(p_element);
	}

	void _copy_from(const RBSet &p_set) {
		clear();
		// not the fastest way, but safeset to write.
		for (Element *I = p_set.front(); I; I = I->next()) {
			insert(I->get());
		}
	}

public:
	const Element *find(const T &p_value) const {
		if (!_data._root) {
			return nullptr;
		}

		const Element *res = _find(p_value);
		return res;
	}

	Element *find(const T &p_value) {
		if (!_data._root) {
			return nullptr;
		}

		Element *res = _find(p_value);
		return res;
	}

	Element *lower_bound(const T &p_value) const {
		if (!_data._root) {
			return nullptr;
		}
		return _lower_bound(p_value);
	}

	bool has(const T &p_value) const {
		return find(p_value) != nullptr;
	}

	Element *insert(const T &p_value) {
		if (!_data._root) {
			_data._create_root();
		}
		return _insert(p_value);
	}

	void erase(Element *p_element) {
		if (!_data._root || !p_element) {
			return;
		}

		_erase(p_element);
		if (_data.size_cache == 0 && _data._root) {
			_data._free_root();
		}
	}

	bool erase(const T &p_value) {
		if (!_data._root) {
			return false;
		}

		Element *e = find(p_value);
		if (!e) {
			return false;
		}

		_erase(e);
		if (_data.size_cache == 0 && _data._root) {
			_data._free_root();
		}
		return true;
	}

	Element *front() const {
		if (!_data._root) {
			return nullptr;
		}

		Element *e = _data._root->left;
		if (e == _data._nil) {
			return nullptr;
		}

		while (e->left != _data._nil) {
			e = e->left;
		}

		return e;
	}

	Element *back() const {
		if (!_data._root) {
			return nullptr;
		}

		Element *e = _data._root->left;
		if (e == _data._nil) {
			return nullptr;
		}

		while (e->right != _data._nil) {
			e = e->right;
		}

		return e;
	}

	inline bool empty() const { return _data.size_cache == 0; }
	inline int size() const { return _data.size_cache; }

	int calculate_depth() const {
		// used for debug mostly
		if (!_data._root) {
			return 0;
		}

		int max_d = 0;
		_calculate_depth(_data._root->left, max_d, 0);
		return max_d;
	}

	void clear() {
		if (!_data._root) {
			return;
		}

		_cleanup_tree(_data._root->left);
		_data._root->left = _data._nil;
		_data.size_cache = 0;
		_data._free_root();
	}

	void operator=(const RBSet &p_set) {
		_copy_from(p_set);
	}

	RBSet(const RBSet &p_set) {
		_copy_from(p_set);
	}

	_FORCE_INLINE_ RBSet() {
	}

	~RBSet() {
		clear();
	}
};

#line 0

#line 1 "sfwl/core/vmap.h"

/*************************************************************************/
/*  vmap.h                                                               */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

template <class T, class V>
class VMap {
public:
	struct Pair {
		T key;
		V value;

		_FORCE_INLINE_ Pair() {}

		_FORCE_INLINE_ Pair(const T &p_key, const V &p_value) {
			key = p_key;
			value = p_value;
		}
	};

private:
	CowData<Pair> _cowdata;

	_FORCE_INLINE_ int _find(const T &p_val, bool &r_exact) const {
		r_exact = false;
		if (_cowdata.empty()) {
			return 0;
		}

		int low = 0;
		int high = _cowdata.size() - 1;
		const Pair *a = _cowdata.ptr();
		int middle = 0;

#ifdef DEBUG_ENABLED
		if (low > high)
			ERR_PRINT("low > high, this may be a bug");
#endif
		while (low <= high) {
			middle = (low + high) / 2;

			if (p_val < a[middle].key) {
				high = middle - 1; //search low end of array
			} else if (a[middle].key < p_val) {
				low = middle + 1; //search high end of array
			} else {
				r_exact = true;
				return middle;
			}
		}

		//return the position where this would be inserted
		if (a[middle].key < p_val) {
			middle++;
		}
		return middle;
	}

	_FORCE_INLINE_ int _find_exact(const T &p_val) const {
		if (_cowdata.empty()) {
			return -1;
		}

		int low = 0;
		int high = _cowdata.size() - 1;
		int middle;
		const Pair *a = _cowdata.ptr();

		while (low <= high) {
			middle = (low + high) / 2;

			if (p_val < a[middle].key) {
				high = middle - 1; //search low end of array
			} else if (a[middle].key < p_val) {
				low = middle + 1; //search high end of array
			} else {
				return middle;
			}
		}

		return -1;
	}

public:
	int insert(const T &p_key, const V &p_val) {
		bool exact;
		int pos = _find(p_key, exact);
		if (exact) {
			_cowdata.get_m(pos).value = p_val;
			return pos;
		}
		_cowdata.insert(pos, Pair(p_key, p_val));
		return pos;
	}

	bool has(const T &p_val) const {
		return _find_exact(p_val) != -1;
	}

	void erase(const T &p_val) {
		int pos = _find_exact(p_val);
		if (pos < 0) {
			return;
		}
		_cowdata.remove(pos);
	}

	int find(const T &p_val) const {
		return _find_exact(p_val);
	}

	int find_nearest(const T &p_val) const {
		if (_cowdata.empty()) {
			return -1;
		}
		bool exact;
		return _find(p_val, exact);
	}

	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	_FORCE_INLINE_ bool empty() const { return _cowdata.empty(); }

	const Pair *get_array() const {
		return _cowdata.ptr();
	}

	Pair *get_array() {
		return _cowdata.ptrw();
	}

	const V &getv(int p_index) const {
		return _cowdata.get(p_index).value;
	}

	V &getv(int p_index) {
		return _cowdata.get_m(p_index).value;
	}

	const T &getk(int p_index) const {
		return _cowdata.get(p_index).key;
	}

	T &getk(int p_index) {
		return _cowdata.get_m(p_index).key;
	}

	inline const V &operator[](const T &p_key) const {
		int pos = _find_exact(p_key);

		CRASH_COND(pos < 0);

		return _cowdata.get(pos).value;
	}

	inline V &operator[](const T &p_key) {
		int pos = _find_exact(p_key);
		if (pos < 0) {
			pos = insert(p_key, V());
		}

		return _cowdata.get_m(pos).value;
	}

	_FORCE_INLINE_ VMap(){};
	_FORCE_INLINE_ VMap(const VMap &p_from) { _cowdata._ref(p_from._cowdata); }
	inline VMap &operator=(const VMap &p_from) {
		_cowdata._ref(p_from._cowdata);
		return *this;
	}
};

#line 0

#line 1 "sfwl/core/vector.h"

/*************************************************************************/
/*  vector.h                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * @class Vector
 * @author Juan Linietsky
 * Vector container. Regular Vector Container. Use with care and for smaller arrays when possible. Use PoolVector for large arrays.
 */

template <class T>
class VectorWriteProxy {
public:
	_FORCE_INLINE_ T &operator[](int p_index) {
		CRASH_BAD_INDEX(p_index, ((Vector<T> *)(this))->_cowdata.size());

		return ((Vector<T> *)(this))->_cowdata.ptrw()[p_index];
	}
};

template <class T>
class Vector {
	friend class VectorWriteProxy<T>;

public:
	VectorWriteProxy<T> write;

private:
	CowData<T> _cowdata;

public:
	bool push_back(T p_elem);

	void remove(int p_index) { _cowdata.remove(p_index); }
	_FORCE_INLINE_ bool erase(const T &p_val) {
		int idx = find(p_val);
		if (idx >= 0) {
			remove(idx);
			return true;
		}
		return false;
	};
	void invert();

	_FORCE_INLINE_ T *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const T *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ bool empty() const { return _cowdata.empty(); }

	_FORCE_INLINE_ T get(int p_index) { return _cowdata.get(p_index); }
	_FORCE_INLINE_ const T &get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const T &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	Error resize(int p_size) { return _cowdata.resize(p_size); }
	_FORCE_INLINE_ const T &operator[](int p_index) const { return _cowdata.get(p_index); }
	Error insert(int p_pos, T p_val) { return _cowdata.insert(p_pos, p_val); }
	int find(const T &p_val, int p_from = 0) const { return _cowdata.find(p_val, p_from); }
	_FORCE_INLINE_ void fill(const T &p_val) { _cowdata.fill(p_val); }

	void append_array(Vector<T> p_other);

	template <class C>
	void sort_custom() {
		int len = _cowdata.size();
		if (len == 0) {
			return;
		}

		T *data = ptrw();
		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<T>>();
	}

	void ordered_insert(const T &p_val) {
		int i;
		for (i = 0; i < _cowdata.size(); i++) {
			if (p_val < operator[](i)) {
				break;
			};
		};
		insert(i, p_val);
	}

	_FORCE_INLINE_ Vector() {}
	_FORCE_INLINE_ Vector(const Vector &p_from) { _cowdata._ref(p_from._cowdata); }
	inline Vector &operator=(const Vector &p_from) {
		_cowdata._ref(p_from._cowdata);
		return *this;
	}

	Vector<uint8_t> to_byte_array() const {
		Vector<uint8_t> ret;
		ret.resize(size() * sizeof(T));
		memcpy(ret.ptrw(), ptr(), sizeof(T) * size());
		return ret;
	}

	Vector<T> slice(int p_begin, int p_end = INT32_MAX) const {
		Vector<T> result;

		const int s = size();

		int begin = CLAMP(p_begin, -s, s);
		if (begin < 0) {
			begin += s;
		}
		int end = CLAMP(p_end, -s, s);
		if (end < 0) {
			end += s;
		}

		ERR_FAIL_COND_V(begin > end, result);

		int result_size = end - begin;
		result.resize(result_size);

		const T *const r = ptr();
		T *const w = result.ptrw();
		for (int i = 0; i < result_size; ++i) {
			w[i] = r[begin + i];
		}

		return result;
	}

	_FORCE_INLINE_ ~Vector() {}
};

template <class T>
void Vector<T>::invert() {
	for (int i = 0; i < size() / 2; i++) {
		T *p = ptrw();
		SWAP(p[i], p[size() - i - 1]);
	}
}

template <class T>
void Vector<T>::append_array(Vector<T> p_other) {
	const int ds = p_other.size();
	if (ds == 0) {
		return;
	}
	const int bs = size();
	resize(bs + ds);
	for (int i = 0; i < ds; ++i) {
		ptrw()[bs + i] = p_other[i];
	}
}

template <class T>
bool Vector<T>::push_back(T p_elem) {
	Error err = resize(size() + 1);
	ERR_FAIL_COND_V(err, true);
	set(size() - 1, p_elem);

	return false;
}

#line 0

#line 1 "sfwl/core/vset.h"

/*************************************************************************/
/*  vset.h                                                               */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

template <class T>
class VSet {
	Vector<T> _data;

	_FORCE_INLINE_ int _find(const T &p_val, bool &r_exact) const {
		r_exact = false;
		if (_data.empty()) {
			return 0;
		}

		int low = 0;
		int high = _data.size() - 1;
		const T *a = &_data[0];
		int middle = 0;

#ifdef DEBUG_ENABLED
		if (low > high)
			ERR_PRINT("low > high, this may be a bug");
#endif

		while (low <= high) {
			middle = (low + high) / 2;

			if (p_val < a[middle]) {
				high = middle - 1; //search low end of array
			} else if (a[middle] < p_val) {
				low = middle + 1; //search high end of array
			} else {
				r_exact = true;
				return middle;
			}
		}

		//return the position where this would be inserted
		if (a[middle] < p_val) {
			middle++;
		}
		return middle;
	}

	_FORCE_INLINE_ int _find_exact(const T &p_val) const {
		if (_data.empty()) {
			return -1;
		}

		int low = 0;
		int high = _data.size() - 1;
		int middle;
		const T *a = &_data[0];

		while (low <= high) {
			middle = (low + high) / 2;

			if (p_val < a[middle]) {
				high = middle - 1; //search low end of array
			} else if (a[middle] < p_val) {
				low = middle + 1; //search high end of array
			} else {
				return middle;
			}
		}

		return -1;
	}

public:
	void insert(const T &p_val) {
		bool exact;
		int pos = _find(p_val, exact);
		if (exact) {
			return;
		}
		_data.insert(pos, p_val);
	}

	bool has(const T &p_val) const {
		return _find_exact(p_val) != -1;
	}

	void erase(const T &p_val) {
		int pos = _find_exact(p_val);
		if (pos < 0) {
			return;
		}
		_data.remove(pos);
	}

	int find(const T &p_val) const {
		return _find_exact(p_val);
	}

	_FORCE_INLINE_ bool empty() const { return _data.empty(); }

	_FORCE_INLINE_ int size() const { return _data.size(); }

	inline T &operator[](int p_index) {
		return _data.write[p_index];
	}

	inline const T &operator[](int p_index) const {
		return _data[p_index];
	}
};

#line 0

#line 1 "sfwl/core/list.h"

/*************************************************************************/
/*  list.h                                                               */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * Generic Templatized Linked List Implementation.
 * The implementation differs from the STL one because
 * a compatible preallocated linked list can be written
 * using the same API, or features such as erasing an element
 * from the iterator.
 */

template <class T, class A = DefaultAllocator>
class List {
	struct _Data;

public:
	class Element {
	private:
		friend class List<T, A>;

		T value;
		Element *next_ptr;
		Element *prev_ptr;
		_Data *data;

	public:
		/**
		 * Get NEXT Element iterator, for constant lists.
		 */
		_FORCE_INLINE_ const Element *next() const {
			return next_ptr;
		};
		/**
		 * Get NEXT Element iterator,
		 */
		_FORCE_INLINE_ Element *next() {
			return next_ptr;
		};

		/**
		 * Get PREV Element iterator, for constant lists.
		 */
		_FORCE_INLINE_ const Element *prev() const {
			return prev_ptr;
		};
		/**
		 * Get PREV Element iterator,
		 */
		_FORCE_INLINE_ Element *prev() {
			return prev_ptr;
		};

		/**
		 * * operator, for using as *iterator, when iterators are defined on stack.
		 */
		_FORCE_INLINE_ const T &operator*() const {
			return value;
		};
		/**
		 * operator->, for using as iterator->, when iterators are defined on stack, for constant lists.
		 */
		_FORCE_INLINE_ const T *operator->() const {
			return &value;
		};
		/**
		 * * operator, for using as *iterator, when iterators are defined on stack,
		 */
		_FORCE_INLINE_ T &operator*() {
			return value;
		};
		/**
		 * operator->, for using as iterator->, when iterators are defined on stack, for constant lists.
		 */
		_FORCE_INLINE_ T *operator->() {
			return &value;
		};

		/**
		 * get the value stored in this element.
		 */
		_FORCE_INLINE_ T &get() {
			return value;
		};
		/**
		 * get the value stored in this element, for constant lists
		 */
		_FORCE_INLINE_ const T &get() const {
			return value;
		};
		/**
		 * set the value stored in this element.
		 */
		_FORCE_INLINE_ void set(const T &p_value) {
			value = (T &)p_value;
		};

		void erase() {
			data->erase(this);
		}

		_FORCE_INLINE_ Element() {
			next_ptr = nullptr;
			prev_ptr = nullptr;
			data = nullptr;
		};
	};

private:
	struct _Data {
		Element *first;
		Element *last;
		int size_cache;

		bool erase(const Element *p_I) {
			ERR_FAIL_COND_V(!p_I, false);
			ERR_FAIL_COND_V(p_I->data != this, false);

			if (first == p_I) {
				first = p_I->next_ptr;
			};

			if (last == p_I) {
				last = p_I->prev_ptr;
			}

			if (p_I->prev_ptr) {
				p_I->prev_ptr->next_ptr = p_I->next_ptr;
			}

			if (p_I->next_ptr) {
				p_I->next_ptr->prev_ptr = p_I->prev_ptr;
			}

			memdelete_allocator<Element, A>(const_cast<Element *>(p_I));
			size_cache--;

			return true;
		}
	};

	_Data *_data;

public:
	/**
	 * return a const iterator to the beginning of the list.
	 */
	_FORCE_INLINE_ const Element *front() const {
		return _data ? _data->first : nullptr;
	};

	/**
	 * return an iterator to the beginning of the list.
	 */
	_FORCE_INLINE_ Element *front() {
		return _data ? _data->first : nullptr;
	};

	/**
	 * return a const iterator to the last member of the list.
	 */
	_FORCE_INLINE_ const Element *back() const {
		return _data ? _data->last : nullptr;
	};

	/**
	 * return an iterator to the last member of the list.
	 */
	_FORCE_INLINE_ Element *back() {
		return _data ? _data->last : nullptr;
	};

	/**
	 * store a new element at the end of the list
	 */
	Element *push_back(const T &value) {
		if (!_data) {
			_data = memnew_allocator(_Data, A);
			_data->first = nullptr;
			_data->last = nullptr;
			_data->size_cache = 0;
		}

		Element *n = memnew_allocator(Element, A);
		n->value = (T &)value;

		n->prev_ptr = _data->last;
		n->next_ptr = nullptr;
		n->data = _data;

		if (_data->last) {
			_data->last->next_ptr = n;
		}

		_data->last = n;

		if (!_data->first) {
			_data->first = n;
		}

		_data->size_cache++;

		return n;
	};

	void pop_back() {
		if (_data && _data->last) {
			erase(_data->last);
		}
	}

	/**
	 * store a new element at the beginning of the list
	 */
	Element *push_front(const T &value) {
		if (!_data) {
			_data = memnew_allocator(_Data, A);
			_data->first = nullptr;
			_data->last = nullptr;
			_data->size_cache = 0;
		}

		Element *n = memnew_allocator(Element, A);
		n->value = (T &)value;
		n->prev_ptr = nullptr;
		n->next_ptr = _data->first;
		n->data = _data;

		if (_data->first) {
			_data->first->prev_ptr = n;
		}

		_data->first = n;

		if (!_data->last) {
			_data->last = n;
		}

		_data->size_cache++;

		return n;
	};

	void pop_front() {
		if (_data && _data->first) {
			erase(_data->first);
		}
	}

	Element *insert_after(Element *p_element, const T &p_value) {
		CRASH_COND(p_element && (!_data || p_element->data != _data));

		if (!p_element) {
			return push_back(p_value);
		}

		Element *n = memnew_allocator(Element, A);
		n->value = (T &)p_value;
		n->prev_ptr = p_element;
		n->next_ptr = p_element->next_ptr;
		n->data = _data;

		if (!p_element->next_ptr) {
			_data->last = n;
		} else {
			p_element->next_ptr->prev_ptr = n;
		}

		p_element->next_ptr = n;

		_data->size_cache++;

		return n;
	}

	Element *insert_before(Element *p_element, const T &p_value) {
		CRASH_COND(p_element && (!_data || p_element->data != _data));

		if (!p_element) {
			return push_back(p_value);
		}

		Element *n = memnew_allocator(Element, A);
		n->value = (T &)p_value;
		n->prev_ptr = p_element->prev_ptr;
		n->next_ptr = p_element;
		n->data = _data;

		if (!p_element->prev_ptr) {
			_data->first = n;
		} else {
			p_element->prev_ptr->next_ptr = n;
		}

		p_element->prev_ptr = n;

		_data->size_cache++;

		return n;
	}

	/**
	 * find an element in the list,
	 */
	template <class T_v>
	Element *find(const T_v &p_val) {
		Element *it = front();
		while (it) {
			if (it->value == p_val) {
				return it;
			}
			it = it->next();
		};

		return nullptr;
	};

	/**
	 * erase an element in the list, by iterator pointing to it. Return true if it was found/erased.
	 */
	bool erase(const Element *p_I) {
		if (_data) {
			bool ret = _data->erase(p_I);

			if (_data->size_cache == 0) {
				memdelete_allocator<_Data, A>(_data);
				_data = nullptr;
			}

			return ret;
		}

		return false;
	};

	/**
	 * erase the first element in the list, that contains value
	 */
	bool erase(const T &value) {
		Element *I = find(value);
		return erase(I);
	};

	/**
	 * return whether the list is empty
	 */
	_FORCE_INLINE_ bool empty() const {
		return (!_data || !_data->size_cache);
	}

	/**
	 * clear the list
	 */
	void clear() {
		while (front()) {
			erase(front());
		};
	};

	_FORCE_INLINE_ int size() const {
		return _data ? _data->size_cache : 0;
	}

	void swap(Element *p_A, Element *p_B) {
		ERR_FAIL_COND(!p_A || !p_B);
		ERR_FAIL_COND(p_A->data != _data);
		ERR_FAIL_COND(p_B->data != _data);

		if (p_A == p_B) {
			return;
		}
		Element *A_prev = p_A->prev_ptr;
		Element *A_next = p_A->next_ptr;
		Element *B_prev = p_B->prev_ptr;
		Element *B_next = p_B->next_ptr;

		if (A_prev) {
			A_prev->next_ptr = p_B;
		} else {
			_data->first = p_B;
		}
		if (B_prev) {
			B_prev->next_ptr = p_A;
		} else {
			_data->first = p_A;
		}
		if (A_next) {
			A_next->prev_ptr = p_B;
		} else {
			_data->last = p_B;
		}
		if (B_next) {
			B_next->prev_ptr = p_A;
		} else {
			_data->last = p_A;
		}
		p_A->prev_ptr = A_next == p_B ? p_B : B_prev;
		p_A->next_ptr = B_next == p_A ? p_B : B_next;
		p_B->prev_ptr = B_next == p_A ? p_A : A_prev;
		p_B->next_ptr = A_next == p_B ? p_A : A_next;
	}
	/**
	 * copy the list
	 */
	void operator=(const List &p_list) {
		clear();
		const Element *it = p_list.front();
		while (it) {
			push_back(it->get());
			it = it->next();
		}
	}

	T &operator[](int p_index) {
		CRASH_BAD_INDEX(p_index, size());

		Element *I = front();
		int c = 0;
		while (I) {
			if (c == p_index) {
				return I->get();
			}
			I = I->next();
			c++;
		}

		CRASH_NOW(); // bug!!
	}

	const T &operator[](int p_index) const {
		CRASH_BAD_INDEX(p_index, size());

		const Element *I = front();
		int c = 0;
		while (I) {
			if (c == p_index) {
				return I->get();
			}
			I = I->next();
			c++;
		}

		CRASH_NOW(); // bug!!
	}

	void move_to_back(Element *p_I) {
		ERR_FAIL_COND(p_I->data != _data);
		if (!p_I->next_ptr) {
			return;
		}

		if (_data->first == p_I) {
			_data->first = p_I->next_ptr;
		};

		if (_data->last == p_I) {
			_data->last = p_I->prev_ptr;
		}

		if (p_I->prev_ptr) {
			p_I->prev_ptr->next_ptr = p_I->next_ptr;
		}

		p_I->next_ptr->prev_ptr = p_I->prev_ptr;

		_data->last->next_ptr = p_I;
		p_I->prev_ptr = _data->last;
		p_I->next_ptr = nullptr;
		_data->last = p_I;
	}

	void invert() {
		int s = size() / 2;
		Element *F = front();
		Element *B = back();
		for (int i = 0; i < s; i++) {
			SWAP(F->value, B->value);
			F = F->next();
			B = B->prev();
		}
	}

	void move_to_front(Element *p_I) {
		ERR_FAIL_COND(p_I->data != _data);
		if (!p_I->prev_ptr) {
			return;
		}

		if (_data->first == p_I) {
			_data->first = p_I->next_ptr;
		};

		if (_data->last == p_I) {
			_data->last = p_I->prev_ptr;
		}

		p_I->prev_ptr->next_ptr = p_I->next_ptr;

		if (p_I->next_ptr) {
			p_I->next_ptr->prev_ptr = p_I->prev_ptr;
		}

		_data->first->prev_ptr = p_I;
		p_I->next_ptr = _data->first;
		p_I->prev_ptr = nullptr;
		_data->first = p_I;
	}

	void move_before(Element *value, Element *where) {
		if (value->prev_ptr) {
			value->prev_ptr->next_ptr = value->next_ptr;
		} else {
			_data->first = value->next_ptr;
		}
		if (value->next_ptr) {
			value->next_ptr->prev_ptr = value->prev_ptr;
		} else {
			_data->last = value->prev_ptr;
		}

		value->next_ptr = where;
		if (!where) {
			value->prev_ptr = _data->last;
			_data->last = value;
			return;
		};

		value->prev_ptr = where->prev_ptr;

		if (where->prev_ptr) {
			where->prev_ptr->next_ptr = value;
		} else {
			_data->first = value;
		};

		where->prev_ptr = value;
	};

	/**
	 * simple insertion sort
	 */

	void sort() {
		sort_custom<Comparator<T>>();
	}

	template <class C>
	void sort_custom_inplace() {
		if (size() < 2) {
			return;
		}

		Element *from = front();
		Element *current = from;
		Element *to = from;

		while (current) {
			Element *next = current->next_ptr;

			if (from != current) {
				current->prev_ptr = NULL;
				current->next_ptr = from;

				Element *find = from;
				C less;
				while (find && less(find->value, current->value)) {
					current->prev_ptr = find;
					current->next_ptr = find->next_ptr;
					find = find->next_ptr;
				}

				if (current->prev_ptr) {
					current->prev_ptr->next_ptr = current;
				} else {
					from = current;
				}

				if (current->next_ptr) {
					current->next_ptr->prev_ptr = current;
				} else {
					to = current;
				}
			} else {
				current->prev_ptr = NULL;
				current->next_ptr = NULL;
			}

			current = next;
		}
		_data->first = from;
		_data->last = to;
	}

	template <class C>
	struct AuxiliaryComparator {
		C compare;
		_FORCE_INLINE_ bool operator()(const Element *a, const Element *b) const {
			return compare(a->value, b->value);
		}
	};

	template <class C>
	void sort_custom() {
		//this version uses auxiliary memory for speed.
		//if you don't want to use auxiliary memory, use the in_place version

		int s = size();
		if (s < 2) {
			return;
		}

		Element **aux_buffer = memnew_arr(Element *, s);

		int idx = 0;
		for (Element *E = front(); E; E = E->next_ptr) {
			aux_buffer[idx] = E;
			idx++;
		}

		SortArray<Element *, AuxiliaryComparator<C>> sort;
		sort.sort(aux_buffer, s);

		_data->first = aux_buffer[0];
		aux_buffer[0]->prev_ptr = nullptr;
		aux_buffer[0]->next_ptr = aux_buffer[1];

		_data->last = aux_buffer[s - 1];
		aux_buffer[s - 1]->prev_ptr = aux_buffer[s - 2];
		aux_buffer[s - 1]->next_ptr = nullptr;

		for (int i = 1; i < s - 1; i++) {
			aux_buffer[i]->prev_ptr = aux_buffer[i - 1];
			aux_buffer[i]->next_ptr = aux_buffer[i + 1];
		}

		memdelete_arr(aux_buffer);
	}

	const void *id() const {
		return (void *)_data;
	}

	/**
	 * copy constructor for the list
	 */
	List(const List &p_list) {
		_data = nullptr;
		const Element *it = p_list.front();
		while (it) {
			push_back(it->get());
			it = it->next();
		}
	}

	List() {
		_data = nullptr;
	};
	~List() {
		clear();
		if (_data) {
			ERR_FAIL_COND(_data->size_cache);
			memdelete_allocator<_Data, A>(_data);
		}
	};
};

#line 0

#line 1 "sfwl/core/ring_buffer.h"

/*************************************************************************/
/*  ring_buffer.h                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

template <typename T>
class RingBuffer {
	Vector<T> data;
	int read_pos;
	int write_pos;
	int size_mask;

	inline int inc(int &p_var, int p_size) const {
		int ret = p_var;
		p_var += p_size;
		p_var = p_var & size_mask;
		return ret;
	};

public:
	T read() {
		ERR_FAIL_COND_V(space_left() < 1, T());
		return data.ptr()[inc(read_pos, 1)];
	};

	int read(T *p_buf, int p_size, bool p_advance = true) {
		int left = data_left();
		p_size = MIN(left, p_size);
		int pos = read_pos;
		int to_read = p_size;
		int dst = 0;
		while (to_read) {
			int end = pos + to_read;
			end = MIN(end, size());
			int total = end - pos;
			const T *read = data.ptr();
			for (int i = 0; i < total; i++) {
				p_buf[dst++] = read[pos + i];
			};
			to_read -= total;
			pos = 0;
		};
		if (p_advance) {
			inc(read_pos, p_size);
		};
		return p_size;
	};

	int copy(T *p_buf, int p_offset, int p_size) const {
		int left = data_left();
		if ((p_offset + p_size) > left) {
			p_size -= left - p_offset;
			if (p_size <= 0) {
				return 0;
			}
		}
		p_size = MIN(left, p_size);
		int pos = read_pos;
		inc(pos, p_offset);
		int to_read = p_size;
		int dst = 0;
		while (to_read) {
			int end = pos + to_read;
			end = MIN(end, size());
			int total = end - pos;
			for (int i = 0; i < total; i++) {
				p_buf[dst++] = data[pos + i];
			};
			to_read -= total;
			pos = 0;
		};
		return p_size;
	};

	int find(const T &t, int p_offset, int p_max_size) const {
		int left = data_left();
		if ((p_offset + p_max_size) > left) {
			p_max_size -= left - p_offset;
			if (p_max_size <= 0) {
				return 0;
			}
		}
		p_max_size = MIN(left, p_max_size);
		int pos = read_pos;
		inc(pos, p_offset);
		int to_read = p_max_size;
		while (to_read) {
			int end = pos + to_read;
			end = MIN(end, size());
			int total = end - pos;
			for (int i = 0; i < total; i++) {
				if (data[pos + i] == t) {
					return i + (p_max_size - to_read);
				}
			};
			to_read -= total;
			pos = 0;
		}
		return -1;
	}

	inline int advance_read(int p_n) {
		p_n = MIN(p_n, data_left());
		inc(read_pos, p_n);
		return p_n;
	};

	inline int decrease_write(int p_n) {
		p_n = MIN(p_n, data_left());
		inc(write_pos, size_mask + 1 - p_n);
		return p_n;
	}

	Error write(const T &p_v) {
		ERR_FAIL_COND_V(space_left() < 1, FAILED);
		data.write[inc(write_pos, 1)] = p_v;
		return OK;
	};

	int write(const T *p_buf, int p_size) {
		int left = space_left();
		p_size = MIN(left, p_size);

		int pos = write_pos;
		int to_write = p_size;
		int src = 0;
		while (to_write) {
			int end = pos + to_write;
			end = MIN(end, size());
			int total = end - pos;

			for (int i = 0; i < total; i++) {
				data.write[pos + i] = p_buf[src++];
			};
			to_write -= total;
			pos = 0;
		};

		inc(write_pos, p_size);
		return p_size;
	};

	inline int space_left() const {
		int left = read_pos - write_pos;
		if (left < 0) {
			return size() + left - 1;
		};
		if (left == 0) {
			return size() - 1;
		};
		return left - 1;
	};
	inline int data_left() const {
		return size() - space_left() - 1;
	};

	inline int size() const {
		return data.size();
	};

	inline void clear() {
		read_pos = 0;
		write_pos = 0;
	}

	void resize(int p_power) {
		int old_size = size();
		int new_size = 1 << p_power;
		int mask = new_size - 1;
		data.resize(1 << p_power);
		if (old_size < new_size && read_pos > write_pos) {
			for (int i = 0; i < write_pos; i++) {
				data.write[(old_size + i) & mask] = data[i];
			};
			write_pos = (old_size + write_pos) & mask;
		} else {
			read_pos = read_pos & mask;
			write_pos = write_pos & mask;
		};

		size_mask = mask;
	};

	RingBuffer<T>(int p_power = 0) {
		read_pos = 0;
		write_pos = 0;
		resize(p_power);
	};
	~RingBuffer<T>(){};
};

#line 0

#line 1 "sfwl/core/paged_allocator.h"

/*************************************************************************/
/*  paged_allocator.h                                                    */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

template <class T, bool thread_safe = false>
class PagedAllocator {
	T **page_pool = nullptr;
	T ***available_pool = nullptr;
	uint32_t pages_allocated = 0;
	uint32_t allocs_available = 0;

	uint32_t page_shift = 0;
	uint32_t page_mask = 0;
	uint32_t page_size = 0;
	SpinLock spin_lock;

public:
	T *alloc() {
		if (thread_safe) {
			spin_lock.lock();
		}
		if (unlikely(allocs_available == 0)) {
			uint32_t pages_used = pages_allocated;

			pages_allocated++;
			page_pool = (T **)memrealloc(page_pool, sizeof(T *) * pages_allocated);
			available_pool = (T ***)memrealloc(available_pool, sizeof(T **) * pages_allocated);

			page_pool[pages_used] = (T *)memalloc(sizeof(T) * page_size);
			available_pool[pages_used] = (T **)memalloc(sizeof(T *) * page_size);

			for (uint32_t i = 0; i < page_size; i++) {
				available_pool[0][i] = &page_pool[pages_used][i];
			}
			allocs_available += page_size;
		}

		allocs_available--;
		T *alloc = available_pool[allocs_available >> page_shift][allocs_available & page_mask];
		if (thread_safe) {
			spin_lock.unlock();
		}
		memnew_placement(alloc, T);
		return alloc;
	}

	void free(T *p_mem) {
		if (thread_safe) {
			spin_lock.lock();
		}
		p_mem->~T();
		available_pool[allocs_available >> page_shift][allocs_available & page_mask] = p_mem;
		if (thread_safe) {
			spin_lock.unlock();
		}
		allocs_available++;
	}

	void reset(bool p_allow_unfreed = false) {
		if (!p_allow_unfreed || !HAS_TRIVIAL_DESTRUCTOR(T)) {
			ERR_FAIL_COND(allocs_available < pages_allocated * page_size);
		}
		if (pages_allocated) {
			for (uint32_t i = 0; i < pages_allocated; i++) {
				memfree(page_pool[i]);
				memfree(available_pool[i]);
			}
			memfree(page_pool);
			memfree(available_pool);
			page_pool = nullptr;
			available_pool = nullptr;
			pages_allocated = 0;
			allocs_available = 0;
		}
	}
	bool is_configured() const {
		return page_size > 0;
	}

	void configure(uint32_t p_page_size) {
		ERR_FAIL_COND(page_pool != nullptr); //sanity check
		ERR_FAIL_COND(p_page_size == 0);
		page_size = nearest_power_of_2_templated(p_page_size);
		page_mask = page_size - 1;
		page_shift = get_shift_from_power_of_2(page_size);
	}

	// Power of 2 recommended because of alignment with OS page sizes.
	// Even if element is bigger, its still a multiple and get rounded amount of pages
	PagedAllocator(uint32_t p_page_size = 4096) {
		configure(p_page_size);
	}

	~PagedAllocator() {
		ERR_FAIL_COND_MSG(allocs_available < pages_allocated * page_size, "Pages in use exist at exit in PagedAllocator");
		reset();
	}
};

#line 0

#line 1 "sfwl/core/pool_allocator.h"

/*************************************************************************/
/*  pool_allocator.h                                                     */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
	@author Juan Linietsky <reduzio@gmail.com>
 * Generic Pool Allocator.
 * This is a generic memory pool allocator, with locking, compacting and alignment. (@TODO alignment)
 * It used as a standard way to manage allocation in a specific region of memory, such as texture memory,
 * audio sample memory, or just any kind of memory overall.
 * (@TODO) abstraction should be greater, because in many platforms, you need to manage a nonreachable memory.
*/

enum {

	POOL_ALLOCATOR_INVALID_ID = -1 ///< default invalid value. use INVALID_ID( id ) to test
};

class PoolAllocator {
public:
	typedef int ID;

private:
	enum {
		CHECK_BITS = 8,
		CHECK_LEN = (1 << CHECK_BITS),
		CHECK_MASK = CHECK_LEN - 1

	};

	struct Entry {
		unsigned int pos;
		unsigned int len;
		unsigned int lock;
		unsigned int check;

		inline void clear() {
			pos = 0;
			len = 0;
			lock = 0;
			check = 0;
		}
		Entry() { clear(); }
	};

	typedef int EntryArrayPos;
	typedef int EntryIndicesPos;

	Entry *entry_array;
	int *entry_indices;
	int entry_max;
	int entry_count;

	uint8_t *pool;
	void *mem_ptr;
	int pool_size;

	int free_mem;
	int free_mem_peak;

	unsigned int check_count;
	int align;

	bool needs_locking;

	inline int entry_end(const Entry &p_entry) const {
		return p_entry.pos + aligned(p_entry.len);
	}
	inline int aligned(int p_size) const {
		int rem = p_size % align;
		if (rem) {
			p_size += align - rem;
		}

		return p_size;
	}

	void compact(int p_up_to = -1);
	void compact_up(int p_from = 0);
	bool get_free_entry(EntryArrayPos *p_pos);
	bool find_hole(EntryArrayPos *p_pos, int p_for_size);
	bool find_entry_index(EntryIndicesPos *p_map_pos, Entry *p_entry);
	Entry *get_entry(ID p_mem);
	const Entry *get_entry(ID p_mem) const;

	void create_pool(void *p_mem, int p_size, int p_max_entries);

protected:
	virtual void mt_lock() const; ///< Reimplement for custom mt locking
	virtual void mt_unlock() const; ///< Reimplement for custom mt locking

public:
	enum {
		DEFAULT_MAX_ALLOCS = 4096,
	};

	ID alloc(int p_size); ///< Alloc memory, get an ID on success, POOL_ALOCATOR_INVALID_ID on failure
	void free(ID p_mem); ///< Free allocated memory
	Error resize(ID p_mem, int p_new_size); ///< resize a memory chunk
	int get_size(ID p_mem) const;

	int get_free_mem(); ///< get free memory
	int get_used_mem() const;
	int get_free_peak(); ///< get free memory

	Error lock(ID p_mem); //@todo move this out
	void *get(ID p_mem);
	const void *get(ID p_mem) const;
	void unlock(ID p_mem);
	bool is_locked(ID p_mem) const;

	PoolAllocator(int p_size, bool p_needs_locking = false, int p_max_entries = DEFAULT_MAX_ALLOCS);
	PoolAllocator(void *p_mem, int p_size, int p_align = 1, bool p_needs_locking = false, int p_max_entries = DEFAULT_MAX_ALLOCS);
	PoolAllocator(int p_align, int p_size, bool p_needs_locking = false, int p_max_entries = DEFAULT_MAX_ALLOCS);

	virtual ~PoolAllocator();
};

#line 0

#line 1 "sfwl/core/char_range.inc"
/*************************************************************************/
/*  char_range.inc                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

struct CharRange {
	char32_t start;
	char32_t end;
};

static CharRange xid_start[] = {
	{ 0x41, 0x5a },
	{ 0x5f, 0x5f },
	{ 0x61, 0x7a },
	{ 0xaa, 0xaa },
	{ 0xb5, 0xb5 },
	{ 0xba, 0xba },
	{ 0xc0, 0xd6 },
	{ 0xd8, 0xf6 },
	{ 0xf8, 0x2c1 },
	{ 0x2c6, 0x2d1 },
	{ 0x2e0, 0x2e4 },
	{ 0x2ec, 0x2ec },
	{ 0x2ee, 0x2ee },
	{ 0x370, 0x374 },
	{ 0x376, 0x377 },
	{ 0x37a, 0x37d },
	{ 0x37f, 0x37f },
	{ 0x386, 0x386 },
	{ 0x388, 0x38a },
	{ 0x38c, 0x38c },
	{ 0x38e, 0x3a1 },
	{ 0x3a3, 0x3f5 },
	{ 0x3f7, 0x481 },
	{ 0x48a, 0x52f },
	{ 0x531, 0x556 },
	{ 0x559, 0x559 },
	{ 0x560, 0x588 },
	{ 0x5d0, 0x5ea },
	{ 0x5ef, 0x5f2 },
	{ 0x620, 0x64a },
	{ 0x66e, 0x66f },
	{ 0x671, 0x6d3 },
	{ 0x6d5, 0x6d5 },
	{ 0x6e5, 0x6e6 },
	{ 0x6ee, 0x6ef },
	{ 0x6fa, 0x6fc },
	{ 0x6ff, 0x6ff },
	{ 0x710, 0x710 },
	{ 0x712, 0x72f },
	{ 0x74d, 0x7a5 },
	{ 0x7b1, 0x7b1 },
	{ 0x7ca, 0x7ea },
	{ 0x7f4, 0x7f5 },
	{ 0x7fa, 0x7fa },
	{ 0x800, 0x815 },
	{ 0x81a, 0x81a },
	{ 0x824, 0x824 },
	{ 0x828, 0x828 },
	{ 0x840, 0x858 },
	{ 0x860, 0x86a },
	{ 0x870, 0x887 },
	{ 0x889, 0x88e },
	{ 0x8a0, 0x8c9 },
	{ 0x904, 0x939 },
	{ 0x93d, 0x93d },
	{ 0x950, 0x950 },
	{ 0x958, 0x961 },
	{ 0x971, 0x980 },
	{ 0x985, 0x98c },
	{ 0x98f, 0x990 },
	{ 0x993, 0x9a8 },
	{ 0x9aa, 0x9b0 },
	{ 0x9b2, 0x9b2 },
	{ 0x9b6, 0x9b9 },
	{ 0x9bd, 0x9bd },
	{ 0x9ce, 0x9ce },
	{ 0x9dc, 0x9dd },
	{ 0x9df, 0x9e1 },
	{ 0x9f0, 0x9f1 },
	{ 0x9fc, 0x9fc },
	{ 0xa05, 0xa0a },
	{ 0xa0f, 0xa10 },
	{ 0xa13, 0xa28 },
	{ 0xa2a, 0xa30 },
	{ 0xa32, 0xa33 },
	{ 0xa35, 0xa36 },
	{ 0xa38, 0xa39 },
	{ 0xa59, 0xa5c },
	{ 0xa5e, 0xa5e },
	{ 0xa72, 0xa74 },
	{ 0xa85, 0xa8d },
	{ 0xa8f, 0xa91 },
	{ 0xa93, 0xaa8 },
	{ 0xaaa, 0xab0 },
	{ 0xab2, 0xab3 },
	{ 0xab5, 0xab9 },
	{ 0xabd, 0xabd },
	{ 0xad0, 0xad0 },
	{ 0xae0, 0xae1 },
	{ 0xaf9, 0xaf9 },
	{ 0xb05, 0xb0c },
	{ 0xb0f, 0xb10 },
	{ 0xb13, 0xb28 },
	{ 0xb2a, 0xb30 },
	{ 0xb32, 0xb33 },
	{ 0xb35, 0xb39 },
	{ 0xb3d, 0xb3d },
	{ 0xb5c, 0xb5d },
	{ 0xb5f, 0xb61 },
	{ 0xb71, 0xb71 },
	{ 0xb83, 0xb83 },
	{ 0xb85, 0xb8a },
	{ 0xb8e, 0xb90 },
	{ 0xb92, 0xb95 },
	{ 0xb99, 0xb9a },
	{ 0xb9c, 0xb9c },
	{ 0xb9e, 0xb9f },
	{ 0xba3, 0xba4 },
	{ 0xba8, 0xbaa },
	{ 0xbae, 0xbb9 },
	{ 0xbd0, 0xbd0 },
	{ 0xc05, 0xc0c },
	{ 0xc0e, 0xc10 },
	{ 0xc12, 0xc28 },
	{ 0xc2a, 0xc39 },
	{ 0xc3d, 0xc3d },
	{ 0xc58, 0xc5a },
	{ 0xc5d, 0xc5d },
	{ 0xc60, 0xc61 },
	{ 0xc80, 0xc80 },
	{ 0xc85, 0xc8c },
	{ 0xc8e, 0xc90 },
	{ 0xc92, 0xca8 },
	{ 0xcaa, 0xcb3 },
	{ 0xcb5, 0xcb9 },
	{ 0xcbd, 0xcbd },
	{ 0xcdd, 0xcde },
	{ 0xce0, 0xce1 },
	{ 0xcf1, 0xcf2 },
	{ 0xd04, 0xd0c },
	{ 0xd0e, 0xd10 },
	{ 0xd12, 0xd3a },
	{ 0xd3d, 0xd3d },
	{ 0xd4e, 0xd4e },
	{ 0xd54, 0xd56 },
	{ 0xd5f, 0xd61 },
	{ 0xd7a, 0xd7f },
	{ 0xd85, 0xd96 },
	{ 0xd9a, 0xdb1 },
	{ 0xdb3, 0xdbb },
	{ 0xdbd, 0xdbd },
	{ 0xdc0, 0xdc6 },
	{ 0xe01, 0xe30 },
	{ 0xe32, 0xe33 },
	{ 0xe40, 0xe46 },
	{ 0xe81, 0xe82 },
	{ 0xe84, 0xe84 },
	{ 0xe86, 0xe8a },
	{ 0xe8c, 0xea3 },
	{ 0xea5, 0xea5 },
	{ 0xea7, 0xeb0 },
	{ 0xeb2, 0xeb3 },
	{ 0xebd, 0xebd },
	{ 0xec0, 0xec4 },
	{ 0xec6, 0xec6 },
	{ 0xedc, 0xedf },
	{ 0xf00, 0xf00 },
	{ 0xf40, 0xf47 },
	{ 0xf49, 0xf6c },
	{ 0xf88, 0xf8c },
	{ 0x1000, 0x102a },
	{ 0x103f, 0x103f },
	{ 0x1050, 0x1055 },
	{ 0x105a, 0x105d },
	{ 0x1061, 0x1061 },
	{ 0x1065, 0x1066 },
	{ 0x106e, 0x1070 },
	{ 0x1075, 0x1081 },
	{ 0x108e, 0x108e },
	{ 0x10a0, 0x10c5 },
	{ 0x10c7, 0x10c7 },
	{ 0x10cd, 0x10cd },
	{ 0x10d0, 0x10fa },
	{ 0x10fc, 0x1248 },
	{ 0x124a, 0x124d },
	{ 0x1250, 0x1256 },
	{ 0x1258, 0x1258 },
	{ 0x125a, 0x125d },
	{ 0x1260, 0x1288 },
	{ 0x128a, 0x128d },
	{ 0x1290, 0x12b0 },
	{ 0x12b2, 0x12b5 },
	{ 0x12b8, 0x12be },
	{ 0x12c0, 0x12c0 },
	{ 0x12c2, 0x12c5 },
	{ 0x12c8, 0x12d6 },
	{ 0x12d8, 0x1310 },
	{ 0x1312, 0x1315 },
	{ 0x1318, 0x135a },
	{ 0x1380, 0x138f },
	{ 0x13a0, 0x13f5 },
	{ 0x13f8, 0x13fd },
	{ 0x1401, 0x166c },
	{ 0x166f, 0x167f },
	{ 0x1681, 0x169a },
	{ 0x16a0, 0x16ea },
	{ 0x16ee, 0x16f8 },
	{ 0x1700, 0x1711 },
	{ 0x171f, 0x1731 },
	{ 0x1740, 0x1751 },
	{ 0x1760, 0x176c },
	{ 0x176e, 0x1770 },
	{ 0x1780, 0x17b3 },
	{ 0x17d7, 0x17d7 },
	{ 0x17dc, 0x17dc },
	{ 0x1820, 0x1878 },
	{ 0x1880, 0x1884 },
	{ 0x1887, 0x18a8 },
	{ 0x18aa, 0x18aa },
	{ 0x18b0, 0x18f5 },
	{ 0x1900, 0x191e },
	{ 0x1950, 0x196d },
	{ 0x1970, 0x1974 },
	{ 0x1980, 0x19ab },
	{ 0x19b0, 0x19c9 },
	{ 0x1a00, 0x1a16 },
	{ 0x1a20, 0x1a54 },
	{ 0x1aa7, 0x1aa7 },
	{ 0x1b05, 0x1b33 },
	{ 0x1b45, 0x1b4c },
	{ 0x1b83, 0x1ba0 },
	{ 0x1bae, 0x1baf },
	{ 0x1bba, 0x1be5 },
	{ 0x1c00, 0x1c23 },
	{ 0x1c4d, 0x1c4f },
	{ 0x1c5a, 0x1c7d },
	{ 0x1c80, 0x1c88 },
	{ 0x1c90, 0x1cba },
	{ 0x1cbd, 0x1cbf },
	{ 0x1ce9, 0x1cec },
	{ 0x1cee, 0x1cf3 },
	{ 0x1cf5, 0x1cf6 },
	{ 0x1cfa, 0x1cfa },
	{ 0x1d00, 0x1dbf },
	{ 0x1e00, 0x1f15 },
	{ 0x1f18, 0x1f1d },
	{ 0x1f20, 0x1f45 },
	{ 0x1f48, 0x1f4d },
	{ 0x1f50, 0x1f57 },
	{ 0x1f59, 0x1f59 },
	{ 0x1f5b, 0x1f5b },
	{ 0x1f5d, 0x1f5d },
	{ 0x1f5f, 0x1f7d },
	{ 0x1f80, 0x1fb4 },
	{ 0x1fb6, 0x1fbc },
	{ 0x1fbe, 0x1fbe },
	{ 0x1fc2, 0x1fc4 },
	{ 0x1fc6, 0x1fcc },
	{ 0x1fd0, 0x1fd3 },
	{ 0x1fd6, 0x1fdb },
	{ 0x1fe0, 0x1fec },
	{ 0x1ff2, 0x1ff4 },
	{ 0x1ff6, 0x1ffc },
	{ 0x2071, 0x2071 },
	{ 0x207f, 0x207f },
	{ 0x2090, 0x209c },
	{ 0x2102, 0x2102 },
	{ 0x2107, 0x2107 },
	{ 0x210a, 0x2113 },
	{ 0x2115, 0x2115 },
	{ 0x2118, 0x211d },
	{ 0x2124, 0x2124 },
	{ 0x2126, 0x2126 },
	{ 0x2128, 0x2128 },
	{ 0x212a, 0x2139 },
	{ 0x213c, 0x213f },
	{ 0x2145, 0x2149 },
	{ 0x214e, 0x214e },
	{ 0x2160, 0x2188 },
	{ 0x2c00, 0x2ce4 },
	{ 0x2ceb, 0x2cee },
	{ 0x2cf2, 0x2cf3 },
	{ 0x2d00, 0x2d25 },
	{ 0x2d27, 0x2d27 },
	{ 0x2d2d, 0x2d2d },
	{ 0x2d30, 0x2d67 },
	{ 0x2d6f, 0x2d6f },
	{ 0x2d80, 0x2d96 },
	{ 0x2da0, 0x2da6 },
	{ 0x2da8, 0x2dae },
	{ 0x2db0, 0x2db6 },
	{ 0x2db8, 0x2dbe },
	{ 0x2dc0, 0x2dc6 },
	{ 0x2dc8, 0x2dce },
	{ 0x2dd0, 0x2dd6 },
	{ 0x2dd8, 0x2dde },
	{ 0x3005, 0x3007 },
	{ 0x3021, 0x3029 },
	{ 0x3031, 0x3035 },
	{ 0x3038, 0x303c },
	{ 0x3041, 0x3096 },
	{ 0x309b, 0x309f },
	{ 0x30a1, 0x30fa },
	{ 0x30fc, 0x30ff },
	{ 0x3105, 0x312f },
	{ 0x3131, 0x318e },
	{ 0x31a0, 0x31bf },
	{ 0x31f0, 0x31ff },
	{ 0x3400, 0x4dbf },
	{ 0x4e00, 0xa48c },
	{ 0xa4d0, 0xa4fd },
	{ 0xa500, 0xa60c },
	{ 0xa610, 0xa61f },
	{ 0xa62a, 0xa62b },
	{ 0xa640, 0xa66e },
	{ 0xa67f, 0xa69d },
	{ 0xa6a0, 0xa6ef },
	{ 0xa717, 0xa71f },
	{ 0xa722, 0xa788 },
	{ 0xa78b, 0xa7ca },
	{ 0xa7d0, 0xa7d1 },
	{ 0xa7d3, 0xa7d3 },
	{ 0xa7d5, 0xa7d9 },
	{ 0xa7f2, 0xa801 },
	{ 0xa803, 0xa805 },
	{ 0xa807, 0xa80a },
	{ 0xa80c, 0xa822 },
	{ 0xa840, 0xa873 },
	{ 0xa882, 0xa8b3 },
	{ 0xa8f2, 0xa8f7 },
	{ 0xa8fb, 0xa8fb },
	{ 0xa8fd, 0xa8fe },
	{ 0xa90a, 0xa925 },
	{ 0xa930, 0xa946 },
	{ 0xa960, 0xa97c },
	{ 0xa984, 0xa9b2 },
	{ 0xa9cf, 0xa9cf },
	{ 0xa9e0, 0xa9e4 },
	{ 0xa9e6, 0xa9ef },
	{ 0xa9fa, 0xa9fe },
	{ 0xaa00, 0xaa28 },
	{ 0xaa40, 0xaa42 },
	{ 0xaa44, 0xaa4b },
	{ 0xaa60, 0xaa76 },
	{ 0xaa7a, 0xaa7a },
	{ 0xaa7e, 0xaaaf },
	{ 0xaab1, 0xaab1 },
	{ 0xaab5, 0xaab6 },
	{ 0xaab9, 0xaabd },
	{ 0xaac0, 0xaac0 },
	{ 0xaac2, 0xaac2 },
	{ 0xaadb, 0xaadd },
	{ 0xaae0, 0xaaea },
	{ 0xaaf2, 0xaaf4 },
	{ 0xab01, 0xab06 },
	{ 0xab09, 0xab0e },
	{ 0xab11, 0xab16 },
	{ 0xab20, 0xab26 },
	{ 0xab28, 0xab2e },
	{ 0xab30, 0xab5a },
	{ 0xab5c, 0xab69 },
	{ 0xab70, 0xabe2 },
	{ 0xac00, 0xd7a3 },
	{ 0xd7b0, 0xd7c6 },
	{ 0xd7cb, 0xd7fb },
	{ 0xf900, 0xfa6d },
	{ 0xfa70, 0xfad9 },
	{ 0xfb00, 0xfb06 },
	{ 0xfb13, 0xfb17 },
	{ 0xfb1d, 0xfb1d },
	{ 0xfb1f, 0xfb28 },
	{ 0xfb2a, 0xfb36 },
	{ 0xfb38, 0xfb3c },
	{ 0xfb3e, 0xfb3e },
	{ 0xfb40, 0xfb41 },
	{ 0xfb43, 0xfb44 },
	{ 0xfb46, 0xfbb1 },
	{ 0xfbd3, 0xfd3d },
	{ 0xfd50, 0xfd8f },
	{ 0xfd92, 0xfdc7 },
	{ 0xfdf0, 0xfdfb },
	{ 0xfe70, 0xfe74 },
	{ 0xfe76, 0xfefc },
	{ 0xff21, 0xff3a },
	{ 0xff41, 0xff5a },
	{ 0xff66, 0xffbe },
	{ 0xffc2, 0xffc7 },
	{ 0xffca, 0xffcf },
	{ 0xffd2, 0xffd7 },
	{ 0xffda, 0xffdc },
	{ 0x10000, 0x1000b },
	{ 0x1000d, 0x10026 },
	{ 0x10028, 0x1003a },
	{ 0x1003c, 0x1003d },
	{ 0x1003f, 0x1004d },
	{ 0x10050, 0x1005d },
	{ 0x10080, 0x100fa },
	{ 0x10140, 0x10174 },
	{ 0x10280, 0x1029c },
	{ 0x102a0, 0x102d0 },
	{ 0x10300, 0x1031f },
	{ 0x1032d, 0x1034a },
	{ 0x10350, 0x10375 },
	{ 0x10380, 0x1039d },
	{ 0x103a0, 0x103c3 },
	{ 0x103c8, 0x103cf },
	{ 0x103d1, 0x103d5 },
	{ 0x10400, 0x1049d },
	{ 0x104b0, 0x104d3 },
	{ 0x104d8, 0x104fb },
	{ 0x10500, 0x10527 },
	{ 0x10530, 0x10563 },
	{ 0x10570, 0x1057a },
	{ 0x1057c, 0x1058a },
	{ 0x1058c, 0x10592 },
	{ 0x10594, 0x10595 },
	{ 0x10597, 0x105a1 },
	{ 0x105a3, 0x105b1 },
	{ 0x105b3, 0x105b9 },
	{ 0x105bb, 0x105bc },
	{ 0x10600, 0x10736 },
	{ 0x10740, 0x10755 },
	{ 0x10760, 0x10767 },
	{ 0x10780, 0x10785 },
	{ 0x10787, 0x107b0 },
	{ 0x107b2, 0x107ba },
	{ 0x10800, 0x10805 },
	{ 0x10808, 0x10808 },
	{ 0x1080a, 0x10835 },
	{ 0x10837, 0x10838 },
	{ 0x1083c, 0x1083c },
	{ 0x1083f, 0x10855 },
	{ 0x10860, 0x10876 },
	{ 0x10880, 0x1089e },
	{ 0x108e0, 0x108f2 },
	{ 0x108f4, 0x108f5 },
	{ 0x10900, 0x10915 },
	{ 0x10920, 0x10939 },
	{ 0x10980, 0x109b7 },
	{ 0x109be, 0x109bf },
	{ 0x10a00, 0x10a00 },
	{ 0x10a10, 0x10a13 },
	{ 0x10a15, 0x10a17 },
	{ 0x10a19, 0x10a35 },
	{ 0x10a60, 0x10a7c },
	{ 0x10a80, 0x10a9c },
	{ 0x10ac0, 0x10ac7 },
	{ 0x10ac9, 0x10ae4 },
	{ 0x10b00, 0x10b35 },
	{ 0x10b40, 0x10b55 },
	{ 0x10b60, 0x10b72 },
	{ 0x10b80, 0x10b91 },
	{ 0x10c00, 0x10c48 },
	{ 0x10c80, 0x10cb2 },
	{ 0x10cc0, 0x10cf2 },
	{ 0x10d00, 0x10d23 },
	{ 0x10e80, 0x10ea9 },
	{ 0x10eb0, 0x10eb1 },
	{ 0x10f00, 0x10f1c },
	{ 0x10f27, 0x10f27 },
	{ 0x10f30, 0x10f45 },
	{ 0x10f70, 0x10f81 },
	{ 0x10fb0, 0x10fc4 },
	{ 0x10fe0, 0x10ff6 },
	{ 0x11003, 0x11037 },
	{ 0x11071, 0x11072 },
	{ 0x11075, 0x11075 },
	{ 0x11083, 0x110af },
	{ 0x110d0, 0x110e8 },
	{ 0x11103, 0x11126 },
	{ 0x11144, 0x11144 },
	{ 0x11147, 0x11147 },
	{ 0x11150, 0x11172 },
	{ 0x11176, 0x11176 },
	{ 0x11183, 0x111b2 },
	{ 0x111c1, 0x111c4 },
	{ 0x111da, 0x111da },
	{ 0x111dc, 0x111dc },
	{ 0x11200, 0x11211 },
	{ 0x11213, 0x1122b },
	{ 0x11280, 0x11286 },
	{ 0x11288, 0x11288 },
	{ 0x1128a, 0x1128d },
	{ 0x1128f, 0x1129d },
	{ 0x1129f, 0x112a8 },
	{ 0x112b0, 0x112de },
	{ 0x11305, 0x1130c },
	{ 0x1130f, 0x11310 },
	{ 0x11313, 0x11328 },
	{ 0x1132a, 0x11330 },
	{ 0x11332, 0x11333 },
	{ 0x11335, 0x11339 },
	{ 0x1133d, 0x1133d },
	{ 0x11350, 0x11350 },
	{ 0x1135d, 0x11361 },
	{ 0x11400, 0x11434 },
	{ 0x11447, 0x1144a },
	{ 0x1145f, 0x11461 },
	{ 0x11480, 0x114af },
	{ 0x114c4, 0x114c5 },
	{ 0x114c7, 0x114c7 },
	{ 0x11580, 0x115ae },
	{ 0x115d8, 0x115db },
	{ 0x11600, 0x1162f },
	{ 0x11644, 0x11644 },
	{ 0x11680, 0x116aa },
	{ 0x116b8, 0x116b8 },
	{ 0x11700, 0x1171a },
	{ 0x11740, 0x11746 },
	{ 0x11800, 0x1182b },
	{ 0x118a0, 0x118df },
	{ 0x118ff, 0x11906 },
	{ 0x11909, 0x11909 },
	{ 0x1190c, 0x11913 },
	{ 0x11915, 0x11916 },
	{ 0x11918, 0x1192f },
	{ 0x1193f, 0x1193f },
	{ 0x11941, 0x11941 },
	{ 0x119a0, 0x119a7 },
	{ 0x119aa, 0x119d0 },
	{ 0x119e1, 0x119e1 },
	{ 0x119e3, 0x119e3 },
	{ 0x11a00, 0x11a00 },
	{ 0x11a0b, 0x11a32 },
	{ 0x11a3a, 0x11a3a },
	{ 0x11a50, 0x11a50 },
	{ 0x11a5c, 0x11a89 },
	{ 0x11a9d, 0x11a9d },
	{ 0x11ab0, 0x11af8 },
	{ 0x11c00, 0x11c08 },
	{ 0x11c0a, 0x11c2e },
	{ 0x11c40, 0x11c40 },
	{ 0x11c72, 0x11c8f },
	{ 0x11d00, 0x11d06 },
	{ 0x11d08, 0x11d09 },
	{ 0x11d0b, 0x11d30 },
	{ 0x11d46, 0x11d46 },
	{ 0x11d60, 0x11d65 },
	{ 0x11d67, 0x11d68 },
	{ 0x11d6a, 0x11d89 },
	{ 0x11d98, 0x11d98 },
	{ 0x11ee0, 0x11ef2 },
	{ 0x11fb0, 0x11fb0 },
	{ 0x12000, 0x12399 },
	{ 0x12400, 0x1246e },
	{ 0x12480, 0x12543 },
	{ 0x12f90, 0x12ff0 },
	{ 0x13000, 0x1342e },
	{ 0x14400, 0x14646 },
	{ 0x16800, 0x16a38 },
	{ 0x16a40, 0x16a5e },
	{ 0x16a70, 0x16abe },
	{ 0x16ad0, 0x16aed },
	{ 0x16b00, 0x16b2f },
	{ 0x16b40, 0x16b43 },
	{ 0x16b63, 0x16b77 },
	{ 0x16b7d, 0x16b8f },
	{ 0x16e40, 0x16e7f },
	{ 0x16f00, 0x16f4a },
	{ 0x16f50, 0x16f50 },
	{ 0x16f93, 0x16f9f },
	{ 0x16fe0, 0x16fe1 },
	{ 0x16fe3, 0x16fe3 },
	{ 0x17000, 0x187f7 },
	{ 0x18800, 0x18cd5 },
	{ 0x18d00, 0x18d08 },
	{ 0x1aff0, 0x1aff3 },
	{ 0x1aff5, 0x1affb },
	{ 0x1affd, 0x1affe },
	{ 0x1b000, 0x1b122 },
	{ 0x1b150, 0x1b152 },
	{ 0x1b164, 0x1b167 },
	{ 0x1b170, 0x1b2fb },
	{ 0x1bc00, 0x1bc6a },
	{ 0x1bc70, 0x1bc7c },
	{ 0x1bc80, 0x1bc88 },
	{ 0x1bc90, 0x1bc99 },
	{ 0x1d400, 0x1d454 },
	{ 0x1d456, 0x1d49c },
	{ 0x1d49e, 0x1d49f },
	{ 0x1d4a2, 0x1d4a2 },
	{ 0x1d4a5, 0x1d4a6 },
	{ 0x1d4a9, 0x1d4ac },
	{ 0x1d4ae, 0x1d4b9 },
	{ 0x1d4bb, 0x1d4bb },
	{ 0x1d4bd, 0x1d4c3 },
	{ 0x1d4c5, 0x1d505 },
	{ 0x1d507, 0x1d50a },
	{ 0x1d50d, 0x1d514 },
	{ 0x1d516, 0x1d51c },
	{ 0x1d51e, 0x1d539 },
	{ 0x1d53b, 0x1d53e },
	{ 0x1d540, 0x1d544 },
	{ 0x1d546, 0x1d546 },
	{ 0x1d54a, 0x1d550 },
	{ 0x1d552, 0x1d6a5 },
	{ 0x1d6a8, 0x1d6c0 },
	{ 0x1d6c2, 0x1d6da },
	{ 0x1d6dc, 0x1d6fa },
	{ 0x1d6fc, 0x1d714 },
	{ 0x1d716, 0x1d734 },
	{ 0x1d736, 0x1d74e },
	{ 0x1d750, 0x1d76e },
	{ 0x1d770, 0x1d788 },
	{ 0x1d78a, 0x1d7a8 },
	{ 0x1d7aa, 0x1d7c2 },
	{ 0x1d7c4, 0x1d7cb },
	{ 0x1df00, 0x1df1e },
	{ 0x1e100, 0x1e12c },
	{ 0x1e137, 0x1e13d },
	{ 0x1e14e, 0x1e14e },
	{ 0x1e290, 0x1e2ad },
	{ 0x1e2c0, 0x1e2eb },
	{ 0x1e7e0, 0x1e7e6 },
	{ 0x1e7e8, 0x1e7eb },
	{ 0x1e7ed, 0x1e7ee },
	{ 0x1e7f0, 0x1e7fe },
	{ 0x1e800, 0x1e8c4 },
	{ 0x1e900, 0x1e943 },
	{ 0x1e94b, 0x1e94b },
	{ 0x1ee00, 0x1ee03 },
	{ 0x1ee05, 0x1ee1f },
	{ 0x1ee21, 0x1ee22 },
	{ 0x1ee24, 0x1ee24 },
	{ 0x1ee27, 0x1ee27 },
	{ 0x1ee29, 0x1ee32 },
	{ 0x1ee34, 0x1ee37 },
	{ 0x1ee39, 0x1ee39 },
	{ 0x1ee3b, 0x1ee3b },
	{ 0x1ee42, 0x1ee42 },
	{ 0x1ee47, 0x1ee47 },
	{ 0x1ee49, 0x1ee49 },
	{ 0x1ee4b, 0x1ee4b },
	{ 0x1ee4d, 0x1ee4f },
	{ 0x1ee51, 0x1ee52 },
	{ 0x1ee54, 0x1ee54 },
	{ 0x1ee57, 0x1ee57 },
	{ 0x1ee59, 0x1ee59 },
	{ 0x1ee5b, 0x1ee5b },
	{ 0x1ee5d, 0x1ee5d },
	{ 0x1ee5f, 0x1ee5f },
	{ 0x1ee61, 0x1ee62 },
	{ 0x1ee64, 0x1ee64 },
	{ 0x1ee67, 0x1ee6a },
	{ 0x1ee6c, 0x1ee72 },
	{ 0x1ee74, 0x1ee77 },
	{ 0x1ee79, 0x1ee7c },
	{ 0x1ee7e, 0x1ee7e },
	{ 0x1ee80, 0x1ee89 },
	{ 0x1ee8b, 0x1ee9b },
	{ 0x1eea1, 0x1eea3 },
	{ 0x1eea5, 0x1eea9 },
	{ 0x1eeab, 0x1eebb },
	{ 0x20000, 0x2a6df },
	{ 0x2a700, 0x2b738 },
	{ 0x2b740, 0x2b81d },
	{ 0x2b820, 0x2cea1 },
	{ 0x2ceb0, 0x2ebe0 },
	{ 0x2f800, 0x2fa1d },
	{ 0x30000, 0x3134a },
	{ 0x0, 0x0 },
};

static CharRange xid_continue[] = {
	{ 0x30, 0x39 },
	{ 0x41, 0x5a },
	{ 0x5f, 0x5f },
	{ 0x61, 0x7a },
	{ 0xaa, 0xaa },
	{ 0xb5, 0xb5 },
	{ 0xb7, 0xb7 },
	{ 0xba, 0xba },
	{ 0xc0, 0xd6 },
	{ 0xd8, 0xf6 },
	{ 0xf8, 0x2c1 },
	{ 0x2c6, 0x2d1 },
	{ 0x2e0, 0x2e4 },
	{ 0x2ec, 0x2ec },
	{ 0x2ee, 0x2ee },
	{ 0x300, 0x374 },
	{ 0x376, 0x377 },
	{ 0x37a, 0x37d },
	{ 0x37f, 0x37f },
	{ 0x386, 0x38a },
	{ 0x38c, 0x38c },
	{ 0x38e, 0x3a1 },
	{ 0x3a3, 0x3f5 },
	{ 0x3f7, 0x481 },
	{ 0x483, 0x487 },
	{ 0x48a, 0x52f },
	{ 0x531, 0x556 },
	{ 0x559, 0x559 },
	{ 0x560, 0x588 },
	{ 0x591, 0x5bd },
	{ 0x5bf, 0x5bf },
	{ 0x5c1, 0x5c2 },
	{ 0x5c4, 0x5c5 },
	{ 0x5c7, 0x5c7 },
	{ 0x5d0, 0x5ea },
	{ 0x5ef, 0x5f2 },
	{ 0x610, 0x61a },
	{ 0x620, 0x669 },
	{ 0x66e, 0x6d3 },
	{ 0x6d5, 0x6dc },
	{ 0x6df, 0x6e8 },
	{ 0x6ea, 0x6fc },
	{ 0x6ff, 0x6ff },
	{ 0x710, 0x74a },
	{ 0x74d, 0x7b1 },
	{ 0x7c0, 0x7f5 },
	{ 0x7fa, 0x7fa },
	{ 0x7fd, 0x7fd },
	{ 0x800, 0x82d },
	{ 0x840, 0x85b },
	{ 0x860, 0x86a },
	{ 0x870, 0x887 },
	{ 0x889, 0x88e },
	{ 0x898, 0x8e1 },
	{ 0x8e3, 0x963 },
	{ 0x966, 0x96f },
	{ 0x971, 0x983 },
	{ 0x985, 0x98c },
	{ 0x98f, 0x990 },
	{ 0x993, 0x9a8 },
	{ 0x9aa, 0x9b0 },
	{ 0x9b2, 0x9b2 },
	{ 0x9b6, 0x9b9 },
	{ 0x9bc, 0x9c4 },
	{ 0x9c7, 0x9c8 },
	{ 0x9cb, 0x9ce },
	{ 0x9d7, 0x9d7 },
	{ 0x9dc, 0x9dd },
	{ 0x9df, 0x9e3 },
	{ 0x9e6, 0x9f1 },
	{ 0x9fc, 0x9fc },
	{ 0x9fe, 0x9fe },
	{ 0xa01, 0xa03 },
	{ 0xa05, 0xa0a },
	{ 0xa0f, 0xa10 },
	{ 0xa13, 0xa28 },
	{ 0xa2a, 0xa30 },
	{ 0xa32, 0xa33 },
	{ 0xa35, 0xa36 },
	{ 0xa38, 0xa39 },
	{ 0xa3c, 0xa3c },
	{ 0xa3e, 0xa42 },
	{ 0xa47, 0xa48 },
	{ 0xa4b, 0xa4d },
	{ 0xa51, 0xa51 },
	{ 0xa59, 0xa5c },
	{ 0xa5e, 0xa5e },
	{ 0xa66, 0xa75 },
	{ 0xa81, 0xa83 },
	{ 0xa85, 0xa8d },
	{ 0xa8f, 0xa91 },
	{ 0xa93, 0xaa8 },
	{ 0xaaa, 0xab0 },
	{ 0xab2, 0xab3 },
	{ 0xab5, 0xab9 },
	{ 0xabc, 0xac5 },
	{ 0xac7, 0xac9 },
	{ 0xacb, 0xacd },
	{ 0xad0, 0xad0 },
	{ 0xae0, 0xae3 },
	{ 0xae6, 0xaef },
	{ 0xaf9, 0xaff },
	{ 0xb01, 0xb03 },
	{ 0xb05, 0xb0c },
	{ 0xb0f, 0xb10 },
	{ 0xb13, 0xb28 },
	{ 0xb2a, 0xb30 },
	{ 0xb32, 0xb33 },
	{ 0xb35, 0xb39 },
	{ 0xb3c, 0xb44 },
	{ 0xb47, 0xb48 },
	{ 0xb4b, 0xb4d },
	{ 0xb55, 0xb57 },
	{ 0xb5c, 0xb5d },
	{ 0xb5f, 0xb63 },
	{ 0xb66, 0xb6f },
	{ 0xb71, 0xb71 },
	{ 0xb82, 0xb83 },
	{ 0xb85, 0xb8a },
	{ 0xb8e, 0xb90 },
	{ 0xb92, 0xb95 },
	{ 0xb99, 0xb9a },
	{ 0xb9c, 0xb9c },
	{ 0xb9e, 0xb9f },
	{ 0xba3, 0xba4 },
	{ 0xba8, 0xbaa },
	{ 0xbae, 0xbb9 },
	{ 0xbbe, 0xbc2 },
	{ 0xbc6, 0xbc8 },
	{ 0xbca, 0xbcd },
	{ 0xbd0, 0xbd0 },
	{ 0xbd7, 0xbd7 },
	{ 0xbe6, 0xbef },
	{ 0xc00, 0xc0c },
	{ 0xc0e, 0xc10 },
	{ 0xc12, 0xc28 },
	{ 0xc2a, 0xc39 },
	{ 0xc3c, 0xc44 },
	{ 0xc46, 0xc48 },
	{ 0xc4a, 0xc4d },
	{ 0xc55, 0xc56 },
	{ 0xc58, 0xc5a },
	{ 0xc5d, 0xc5d },
	{ 0xc60, 0xc63 },
	{ 0xc66, 0xc6f },
	{ 0xc80, 0xc83 },
	{ 0xc85, 0xc8c },
	{ 0xc8e, 0xc90 },
	{ 0xc92, 0xca8 },
	{ 0xcaa, 0xcb3 },
	{ 0xcb5, 0xcb9 },
	{ 0xcbc, 0xcc4 },
	{ 0xcc6, 0xcc8 },
	{ 0xcca, 0xccd },
	{ 0xcd5, 0xcd6 },
	{ 0xcdd, 0xcde },
	{ 0xce0, 0xce3 },
	{ 0xce6, 0xcef },
	{ 0xcf1, 0xcf2 },
	{ 0xd00, 0xd0c },
	{ 0xd0e, 0xd10 },
	{ 0xd12, 0xd44 },
	{ 0xd46, 0xd48 },
	{ 0xd4a, 0xd4e },
	{ 0xd54, 0xd57 },
	{ 0xd5f, 0xd63 },
	{ 0xd66, 0xd6f },
	{ 0xd7a, 0xd7f },
	{ 0xd81, 0xd83 },
	{ 0xd85, 0xd96 },
	{ 0xd9a, 0xdb1 },
	{ 0xdb3, 0xdbb },
	{ 0xdbd, 0xdbd },
	{ 0xdc0, 0xdc6 },
	{ 0xdca, 0xdca },
	{ 0xdcf, 0xdd4 },
	{ 0xdd6, 0xdd6 },
	{ 0xdd8, 0xddf },
	{ 0xde6, 0xdef },
	{ 0xdf2, 0xdf3 },
	{ 0xe01, 0xe3a },
	{ 0xe40, 0xe4e },
	{ 0xe50, 0xe59 },
	{ 0xe81, 0xe82 },
	{ 0xe84, 0xe84 },
	{ 0xe86, 0xe8a },
	{ 0xe8c, 0xea3 },
	{ 0xea5, 0xea5 },
	{ 0xea7, 0xebd },
	{ 0xec0, 0xec4 },
	{ 0xec6, 0xec6 },
	{ 0xec8, 0xecd },
	{ 0xed0, 0xed9 },
	{ 0xedc, 0xedf },
	{ 0xf00, 0xf00 },
	{ 0xf18, 0xf19 },
	{ 0xf20, 0xf29 },
	{ 0xf35, 0xf35 },
	{ 0xf37, 0xf37 },
	{ 0xf39, 0xf39 },
	{ 0xf3e, 0xf47 },
	{ 0xf49, 0xf6c },
	{ 0xf71, 0xf84 },
	{ 0xf86, 0xf97 },
	{ 0xf99, 0xfbc },
	{ 0xfc6, 0xfc6 },
	{ 0x1000, 0x1049 },
	{ 0x1050, 0x109d },
	{ 0x10a0, 0x10c5 },
	{ 0x10c7, 0x10c7 },
	{ 0x10cd, 0x10cd },
	{ 0x10d0, 0x10fa },
	{ 0x10fc, 0x1248 },
	{ 0x124a, 0x124d },
	{ 0x1250, 0x1256 },
	{ 0x1258, 0x1258 },
	{ 0x125a, 0x125d },
	{ 0x1260, 0x1288 },
	{ 0x128a, 0x128d },
	{ 0x1290, 0x12b0 },
	{ 0x12b2, 0x12b5 },
	{ 0x12b8, 0x12be },
	{ 0x12c0, 0x12c0 },
	{ 0x12c2, 0x12c5 },
	{ 0x12c8, 0x12d6 },
	{ 0x12d8, 0x1310 },
	{ 0x1312, 0x1315 },
	{ 0x1318, 0x135a },
	{ 0x135d, 0x135f },
	{ 0x1369, 0x1369 },
	{ 0x1371, 0x1371 },
	{ 0x1380, 0x138f },
	{ 0x13a0, 0x13f5 },
	{ 0x13f8, 0x13fd },
	{ 0x1401, 0x166c },
	{ 0x166f, 0x167f },
	{ 0x1681, 0x169a },
	{ 0x16a0, 0x16ea },
	{ 0x16ee, 0x16f8 },
	{ 0x1700, 0x1715 },
	{ 0x171f, 0x1734 },
	{ 0x1740, 0x1753 },
	{ 0x1760, 0x176c },
	{ 0x176e, 0x1770 },
	{ 0x1772, 0x1773 },
	{ 0x1780, 0x17d3 },
	{ 0x17d7, 0x17d7 },
	{ 0x17dc, 0x17dd },
	{ 0x17e0, 0x17e9 },
	{ 0x180b, 0x180d },
	{ 0x180f, 0x1819 },
	{ 0x1820, 0x1878 },
	{ 0x1880, 0x18aa },
	{ 0x18b0, 0x18f5 },
	{ 0x1900, 0x191e },
	{ 0x1920, 0x192b },
	{ 0x1930, 0x193b },
	{ 0x1946, 0x196d },
	{ 0x1970, 0x1974 },
	{ 0x1980, 0x19ab },
	{ 0x19b0, 0x19c9 },
	{ 0x19d0, 0x19da },
	{ 0x1a00, 0x1a1b },
	{ 0x1a20, 0x1a5e },
	{ 0x1a60, 0x1a7c },
	{ 0x1a7f, 0x1a89 },
	{ 0x1a90, 0x1a99 },
	{ 0x1aa7, 0x1aa7 },
	{ 0x1ab0, 0x1abd },
	{ 0x1abf, 0x1ace },
	{ 0x1b00, 0x1b4c },
	{ 0x1b50, 0x1b59 },
	{ 0x1b6b, 0x1b73 },
	{ 0x1b80, 0x1bf3 },
	{ 0x1c00, 0x1c37 },
	{ 0x1c40, 0x1c49 },
	{ 0x1c4d, 0x1c7d },
	{ 0x1c80, 0x1c88 },
	{ 0x1c90, 0x1cba },
	{ 0x1cbd, 0x1cbf },
	{ 0x1cd0, 0x1cd2 },
	{ 0x1cd4, 0x1cfa },
	{ 0x1d00, 0x1f15 },
	{ 0x1f18, 0x1f1d },
	{ 0x1f20, 0x1f45 },
	{ 0x1f48, 0x1f4d },
	{ 0x1f50, 0x1f57 },
	{ 0x1f59, 0x1f59 },
	{ 0x1f5b, 0x1f5b },
	{ 0x1f5d, 0x1f5d },
	{ 0x1f5f, 0x1f7d },
	{ 0x1f80, 0x1fb4 },
	{ 0x1fb6, 0x1fbc },
	{ 0x1fbe, 0x1fbe },
	{ 0x1fc2, 0x1fc4 },
	{ 0x1fc6, 0x1fcc },
	{ 0x1fd0, 0x1fd3 },
	{ 0x1fd6, 0x1fdb },
	{ 0x1fe0, 0x1fec },
	{ 0x1ff2, 0x1ff4 },
	{ 0x1ff6, 0x1ffc },
	{ 0x203f, 0x2040 },
	{ 0x2054, 0x2054 },
	{ 0x2071, 0x2071 },
	{ 0x207f, 0x207f },
	{ 0x2090, 0x209c },
	{ 0x20d0, 0x20dc },
	{ 0x20e1, 0x20e1 },
	{ 0x20e5, 0x20f0 },
	{ 0x2102, 0x2102 },
	{ 0x2107, 0x2107 },
	{ 0x210a, 0x2113 },
	{ 0x2115, 0x2115 },
	{ 0x2118, 0x211d },
	{ 0x2124, 0x2124 },
	{ 0x2126, 0x2126 },
	{ 0x2128, 0x2128 },
	{ 0x212a, 0x2139 },
	{ 0x213c, 0x213f },
	{ 0x2145, 0x2149 },
	{ 0x214e, 0x214e },
	{ 0x2160, 0x2188 },
	{ 0x2c00, 0x2ce4 },
	{ 0x2ceb, 0x2cf3 },
	{ 0x2d00, 0x2d25 },
	{ 0x2d27, 0x2d27 },
	{ 0x2d2d, 0x2d2d },
	{ 0x2d30, 0x2d67 },
	{ 0x2d6f, 0x2d6f },
	{ 0x2d7f, 0x2d96 },
	{ 0x2da0, 0x2da6 },
	{ 0x2da8, 0x2dae },
	{ 0x2db0, 0x2db6 },
	{ 0x2db8, 0x2dbe },
	{ 0x2dc0, 0x2dc6 },
	{ 0x2dc8, 0x2dce },
	{ 0x2dd0, 0x2dd6 },
	{ 0x2dd8, 0x2dde },
	{ 0x2de0, 0x2dff },
	{ 0x3005, 0x3007 },
	{ 0x3021, 0x302f },
	{ 0x3031, 0x3035 },
	{ 0x3038, 0x303c },
	{ 0x3041, 0x3096 },
	{ 0x3099, 0x309f },
	{ 0x30a1, 0x30fa },
	{ 0x30fc, 0x30ff },
	{ 0x3105, 0x312f },
	{ 0x3131, 0x318e },
	{ 0x31a0, 0x31bf },
	{ 0x31f0, 0x31ff },
	{ 0x3400, 0x4dbf },
	{ 0x4e00, 0xa48c },
	{ 0xa4d0, 0xa4fd },
	{ 0xa500, 0xa60c },
	{ 0xa610, 0xa62b },
	{ 0xa640, 0xa66f },
	{ 0xa674, 0xa67d },
	{ 0xa67f, 0xa6f1 },
	{ 0xa717, 0xa71f },
	{ 0xa722, 0xa788 },
	{ 0xa78b, 0xa7ca },
	{ 0xa7d0, 0xa7d1 },
	{ 0xa7d3, 0xa7d3 },
	{ 0xa7d5, 0xa7d9 },
	{ 0xa7f2, 0xa827 },
	{ 0xa82c, 0xa82c },
	{ 0xa840, 0xa873 },
	{ 0xa880, 0xa8c5 },
	{ 0xa8d0, 0xa8d9 },
	{ 0xa8e0, 0xa8f7 },
	{ 0xa8fb, 0xa8fb },
	{ 0xa8fd, 0xa92d },
	{ 0xa930, 0xa953 },
	{ 0xa960, 0xa97c },
	{ 0xa980, 0xa9c0 },
	{ 0xa9cf, 0xa9d9 },
	{ 0xa9e0, 0xa9fe },
	{ 0xaa00, 0xaa36 },
	{ 0xaa40, 0xaa4d },
	{ 0xaa50, 0xaa59 },
	{ 0xaa60, 0xaa76 },
	{ 0xaa7a, 0xaac2 },
	{ 0xaadb, 0xaadd },
	{ 0xaae0, 0xaaef },
	{ 0xaaf2, 0xaaf6 },
	{ 0xab01, 0xab06 },
	{ 0xab09, 0xab0e },
	{ 0xab11, 0xab16 },
	{ 0xab20, 0xab26 },
	{ 0xab28, 0xab2e },
	{ 0xab30, 0xab5a },
	{ 0xab5c, 0xab69 },
	{ 0xab70, 0xabea },
	{ 0xabec, 0xabed },
	{ 0xabf0, 0xabf9 },
	{ 0xac00, 0xd7a3 },
	{ 0xd7b0, 0xd7c6 },
	{ 0xd7cb, 0xd7fb },
	{ 0xf900, 0xfa6d },
	{ 0xfa70, 0xfad9 },
	{ 0xfb00, 0xfb06 },
	{ 0xfb13, 0xfb17 },
	{ 0xfb1d, 0xfb28 },
	{ 0xfb2a, 0xfb36 },
	{ 0xfb38, 0xfb3c },
	{ 0xfb3e, 0xfb3e },
	{ 0xfb40, 0xfb41 },
	{ 0xfb43, 0xfb44 },
	{ 0xfb46, 0xfbb1 },
	{ 0xfbd3, 0xfd3d },
	{ 0xfd50, 0xfd8f },
	{ 0xfd92, 0xfdc7 },
	{ 0xfdf0, 0xfdfb },
	{ 0xfe00, 0xfe0f },
	{ 0xfe20, 0xfe2f },
	{ 0xfe33, 0xfe34 },
	{ 0xfe4d, 0xfe4f },
	{ 0xfe70, 0xfe74 },
	{ 0xfe76, 0xfefc },
	{ 0xff10, 0xff19 },
	{ 0xff21, 0xff3a },
	{ 0xff3f, 0xff3f },
	{ 0xff41, 0xff5a },
	{ 0xff66, 0xffbe },
	{ 0xffc2, 0xffc7 },
	{ 0xffca, 0xffcf },
	{ 0xffd2, 0xffd7 },
	{ 0xffda, 0xffdc },
	{ 0x10000, 0x1000b },
	{ 0x1000d, 0x10026 },
	{ 0x10028, 0x1003a },
	{ 0x1003c, 0x1003d },
	{ 0x1003f, 0x1004d },
	{ 0x10050, 0x1005d },
	{ 0x10080, 0x100fa },
	{ 0x10140, 0x10174 },
	{ 0x101fd, 0x101fd },
	{ 0x10280, 0x1029c },
	{ 0x102a0, 0x102d0 },
	{ 0x102e0, 0x102e0 },
	{ 0x10300, 0x1031f },
	{ 0x1032d, 0x1034a },
	{ 0x10350, 0x1037a },
	{ 0x10380, 0x1039d },
	{ 0x103a0, 0x103c3 },
	{ 0x103c8, 0x103cf },
	{ 0x103d1, 0x103d5 },
	{ 0x10400, 0x1049d },
	{ 0x104a0, 0x104a9 },
	{ 0x104b0, 0x104d3 },
	{ 0x104d8, 0x104fb },
	{ 0x10500, 0x10527 },
	{ 0x10530, 0x10563 },
	{ 0x10570, 0x1057a },
	{ 0x1057c, 0x1058a },
	{ 0x1058c, 0x10592 },
	{ 0x10594, 0x10595 },
	{ 0x10597, 0x105a1 },
	{ 0x105a3, 0x105b1 },
	{ 0x105b3, 0x105b9 },
	{ 0x105bb, 0x105bc },
	{ 0x10600, 0x10736 },
	{ 0x10740, 0x10755 },
	{ 0x10760, 0x10767 },
	{ 0x10780, 0x10785 },
	{ 0x10787, 0x107b0 },
	{ 0x107b2, 0x107ba },
	{ 0x10800, 0x10805 },
	{ 0x10808, 0x10808 },
	{ 0x1080a, 0x10835 },
	{ 0x10837, 0x10838 },
	{ 0x1083c, 0x1083c },
	{ 0x1083f, 0x10855 },
	{ 0x10860, 0x10876 },
	{ 0x10880, 0x1089e },
	{ 0x108e0, 0x108f2 },
	{ 0x108f4, 0x108f5 },
	{ 0x10900, 0x10915 },
	{ 0x10920, 0x10939 },
	{ 0x10980, 0x109b7 },
	{ 0x109be, 0x109bf },
	{ 0x10a00, 0x10a03 },
	{ 0x10a05, 0x10a06 },
	{ 0x10a0c, 0x10a13 },
	{ 0x10a15, 0x10a17 },
	{ 0x10a19, 0x10a35 },
	{ 0x10a38, 0x10a3a },
	{ 0x10a3f, 0x10a3f },
	{ 0x10a60, 0x10a7c },
	{ 0x10a80, 0x10a9c },
	{ 0x10ac0, 0x10ac7 },
	{ 0x10ac9, 0x10ae6 },
	{ 0x10b00, 0x10b35 },
	{ 0x10b40, 0x10b55 },
	{ 0x10b60, 0x10b72 },
	{ 0x10b80, 0x10b91 },
	{ 0x10c00, 0x10c48 },
	{ 0x10c80, 0x10cb2 },
	{ 0x10cc0, 0x10cf2 },
	{ 0x10d00, 0x10d27 },
	{ 0x10d30, 0x10d39 },
	{ 0x10e80, 0x10ea9 },
	{ 0x10eab, 0x10eac },
	{ 0x10eb0, 0x10eb1 },
	{ 0x10f00, 0x10f1c },
	{ 0x10f27, 0x10f27 },
	{ 0x10f30, 0x10f50 },
	{ 0x10f70, 0x10f85 },
	{ 0x10fb0, 0x10fc4 },
	{ 0x10fe0, 0x10ff6 },
	{ 0x11000, 0x11046 },
	{ 0x11066, 0x11075 },
	{ 0x1107f, 0x110ba },
	{ 0x110c2, 0x110c2 },
	{ 0x110d0, 0x110e8 },
	{ 0x110f0, 0x110f9 },
	{ 0x11100, 0x11134 },
	{ 0x11136, 0x1113f },
	{ 0x11144, 0x11147 },
	{ 0x11150, 0x11173 },
	{ 0x11176, 0x11176 },
	{ 0x11180, 0x111c4 },
	{ 0x111c9, 0x111cc },
	{ 0x111ce, 0x111da },
	{ 0x111dc, 0x111dc },
	{ 0x11200, 0x11211 },
	{ 0x11213, 0x11237 },
	{ 0x1123e, 0x1123e },
	{ 0x11280, 0x11286 },
	{ 0x11288, 0x11288 },
	{ 0x1128a, 0x1128d },
	{ 0x1128f, 0x1129d },
	{ 0x1129f, 0x112a8 },
	{ 0x112b0, 0x112ea },
	{ 0x112f0, 0x112f9 },
	{ 0x11300, 0x11303 },
	{ 0x11305, 0x1130c },
	{ 0x1130f, 0x11310 },
	{ 0x11313, 0x11328 },
	{ 0x1132a, 0x11330 },
	{ 0x11332, 0x11333 },
	{ 0x11335, 0x11339 },
	{ 0x1133b, 0x11344 },
	{ 0x11347, 0x11348 },
	{ 0x1134b, 0x1134d },
	{ 0x11350, 0x11350 },
	{ 0x11357, 0x11357 },
	{ 0x1135d, 0x11363 },
	{ 0x11366, 0x1136c },
	{ 0x11370, 0x11374 },
	{ 0x11400, 0x1144a },
	{ 0x11450, 0x11459 },
	{ 0x1145e, 0x11461 },
	{ 0x11480, 0x114c5 },
	{ 0x114c7, 0x114c7 },
	{ 0x114d0, 0x114d9 },
	{ 0x11580, 0x115b5 },
	{ 0x115b8, 0x115c0 },
	{ 0x115d8, 0x115dd },
	{ 0x11600, 0x11640 },
	{ 0x11644, 0x11644 },
	{ 0x11650, 0x11659 },
	{ 0x11680, 0x116b8 },
	{ 0x116c0, 0x116c9 },
	{ 0x11700, 0x1171a },
	{ 0x1171d, 0x1172b },
	{ 0x11730, 0x11739 },
	{ 0x11740, 0x11746 },
	{ 0x11800, 0x1183a },
	{ 0x118a0, 0x118e9 },
	{ 0x118ff, 0x11906 },
	{ 0x11909, 0x11909 },
	{ 0x1190c, 0x11913 },
	{ 0x11915, 0x11916 },
	{ 0x11918, 0x11935 },
	{ 0x11937, 0x11938 },
	{ 0x1193b, 0x11943 },
	{ 0x11950, 0x11959 },
	{ 0x119a0, 0x119a7 },
	{ 0x119aa, 0x119d7 },
	{ 0x119da, 0x119e1 },
	{ 0x119e3, 0x119e4 },
	{ 0x11a00, 0x11a3e },
	{ 0x11a47, 0x11a47 },
	{ 0x11a50, 0x11a99 },
	{ 0x11a9d, 0x11a9d },
	{ 0x11ab0, 0x11af8 },
	{ 0x11c00, 0x11c08 },
	{ 0x11c0a, 0x11c36 },
	{ 0x11c38, 0x11c40 },
	{ 0x11c50, 0x11c59 },
	{ 0x11c72, 0x11c8f },
	{ 0x11c92, 0x11ca7 },
	{ 0x11ca9, 0x11cb6 },
	{ 0x11d00, 0x11d06 },
	{ 0x11d08, 0x11d09 },
	{ 0x11d0b, 0x11d36 },
	{ 0x11d3a, 0x11d3a },
	{ 0x11d3c, 0x11d3d },
	{ 0x11d3f, 0x11d47 },
	{ 0x11d50, 0x11d59 },
	{ 0x11d60, 0x11d65 },
	{ 0x11d67, 0x11d68 },
	{ 0x11d6a, 0x11d8e },
	{ 0x11d90, 0x11d91 },
	{ 0x11d93, 0x11d98 },
	{ 0x11da0, 0x11da9 },
	{ 0x11ee0, 0x11ef6 },
	{ 0x11fb0, 0x11fb0 },
	{ 0x12000, 0x12399 },
	{ 0x12400, 0x1246e },
	{ 0x12480, 0x12543 },
	{ 0x12f90, 0x12ff0 },
	{ 0x13000, 0x1342e },
	{ 0x14400, 0x14646 },
	{ 0x16800, 0x16a38 },
	{ 0x16a40, 0x16a5e },
	{ 0x16a60, 0x16a69 },
	{ 0x16a70, 0x16abe },
	{ 0x16ac0, 0x16ac9 },
	{ 0x16ad0, 0x16aed },
	{ 0x16af0, 0x16af4 },
	{ 0x16b00, 0x16b36 },
	{ 0x16b40, 0x16b43 },
	{ 0x16b50, 0x16b59 },
	{ 0x16b63, 0x16b77 },
	{ 0x16b7d, 0x16b8f },
	{ 0x16e40, 0x16e7f },
	{ 0x16f00, 0x16f4a },
	{ 0x16f4f, 0x16f87 },
	{ 0x16f8f, 0x16f9f },
	{ 0x16fe0, 0x16fe1 },
	{ 0x16fe3, 0x16fe4 },
	{ 0x16ff0, 0x16ff1 },
	{ 0x17000, 0x187f7 },
	{ 0x18800, 0x18cd5 },
	{ 0x18d00, 0x18d08 },
	{ 0x1aff0, 0x1aff3 },
	{ 0x1aff5, 0x1affb },
	{ 0x1affd, 0x1affe },
	{ 0x1b000, 0x1b122 },
	{ 0x1b150, 0x1b152 },
	{ 0x1b164, 0x1b167 },
	{ 0x1b170, 0x1b2fb },
	{ 0x1bc00, 0x1bc6a },
	{ 0x1bc70, 0x1bc7c },
	{ 0x1bc80, 0x1bc88 },
	{ 0x1bc90, 0x1bc99 },
	{ 0x1bc9d, 0x1bc9e },
	{ 0x1cf00, 0x1cf2d },
	{ 0x1cf30, 0x1cf46 },
	{ 0x1d165, 0x1d169 },
	{ 0x1d16d, 0x1d172 },
	{ 0x1d17b, 0x1d182 },
	{ 0x1d185, 0x1d18b },
	{ 0x1d1aa, 0x1d1ad },
	{ 0x1d242, 0x1d244 },
	{ 0x1d400, 0x1d454 },
	{ 0x1d456, 0x1d49c },
	{ 0x1d49e, 0x1d49f },
	{ 0x1d4a2, 0x1d4a2 },
	{ 0x1d4a5, 0x1d4a6 },
	{ 0x1d4a9, 0x1d4ac },
	{ 0x1d4ae, 0x1d4b9 },
	{ 0x1d4bb, 0x1d4bb },
	{ 0x1d4bd, 0x1d4c3 },
	{ 0x1d4c5, 0x1d505 },
	{ 0x1d507, 0x1d50a },
	{ 0x1d50d, 0x1d514 },
	{ 0x1d516, 0x1d51c },
	{ 0x1d51e, 0x1d539 },
	{ 0x1d53b, 0x1d53e },
	{ 0x1d540, 0x1d544 },
	{ 0x1d546, 0x1d546 },
	{ 0x1d54a, 0x1d550 },
	{ 0x1d552, 0x1d6a5 },
	{ 0x1d6a8, 0x1d6c0 },
	{ 0x1d6c2, 0x1d6da },
	{ 0x1d6dc, 0x1d6fa },
	{ 0x1d6fc, 0x1d714 },
	{ 0x1d716, 0x1d734 },
	{ 0x1d736, 0x1d74e },
	{ 0x1d750, 0x1d76e },
	{ 0x1d770, 0x1d788 },
	{ 0x1d78a, 0x1d7a8 },
	{ 0x1d7aa, 0x1d7c2 },
	{ 0x1d7c4, 0x1d7cb },
	{ 0x1d7ce, 0x1d7ff },
	{ 0x1da00, 0x1da36 },
	{ 0x1da3b, 0x1da6c },
	{ 0x1da75, 0x1da75 },
	{ 0x1da84, 0x1da84 },
	{ 0x1da9b, 0x1da9f },
	{ 0x1daa1, 0x1daaf },
	{ 0x1df00, 0x1df1e },
	{ 0x1e000, 0x1e006 },
	{ 0x1e008, 0x1e018 },
	{ 0x1e01b, 0x1e021 },
	{ 0x1e023, 0x1e024 },
	{ 0x1e026, 0x1e02a },
	{ 0x1e100, 0x1e12c },
	{ 0x1e130, 0x1e13d },
	{ 0x1e140, 0x1e149 },
	{ 0x1e14e, 0x1e14e },
	{ 0x1e290, 0x1e2ae },
	{ 0x1e2c0, 0x1e2f9 },
	{ 0x1e7e0, 0x1e7e6 },
	{ 0x1e7e8, 0x1e7eb },
	{ 0x1e7ed, 0x1e7ee },
	{ 0x1e7f0, 0x1e7fe },
	{ 0x1e800, 0x1e8c4 },
	{ 0x1e8d0, 0x1e8d6 },
	{ 0x1e900, 0x1e94b },
	{ 0x1e950, 0x1e959 },
	{ 0x1ee00, 0x1ee03 },
	{ 0x1ee05, 0x1ee1f },
	{ 0x1ee21, 0x1ee22 },
	{ 0x1ee24, 0x1ee24 },
	{ 0x1ee27, 0x1ee27 },
	{ 0x1ee29, 0x1ee32 },
	{ 0x1ee34, 0x1ee37 },
	{ 0x1ee39, 0x1ee39 },
	{ 0x1ee3b, 0x1ee3b },
	{ 0x1ee42, 0x1ee42 },
	{ 0x1ee47, 0x1ee47 },
	{ 0x1ee49, 0x1ee49 },
	{ 0x1ee4b, 0x1ee4b },
	{ 0x1ee4d, 0x1ee4f },
	{ 0x1ee51, 0x1ee52 },
	{ 0x1ee54, 0x1ee54 },
	{ 0x1ee57, 0x1ee57 },
	{ 0x1ee59, 0x1ee59 },
	{ 0x1ee5b, 0x1ee5b },
	{ 0x1ee5d, 0x1ee5d },
	{ 0x1ee5f, 0x1ee5f },
	{ 0x1ee61, 0x1ee62 },
	{ 0x1ee64, 0x1ee64 },
	{ 0x1ee67, 0x1ee6a },
	{ 0x1ee6c, 0x1ee72 },
	{ 0x1ee74, 0x1ee77 },
	{ 0x1ee79, 0x1ee7c },
	{ 0x1ee7e, 0x1ee7e },
	{ 0x1ee80, 0x1ee89 },
	{ 0x1ee8b, 0x1ee9b },
	{ 0x1eea1, 0x1eea3 },
	{ 0x1eea5, 0x1eea9 },
	{ 0x1eeab, 0x1eebb },
	{ 0x1fbf0, 0x1fbf9 },
	{ 0x20000, 0x2a6df },
	{ 0x2a700, 0x2b738 },
	{ 0x2b740, 0x2b81d },
	{ 0x2b820, 0x2cea1 },
	{ 0x2ceb0, 0x2ebe0 },
	{ 0x2f800, 0x2fa1d },
	{ 0x30000, 0x3134a },
	{ 0xe0100, 0xe01ef },
	{ 0x0, 0x0 },
};

#line 0

#line 1 "sfwl/core/char_utils.h"
/*************************************************************************/
/*  char_utils.h                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

static _FORCE_INLINE_ bool is_unicode_identifier_start(char32_t c) {
	for (int i = 0; xid_start[i].start != 0; i++) {
		if (c >= xid_start[i].start && c <= xid_start[i].end) {
			return true;
		}
	}
	return false;
}

static _FORCE_INLINE_ bool is_unicode_identifier_continue(char32_t c) {
	for (int i = 0; xid_continue[i].start != 0; i++) {
		if (c >= xid_continue[i].start && c <= xid_continue[i].end) {
			return true;
		}
	}
	return false;
}

static _FORCE_INLINE_ bool is_ascii_upper_case(char32_t c) {
	return (c >= 'A' && c <= 'Z');
}

static _FORCE_INLINE_ bool is_ascii_lower_case(char32_t c) {
	return (c >= 'a' && c <= 'z');
}

static _FORCE_INLINE_ bool is_digit(char32_t c) {
	return (c >= '0' && c <= '9');
}

static _FORCE_INLINE_ bool is_hex_digit(char32_t c) {
	return (is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
}

static _FORCE_INLINE_ bool is_binary_digit(char32_t c) {
	return (c == '0' || c == '1');
}

static _FORCE_INLINE_ bool is_ascii_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static _FORCE_INLINE_ bool is_ascii_alphanumeric_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

static _FORCE_INLINE_ bool is_ascii_identifier_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static _FORCE_INLINE_ bool is_symbol(char32_t c) {
	return c != '_' && ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~') || c == '\t' || c == ' ');
}

static _FORCE_INLINE_ bool is_control(char32_t p_char) {
	return (p_char <= 0x001f) || (p_char >= 0x007f && p_char <= 0x009f);
}

static _FORCE_INLINE_ bool is_whitespace(char32_t p_char) {
	return (p_char == ' ') || (p_char == 0x00a0) || (p_char == 0x1680) || (p_char >= 0x2000 && p_char <= 0x200a) || (p_char == 0x202f) || (p_char == 0x205f) || (p_char == 0x3000) || (p_char == 0x2028) || (p_char == 0x2029) || (p_char >= 0x0009 && p_char <= 0x000d) || (p_char == 0x0085);
}

static _FORCE_INLINE_ bool is_linebreak(char32_t p_char) {
	return (p_char >= 0x000a && p_char <= 0x000d) || (p_char == 0x0085) || (p_char == 0x2028) || (p_char == 0x2029);
}

static _FORCE_INLINE_ bool is_punct(char32_t p_char) {
	return (p_char >= ' ' && p_char <= '/') || (p_char >= ':' && p_char <= '@') || (p_char >= '[' && p_char <= '^') || (p_char == '`') || (p_char >= '{' && p_char <= '~') || (p_char >= 0x2000 && p_char <= 0x206f) || (p_char >= 0x3000 && p_char <= 0x303f);
}

static _FORCE_INLINE_ bool is_underscore(char32_t p_char) {
	return (p_char == '_');
}

#line 0

#line 1 "sfwl/core/ustring.h"

/*************************************************************************/
/*  ustring.h                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/*************************************************************************/
/*  CharProxy                                                            */
/*************************************************************************/

template <class T>
class CharProxy {
	friend class Char16String;
	friend class CharString;
	friend class String;

	const int _index;
	CowData<T> &_cowdata;
	static const T _null = 0;

	_FORCE_INLINE_ CharProxy(const int &p_index, CowData<T> &p_cowdata) :
			_index(p_index),
			_cowdata(p_cowdata) {}

public:
	_FORCE_INLINE_ CharProxy(const CharProxy<T> &p_other) :
			_index(p_other._index),
			_cowdata(p_other._cowdata) {}

	_FORCE_INLINE_ operator T() const {
		if (unlikely(_cowdata.size() == 0)) {
			return _null;
		}

		return _cowdata.get(_index);
	}

	_FORCE_INLINE_ const T *operator&() const {
		return _cowdata.ptr() + _index;
	}

	_FORCE_INLINE_ void operator=(const T &p_other) const {
		_cowdata.set(_index, p_other);
	}

	_FORCE_INLINE_ void operator=(const CharProxy<T> &p_other) const {
		_cowdata.set(_index, p_other.operator T());
	}
};

/*************************************************************************/
/*  Char16String                                                         */
/*************************************************************************/

class Char16String {
	CowData<char16_t> _cowdata;
	static const char16_t _null;

public:
	_FORCE_INLINE_ char16_t *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const char16_t *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	Error resize(int p_size) {
		Error err = _cowdata.resize(p_size);

		//Ensure null terminator
		if (err == OK && _cowdata.size() > 0) {
			_cowdata.set(_cowdata.size() - 1, '\0');
		}

		return err;
	}

	_FORCE_INLINE_ char16_t get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const char16_t &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ const char16_t &operator[](int p_index) const {
		if (unlikely(_cowdata.size() == 0)) {
			return _null;
		}

		return _cowdata.get(p_index);
	}
	_FORCE_INLINE_ CharProxy<char16_t> operator[](int p_index) { return CharProxy<char16_t>(p_index, _cowdata); }

	_FORCE_INLINE_ Char16String() {}
	_FORCE_INLINE_ Char16String(const Char16String &p_str) { _cowdata._ref(p_str._cowdata); }
	_FORCE_INLINE_ void operator=(const Char16String &p_str) { _cowdata._ref(p_str._cowdata); }
	_FORCE_INLINE_ Char16String(const char16_t *p_cstr) { copy_from(p_cstr); }

	void operator=(const char16_t *p_cstr);
	bool operator<(const Char16String &p_right) const;
	Char16String &operator+=(char16_t p_char);
	int length() const { return size() ? size() - 1 : 0; }
	const char16_t *get_data() const;
	operator const char16_t *() const { return get_data(); };

protected:
	void copy_from(const char16_t *p_cstr);
};

/*************************************************************************/
/*  CharString                                                           */
/*************************************************************************/

class CharString {
	CowData<char> _cowdata;
	static const char _null;

public:
	_FORCE_INLINE_ char *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const char *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	Error resize(int p_size) {
		Error err = _cowdata.resize(p_size);

		//Ensure null terminator
		if (err == OK && _cowdata.size() > 0) {
			_cowdata.set(_cowdata.size() - 1, '\0');
		}

		return err;
	}

	_FORCE_INLINE_ char get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const char &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ const char &operator[](int p_index) const {
		if (unlikely(_cowdata.size() == 0)) {
			return _null;
		}

		return _cowdata.get(p_index);
	}
	_FORCE_INLINE_ CharProxy<char> operator[](int p_index) { return CharProxy<char>(p_index, _cowdata); }

	_FORCE_INLINE_ CharString() {}
	_FORCE_INLINE_ CharString(const CharString &p_str) { _cowdata._ref(p_str._cowdata); }
	_FORCE_INLINE_ void operator=(const CharString &p_str) { _cowdata._ref(p_str._cowdata); }
	_FORCE_INLINE_ CharString(const char *p_cstr) { copy_from(p_cstr); }

	void operator=(const char *p_cstr);
	bool operator<(const CharString &p_right) const;
	CharString &operator+=(char p_char);
	int length() const { return size() ? size() - 1 : 0; }
	const char *get_data() const;
	operator const char *() const { return get_data(); };

protected:
	void copy_from(const char *p_cstr);
};

/*************************************************************************/
/*  String                                                               */
/*************************************************************************/

typedef char32_t CharType;

struct StrRange {
	const CharType *c_str;
	int len;

	StrRange(const CharType *p_c_str = nullptr, int p_len = 0) {
		c_str = p_c_str;
		len = p_len;
	}
};

class String {
public:
	enum {
		npos = -1 ///<for "some" compatibility with std::string (npos is a huge value in std::string)
	};

	_FORCE_INLINE_ CharType *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const CharType *ptr() const { return _cowdata.ptr(); }

	_FORCE_INLINE_ void remove(int p_index) {
		_cowdata.remove(p_index);

		//Ensure null terminator
		if (_cowdata.size() > 0) {
			_cowdata.set(_cowdata.size() - 1, '\0');
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }

	_FORCE_INLINE_ CharType get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const CharType &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	_FORCE_INLINE_ Error resize(int p_size) {
		Error err = _cowdata.resize(p_size);

		//Ensure null terminator
		if (err == OK && _cowdata.size() > 0) {
			_cowdata.set(_cowdata.size() - 1, '\0');
		}

		return err;
	}
	_FORCE_INLINE_ Error set_length(int p_length) { return resize(p_length + 1); }

	_FORCE_INLINE_ const CharType &operator[](int p_index) const {
		return _cowdata.get(p_index);
	}
	_FORCE_INLINE_ CharProxy<CharType> operator[](int p_index) { return CharProxy<CharType>(p_index, _cowdata); }

	bool operator==(const String &p_str) const;
	bool operator!=(const String &p_str) const;
	String operator+(const String &p_str) const;
	//String operator+(CharType p_char) const;

	String &operator+=(const String &);
	String &operator+=(CharType p_char);
	String &operator+=(const char *p_str);
	String &operator+=(const wchar_t *p_str);
	String &operator+=(const CharType *p_str);

	/* Compatibility Operators */

	void operator=(const char *p_str);
	void operator=(const wchar_t *p_str);
	void operator=(const CharType *p_str);

	bool operator==(const char *p_str) const;
	bool operator==(const wchar_t *p_str) const;
	bool operator==(const CharType *p_str) const;
	bool operator==(const StrRange &p_str_range) const;

	bool operator!=(const char *p_str) const;
	bool operator!=(const wchar_t *p_str) const;
	bool operator!=(const CharType *p_str) const;

	bool operator<(const CharType *p_str) const;
	bool operator<(const char *p_str) const;
	bool operator<(const wchar_t *p_str) const;

	bool operator<(const String &p_str) const;
	bool operator<=(const String &p_str) const;
	bool operator>(const String &p_str) const;
	bool operator>=(const String &p_str) const;

	signed char casecmp_to(const String &p_str) const;
	signed char nocasecmp_to(const String &p_str) const;
	signed char naturalnocasecmp_to(const String &p_str) const;

	const CharType *get_data() const;

	/* standard size stuff */

	_FORCE_INLINE_ int length() const {
		int s = size();
		return s ? (s - 1) : 0; // length does not include zero
	}

	bool is_valid_string() const;

	/* debug, error messages */
	void print_unicode_error(const String &p_message, bool p_critical = false) const;

	/* complex helpers */
	String substr(int p_from, int p_chars = -1) const;
	String substr_index(const int p_start_index, const int p_end_index) const; //end_index is not included

	int find(const String &p_str, int p_from = 0) const; ///< return <0 if failed
	int find(const char *p_str, int p_from = 0) const; ///< return <0 if failed
	int find_char(const CharType &p_char, int p_from = 0) const; ///< return <0 if failed
	int find_last(const String &p_str) const; ///< return <0 if failed
	int findn(const String &p_str, int p_from = 0) const; ///< return <0 if failed, case insensitive
	int rfind(const String &p_str, int p_from = -1) const; ///< return <0 if failed
	int rfindn(const String &p_str, int p_from = -1) const; ///< return <0 if failed, case insensitive
	int findmk(const Vector<String> &p_keys, int p_from = 0, int *r_key = nullptr) const; ///< return <0 if failed
	int find_first_difference_index(const String &p_str) const;

	bool is_word_at(const int p_index, const char *p_str) const;
	bool is_word_at(const int p_index, const String &p_str) const;

	bool match(const String &p_wildcard) const;
	bool matchn(const String &p_wildcard) const;

	bool begins_with(const String &p_string) const;
	bool begins_with(const char *p_string) const;
	bool ends_with(const String &p_string) const;

	bool is_enclosed_in(const String &p_string) const;
	bool is_subsequence_of(const String &p_string) const;
	bool is_subsequence_ofi(const String &p_string) const;
	bool is_quoted() const;

	Vector<String> bigrams() const;
	float similarity(const String &p_string) const;
	//String format(const Variant &values, String placeholder = "{_}") const;

	String replace_first(const String &p_key, const String &p_with) const;
	String replace(const String &p_key, const String &p_with) const;
	String replace(const char *p_key, const char *p_with) const;
	String replacen(const String &p_key, const String &p_with) const;

	String newline_to_br() const;

	String repeat(int p_count) const;
	String insert(int p_at_pos, const String &p_string) const;

	String pad_decimals(int p_digits) const;
	String pad_zeros(int p_digits) const;

	String trim_prefix(const String &p_prefix) const;
	String trim_suffix(const String &p_suffix) const;

	String lpad(int min_length, const String &character = " ") const;
	String rpad(int min_length, const String &character = " ") const;

	//Moved this to Array to simplify the library's design
	//String sprintf(const Array &values, bool *error) const;

	String quote(String quotechar = "\"") const;
	String unquote() const;

	static String num(double p_num, int p_decimals = -1);
	static String num_scientific(double p_num);
	static String num_real(double p_num);
	static String num_int64(int64_t p_num, int base = 10, bool capitalize_hex = false);
	static String num_uint64(uint64_t p_num, int base = 10, bool capitalize_hex = false);

	static String chr(CharType p_char);

	static String md5(const uint8_t *p_md5);

	static String hex_encode_buffer(const uint8_t *p_buffer, int p_len);

	static String bool_num(bool p_val);
	static String bool_str(bool p_val);

	bool is_numeric() const;
	bool is_zero() const;

	double to_double() const;
	float to_float() const;
	int to_int() const;
	bool to_bool() const;
	uint32_t to_uint() const;

	int hex_to_int(bool p_with_prefix = true) const;
	int64_t hex_to_int64(bool p_with_prefix = true) const;
	int64_t bin_to_int64(bool p_with_prefix = true) const;
	int64_t to_int64() const;

	static int64_t to_int(const char *p_str, int p_len = -1);
	static int64_t to_int(const wchar_t *p_str, int p_len = -1);
	static int64_t to_int(const CharType *p_str, int p_len = -1, bool p_clamp = false);

	static double to_float(const char *p_str);
	static double to_float(const wchar_t *p_str, const wchar_t **r_end = nullptr);
	static double to_float(const CharType *p_str, const CharType **r_end = nullptr);

	static double to_double(const char *p_str);
	static double to_double(const wchar_t *p_str, const wchar_t **r_end = nullptr);
	static double to_double(const CharType *p_str, const CharType **r_end = nullptr);

	static uint32_t num_characters(int64_t p_int);

	String capitalize() const;
	String camelcase_to_underscore(bool lowercase = true) const;

	int get_slice_count(String p_splitter) const;
	String get_slice(String p_splitter, int p_slice) const;
	String get_slicec(CharType p_splitter, int p_slice) const;

	Vector<String> split(const String &p_splitter, bool p_allow_empty = true, int p_maxsplit = 0) const;
	Vector<String> rsplit(const String &p_splitter, bool p_allow_empty = true, int p_maxsplit = 0) const;
	Vector<String> split_spaces() const;
	Vector<float> split_floats(const String &p_splitter, bool p_allow_empty = true) const;
	Vector<float> split_floats_mk(const Vector<String> &p_splitters, bool p_allow_empty = true) const;
	Vector<int> split_ints(const String &p_splitter, bool p_allow_empty = true) const;
	Vector<int> split_ints_mk(const Vector<String> &p_splitters, bool p_allow_empty = true) const;

	String join(const Vector<String> &parts) const;

	static CharType char_uppercase(CharType p_char);
	static CharType char_lowercase(CharType p_char);
	String to_upper() const;
	String to_lower() const;

	int count(const String &p_string, int p_from = 0, int p_to = 0) const;
	int countn(const String &p_string, int p_from = 0, int p_to = 0) const;

	String left(int p_pos) const;
	String right(int p_pos) const;
	String indent(const String &p_prefix) const;
	String dedent() const;

	String strip_edges(bool left = true, bool right = true) const;
	String strip_escapes() const;
	String lstrip(const String &p_chars) const;
	String rstrip(const String &p_chars) const;

	String get_extension() const;
	String get_basename() const;
	String plus_file(const String &p_file) const;

	CharType unicode_at(int p_idx) const;
	CharType ord_at(int p_idx) const;

	void erase(int p_pos, int p_chars);

	CharString ascii(bool p_allow_extended = false) const;
	CharString utf8() const;
	Error parse_utf8(const char *p_utf8, int p_len = -1, bool p_skip_cr = false); //return true on error
	static String utf8(const char *p_utf8, int p_len = -1);
	int utf8_byte_length() const;

	Char16String utf16() const;
	Error parse_utf16(const char16_t *p_utf16, int p_len = -1);
	static String utf16(const char16_t *p_utf16, int p_len = -1);
	int utf16_byte_length() const;

	static uint32_t hash(const char *p_cstr); /* hash the string */
	static uint32_t hash(const char *p_cstr, int p_len); /* hash the string */
	static uint32_t hash(const wchar_t *p_cstr); /* hash the string */
	static uint32_t hash(const wchar_t *p_cstr, int p_len); /* hash the string */
	static uint32_t hash(const CharType *p_cstr); /* hash the string */
	static uint32_t hash(const CharType *p_cstr, int p_len); /* hash the string */

	uint32_t hash() const; /* hash the string */
	uint64_t hash64() const; /* hash the string */

	_FORCE_INLINE_ bool empty() const { return length() == 0; }
	_FORCE_INLINE_ bool contains(const char *p_str) const { return find(p_str) != -1; }
	_FORCE_INLINE_ bool contains(const String &p_str) const { return find(p_str) != -1; }

	// path functions
	bool is_abs_path() const;
	bool is_rel_path() const;
	bool is_resource_file() const;
	String path_to(const String &p_path) const;
	String path_to_file(const String &p_path) const;
	String get_base_dir() const;
	String get_file() const;
	static String humanize_size(uint64_t p_size);
	String simplify_path() const;
	bool is_network_share_path() const;
	String append_path(const char *p_path) const;
	String append_path(const String &p_path) const;
	String path_clean_end_slash() const;
	String path_ensure_end_slash() const;
	String path_get_prev_dir() const;

	String xml_escape(bool p_escape_quotes = false) const;
	String xml_unescape() const;
	String http_escape() const;
	String http_unescape() const;
	String uri_encode() const;
	String uri_decode() const;
	String c_escape() const;
	String c_escape_multiline() const;
	String c_unescape() const;
	String json_escape() const;
	String word_wrap(int p_chars_per_line) const;
	Error parse_url(String &r_scheme, String &r_host, int &r_port, String &r_path) const;

	String percent_encode() const;
	String percent_decode() const;

	String property_name_encode() const;

	// node functions
	static String get_invalid_node_name_characters();
	String validate_node_name() const;
	String validate_identifier() const;

	bool is_valid_identifier() const;
	bool is_valid_integer() const;
	bool is_valid_float() const;
	bool is_valid_hex_number(bool p_with_prefix) const;
	bool is_valid_html_color() const;
	bool is_valid_ip_address() const;
	bool is_valid_filename() const;
	bool is_valid_bool() const;
	bool is_valid_unsigned_integer() const;

	/**
	 * The constructors must not depend on other overloads
	 */

	/*	String(CharType p_char);*/
	_FORCE_INLINE_ String() {}
	_FORCE_INLINE_ String(const String &p_str) { _cowdata._ref(p_str._cowdata); }

	void operator=(const String &p_str) {
		_cowdata._ref(p_str._cowdata);
	}

	Vector<uint8_t> to_ascii_buffer() const;
	Vector<uint8_t> to_utf8_buffer() const;
	Vector<uint8_t> to_utf16_buffer() const;
	Vector<uint8_t> to_utf32_buffer() const;

	String(const char *p_str);
	String(const wchar_t *p_str);
	String(const CharType *p_str);
	String(const char *p_str, int p_clip_to_len);
	String(const wchar_t *p_str, int p_clip_to_len);
	String(const CharType *p_str, int p_clip_to_len);
	String(const StrRange &p_range);

private:
	CowData<CharType> _cowdata;
	static const CharType _null;

	void copy_from(const char *p_cstr);
	void copy_from(const char *p_cstr, const int p_clip_to);
	void copy_from(const wchar_t *p_cstr);
	void copy_from(const wchar_t *p_cstr, const int p_clip_to);
	void copy_from(const CharType *p_cstr);
	void copy_from(const CharType *p_cstr, const int p_clip_to);

	void copy_from(const CharType &p_char);

	void copy_from_unchecked(const CharType *p_char, const int p_length);

	bool _base_is_subsequence_of(const String &p_string, bool case_insensitive) const;
	int _count(const String &p_string, int p_from, int p_to, bool p_case_insensitive) const;
};

bool operator==(const char *p_chr, const String &p_str);
bool operator==(const wchar_t *p_chr, const String &p_str);
bool operator!=(const char *p_chr, const String &p_str);
bool operator!=(const wchar_t *p_chr, const String &p_str);

String operator+(const char *p_chr, const String &p_str);
String operator+(const wchar_t *p_chr, const String &p_str);
String operator+(CharType p_chr, const String &p_str);

String itos(int64_t p_val);
String uitos(uint64_t p_val);
String rtos(double p_val);
String rtoss(double p_val); //scientific version

struct NoCaseComparator {
	bool operator()(const String &p_a, const String &p_b) const {
		return p_a.nocasecmp_to(p_b) < 0;
	}
};

struct NaturalNoCaseComparator {
	bool operator()(const String &p_a, const String &p_b) const {
		return p_a.naturalnocasecmp_to(p_b) < 0;
	}
};

template <typename L, typename R>
_FORCE_INLINE_ bool is_str_less(const L *l_ptr, const R *r_ptr) {
	while (true) {
		const CharType l = *l_ptr;
		const CharType r = *r_ptr;

		if (l == 0 && r == 0) {
			return false;
		} else if (l == 0) {
			return true;
		} else if (r == 0) {
			return false;
		} else if (l < r) {
			return true;
		} else if (l > r) {
			return false;
		}

		l_ptr++;
		r_ptr++;
	}
}

/* end of namespace */

// Tool translate (TTR and variants) for the editor UI,
// and doc translate for the class reference (DTR).
#ifdef TOOLS_ENABLED
// Gets parsed.
String TTR(const String &p_text, const String &p_context = "");
String DTR(const String &);
// Use for C strings.
#define TTRC(m_value) (m_value)
// Use to avoid parsing (for use later with C strings).
#define TTRGET(m_value) TTR(m_value)

#else
#define TTR(m_value) (String())
#define DTR(m_value) (String())
#define TTRC(m_value) (m_value)
#define TTRGET(m_value) (m_value)
#endif

// Runtime translate for the public node API.
String RTR(const String &);

bool select_word(const String &p_s, int p_col, int &r_beg, int &r_end);

#line 0

#line 1 "sfwl/core/string_name.h"

/*************************************************************************/
/*  string_name.h                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

struct StaticCString {
	const char *ptr;
	static StaticCString create(const char *p_ptr);
};

class StringName {
	enum {
		STRING_TABLE_BITS = 14,
		STRING_TABLE_LEN = 1 << STRING_TABLE_BITS,
		STRING_TABLE_MASK = STRING_TABLE_LEN - 1
	};

	struct _Data {
		SafeRefCount refcount;
		SafeNumeric<uint32_t> static_count;
		const char *cname;
		String name;

#ifdef DEBUG_ENABLED
		uint32_t debug_references;
#endif

		String get_name() const {
			return cname ? String(cname) : name;
		}

		int idx;
		uint32_t hash;
		_Data *prev;
		_Data *next;

		_Data() {
#ifdef DEBUG_ENABLED
			debug_references = 0;
#endif
			cname = nullptr;
			prev = nullptr;
			next = nullptr;
			idx = 0;
			hash = 0;
		}
	};

	static _Data *_table[STRING_TABLE_LEN];

	_Data *_data;

	union _HashUnion {
		_Data *ptr;
		uint32_t hash;

		_HashUnion() {
			ptr = nullptr;
		}
	};

	void unref();
	friend void register_core_types();
	friend void unregister_core_types();

	static Mutex lock;
	static bool configured;
#ifdef DEBUG_ENABLED
	struct DebugSortReferences {
		bool operator()(const _Data *p_left, const _Data *p_right) const {
			return p_left->debug_references > p_right->debug_references;
		}
	};

	static bool debug_stringname;
#endif

	StringName(_Data *p_data) {
		_data = p_data;
	}

public:
	operator const void *() const {
		return (_data && (_data->cname || !_data->name.empty())) ? (void *)1 : nullptr;
	}

	bool operator==(const String &p_name) const;
	bool operator==(const char *p_name) const;
	bool operator!=(const String &p_name) const;

	_FORCE_INLINE_ bool is_node_unique_name() const {
		if (!_data) {
			return false;
		}

		if (_data->cname != nullptr) {
			return _data->cname[0] == '%';
		} else {
			return _data->name[0] == '%';
		}
	}

	_FORCE_INLINE_ bool operator<(const StringName &p_name) const {
		return _data < p_name._data;
	}
	_FORCE_INLINE_ bool operator==(const StringName &p_name) const {
		// the real magic of all this mess happens here.
		// this is why path comparisons are very fast
		return _data == p_name._data;
	}
	_FORCE_INLINE_ uint32_t hash() const {
		if (_data) {
			return _data->hash;
		} else {
			return 0;
		}
	}
	_FORCE_INLINE_ const void *data_unique_pointer() const {
		return (void *)_data;
	}
	bool operator!=(const StringName &p_name) const;

	_FORCE_INLINE_ operator String() const {
		if (_data) {
			if (_data->cname) {
				return String(_data->cname);
			} else {
				return _data->name;
			}
		}

		return String();
	}

	static StringName search(const char *p_name);
	static StringName search(const CharType *p_name);
	static StringName search(const String &p_name);

	struct AlphCompare {
		_FORCE_INLINE_ bool operator()(const StringName &l, const StringName &r) const {
			const char *l_cname = l._data ? l._data->cname : "";
			const char *r_cname = r._data ? r._data->cname : "";

			if (l_cname) {
				if (r_cname) {
					return is_str_less(l_cname, r_cname);
				} else {
					return is_str_less(l_cname, r._data->name.ptr());
				}
			} else {
				if (r_cname) {
					return is_str_less(l._data->name.ptr(), r_cname);
				} else {
					return is_str_less(l._data->name.ptr(), r._data->name.ptr());
				}
			}
		}
	};

	void operator=(const StringName &p_name);
	StringName(const char *p_name, bool p_static = false);
	StringName(const StringName &p_name);
	StringName(const String &p_name, bool p_static = false);
	StringName(const StaticCString &p_static_string, bool p_static = false);
	StringName();
	_FORCE_INLINE_ ~StringName() {
		if (likely(configured) && _data) { //only free if configured
			unref();
		}
	}

	static void setup();
	static void cleanup();
};

/*
bool operator==(const String &p_name, const StringName &p_string_name);
bool operator!=(const String &p_name, const StringName &p_string_name);
bool operator==(const char *p_name, const StringName &p_string_name);
bool operator!=(const char *p_name, const StringName &p_string_name);
*/

StringName _scs_create(const char *p_chr, bool p_static = false);

/*
 * The SNAME macro is used to speed up StringName creation, as it allows caching it after the first usage in a very efficient way.
 * It should NOT be used everywhere, but instead in places where high performance is required and the creation of a StringName
 * can be costly. Places where it should be used are:
 * - Control::get_theme_*(<name> and Window::get_theme_*(<name> functions.
 * - emit_signal(<name>,..) function
 * - call_deferred(<name>,..) function
 * - Comparisons to a StringName in overridden _set and _get methods.
 *
 * Use in places that can be called hundreds of times per frame (or more) is recommended, but this situation is very rare. If in doubt, do not use.
 */

//#define SNAME(m_arg) ([]() -> const StringName & { static StringName sname = _scs_create(m_arg, true); return sname; })()

#line 0

#line 1 "sfwl/core/color.h"

/*************************************************************************/
/*  color.h                                                              */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

struct _NO_DISCARD_CLASS_ Color {
	union {
		struct {
			float r;
			float g;
			float b;
			float a;
		};
		float components[4];
	};

	bool operator==(const Color &p_color) const { return (r == p_color.r && g == p_color.g && b == p_color.b && a == p_color.a); }
	bool operator!=(const Color &p_color) const { return (r != p_color.r || g != p_color.g || b != p_color.b || a != p_color.a); }

	uint32_t to_rgba32() const;
	uint32_t to_argb32() const;
	uint32_t to_abgr32() const;
	uint64_t to_rgba64() const;
	uint64_t to_argb64() const;
	uint64_t to_abgr64() const;
	float get_h() const;
	float get_s() const;
	float get_v() const;
	void set_hsv(float p_h, float p_s, float p_v, float p_alpha = 1.0);

	_FORCE_INLINE_ float &operator[](int idx) {
		return components[idx];
	}
	_FORCE_INLINE_ const float &operator[](int idx) const {
		return components[idx];
	}

	Color operator+(const Color &p_color) const;
	void operator+=(const Color &p_color);

	Color operator-() const;
	Color operator-(const Color &p_color) const;
	void operator-=(const Color &p_color);

	Color operator*(const Color &p_color) const;
	Color operator*(const real_t &rvalue) const;
	void operator*=(const Color &p_color);
	void operator*=(const real_t &rvalue);

	Color operator/(const Color &p_color) const;
	Color operator/(const real_t &rvalue) const;
	void operator/=(const Color &p_color);
	void operator/=(const real_t &rvalue);

	bool is_equal_approx(const Color &p_color) const;

	Color clamp(const Color &p_min = Color(0, 0, 0, 0), const Color &p_max = Color(1, 1, 1, 1)) const;

	void invert();
	void contrast();
	Color inverted() const;
	Color contrasted() const;

	_FORCE_INLINE_ float get_luminance() const {
		return 0.2126 * r + 0.7152 * g + 0.0722 * b;
	}

	_FORCE_INLINE_ Color linear_interpolate(const Color &p_to, float p_weight) const {
		Color res = *this;

		res.r += (p_weight * (p_to.r - r));
		res.g += (p_weight * (p_to.g - g));
		res.b += (p_weight * (p_to.b - b));
		res.a += (p_weight * (p_to.a - a));

		return res;
	}

	_FORCE_INLINE_ Color darkened(float p_amount) const {
		Color res = *this;
		res.r = res.r * (1.0f - p_amount);
		res.g = res.g * (1.0f - p_amount);
		res.b = res.b * (1.0f - p_amount);
		return res;
	}

	_FORCE_INLINE_ Color lightened(float p_amount) const {
		Color res = *this;
		res.r = res.r + (1.0f - res.r) * p_amount;
		res.g = res.g + (1.0f - res.g) * p_amount;
		res.b = res.b + (1.0f - res.b) * p_amount;
		return res;
	}

	_FORCE_INLINE_ uint32_t to_rgbe9995() const {
		const float pow2to9 = 512.0f;
		const float B = 15.0f;
		//const float Emax = 31.0f;
		const float N = 9.0f;

		float sharedexp = 65408.000f; //(( pow2to9  - 1.0f)/ pow2to9)*powf( 2.0f, 31.0f - 15.0f);

		float cRed = MAX(0.0f, MIN(sharedexp, r));
		float cGreen = MAX(0.0f, MIN(sharedexp, g));
		float cBlue = MAX(0.0f, MIN(sharedexp, b));

		float cMax = MAX(cRed, MAX(cGreen, cBlue));

		// expp = MAX(-B - 1, log2(maxc)) + 1 + B

		float expp = MAX(-B - 1.0f, floor(Math::log(cMax) / Math_LN2)) + 1.0f + B;

		float sMax = (float)floor((cMax / Math::pow(2.0f, expp - B - N)) + 0.5f);

		float exps = expp + 1.0f;

		if (0.0 <= sMax && sMax < pow2to9) {
			exps = expp;
		}

		float sRed = Math::floor((cRed / pow(2.0f, exps - B - N)) + 0.5f);
		float sGreen = Math::floor((cGreen / pow(2.0f, exps - B - N)) + 0.5f);
		float sBlue = Math::floor((cBlue / pow(2.0f, exps - B - N)) + 0.5f);

		return (uint32_t(Math::fast_ftoi(sRed)) & 0x1FF) | ((uint32_t(Math::fast_ftoi(sGreen)) & 0x1FF) << 9) | ((uint32_t(Math::fast_ftoi(sBlue)) & 0x1FF) << 18) | ((uint32_t(Math::fast_ftoi(exps)) & 0x1F) << 27);
	}

	_FORCE_INLINE_ Color blend(const Color &p_over) const {
		Color res;
		float sa = 1.0 - p_over.a;
		res.a = a * sa + p_over.a;
		if (res.a == 0) {
			return Color(0, 0, 0, 0);
		} else {
			res.r = (r * a * sa + p_over.r * p_over.a) / res.a;
			res.g = (g * a * sa + p_over.g * p_over.a) / res.a;
			res.b = (b * a * sa + p_over.b * p_over.a) / res.a;
		}
		return res;
	}

	_FORCE_INLINE_ Color to_linear() const {
		return Color(
				r < 0.04045 ? r * (1.0 / 12.92) : Math::pow((r + 0.055) * (1.0 / (1 + 0.055)), 2.4),
				g < 0.04045 ? g * (1.0 / 12.92) : Math::pow((g + 0.055) * (1.0 / (1 + 0.055)), 2.4),
				b < 0.04045 ? b * (1.0 / 12.92) : Math::pow((b + 0.055) * (1.0 / (1 + 0.055)), 2.4),
				a);
	}
	_FORCE_INLINE_ Color to_srgb() const {
		return Color(
				r < 0.0031308 ? 12.92 * r : (1.0 + 0.055) * Math::pow(r, 1.0f / 2.4f) - 0.055,
				g < 0.0031308 ? 12.92 * g : (1.0 + 0.055) * Math::pow(g, 1.0f / 2.4f) - 0.055,
				b < 0.0031308 ? 12.92 * b : (1.0 + 0.055) * Math::pow(b, 1.0f / 2.4f) - 0.055, a);
	}

	static Color hex(uint32_t p_hex);
	static Color hex64(uint64_t p_hex);
	static Color html(const String &p_color);
	static bool html_is_valid(const String &p_color);
	String to_html(bool p_alpha = true) const;
	Color from_hsv(float p_h, float p_s, float p_v, float p_a) const;
	static Color from_rgbe9995(uint32_t p_rgbe);

	_FORCE_INLINE_ bool operator<(const Color &p_color) const; //used in set keys
	operator String() const;

	static _FORCE_INLINE_ Color color8(int r, int g, int b) {
		return Color(static_cast<float>(r) / 255.0f, static_cast<float>(g) / 255.0f, static_cast<float>(b) / 255.0f);
	}

	static _FORCE_INLINE_ Color color8(int r, int g, int b, int a) {
		return Color(static_cast<float>(r) / 255.0f, static_cast<float>(g) / 255.0f, static_cast<float>(b) / 255.0f, static_cast<float>(a) / 255.0f);
	}

	_FORCE_INLINE_ void set_r8(int32_t r8) { r = (CLAMP(r8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_r8() const { return int32_t(CLAMP(Math::round(r * 255.0f), 0.0f, 255.0f)); }
	_FORCE_INLINE_ void set_g8(int32_t g8) { g = (CLAMP(g8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_g8() const { return int32_t(CLAMP(Math::round(g * 255.0f), 0.0f, 255.0f)); }
	_FORCE_INLINE_ void set_b8(int32_t b8) { b = (CLAMP(b8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_b8() const { return int32_t(CLAMP(Math::round(b * 255.0f), 0.0f, 255.0f)); }
	_FORCE_INLINE_ void set_a8(int32_t a8) { a = (CLAMP(a8, 0, 255) / 255.0f); }
	_FORCE_INLINE_ int32_t get_a8() const { return int32_t(CLAMP(Math::round(a * 255.0f), 0.0f, 255.0f)); }

	_FORCE_INLINE_ void set_h(float p_h) { set_hsv(p_h, get_s(), get_v(), a); }
	_FORCE_INLINE_ void set_s(float p_s) { set_hsv(get_h(), p_s, get_v(), a); }
	_FORCE_INLINE_ void set_v(float p_v) { set_hsv(get_h(), get_s(), p_v, a); }

	/**
	 * No construct parameters, r=0, g=0, b=0. a=255
	 */
	_FORCE_INLINE_ Color() {
		r = 0;
		g = 0;
		b = 0;
		a = 1.0;
	}

	/**
	 * RGB / RGBA construct parameters. Alpha is optional, but defaults to 1.0
	 */
	_FORCE_INLINE_ Color(float p_r, float p_g, float p_b, float p_a = 1.0) {
		r = p_r;
		g = p_g;
		b = p_b;
		a = p_a;
	}

	/**
	 * Construct a Color from another Color, but with the specified alpha value.
	 */
	_FORCE_INLINE_ Color(const Color &p_c, float p_a) {
		r = p_c.r;
		g = p_c.g;
		b = p_c.b;
		a = p_a;
	}
};

bool Color::operator<(const Color &p_color) const {
	if (r == p_color.r) {
		if (g == p_color.g) {
			if (b == p_color.b) {
				return (a < p_color.a);
			} else {
				return (b < p_color.b);
			}
		} else {
			return g < p_color.g;
		}
	} else {
		return r < p_color.r;
	}
}

#line 0

#line 1 "sfwl/core/vector2i.h"

/*************************************************************************/
/*  vector2i.h                                                           */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class String;

struct _NO_DISCARD_CLASS_ Vector2i {
	enum Axis {
		AXIS_X,
		AXIS_Y,
	};

	union {
		struct {
			union {
				int x;
				int width;
			};
			union {
				int y;
				int height;
			};
		};

		int coord[2];
	};

	_FORCE_INLINE_ int &operator[](int p_idx) {
		DEV_ASSERT((unsigned int)p_idx < 2);
		return coord[p_idx];
	}
	_FORCE_INLINE_ const int &operator[](int p_idx) const {
		DEV_ASSERT((unsigned int)p_idx < 2);
		return coord[p_idx];
	}

	_FORCE_INLINE_ void set_all(int p_value) {
		x = y = p_value;
	}

	_FORCE_INLINE_ int min_axis() const {
		return x < y ? 0 : 1;
	}

	_FORCE_INLINE_ int max_axis() const {
		return x < y ? 1 : 0;
	}

	Vector2i min(const Vector2i &p_vector2i) const {
		return Vector2i(MIN(x, p_vector2i.x), MIN(y, p_vector2i.y));
	}

	Vector2i max(const Vector2i &p_vector2i) const {
		return Vector2i(MAX(x, p_vector2i.x), MAX(y, p_vector2i.y));
	}

	_FORCE_INLINE_ static Vector2i linear_interpolate(const Vector2i &p_a, const Vector2i &p_b, real_t p_weight);
	_FORCE_INLINE_ Vector2i linear_interpolate(const Vector2i &p_to, real_t p_weight) const;

	Vector2i operator+(const Vector2i &p_v) const;
	void operator+=(const Vector2i &p_v);
	Vector2i operator-(const Vector2i &p_v) const;
	void operator-=(const Vector2i &p_v);
	Vector2i operator*(const Vector2i &p_v1) const;

	Vector2i operator*(const int &rvalue) const;
	void operator*=(const int &rvalue);

	Vector2i operator/(const Vector2i &p_v1) const;

	Vector2i operator/(const int &rvalue) const;

	void operator/=(const int &rvalue);

	Vector2i operator-() const;
	bool operator<(const Vector2i &p_vec2) const { return (x == p_vec2.x) ? (y < p_vec2.y) : (x < p_vec2.x); }
	bool operator>(const Vector2i &p_vec2) const { return (x == p_vec2.x) ? (y > p_vec2.y) : (x > p_vec2.x); }
	bool operator<=(const Vector2i &p_vec2) const { return x == p_vec2.x ? (y <= p_vec2.y) : (x < p_vec2.x); }
	bool operator>=(const Vector2i &p_vec2) const { return x == p_vec2.x ? (y >= p_vec2.y) : (x > p_vec2.x); }

	bool operator==(const Vector2i &p_vec2) const;
	bool operator!=(const Vector2i &p_vec2) const;

	int64_t length_squared() const;
	double length() const;

	real_t aspect() const { return width / (real_t)height; }
	Vector2i sign() const { return Vector2i(SGN(x), SGN(y)); }
	Vector2i abs() const { return Vector2i(ABS(x), ABS(y)); }
	Vector2i clamp(const Vector2i &p_min, const Vector2i &p_max) const;

	operator String() const;

	inline Vector2i(const Vector2i &p_vec2) {
		x = p_vec2.x;
		y = p_vec2.y;
	}
	inline Vector2i(int p_x, int p_y) {
		x = p_x;
		y = p_y;
	}
	inline Vector2i() {
		x = 0;
		y = 0;
	}
};

Vector2i Vector2i::linear_interpolate(const Vector2i &p_a, const Vector2i &p_b, real_t p_weight) {
	Vector2i res = p_a;

	res.x += (p_weight * (p_b.x - p_a.x));
	res.y += (p_weight * (p_b.y - p_a.y));

	return res;
}

Vector2i Vector2i::linear_interpolate(const Vector2i &p_to, real_t p_weight) const {
	real_t res_x = x;
	real_t res_y = y;

	res_x += (p_weight * (p_to.x - x));
	res_y += (p_weight * (p_to.y - y));

	return Vector2i(res_x, res_y);
}

typedef Vector2i Size2i;
typedef Vector2i Point2i;

#line 0

#line 1 "sfwl/core/rect2i.h"

/*************************************************************************/
/*  rect2i.h                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

struct _NO_DISCARD_CLASS_ Rect2i {
	Point2i position;
	Size2i size;

	const Point2i &get_position() const { return position; }
	void set_position(const Point2i &p_position) { position = p_position; }
	const Size2i &get_size() const { return size; }
	void set_size(const Size2i &p_size) { size = p_size; }

	int get_area() const { return size.width * size.height; }

	_FORCE_INLINE_ Vector2i get_center() const { return position + (size / 2); }

	inline bool intersects(const Rect2i &p_rect) const {
		if (position.x > (p_rect.position.x + p_rect.size.width)) {
			return false;
		}
		if ((position.x + size.width) < p_rect.position.x) {
			return false;
		}
		if (position.y > (p_rect.position.y + p_rect.size.height)) {
			return false;
		}
		if ((position.y + size.height) < p_rect.position.y) {
			return false;
		}

		return true;
	}

	inline bool encloses(const Rect2i &p_rect) const {
		return (p_rect.position.x >= position.x) && (p_rect.position.y >= position.y) &&
				((p_rect.position.x + p_rect.size.x) < (position.x + size.x)) &&
				((p_rect.position.y + p_rect.size.y) < (position.y + size.y));
	}

	_FORCE_INLINE_ bool has_no_area() const {
		return (size.x <= 0 || size.y <= 0);
	}

	inline Rect2i clip(const Rect2i &p_rect) const { /// return a clipped rect

		Rect2i new_rect = p_rect;

		if (!intersects(new_rect)) {
			return Rect2i();
		}

		new_rect.position.x = MAX(p_rect.position.x, position.x);
		new_rect.position.y = MAX(p_rect.position.y, position.y);

		Point2i p_rect_end = p_rect.position + p_rect.size;
		Point2i end = position + size;

		new_rect.size.x = (int)(MIN(p_rect_end.x, end.x) - new_rect.position.x);
		new_rect.size.y = (int)(MIN(p_rect_end.y, end.y) - new_rect.position.y);

		return new_rect;
	}

	// Returns the instersection between two Rect2is or an empty Rect2i if there is no intersection
	inline Rect2i intersection(const Rect2i &p_rect) const {
		Rect2i new_rect = p_rect;

		if (!intersects(new_rect)) {
			return Rect2i();
		}

		new_rect.position.x = MAX(p_rect.position.x, position.x);
		new_rect.position.y = MAX(p_rect.position.y, position.y);

		Point2i p_rect_end = p_rect.position + p_rect.size;
		Point2i end = position + size;

		new_rect.size.x = MIN(p_rect_end.x, end.x) - new_rect.position.x;
		new_rect.size.y = MIN(p_rect_end.y, end.y) - new_rect.position.y;

		return new_rect;
	}

	inline Rect2i merge(const Rect2i &p_rect) const { ///< return a merged rect

		Rect2i new_rect;

		new_rect.position.x = MIN(p_rect.position.x, position.x);
		new_rect.position.y = MIN(p_rect.position.y, position.y);

		new_rect.size.x = MAX(p_rect.position.x + p_rect.size.x, position.x + size.x);
		new_rect.size.y = MAX(p_rect.position.y + p_rect.size.y, position.y + size.y);

		new_rect.size = new_rect.size - new_rect.position; //make relative again

		return new_rect;
	}

	bool has_point(const Point2i &p_point) const {
		if (p_point.x < position.x) {
			return false;
		}
		if (p_point.y < position.y) {
			return false;
		}

		if (p_point.x >= (position.x + size.x)) {
			return false;
		}
		if (p_point.y >= (position.y + size.y)) {
			return false;
		}

		return true;
	}

	bool operator==(const Rect2i &p_rect) const { return position == p_rect.position && size == p_rect.size; }
	bool operator!=(const Rect2i &p_rect) const { return position != p_rect.position || size != p_rect.size; }

	Rect2i grow(int p_by) const {
		Rect2i g = *this;
		g.position.x -= p_by;
		g.position.y -= p_by;
		g.size.width += p_by * 2;
		g.size.height += p_by * 2;
		return g;
	}

	void grow_by(int p_by) {
		position.x -= p_by;
		position.y -= p_by;
		size.width += p_by * 2;
		size.height += p_by * 2;
	}

	inline Rect2i grow_margin(Margin p_margin, int p_amount) const {
		Rect2i g = *this;
		g = g.grow_individual((MARGIN_LEFT == p_margin) ? p_amount : 0,
				(MARGIN_TOP == p_margin) ? p_amount : 0,
				(MARGIN_RIGHT == p_margin) ? p_amount : 0,
				(MARGIN_BOTTOM == p_margin) ? p_amount : 0);
		return g;
	}

	inline Rect2i grow_side(Side p_side, int p_amount) const {
		Rect2i g = *this;
		g = g.grow_individual((SIDE_LEFT == p_side) ? p_amount : 0,
				(SIDE_TOP == p_side) ? p_amount : 0,
				(SIDE_RIGHT == p_side) ? p_amount : 0,
				(SIDE_BOTTOM == p_side) ? p_amount : 0);
		return g;
	}

	inline Rect2i grow_individual(int p_left, int p_top, int p_right, int p_bottom) const {
		Rect2i g = *this;
		g.position.x -= p_left;
		g.position.y -= p_top;
		g.size.width += p_left + p_right;
		g.size.height += p_top + p_bottom;

		return g;
	}

	_FORCE_INLINE_ Rect2i expand(const Vector2i &p_vector) const {
		Rect2i r = *this;
		r.expand_to(p_vector);
		return r;
	}

	inline void expand_to(const Point2i &p_vector) {
		Point2i begin = position;
		Point2i end = position + size;

		if (p_vector.x < begin.x) {
			begin.x = p_vector.x;
		}
		if (p_vector.y < begin.y) {
			begin.y = p_vector.y;
		}

		if (p_vector.x > end.x) {
			end.x = p_vector.x;
		}
		if (p_vector.y > end.y) {
			end.y = p_vector.y;
		}

		position = begin;
		size = end - begin;
	}

	_FORCE_INLINE_ Rect2i abs() const {
		return Rect2i(Point2i(position.x + MIN(size.x, 0), position.y + MIN(size.y, 0)), size.abs());
	}

	_FORCE_INLINE_ void set_end(const Vector2i &p_end) {
		size = p_end - position;
	}

	_FORCE_INLINE_ Vector2i get_end() const {
		return position + size;
	}

	operator String() const;

	Rect2i() {}
	Rect2i(int p_x, int p_y, int p_width, int p_height) :
			position(Point2i(p_x, p_y)),
			size(Size2i(p_width, p_height)) {
	}
	Rect2i(const Point2i &p_pos, const Size2i &p_size) :
			position(p_pos),
			size(p_size) {
	}
};

#line 0

#line 1 "sfwl/core/pool_vector.h"

/*************************************************************************/
/*  pool_vector.h                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

struct MemoryPool {
	//avoid accessing these directly, must be public for template access

	static PoolAllocator *memory_pool;
	static uint8_t *pool_memory;
	static size_t *pool_size;

	struct Alloc {
		SafeRefCount refcount;
		SafeNumeric<uint32_t> lock;
		void *mem;
		PoolAllocator::ID pool_id;
		size_t size;

		Alloc *free_list;

		Alloc() :
				lock(0),
				mem(nullptr),
				pool_id(POOL_ALLOCATOR_INVALID_ID),
				size(0),
				free_list(nullptr) {
		}
	};

	static Alloc *allocs;
	static Alloc *free_list;
	static uint32_t alloc_count;
	static uint32_t allocs_used;
	static Mutex alloc_mutex;
	static size_t total_memory;
	static size_t max_memory;

	static void setup(uint32_t p_max_allocs = (1 << 16));
	static void cleanup();
};

template <class T>
class PoolVector {
	MemoryPool::Alloc *alloc;

	void _copy_on_write() {
		if (!alloc) {
			return;
		}

		//		ERR_FAIL_COND(alloc->lock>0); should not be illegal to lock this for copy on write, as it's a copy on write after all

		// Refcount should not be zero, otherwise it's a misuse of COW
		if (alloc->refcount.get() == 1) {
			return; //nothing to do
		}

		//must allocate something

		MemoryPool::alloc_mutex.lock();
		if (MemoryPool::allocs_used == MemoryPool::alloc_count) {
			MemoryPool::alloc_mutex.unlock();
			ERR_FAIL_MSG("All memory pool allocations are in use, can't COW.");
		}

		MemoryPool::Alloc *old_alloc = alloc;

		//take one from the free list
		alloc = MemoryPool::free_list;
		MemoryPool::free_list = alloc->free_list;
		//increment the used counter
		MemoryPool::allocs_used++;

		//copy the alloc data
		alloc->size = old_alloc->size;
		alloc->refcount.init();
		alloc->pool_id = POOL_ALLOCATOR_INVALID_ID;
		alloc->lock.set(0);

#ifdef DEBUG_ENABLED
		MemoryPool::total_memory += alloc->size;
		if (MemoryPool::total_memory > MemoryPool::max_memory) {
			MemoryPool::max_memory = MemoryPool::total_memory;
		}
#endif

		MemoryPool::alloc_mutex.unlock();

		if (MemoryPool::memory_pool) {
		} else {
			alloc->mem = memalloc(alloc->size);
		}

		{
			Write w;
			w._ref(alloc);
			Read r;
			r._ref(old_alloc);

			int cur_elements = alloc->size / sizeof(T);
			T *dst = (T *)w.ptr();
			const T *src = (const T *)r.ptr();
			for (int i = 0; i < cur_elements; i++) {
				memnew_placement(&dst[i], T(src[i]));
			}
		}

		if (old_alloc->refcount.unref()) {
			//this should never happen but..

#ifdef DEBUG_ENABLED
			MemoryPool::alloc_mutex.lock();
			MemoryPool::total_memory -= old_alloc->size;
			MemoryPool::alloc_mutex.unlock();
#endif

			{
				Write w;
				w._ref(old_alloc);

				int cur_elements = old_alloc->size / sizeof(T);
				T *elems = (T *)w.ptr();
				for (int i = 0; i < cur_elements; i++) {
					elems[i].~T();
				}
			}

			if (MemoryPool::memory_pool) {
				//resize memory pool
				//if none, create
				//if some resize
			} else {
				memfree(old_alloc->mem);
				old_alloc->mem = nullptr;
				old_alloc->size = 0;

				MemoryPool::alloc_mutex.lock();
				old_alloc->free_list = MemoryPool::free_list;
				MemoryPool::free_list = old_alloc;
				MemoryPool::allocs_used--;
				MemoryPool::alloc_mutex.unlock();
			}
		}
	}

	void _reference(const PoolVector &p_pool_vector) {
		if (alloc == p_pool_vector.alloc) {
			return;
		}

		_unreference();

		if (!p_pool_vector.alloc) {
			return;
		}

		if (p_pool_vector.alloc->refcount.ref()) {
			alloc = p_pool_vector.alloc;
		}
	}

	void _unreference() {
		if (!alloc) {
			return;
		}

		if (!alloc->refcount.unref()) {
			alloc = nullptr;
			return;
		}

		//must be disposed!

		{
			int cur_elements = alloc->size / sizeof(T);

			// Don't use write() here because it could otherwise provoke COW,
			// which is not desirable here because we are destroying the last reference anyways
			Write w;
			// Reference to still prevent other threads from touching the alloc
			w._ref(alloc);

			for (int i = 0; i < cur_elements; i++) {
				w[i].~T();
			}
		}

#ifdef DEBUG_ENABLED
		MemoryPool::alloc_mutex.lock();
		MemoryPool::total_memory -= alloc->size;
		MemoryPool::alloc_mutex.unlock();
#endif

		if (MemoryPool::memory_pool) {
			//resize memory pool
			//if none, create
			//if some resize
		} else {
			memfree(alloc->mem);
			alloc->mem = nullptr;
			alloc->size = 0;

			MemoryPool::alloc_mutex.lock();
			alloc->free_list = MemoryPool::free_list;
			MemoryPool::free_list = alloc;
			MemoryPool::allocs_used--;
			MemoryPool::alloc_mutex.unlock();
		}

		alloc = nullptr;
	}

public:
	class Access {
		friend class PoolVector;

	protected:
		MemoryPool::Alloc *alloc;
		T *mem;

		_FORCE_INLINE_ void _ref(MemoryPool::Alloc *p_alloc) {
			alloc = p_alloc;
			if (alloc) {
				if (alloc->lock.increment() == 1) {
					if (MemoryPool::memory_pool) {
						//lock it and get mem
					}
				}

				mem = (T *)alloc->mem;
			}
		}

		_FORCE_INLINE_ void _unref() {
			if (alloc) {
				if (alloc->lock.decrement() == 0) {
					if (MemoryPool::memory_pool) {
						//put mem back
					}
				}

				mem = nullptr;
				alloc = nullptr;
			}
		}

		Access() {
			alloc = nullptr;
			mem = nullptr;
		}

	public:
		virtual ~Access() {
			_unref();
		}

		void release() {
			_unref();
		}
	};

	class Read : public Access {
	public:
		_FORCE_INLINE_ const T &operator[](int p_index) const { return this->mem[p_index]; }
		_FORCE_INLINE_ const T *ptr() const { return this->mem; }

		void operator=(const Read &p_read) {
			if (this->alloc == p_read.alloc) {
				return;
			}
			this->_unref();
			this->_ref(p_read.alloc);
		}

		Read(const Read &p_read) {
			this->_ref(p_read.alloc);
		}

		Read() {}
	};

	class Write : public Access {
	public:
		_FORCE_INLINE_ T &operator[](int p_index) const { return this->mem[p_index]; }
		_FORCE_INLINE_ T *ptr() const { return this->mem; }

		void operator=(const Write &p_read) {
			if (this->alloc == p_read.alloc) {
				return;
			}
			this->_unref();
			this->_ref(p_read.alloc);
		}

		Write(const Write &p_read) {
			this->_ref(p_read.alloc);
		}

		Write() {}
	};

	Read read() const {
		Read r;
		if (alloc) {
			r._ref(alloc);
		}
		return r;
	}
	Write write() {
		Write w;
		if (alloc) {
			_copy_on_write(); //make sure there is only one being accessed
			w._ref(alloc);
		}
		return w;
	}

	template <class MC>
	void fill_with(const MC &p_mc) {
		int c = p_mc.size();
		resize(c);
		Write w = write();
		int idx = 0;
		for (const typename MC::Element *E = p_mc.front(); E; E = E->next()) {
			w[idx++] = E->get();
		}
	}

	void remove(int p_index) {
		int s = size();
		ERR_FAIL_INDEX(p_index, s);
		Write w = write();
		for (int i = p_index; i < s - 1; i++) {
			w[i] = w[i + 1];
		};
		w = Write();
		resize(s - 1);
	}

	inline int size() const;
	inline bool empty() const;
	T get(int p_index) const;
	void set(int p_index, const T &p_val);
	void fill(const T &p_val);
	void push_back(const T &p_val);
	void append(const T &p_val) {
		push_back(p_val);
	}
	void append_array(const PoolVector<T> &p_arr) {
		int ds = p_arr.size();
		if (ds == 0) {
			return;
		}
		int bs = size();
		resize(bs + ds);
		Write w = write();
		Read r = p_arr.read();
		for (int i = 0; i < ds; i++) {
			w[bs + i] = r[i];
		}
	}

	PoolVector<T> subarray(int p_from, int p_to) const {
		if (p_from < 0) {
			p_from = size() + p_from;
		}
		if (p_to < 0) {
			p_to = size() + p_to;
		}

		ERR_FAIL_INDEX_V(p_from, size(), PoolVector<T>());
		ERR_FAIL_INDEX_V(p_to, size(), PoolVector<T>());

		PoolVector<T> slice;
		int span = 1 + p_to - p_from;
		slice.resize(span);
		Read r = read();
		Write w = slice.write();
		for (int i = 0; i < span; ++i) {
			w[i] = r[p_from + i];
		}

		return slice;
	}

	Error insert(int p_pos, const T &p_val) {
		int s = size();
		ERR_FAIL_INDEX_V(p_pos, s + 1, ERR_INVALID_PARAMETER);
		resize(s + 1);
		{
			Write w = write();
			for (int i = s; i > p_pos; i--) {
				w[i] = w[i - 1];
			}
			w[p_pos] = p_val;
		}

		return OK;
	}

	String join(const String &delimiter) const {
		String rs = "";
		int s = size();
		Read r = read();
		for (int i = 0; i < s; i++) {
			rs += r[i] + delimiter;
		}
		rs.erase(rs.length() - delimiter.length(), delimiter.length());
		return rs;
	}

	bool contains(const T &p_val) const;
	int find(const T &p_val, int p_from = 0) const;
	int rfind(const T &p_val, int p_from = -1) const;
	int count(const T &p_val) const;
	bool has(const T &p_val) const;

	bool is_locked() const {
		return alloc && alloc->lock.get() > 0;
	}

	inline T operator[](int p_index) const;

	Error resize(int p_size);
	Error clear() {
		return resize(0);
	}

	void invert();
	void sort();

	void operator=(const PoolVector &p_pool_vector) {
		_reference(p_pool_vector);
	}
	PoolVector() {
		alloc = nullptr;
	}
	PoolVector(const PoolVector &p_pool_vector) {
		alloc = nullptr;
		_reference(p_pool_vector);
	}
	~PoolVector() {
		_unreference();
	}
};

template <class T>
int PoolVector<T>::size() const {
	return alloc ? alloc->size / sizeof(T) : 0;
}

template <class T>
bool PoolVector<T>::empty() const {
	return alloc ? alloc->size == 0 : true;
}

template <class T>
T PoolVector<T>::get(int p_index) const {
	return operator[](p_index);
}

template <class T>
void PoolVector<T>::set(int p_index, const T &p_val) {
	ERR_FAIL_INDEX(p_index, size());

	Write w = write();
	w[p_index] = p_val;
}

template <class T>
void PoolVector<T>::fill(const T &p_val) {
	Write w = write();
	for (int i = 0; i < size(); i++) {
		w[i] = p_val;
	}
}

template <class T>
void PoolVector<T>::push_back(const T &p_val) {
	resize(size() + 1);
	set(size() - 1, p_val);
}

template <class T>
bool PoolVector<T>::contains(const T &p_val) const {
	Read r = read();
	int s = size();

	for (int i = 0; i < s; ++i) {
		if (r[i] == p_val) {
			return true;
		}
	}

	return false;
}

template <class T>
int PoolVector<T>::find(const T &p_val, int p_from) const {
	if (p_from < 0) {
		return -1;
	}

	const int s = size();
	const Read r = read();

	for (int i = p_from; i < s; i++) {
		if (r[i] == p_val) {
			return i;
		}
	}
	return -1;
}

template <class T>
int PoolVector<T>::rfind(const T &p_val, int p_from) const {
	const int s = size();
	const Read r = read();

	if (p_from < 0) {
		p_from = s + p_from;
	}
	if (p_from < 0 || p_from >= s) {
		p_from = s - 1;
	}

	for (int i = p_from; i >= 0; i--) {
		if (r[i] == p_val) {
			return i;
		}
	}
	return -1;
}

template <class T>
int PoolVector<T>::count(const T &p_val) const {
	const int s = size();
	const Read r = read();
	int amount = 0;
	for (int i = 0; i < s; i++) {
		if (r[i] == p_val) {
			amount++;
		}
	}
	return amount;
}

template <class T>
bool PoolVector<T>::has(const T &p_val) const {
	return find(p_val) != -1;
}

template <class T>
T PoolVector<T>::operator[](int p_index) const {
	CRASH_BAD_INDEX(p_index, size());

	Read r = read();
	return r[p_index];
}

template <class T>
Error PoolVector<T>::resize(int p_size) {
	ERR_FAIL_COND_V_MSG(p_size < 0, ERR_INVALID_PARAMETER, "Size of PoolVector cannot be negative.");

	if (alloc == nullptr) {
		if (p_size == 0) {
			return OK; //nothing to do here
		}

		//must allocate something
		MemoryPool::alloc_mutex.lock();
		if (MemoryPool::allocs_used == MemoryPool::alloc_count) {
			MemoryPool::alloc_mutex.unlock();
			ERR_FAIL_V_MSG(ERR_OUT_OF_MEMORY, "All memory pool allocations are in use.");
		}

		//take one from the free list
		alloc = MemoryPool::free_list;
		MemoryPool::free_list = alloc->free_list;
		//increment the used counter
		MemoryPool::allocs_used++;

		//cleanup the alloc
		alloc->size = 0;
		alloc->refcount.init();
		alloc->pool_id = POOL_ALLOCATOR_INVALID_ID;
		MemoryPool::alloc_mutex.unlock();

	} else {
		ERR_FAIL_COND_V_MSG(alloc->lock.get() > 0, ERR_LOCKED, "Can't resize PoolVector if locked."); //can't resize if locked!
	}

	size_t new_size = sizeof(T) * p_size;

	if (alloc->size == new_size) {
		return OK; //nothing to do
	}

	if (p_size == 0) {
		_unreference();
		return OK;
	}

	_copy_on_write(); // make it unique

#ifdef DEBUG_ENABLED
	MemoryPool::alloc_mutex.lock();
	MemoryPool::total_memory -= alloc->size;
	MemoryPool::total_memory += new_size;
	if (MemoryPool::total_memory > MemoryPool::max_memory) {
		MemoryPool::max_memory = MemoryPool::total_memory;
	}
	MemoryPool::alloc_mutex.unlock();
#endif

	int cur_elements = alloc->size / sizeof(T);

	if (p_size > cur_elements) {
		if (MemoryPool::memory_pool) {
			//resize memory pool
			//if none, create
			//if some resize
		} else {
			if (alloc->size == 0) {
				alloc->mem = memalloc(new_size);
			} else {
				alloc->mem = memrealloc(alloc->mem, new_size);
			}
		}

		alloc->size = new_size;

		Write w = write();

		for (int i = cur_elements; i < p_size; i++) {
			memnew_placement(&w[i], T);
		}

	} else {
		{
			Write w = write();
			for (int i = p_size; i < cur_elements; i++) {
				w[i].~T();
			}
		}

		if (MemoryPool::memory_pool) {
			//resize memory pool
			//if none, create
			//if some resize
		} else {
			if (new_size == 0) {
				memfree(alloc->mem);
				alloc->mem = nullptr;
				alloc->size = 0;

				MemoryPool::alloc_mutex.lock();
				alloc->free_list = MemoryPool::free_list;
				MemoryPool::free_list = alloc;
				MemoryPool::allocs_used--;
				MemoryPool::alloc_mutex.unlock();

			} else {
				alloc->mem = memrealloc(alloc->mem, new_size);
				alloc->size = new_size;
			}
		}
	}

	return OK;
}

template <class T>
void PoolVector<T>::invert() {
	T temp;
	Write w = write();
	int s = size();
	int half_s = s / 2;

	for (int i = 0; i < half_s; i++) {
		temp = w[i];
		w[i] = w[s - i - 1];
		w[s - i - 1] = temp;
	}
}

template <class T>
void PoolVector<T>::sort() {
	int len = size();
	if (len == 0) {
		return;
	}

	Write w = write();
	SortArray<T> sorter;
	sorter.sort(w.ptr(), len);
}

#line 0

#line 1 "sfwl/core/tight_local_vector.h"

/*************************************************************************/
/*  tight_local_vector.h                                                 */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

// It grows strictly as much as needed. (The vanilla LocalVector is what you want in most cases).
template <class T, class U = uint32_t, bool force_trivial = false>
class TightLocalVector {
private:
	U count = 0;
	U capacity = 0;
	T *data = nullptr;

public:
	T *ptr() {
		return data;
	}

	const T *ptr() const {
		return data;
	}

	_FORCE_INLINE_ void push_back(T p_elem) {
		if (unlikely(count == capacity)) {
			if (capacity == 0) {
				capacity = 1;
			} else {
				capacity <<= 1;
			}
			data = (T *)memrealloc(data, capacity * sizeof(T));
			CRASH_COND_MSG(!data, "Out of memory");
		}

		if (!HAS_TRIVIAL_CONSTRUCTOR(T) && !force_trivial) {
			memnew_placement(&data[count++], T(p_elem));
		} else {
			data[count++] = p_elem;
		}
	}

	void remove(U p_index) {
		ERR_FAIL_UNSIGNED_INDEX(p_index, count);
		count--;
		for (U i = p_index; i < count; i++) {
			data[i] = data[i + 1];
		}
		if (!HAS_TRIVIAL_DESTRUCTOR(T) && !force_trivial) {
			data[count].~T();
		}
	}

	/// Removes the item copying the last value into the position of the one to
	/// remove. It's generally faster than `remove`.
	void remove_unordered(U p_index) {
		ERR_FAIL_INDEX(p_index, count);
		count--;
		if (count > p_index) {
			data[p_index] = data[count];
		}
		if (!HAS_TRIVIAL_DESTRUCTOR(T) && !force_trivial) {
			data[count].~T();
		}
	}

	void erase(const T &p_val) {
		int64_t idx = find(p_val);
		if (idx >= 0) {
			remove(idx);
		}
	}

	U erase_multiple_unordered(const T &p_val) {
		U from = 0;
		U count = 0;
		while (true) {
			int64_t idx = find(p_val, from);

			if (idx == -1) {
				break;
			}
			remove_unordered(idx);
			from = idx;
			count++;
		}
		return count;
	}

	void invert() {
		for (U i = 0; i < count / 2; i++) {
			SWAP(data[i], data[count - i - 1]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ void reset() {
		clear();
		if (data) {
			memfree(data);
			data = nullptr;
			capacity = 0;
		}
	}
	_FORCE_INLINE_ bool empty() const { return count == 0; }
	_FORCE_INLINE_ U get_capacity() const { return capacity; }
	_FORCE_INLINE_ void reserve(U p_size) {
		if (p_size > capacity) {
			capacity = p_size;
			data = (T *)memrealloc(data, capacity * sizeof(T));
			CRASH_COND_MSG(!data, "Out of memory");
		}
	}

	_FORCE_INLINE_ U size() const { return count; }
	void resize(U p_size) {
		if (p_size < count) {
			if (!HAS_TRIVIAL_DESTRUCTOR(T) && !force_trivial) {
				for (U i = p_size; i < count; i++) {
					data[i].~T();
				}
			}
			count = p_size;
		} else if (p_size > count) {
			if (unlikely(p_size > capacity)) {
				if (capacity == 0) {
					capacity = 1;
				}
				while (capacity < p_size) {
					capacity <<= 1;
				}
				data = (T *)memrealloc(data, capacity * sizeof(T));
				CRASH_COND_MSG(!data, "Out of memory");
			}
			if (!HAS_TRIVIAL_CONSTRUCTOR(T) && !force_trivial) {
				for (U i = count; i < p_size; i++) {
					memnew_placement(&data[i], T);
				}
			}
			count = p_size;
		}
	}
	_FORCE_INLINE_ const T &operator[](U p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}
	_FORCE_INLINE_ T &operator[](U p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}

	void fill(T p_val) {
		for (U i = 0; i < count; i++) {
			data[i] = p_val;
		}
	}

	void insert(U p_pos, T p_val) {
		ERR_FAIL_UNSIGNED_INDEX(p_pos, count + 1);
		if (p_pos == count) {
			push_back(p_val);
		} else {
			resize(count + 1);
			for (U i = count - 1; i > p_pos; i--) {
				data[i] = data[i - 1];
			}
			data[p_pos] = p_val;
		}
	}

	int64_t find(const T &p_val, U p_from = 0) const {
		for (U i = p_from; i < count; i++) {
			if (data[i] == p_val) {
				return int64_t(i);
			}
		}
		return -1;
	}

	template <class C>
	void sort_custom() {
		U len = count;
		if (len == 0) {
			return;
		}

		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<T>>();
	}

	void ordered_insert(T p_val) {
		U i;
		for (i = 0; i < count; i++) {
			if (p_val < data[i]) {
				break;
			}
		}
		insert(i, p_val);
	}

	operator Vector<T>() const {
		Vector<T> ret;
		ret.resize(size());
		T *w = ret.ptrw();
		memcpy(w, data, sizeof(T) * count);
		return ret;
	}

	operator PoolVector<T>() const {
		PoolVector<T> pl;
		if (size()) {
			pl.resize(size());
			typename PoolVector<T>::Write w = pl.write();
			T *dest = w.ptr();
			memcpy(dest, data, sizeof(T) * count);
		}
		return pl;
	}

	Vector<uint8_t> to_byte_array() const { //useful to pass stuff to gpu or variant
		Vector<uint8_t> ret;
		ret.resize(count * sizeof(T));
		uint8_t *w = ret.ptrw();
		memcpy(w, data, sizeof(T) * count);
		return ret;
	}

	_FORCE_INLINE_ TightLocalVector() {}
	_FORCE_INLINE_ TightLocalVector(const TightLocalVector &p_from) {
		resize(p_from.size());
		for (U i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
	}
	TightLocalVector(const Vector<T> &p_from) {
		resize(p_from.size());
		for (U i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}
	TightLocalVector(const PoolVector<T> &p_from) {
		resize(p_from.size());
		typename PoolVector<T>::Read r = p_from.read();
		for (U i = 0; i < count; i++) {
			data[i] = r[i];
		}
	}

	inline void operator=(const TightLocalVector &p_from) {
		resize(p_from.size());
		for (U i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
	}
	inline void operator=(const Vector<T> &p_from) {
		resize(p_from.size());
		for (U i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}
	inline TightLocalVector &operator=(const PoolVector<T> &p_from) {
		resize(p_from.size());
		typename PoolVector<T>::Read r = p_from.read();
		for (U i = 0; i < count; i++) {
			data[i] = r[i];
		}
		return *this;
	}

	_FORCE_INLINE_ ~TightLocalVector() {
		if (data) {
			reset();
		}
	}
};

// Integer default version
template <class T, class I = int32_t, bool force_trivial = false>
class TightLocalVectori : public TightLocalVector<T, I, force_trivial> {
};

#line 0

#line 1 "sfwl/core/local_vector.h"

/*************************************************************************/
/*  local_vector.h                                                       */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

template <class T, class U = uint32_t, bool force_trivial = false>
class LocalVector {
protected:
	U count = 0;
	U capacity = 0;
	T *data = nullptr;

public:
	T *ptr() {
		return data;
	}

	const T *ptr() const {
		return data;
	}

	_FORCE_INLINE_ void push_back(T p_elem) {
		if (unlikely(count == capacity)) {
			if (capacity == 0) {
				capacity = 1;
			} else {
				capacity <<= 1;
			}
			data = (T *)memrealloc(data, capacity * sizeof(T));
			CRASH_COND_MSG(!data, "Out of memory");
		}

		if (!HAS_TRIVIAL_CONSTRUCTOR(T) && !force_trivial) {
			memnew_placement(&data[count++], T(p_elem));
		} else {
			data[count++] = p_elem;
		}
	}

	void remove(U p_index) {
		ERR_FAIL_UNSIGNED_INDEX(p_index, count);
		count--;
		for (U i = p_index; i < count; i++) {
			data[i] = data[i + 1];
		}
		if (!HAS_TRIVIAL_DESTRUCTOR(T) && !force_trivial) {
			data[count].~T();
		}
	}

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_unordered(U p_index) {
		ERR_FAIL_INDEX(p_index, count);
		count--;
		if (count > p_index) {
			data[p_index] = data[count];
		}
		if (!HAS_TRIVIAL_DESTRUCTOR(T) && !force_trivial) {
			data[count].~T();
		}
	}

	_FORCE_INLINE_ bool erase(const T &p_val) {
		int64_t idx = find(p_val);
		if (idx >= 0) {
			remove(idx);
			return true;
		}
		return false;
	}

	U erase_multiple_unordered(const T &p_val) {
		U from = 0;
		U count = 0;
		while (true) {
			int64_t idx = find(p_val, from);

			if (idx == -1) {
				break;
			}
			remove_unordered(idx);
			from = idx;
			count++;
		}
		return count;
	}

	void invert() {
		for (U i = 0; i < count / 2; i++) {
			SWAP(data[i], data[count - i - 1]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ void reset() {
		clear();
		if (data) {
			memfree(data);
			data = nullptr;
			capacity = 0;
		}
	}
	_FORCE_INLINE_ bool empty() const { return count == 0; }
	_FORCE_INLINE_ U get_capacity() const { return capacity; }
	_FORCE_INLINE_ void reserve(U p_size, bool p_allow_shrink = false) {
		p_size = nearest_power_of_2_templated(p_size);
		if (!p_allow_shrink ? p_size > capacity : ((p_size >= count) && (p_size != capacity))) {
			capacity = p_size;
			data = (T *)memrealloc(data, capacity * sizeof(T));
			CRASH_COND_MSG(!data, "Out of memory");
		}
	}

	_FORCE_INLINE_ U size() const { return count; }
	void resize(U p_size) {
		if (p_size < count) {
			if (!HAS_TRIVIAL_DESTRUCTOR(T) && !force_trivial) {
				for (U i = p_size; i < count; i++) {
					data[i].~T();
				}
			}
			count = p_size;
		} else if (p_size > count) {
			if (unlikely(p_size > capacity)) {
				if (capacity == 0) {
					capacity = 1;
				}
				while (capacity < p_size) {
					capacity <<= 1;
				}
				data = (T *)memrealloc(data, capacity * sizeof(T));
				CRASH_COND_MSG(!data, "Out of memory");
			}
			if (!HAS_TRIVIAL_CONSTRUCTOR(T) && !force_trivial) {
				for (U i = count; i < p_size; i++) {
					memnew_placement(&data[i], T);
				}
			}
			count = p_size;
		}
	}
	_FORCE_INLINE_ const T &operator[](U p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}
	_FORCE_INLINE_ T &operator[](U p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}

	void fill(T p_val) {
		for (U i = 0; i < count; i++) {
			data[i] = p_val;
		}
	}

	void insert(U p_pos, T p_val) {
		ERR_FAIL_UNSIGNED_INDEX(p_pos, count + 1);
		if (p_pos == count) {
			push_back(p_val);
		} else {
			resize(count + 1);
			for (U i = count - 1; i > p_pos; i--) {
				data[i] = data[i - 1];
			}
			data[p_pos] = p_val;
		}
	}

	int64_t find(const T &p_val, U p_from = 0) const {
		for (U i = p_from; i < count; i++) {
			if (data[i] == p_val) {
				return int64_t(i);
			}
		}
		return -1;
	}

	template <class C>
	void sort_custom() {
		U len = count;
		if (len == 0) {
			return;
		}

		SortArray<T, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<T>>();
	}

	void ordered_insert(T p_val) {
		U i;
		for (i = 0; i < count; i++) {
			if (p_val < data[i]) {
				break;
			}
		}
		insert(i, p_val);
	}

	operator Vector<T>() const {
		Vector<T> ret;
		ret.resize(size());
		T *w = ret.ptrw();
		memcpy(w, data, sizeof(T) * count);
		return ret;
	}

	operator PoolVector<T>() const {
		PoolVector<T> pl;
		if (size()) {
			pl.resize(size());
			typename PoolVector<T>::Write w = pl.write();
			T *dest = w.ptr();
			memcpy(dest, data, sizeof(T) * count);
		}
		return pl;
	}

	Vector<uint8_t> to_byte_array() const { //useful to pass stuff to gpu or variant
		Vector<uint8_t> ret;
		ret.resize(count * sizeof(T));
		uint8_t *w = ret.ptrw();
		memcpy(w, data, sizeof(T) * count);
		return ret;
	}

	_FORCE_INLINE_ LocalVector() {}
	_FORCE_INLINE_ LocalVector(const LocalVector &p_from) {
		resize(p_from.size());
		for (U i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
	}
	LocalVector(const Vector<T> &p_from) {
		resize(p_from.size());
		for (U i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}
	LocalVector(const PoolVector<T> &p_from) {
		resize(p_from.size());
		typename PoolVector<T>::Read r = p_from.read();
		for (U i = 0; i < count; i++) {
			data[i] = r[i];
		}
	}

	inline LocalVector &operator=(const LocalVector &p_from) {
		resize(p_from.size());
		for (U i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
		return *this;
	}
	inline LocalVector &operator=(const Vector<T> &p_from) {
		resize(p_from.size());
		for (U i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
		return *this;
	}
	inline LocalVector &operator=(const PoolVector<T> &p_from) {
		resize(p_from.size());
		typename PoolVector<T>::Read r = p_from.read();
		for (U i = 0; i < count; i++) {
			data[i] = r[i];
		}
		return *this;
	}

	_FORCE_INLINE_ ~LocalVector() {
		if (data) {
			reset();
		}
	}
};

// Integer default version
template <class T, class I = int32_t, bool force_trivial = false>
class LocalVectori : public LocalVector<T, I, force_trivial> {
};

#line 0

#line 1 "sfwl/core/hashfuncs.h"

/*************************************************************************/
/*  hashfuncs.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * Hashing functions
 */

/**
 * DJB2 Hash function
 * @param C String
 * @return 32-bits hashcode
 */
static inline uint32_t hash_djb2(const char *p_cstr) {
	const unsigned char *chr = (const unsigned char *)p_cstr;
	uint32_t hash = 5381;
	uint32_t c;

	while ((c = *chr++)) {
		hash = ((hash << 5) + hash) ^ c; /* hash * 33 ^ c */
	}

	return hash;
}

static inline uint32_t hash_djb2_buffer(const uint8_t *p_buff, int p_len, uint32_t p_prev = 5381) {
	uint32_t hash = p_prev;

	for (int i = 0; i < p_len; i++) {
		hash = ((hash << 5) + hash) ^ p_buff[i]; /* hash * 33 ^ c */
	}

	return hash;
}

static inline uint32_t hash_djb2_one_32(uint32_t p_in, uint32_t p_prev = 5381) {
	return ((p_prev << 5) + p_prev) ^ p_in;
}

/**
 * Thomas Wang's 64-bit to 32-bit Hash function:
 * https://web.archive.org/web/20071223173210/https:/www.concentric.net/~Ttwang/tech/inthash.htm
 *
 * @param p_int - 64-bit unsigned integer key to be hashed
 * @return unsigned 32-bit value representing hashcode
 */
static inline uint32_t hash_one_uint64(const uint64_t p_int) {
	uint64_t v = p_int;
	v = (~v) + (v << 18); // v = (v << 18) - v - 1;
	v = v ^ (v >> 31);
	v = v * 21; // v = (v + (v << 2)) + (v << 4);
	v = v ^ (v >> 11);
	v = v + (v << 6);
	v = v ^ (v >> 22);
	return (uint32_t)v;
}

#define HASH_MURMUR3_SEED 0x7F07C65
// Murmurhash3 32-bit version.
// All MurmurHash versions are public domain software, and the author disclaims all copyright to their code.

static _FORCE_INLINE_ uint32_t hash_murmur3_one_32(uint32_t p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
	p_in *= 0xcc9e2d51;
	p_in = (p_in << 15) | (p_in >> 17);
	p_in *= 0x1b873593;

	p_seed ^= p_in;
	p_seed = (p_seed << 13) | (p_seed >> 19);
	p_seed = p_seed * 5 + 0xe6546b64;

	return p_seed;
}

static _FORCE_INLINE_ uint32_t hash_murmur3_one_float(float p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
	union {
		float f;
		uint32_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f) {
		u.f = 0.0;
	} else if (Math::is_nan(p_in)) {
		u.f = NAN;
	} else {
		u.f = p_in;
	}

	return hash_murmur3_one_32(u.i, p_seed);
}

static _FORCE_INLINE_ uint32_t hash_murmur3_one_64(uint64_t p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
	p_seed = hash_murmur3_one_32(p_in & 0xFFFFFFFF, p_seed);
	return hash_murmur3_one_32(p_in >> 32, p_seed);
}

static _FORCE_INLINE_ uint32_t hash_murmur3_one_double(double p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
	union {
		double d;
		uint64_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f) {
		u.d = 0.0;
	} else if (Math::is_nan(p_in)) {
		u.d = NAN;
	} else {
		u.d = p_in;
	}

	return hash_murmur3_one_64(u.i, p_seed);
}

static _FORCE_INLINE_ uint32_t hash_murmur3_one_real(real_t p_in, uint32_t p_seed = HASH_MURMUR3_SEED) {
#ifdef REAL_T_IS_DOUBLE
	return hash_murmur3_one_double(p_in, p_seed);
#else
	return hash_murmur3_one_float(p_in, p_seed);
#endif
}

static _FORCE_INLINE_ uint32_t hash_rotl32(uint32_t x, int8_t r) {
	return (x << r) | (x >> (32 - r));
}

static _FORCE_INLINE_ uint32_t hash_fmix32(uint32_t h) {
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;

	return h;
}

static _FORCE_INLINE_ uint32_t hash_murmur3_buffer(const void *key, int length, const uint32_t seed = HASH_MURMUR3_SEED) {
	// Although not required, this is a random prime number.
	const uint8_t *data = (const uint8_t *)key;
	const int nblocks = length / 4;

	uint32_t h1 = seed;

	const uint32_t c1 = 0xcc9e2d51;
	const uint32_t c2 = 0x1b873593;

	const uint32_t *blocks = (const uint32_t *)(data + nblocks * 4);

	for (int i = -nblocks; i; i++) {
		uint32_t k1 = blocks[i];

		k1 *= c1;
		k1 = hash_rotl32(k1, 15);
		k1 *= c2;

		h1 ^= k1;
		h1 = hash_rotl32(h1, 13);
		h1 = h1 * 5 + 0xe6546b64;
	}

	const uint8_t *tail = (const uint8_t *)(data + nblocks * 4);

	uint32_t k1 = 0;

	switch (length & 3) {
		case 3:
			k1 ^= tail[2] << 16;
			FALLTHROUGH;
		case 2:
			k1 ^= tail[1] << 8;
			FALLTHROUGH;
		case 1:
			k1 ^= tail[0];
			k1 *= c1;
			k1 = hash_rotl32(k1, 15);
			k1 *= c2;
			h1 ^= k1;
	};

	// Finalize with additional bit mixing.
	h1 ^= length;
	return hash_fmix32(h1);
}

static inline uint32_t hash_djb2_one_float(double p_in, uint32_t p_prev = 5381) {
	union {
		double d;
		uint64_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f) {
		u.d = 0.0;
	} else if (Math::is_nan(p_in)) {
		u.d = Math_NAN;
	} else {
		u.d = p_in;
	}

	return ((p_prev << 5) + p_prev) + hash_one_uint64(u.i);
}

template <class T>
static inline uint32_t make_uint32_t(T p_in) {
	union {
		T t;
		uint32_t _u32;
	} _u;
	_u._u32 = 0;
	_u.t = p_in;
	return _u._u32;
}

static _FORCE_INLINE_ uint64_t hash_djb2_one_float_64(double p_in, uint64_t p_prev = 5381) {
	union {
		double d;
		uint64_t i;
	} u;

	// Normalize +/- 0.0 and NaN values so they hash the same.
	if (p_in == 0.0f) {
		u.d = 0.0;
	} else if (Math::is_nan(p_in)) {
		u.d = NAN;
	} else {
		u.d = p_in;
	}

	return ((p_prev << 5) + p_prev) + u.i;
}

static _FORCE_INLINE_ uint64_t hash_djb2_one_64(uint64_t p_in, uint64_t p_prev = 5381) {
	return ((p_prev << 5) + p_prev) ^ p_in;
}

template <class T>
static _FORCE_INLINE_ uint64_t hash_make_uint64_t(T p_in) {
	union {
		T t;
		uint64_t _u64;
	} _u;
	_u._u64 = 0; // in case p_in is smaller

	_u.t = p_in;
	return _u._u64;
}

template <class T>
static inline uint64_t make_uint64_t(T p_in) {
	union {
		T t;
		uint64_t _u64;
	} _u;
	_u._u64 = 0; // in case p_in is smaller

	_u.t = p_in;
	return _u._u64;
}

template <class T>
class Ref;

struct HashMapHasherDefault {
	// Generic hash function for any type.
	template <class T>
	static _FORCE_INLINE_ uint32_t hash(const T *p_pointer) { return hash_one_uint64((uint64_t)p_pointer); }

	template <class T>
	static _FORCE_INLINE_ uint32_t hash(const Ref<T> &p_ref) { return hash_one_uint64((uint64_t)p_ref.operator->()); }

	static _FORCE_INLINE_ uint32_t hash(const String &p_string) { return p_string.hash(); }
	static _FORCE_INLINE_ uint32_t hash(const char *p_cstr) { return hash_djb2(p_cstr); }
	static _FORCE_INLINE_ uint32_t hash(const wchar_t p_wchar) { return hash_fmix32(p_wchar); }
	static _FORCE_INLINE_ uint32_t hash(const char16_t p_uchar) { return hash_fmix32(p_uchar); }
	static _FORCE_INLINE_ uint32_t hash(const char32_t p_uchar) { return hash_fmix32(p_uchar); }
	static _FORCE_INLINE_ uint32_t hash(const StringName &p_string_name) { return p_string_name.hash(); }

	static _FORCE_INLINE_ uint32_t hash(const uint64_t p_int) { return hash_one_uint64(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const int64_t p_int) { return hash_one_uint64(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const float p_float) { return hash_murmur3_one_float(p_float); }
	static _FORCE_INLINE_ uint32_t hash(const double p_double) { return hash_murmur3_one_double(p_double); }
	static _FORCE_INLINE_ uint32_t hash(const uint32_t p_int) { return hash_fmix32(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const int32_t p_int) { return hash_fmix32(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const uint16_t p_int) { return hash_fmix32(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const int16_t p_int) { return hash_fmix32(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const uint8_t p_int) { return hash_fmix32(p_int); }
	static _FORCE_INLINE_ uint32_t hash(const int8_t p_int) { return hash_fmix32(p_int); }

	static _FORCE_INLINE_ uint32_t hash(const Vector2i &p_vec) {
		uint32_t h = hash_murmur3_one_32(p_vec.x);
		h = hash_murmur3_one_32(p_vec.y, h);
		return hash_fmix32(h);
	}
	static _FORCE_INLINE_ uint32_t hash(const Rect2i &p_rect) {
		uint32_t h = hash_murmur3_one_32(p_rect.position.x);
		h = hash_murmur3_one_32(p_rect.position.y, h);
		h = hash_murmur3_one_32(p_rect.size.x, h);
		h = hash_murmur3_one_32(p_rect.size.y, h);
		return hash_fmix32(h);
	}
};

template <typename T>
struct HashMapComparatorDefault {
	static bool compare(const T &p_lhs, const T &p_rhs) {
		return p_lhs == p_rhs;
	}
};

template <>
struct HashMapComparatorDefault<float> {
	static bool compare(const float &p_lhs, const float &p_rhs) {
		return (p_lhs == p_rhs) || (Math::is_nan(p_lhs) && Math::is_nan(p_rhs));
	}
};

template <>
struct HashMapComparatorDefault<double> {
	static bool compare(const double &p_lhs, const double &p_rhs) {
		return (p_lhs == p_rhs) || (Math::is_nan(p_lhs) && Math::is_nan(p_rhs));
	}
};

constexpr uint32_t HASH_TABLE_SIZE_MAX = 29;

const uint32_t hash_table_size_primes[HASH_TABLE_SIZE_MAX] = {
	5,
	13,
	23,
	47,
	97,
	193,
	389,
	769,
	1543,
	3079,
	6151,
	12289,
	24593,
	49157,
	98317,
	196613,
	393241,
	786433,
	1572869,
	3145739,
	6291469,
	12582917,
	25165843,
	50331653,
	100663319,
	201326611,
	402653189,
	805306457,
	1610612741,
};

// Computed with elem_i = UINT64_C (0 x FFFFFFFF FFFFFFFF ) / d_i + 1, where d_i is the i-th element of the above array.
const uint64_t hash_table_size_primes_inv[HASH_TABLE_SIZE_MAX] = {
	3689348814741910324,
	1418980313362273202,
	802032351030850071,
	392483916461905354,
	190172619316593316,
	95578984837873325,
	47420935922132524,
	23987963684927896,
	11955116055547344,
	5991147799191151,
	2998982941588287,
	1501077717772769,
	750081082979285,
	375261795343686,
	187625172388393,
	93822606204624,
	46909513691883,
	23456218233098,
	11728086747027,
	5864041509391,
	2932024948977,
	1466014921160,
	733007198436,
	366503839517,
	183251896093,
	91625960335,
	45812983922,
	22906489714,
	11453246088
};

/**
 * Fastmod computes ( n mod d ) given the precomputed c much faster than n % d.
 * The implementation of fastmod is based on the following paper by Daniel Lemire et al.
 * Faster Remainder by Direct Computation: Applications to Compilers and Software Libraries
 * https://arxiv.org/abs/1902.01961
 */
static _FORCE_INLINE_ uint32_t fastmod(const uint32_t n, const uint64_t c, const uint32_t d) {
#if defined(_MSC_VER)
	// Returns the upper 64 bits of the product of two 64-bit unsigned integers.
	// This intrinsic function is required since MSVC does not support unsigned 128-bit integers.
#if defined(_M_X64) || defined(_M_ARM64)
	return __umulh(c * n, d);
#else
	// Fallback to the slower method for 32-bit platforms.
	return n % d;
#endif // _M_X64 || _M_ARM64
#else
#ifdef __SIZEOF_INT128__
	// Prevent compiler warning, because we know what we are doing.
	uint64_t lowbits = c * n;
	__extension__ typedef unsigned __int128 uint128;
	return static_cast<uint64_t>(((uint128)lowbits * d) >> 64);
#else
	// Fallback to the slower method if no 128-bit unsigned integer type is available.
	return n % d;
#endif // __SIZEOF_INT128__
#endif // _MSC_VER
}

#line 0

#line 1 "sfwl/core/pair.h"

/*************************************************************************/
/*  pair.h                                                               */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

template <class F, class S>
struct Pair {
	F first;
	S second;

	Pair() :
			first(),
			second() {
	}

	Pair(F p_first, const S &p_second) :
			first(p_first),
			second(p_second) {
	}
};

template <class F, class S>
bool operator==(const Pair<F, S> &pair, const Pair<F, S> &other) {
	return (pair.first == other.first) && (pair.second == other.second);
}

template <class F, class S>
bool operator!=(const Pair<F, S> &pair, const Pair<F, S> &other) {
	return (pair.first != other.first) || (pair.second != other.second);
}

template <class F, class S>
struct PairSort {
	bool operator()(const Pair<F, S> &A, const Pair<F, S> &B) const {
		if (A.first != B.first) {
			return A.first < B.first;
		}
		return A.second < B.second;
	}
};

template <class F, class S>
struct PairHash {
	static uint32_t hash(const Pair<F, S> &P) {
		uint64_t h1 = HashMapHasherDefault::hash(P.first);
		uint64_t h2 = HashMapHasherDefault::hash(P.second);
		return hash_one_uint64((h1 << 32) | h2);
	}
};

template <class K, class V>
struct KeyValue {
	const K key;
	V value;

	void operator=(const KeyValue &p_kv) = delete;
	_FORCE_INLINE_ KeyValue(const KeyValue &p_kv) :
			key(p_kv.key),
			value(p_kv.value) {
	}
	_FORCE_INLINE_ KeyValue(const K &p_key, const V &p_value) :
			key(p_key),
			value(p_value) {
	}
};

template <class K, class V>
bool operator==(const KeyValue<K, V> &pair, const KeyValue<K, V> &other) {
	return (pair.key == other.key) && (pair.value == other.value);
}

template <class K, class V>
bool operator!=(const KeyValue<K, V> &pair, const KeyValue<K, V> &other) {
	return (pair.key != other.key) || (pair.value != other.value);
}

template <class K, class V>
struct KeyValueSort {
	bool operator()(const KeyValue<K, V> &A, const KeyValue<K, V> &B) const {
		return A.key < B.key;
	}
};

#line 0

#line 1 "sfwl/core/og_hash_map.h"

/*************************************************************************/
/*  og_hash_map.h                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * @class OGHashMap
 * @author Juan Linietsky <reduzio@gmail.com>
 *
 * Implementation of a standard Hashing HashMap, for quick lookups of Data associated with a Key.
 * The implementation provides hashers for the default types, if you need a special kind of hasher, provide
 * your own.
 * @param TKey  Key, search is based on it, needs to be hasheable. It is unique in this container.
 * @param TData Data, data associated with the key
 * @param Hasher Hasher object, needs to provide a valid static hash function for TKey
 * @param Comparator comparator object, needs to be able to safely compare two TKey values. It needs to ensure that x == x for any items inserted in the map. Bear in mind that nan != nan when implementing an equality check.
 * @param MIN_HASH_TABLE_POWER Miminum size of the hash table, as a power of two. You rarely need to change this parameter.
 * @param RELATIONSHIP Relationship at which the hash table is resized. if amount of elements is RELATIONSHIP
 * times bigger than the hash table, table is resized to solve this condition. if RELATIONSHIP is zero, table is always MIN_HASH_TABLE_POWER.
 *
 */

template <class TKey, class TData, class Hasher = HashMapHasherDefault, class Comparator = HashMapComparatorDefault<TKey>, uint8_t MIN_HASH_TABLE_POWER = 3, uint8_t RELATIONSHIP = 8>
class OGHashMap {
public:
	struct Pair {
		TKey key;
		TData data;

		Pair(const TKey &p_key) :
				key(p_key),
				data() {}
		Pair(const TKey &p_key, const TData &p_data) :
				key(p_key),
				data(p_data) {
		}
	};

	struct Element {
	private:
		friend class OGHashMap;

		uint32_t hash;
		Element *next;
		Element() { next = nullptr; }
		Pair pair;

	public:
		const TKey &key() const {
			return pair.key;
		}

		TData &value() {
			return pair.data;
		}

		const TData &value() const {
			return pair.data;
		}

		TData &get() {
			return pair.data;
		};
		const TData &get() const {
			return pair.data;
		};

		Element(const TKey &p_key) :
				pair(p_key) {}
		Element(const Element &p_other) :
				hash(p_other.hash),
				pair(p_other.pair.key, p_other.pair.data) {}
	};

private:
	Element **hash_table;
	uint8_t hash_table_power;
	uint32_t elements;

	void make_hash_table() {
		ERR_FAIL_COND(hash_table);

		hash_table = memnew_arr(Element *, (1 << MIN_HASH_TABLE_POWER));

		hash_table_power = MIN_HASH_TABLE_POWER;
		elements = 0;
		for (int i = 0; i < (1 << MIN_HASH_TABLE_POWER); i++) {
			hash_table[i] = nullptr;
		}
	}

	void erase_hash_table() {
		ERR_FAIL_COND_MSG(elements, "Cannot erase hash table if there are still elements inside.");

		memdelete_arr(hash_table);
		hash_table = nullptr;
		hash_table_power = 0;
		elements = 0;
	}

	void check_hash_table() {
		int new_hash_table_power = -1;

		if ((int)elements > ((1 << hash_table_power) * RELATIONSHIP)) {
			/* rehash up */
			new_hash_table_power = hash_table_power + 1;

			while ((int)elements > ((1 << new_hash_table_power) * RELATIONSHIP)) {
				new_hash_table_power++;
			}

		} else if ((hash_table_power > (int)MIN_HASH_TABLE_POWER) && ((int)elements < ((1 << (hash_table_power - 1)) * RELATIONSHIP))) {
			/* rehash down */
			new_hash_table_power = hash_table_power - 1;

			while ((int)elements < ((1 << (new_hash_table_power - 1)) * RELATIONSHIP)) {
				new_hash_table_power--;
			}

			if (new_hash_table_power < (int)MIN_HASH_TABLE_POWER) {
				new_hash_table_power = MIN_HASH_TABLE_POWER;
			}
		}

		if (new_hash_table_power == -1) {
			return;
		}

		Element **new_hash_table = memnew_arr(Element *, ((uint64_t)1 << new_hash_table_power));
		ERR_FAIL_COND_MSG(!new_hash_table, "Out of memory.");

		for (int i = 0; i < (1 << new_hash_table_power); i++) {
			new_hash_table[i] = nullptr;
		}

		if (hash_table) {
			for (int i = 0; i < (1 << hash_table_power); i++) {
				while (hash_table[i]) {
					Element *se = hash_table[i];
					hash_table[i] = se->next;
					int new_pos = se->hash & ((1 << new_hash_table_power) - 1);
					se->next = new_hash_table[new_pos];
					new_hash_table[new_pos] = se;
				}
			}

			memdelete_arr(hash_table);
		}
		hash_table = new_hash_table;
		hash_table_power = new_hash_table_power;
	}

	/* I want to have only one function.. */
	_FORCE_INLINE_ const Element *get_element(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_key)) {
				/* the pair exists in this hashtable, so just update data */
				return e;
			}

			e = e->next;
		}

		return nullptr;
	}

	Element *create_element(const TKey &p_key) {
		/* if element doesn't exist, create it */
		Element *e = memnew(Element(p_key));
		ERR_FAIL_COND_V_MSG(!e, nullptr, "Out of memory.");
		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);
		e->next = hash_table[index];
		e->hash = hash;

		hash_table[index] = e;
		elements++;

		return e;
	}

	void copy_from(const OGHashMap &p_t) {
		if (&p_t == this) {
			return; /* much less bother with that */
		}

		clear();

		if (!p_t.hash_table || p_t.hash_table_power == 0) {
			return; /* not copying from empty table */
		}

		hash_table = memnew_arr(Element *, (uint64_t)1 << p_t.hash_table_power);
		hash_table_power = p_t.hash_table_power;
		elements = p_t.elements;

		for (int i = 0; i < (1 << p_t.hash_table_power); i++) {
			hash_table[i] = nullptr;

			const Element *e = p_t.hash_table[i];

			while (e) {
				Element *le = memnew(Element(*e)); /* local element */

				/* add to list and reassign pointers */
				le->next = hash_table[i];
				hash_table[i] = le;

				e = e->next;
			}
		}
	}

public:
	Element *set(const TKey &p_key, const TData &p_data) {
		return set(Pair(p_key, p_data));
	}

	Element *set(const Pair &p_pair) {
		Element *e = nullptr;
		if (!hash_table) {
			make_hash_table(); // if no table, make one
		} else {
			e = const_cast<Element *>(get_element(p_pair.key));
		}

		/* if we made it up to here, the pair doesn't exist, create and assign */

		if (!e) {
			e = create_element(p_pair.key);
			if (!e) {
				return nullptr;
			}
			check_hash_table(); // perform mantenience routine
		}

		e->pair.data = p_pair.data;
		return e;
	}

	bool has(const TKey &p_key) const {
		return getptr(p_key) != nullptr;
	}

	/**
	 * Get a key from data, return a const reference.
	 * WARNING: this doesn't check errors, use either getptr and check NULL, or check
	 * first with has(key)
	 */

	const TData &get(const TKey &p_key) const {
		const TData *res = getptr(p_key);
		CRASH_COND_MSG(!res, "Map key not found.");
		return *res;
	}

	TData &get(const TKey &p_key) {
		TData *res = getptr(p_key);
		CRASH_COND_MSG(!res, "Map key not found.");
		return *res;
	}

	/**
	 * Same as get, except it can return NULL when item was not found.
	 * This is mainly used for speed purposes.
	 */

	_FORCE_INLINE_ TData *getptr(const TKey &p_key) {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		Element *e = const_cast<Element *>(get_element(p_key));

		if (e) {
			return &e->pair.data;
		}

		return nullptr;
	}

	_FORCE_INLINE_ const TData *getptr(const TKey &p_key) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		const Element *e = const_cast<Element *>(get_element(p_key));

		if (e) {
			return &e->pair.data;
		}

		return nullptr;
	}

	const Element *find(const TKey &p_key) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		const Element *e = const_cast<Element *>(get_element(p_key));

		return e;
	}

	Element *find(const TKey &p_key) {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		Element *e = const_cast<Element *>(get_element(p_key));

		return e;
	}

	/**
	 * Same as get, except it can return NULL when item was not found.
	 * This version is custom, will take a hash and a custom key (that should support operator==()
	 */

	template <class C>
	_FORCE_INLINE_ TData *custom_getptr(C p_custom_key, uint32_t p_custom_hash) {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_custom_key)) {
				/* the pair exists in this hashtable, so just update data */
				return &e->pair.data;
			}

			e = e->next;
		}

		return nullptr;
	}

	template <class C>
	_FORCE_INLINE_ const TData *custom_getptr(C p_custom_key, uint32_t p_custom_hash) const {
		if (unlikely(!hash_table)) {
			return NULL;
		}

		uint32_t hash = p_custom_hash;
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		const Element *e = hash_table[index];

		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_custom_key)) {
				/* the pair exists in this hashtable, so just update data */
				return &e->pair.data;
			}

			e = e->next;
		}

		return NULL;
	}

	/**
	 * Erase an item, return true if erasing was successful
	 */

	bool erase(const TKey &p_key) {
		if (unlikely(!hash_table)) {
			return false;
		}

		uint32_t hash = Hasher::hash(p_key);
		uint32_t index = hash & ((1 << hash_table_power) - 1);

		Element *e = hash_table[index];
		Element *p = nullptr;
		while (e) {
			/* checking hash first avoids comparing key, which may take longer */
			if (e->hash == hash && Comparator::compare(e->pair.key, p_key)) {
				if (p) {
					p->next = e->next;
				} else {
					//begin of list
					hash_table[index] = e->next;
				}

				memdelete(e);
				elements--;

				if (elements == 0) {
					erase_hash_table();
				} else {
					check_hash_table();
				}
				return true;
			}

			p = e;
			e = e->next;
		}

		return false;
	}

	inline const TData &operator[](const TKey &p_key) const { //constref

		return get(p_key);
	}
	inline TData &operator[](const TKey &p_key) { //assignment

		Element *e = nullptr;
		if (!hash_table) {
			make_hash_table(); // if no table, make one
		} else {
			e = const_cast<Element *>(get_element(p_key));
		}

		/* if we made it up to here, the pair doesn't exist, create */
		if (!e) {
			e = create_element(p_key);
			CRASH_COND(!e);
			check_hash_table(); // perform mantenience routine
		}

		return e->pair.data;
	}

	/**
	 * Get the next key to p_key, and the first key if p_key is null.
	 * Returns a pointer to the next key if found, NULL otherwise.
	 * Adding/Removing elements while iterating will, of course, have unexpected results, don't do it.
	 *
	 * Example:
	 *
	 * 	const TKey *k=NULL;
	 *
	 * 	while( (k=table.next(k)) ) {
	 *
	 * 		print( *k );
	 * 	}
	 *
	 */
	const TKey *next(const TKey *p_key) const {
		if (unlikely(!hash_table)) {
			return nullptr;
		}

		if (!p_key) { /* get the first key */

			for (int i = 0; i < (1 << hash_table_power); i++) {
				if (hash_table[i]) {
					return &hash_table[i]->pair.key;
				}
			}

		} else { /* get the next key */

			const Element *e = get_element(*p_key);
			ERR_FAIL_COND_V_MSG(!e, nullptr, "Invalid key supplied.");
			if (e->next) {
				/* if there is a "next" in the list, return that */
				return &e->next->pair.key;
			} else {
				/* go to next elements */
				uint32_t index = e->hash & ((1 << hash_table_power) - 1);
				index++;
				for (int i = index; i < (1 << hash_table_power); i++) {
					if (hash_table[i]) {
						return &hash_table[i]->pair.key;
					}
				}
			}

			/* nothing found, was at end */
		}

		return nullptr; /* nothing found */
	}

	inline unsigned int size() const {
		return elements;
	}

	inline bool empty() const {
		return elements == 0;
	}

	void clear() {
		/* clean up */
		if (hash_table) {
			for (int i = 0; i < (1 << hash_table_power); i++) {
				while (hash_table[i]) {
					Element *e = hash_table[i];
					hash_table[i] = e->next;
					memdelete(e);
				}
			}

			memdelete_arr(hash_table);
		}

		hash_table = nullptr;
		hash_table_power = 0;
		elements = 0;
	}

	void operator=(const OGHashMap &p_table) {
		copy_from(p_table);
	}

	OGHashMap() {
		hash_table = nullptr;
		elements = 0;
		hash_table_power = 0;
	}

	void get_key_value_ptr_array(const Pair **p_pairs) const {
		if (unlikely(!hash_table)) {
			return;
		}
		for (int i = 0; i < (1 << hash_table_power); i++) {
			Element *e = hash_table[i];
			while (e) {
				*p_pairs = &e->pair;
				p_pairs++;
				e = e->next;
			}
		}
	}

	void get_key_list(List<TKey> *p_keys) const {
		if (unlikely(!hash_table)) {
			return;
		}
		for (int i = 0; i < (1 << hash_table_power); i++) {
			Element *e = hash_table[i];
			while (e) {
				p_keys->push_back(e->pair.key);
				e = e->next;
			}
		}
	}

	OGHashMap(const OGHashMap &p_table) {
		hash_table = nullptr;
		elements = 0;
		hash_table_power = 0;

		copy_from(p_table);
	}

	~OGHashMap() {
		clear();
	}
};

#line 0

#line 1 "sfwl/core/ordered_hash_map.h"

/*************************************************************************/
/*  ordered_hash_map.h                                                   */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * A hash map which allows to iterate elements in insertion order.
 * Insertion, lookup, deletion have O(1) complexity.
 * The API aims to be consistent with Map rather than HashMap, because the
 * former is more frequently used and is more coherent with the rest of the
 * codebase.
 * Deletion during iteration is safe and will preserve the order.
 */
template <class K, class V, class Hasher = HashMapHasherDefault, class Comparator = HashMapComparatorDefault<K>, uint8_t MIN_HASH_TABLE_POWER = 3, uint8_t RELATIONSHIP = 8>
class OrderedHashMap {
	typedef List<Pair<const K *, V>> InternalList;
	typedef OGHashMap<K, typename InternalList::Element *, Hasher, Comparator, MIN_HASH_TABLE_POWER, RELATIONSHIP> InternalMap;

	InternalList list;
	InternalMap map;

public:
	class Element {
		friend class OrderedHashMap<K, V, Hasher, Comparator, MIN_HASH_TABLE_POWER, RELATIONSHIP>;

		typename InternalList::Element *list_element;
		typename InternalList::Element *prev_element;
		typename InternalList::Element *next_element;

		Element(typename InternalList::Element *p_element) {
			list_element = p_element;

			if (list_element) {
				next_element = list_element->next();
				prev_element = list_element->prev();
			}
		}

	public:
		_FORCE_INLINE_ Element() :
				list_element(nullptr),
				prev_element(nullptr),
				next_element(nullptr) {
		}

		Element next() const {
			return Element(next_element);
		}

		Element prev() const {
			return Element(prev_element);
		}

		Element(const Element &other) :
				list_element(other.list_element),
				prev_element(other.prev_element),
				next_element(other.next_element) {
		}

		Element &operator=(const Element &other) {
			list_element = other.list_element;
			next_element = other.next_element;
			prev_element = other.prev_element;
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Element &p_other) const {
			return this->list_element == p_other.list_element;
		}
		_FORCE_INLINE_ bool operator!=(const Element &p_other) const {
			return this->list_element != p_other.list_element;
		}

		operator bool() const {
			return (list_element != nullptr);
		}

		const K &key() const {
			CRASH_COND(!list_element);
			return *(list_element->get().first);
		};

		V &value() {
			CRASH_COND(!list_element);
			return list_element->get().second;
		};

		const V &value() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		};

		V &get() {
			CRASH_COND(!list_element);
			return list_element->get().second;
		};

		const V &get() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		};
	};

	class ConstElement {
		friend class OrderedHashMap<K, V, Hasher, Comparator, MIN_HASH_TABLE_POWER, RELATIONSHIP>;

		const typename InternalList::Element *list_element;

		ConstElement(const typename InternalList::Element *p_element) :
				list_element(p_element) {
		}

	public:
		_FORCE_INLINE_ ConstElement() :
				list_element(NULL) {
		}

		ConstElement(const ConstElement &other) :
				list_element(other.list_element) {
		}

		ConstElement &operator=(const ConstElement &other) {
			list_element = other.list_element;
			return *this;
		}

		ConstElement next() const {
			return ConstElement(list_element ? list_element->next() : nullptr);
		}

		ConstElement prev() const {
			return ConstElement(list_element ? list_element->prev() : NULL);
		}

		_FORCE_INLINE_ bool operator==(const ConstElement &p_other) const {
			return this->list_element == p_other.list_element;
		}
		_FORCE_INLINE_ bool operator!=(const ConstElement &p_other) const {
			return this->list_element != p_other.list_element;
		}

		operator bool() const {
			return (list_element != nullptr);
		}

		const K &key() const {
			CRASH_COND(!list_element);
			return *(list_element->get().first);
		};

		const V &value() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		};

		const V &get() const {
			CRASH_COND(!list_element);
			return list_element->get().second;
		};
	};

	ConstElement find(const K &p_key) const {
		typename InternalList::Element *const *list_element = map.getptr(p_key);
		if (list_element) {
			return ConstElement(*list_element);
		}
		return ConstElement(nullptr);
	}

	Element find(const K &p_key) {
		typename InternalList::Element **list_element = map.getptr(p_key);
		if (list_element) {
			return Element(*list_element);
		}
		return Element(nullptr);
	}

	Element insert(const K &p_key, const V &p_value) {
		typename InternalList::Element **list_element = map.getptr(p_key);
		if (list_element) {
			(*list_element)->get().second = p_value;
			return Element(*list_element);
		}
		// Incorrectly set the first value of the pair with a value that will
		// be invalid as soon as we leave this function...
		typename InternalList::Element *new_element = list.push_back(Pair<const K *, V>(&p_key, p_value));
		// ...this is needed here in case the hashmap recursively reference itself...
		typename InternalMap::Element *e = map.set(p_key, new_element);
		// ...now we can set the right value !
		new_element->get().first = &e->key();

		return Element(new_element);
	}

	void erase(Element &p_element) {
		map.erase(p_element.key());
		list.erase(p_element.list_element);
		p_element.list_element = nullptr;
	}

	bool erase(const K &p_key) {
		typename InternalList::Element **list_element = map.getptr(p_key);
		if (list_element) {
			list.erase(*list_element);
			map.erase(p_key);
			return true;
		}
		return false;
	}

	inline bool has(const K &p_key) const {
		return map.has(p_key);
	}

	const V &operator[](const K &p_key) const {
		ConstElement e = find(p_key);
		CRASH_COND(!e);
		return e.value();
	}

	V &operator[](const K &p_key) {
		Element e = find(p_key);
		if (!e) {
			// consistent with Map behaviour
			e = insert(p_key, V());
		}
		return e.value();
	}

	inline Element front() {
		return Element(list.front());
	}

	inline Element back() {
		return Element(list.back());
	}

	inline ConstElement front() const {
		return ConstElement(list.front());
	}

	inline ConstElement back() const {
		return ConstElement(list.back());
	}

	inline bool empty() const { return list.empty(); }
	inline int size() const { return list.size(); }

	const void *id() const {
		return list.id();
	}

	void clear() {
		map.clear();
		list.clear();
	}

private:
	void _copy_from(const OrderedHashMap &p_map) {
		for (ConstElement E = p_map.front(); E; E = E.next()) {
			insert(E.key(), E.value());
		}
	}

public:
	void operator=(const OrderedHashMap &p_map) {
		_copy_from(p_map);
	}

	OrderedHashMap(const OrderedHashMap &p_map) {
		_copy_from(p_map);
	}

	_FORCE_INLINE_ OrderedHashMap() {
	}
};

#line 0

#line 1 "sfwl/core/hash_map.h"

/*************************************************************************/
/*  hash_map.h                                                           */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * A HashMap implementation that uses open addressing with Robin Hood hashing.
 * Robin Hood hashing swaps out entries that have a smaller probing distance
 * than the to-be-inserted entry, that evens out the average probing distance
 * and enables faster lookups. Backward shift deletion is employed to further
 * improve the performance and to avoid infinite loops in rare cases.
 *
 * Keys and values are stored in a double linked list by insertion order. This
 * has a slight performance overhead on lookup, which can be mostly compensated
 * using a paged allocator if required.
 *
 * The assignment operator copy the pairs from one map to the other.
 */

template <class TKey, class TValue, class Hasher = HashMapHasherDefault, class Comparator = HashMapComparatorDefault<TKey>>
class HashMap {
public:
	const uint32_t MIN_CAPACITY_INDEX = 2; // Use a prime.
	const float MAX_OCCUPANCY = 0.75;
	const uint32_t EMPTY_HASH = 0;

public:
	struct Element {
		Element *next = nullptr;
		Element *prev = nullptr;
		KeyValue<TKey, TValue> data;

		const TKey &key() const {
			return data.key;
		}

		TValue &value() {
			return data.value;
		}

		const TValue &value() const {
			return data.value;
		}

		TValue &get() {
			return data.value;
		};
		const TValue &get() const {
			return data.value;
		};

		Element() {}
		Element(const TKey &p_key, const TValue &p_value) :
				data(p_key, p_value) {}
	};

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return hash_table_size_primes[capacity_index]; }
	_FORCE_INLINE_ uint32_t size() const { return num_elements; }

	/* Standard Godot Container API */

	bool empty() const {
		return num_elements == 0;
	}

	void clear() {
		if (elements == nullptr || num_elements == 0) {
			return;
		}
		uint32_t capacity = hash_table_size_primes[capacity_index];
		for (uint32_t i = 0; i < capacity; i++) {
			if (hashes[i] == EMPTY_HASH) {
				continue;
			}

			hashes[i] = EMPTY_HASH;
			memdelete(elements[i]);
			elements[i] = nullptr;
		}

		tail_element = nullptr;
		head_element = nullptr;
		num_elements = 0;
	}

	TValue &get(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		CRASH_COND_MSG(!exists, "HashMap key not found.");
		return elements[pos]->data.value;
	}

	const TValue &get(const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		CRASH_COND_MSG(!exists, "HashMap key not found.");
		return elements[pos]->data.value;
	}

	const TValue *getptr(const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			return &elements[pos]->data.value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			return &elements[pos]->data.value;
		}
		return nullptr;
	}

	const Element *get_element(const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			return elements[pos];
		}

		return NULL;
	}

	Element *get_element(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			return elements[pos];
		}

		return NULL;
	}

	_FORCE_INLINE_ const Element *find(const TKey &p_key) const {
		return get_element(p_key);
	}

	_FORCE_INLINE_ Element *find(const TKey &p_key) {
		return get_element(p_key);
	}

	/**
	 * Same as get, except it can return NULL when item was not found.
	 * This version is custom, will take a hash and a custom key (that should support operator==()
	 */

	template <class C>
	_FORCE_INLINE_ TValue *custom_getptr(C p_custom_key, uint32_t p_custom_hash) {
		if (unlikely(!elements)) {
			return NULL;
		}

		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t hash = p_custom_hash;
		uint32_t pos = fastmod(hash, capacity_inv, capacity);
		uint32_t distance = 0;

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				return NULL;
			}

			if (distance > _get_probe_length(pos, hashes[pos], capacity, capacity_inv)) {
				return NULL;
			}

			if (hashes[pos] == hash && Comparator::compare(elements[pos]->data.key, p_custom_key)) {
				return &elements[pos]->data.value;
			}

			pos = fastmod((pos + 1), capacity_inv, capacity);
			distance++;
		}

		return NULL;
	}

	template <class C>
	_FORCE_INLINE_ const TValue *custom_getptr(C p_custom_key, uint32_t p_custom_hash) const {
		if (unlikely(!elements)) {
			return NULL;
		}

		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t hash = p_custom_hash;
		uint32_t pos = fastmod(hash, capacity_inv, capacity);
		uint32_t distance = 0;

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				return NULL;
			}

			if (distance > _get_probe_length(pos, hashes[pos], capacity, capacity_inv)) {
				return NULL;
			}

			if (hashes[pos] == hash && Comparator::compare(elements[pos]->data.key, p_custom_key)) {
				return &elements[pos]->data.value;
			}

			pos = fastmod((pos + 1), capacity_inv, capacity);
			distance++;
		}

		return NULL;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) const {
		uint32_t _pos = 0;
		return _lookup_pos(p_key, _pos);
	}

	bool erase(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (!exists) {
			return false;
		}

		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t next_pos = fastmod((pos + 1), capacity_inv, capacity);
		while (hashes[next_pos] != EMPTY_HASH && _get_probe_length(next_pos, hashes[next_pos], capacity, capacity_inv) != 0) {
			SWAP(hashes[next_pos], hashes[pos]);
			SWAP(elements[next_pos], elements[pos]);
			pos = next_pos;
			next_pos = fastmod((pos + 1), capacity_inv, capacity);
		}

		hashes[pos] = EMPTY_HASH;

		if (head_element == elements[pos]) {
			head_element = elements[pos]->next;
		}

		if (tail_element == elements[pos]) {
			tail_element = elements[pos]->prev;
		}

		if (elements[pos]->prev) {
			elements[pos]->prev->next = elements[pos]->next;
		}

		if (elements[pos]->next) {
			elements[pos]->next->prev = elements[pos]->prev;
		}

		memdelete(elements[pos]);
		elements[pos] = nullptr;

		num_elements--;
		return true;
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		uint32_t new_index = capacity_index;

		while (hash_table_size_primes[new_index] < p_new_capacity) {
			ERR_FAIL_COND_MSG(new_index + 1 == (uint32_t)HASH_TABLE_SIZE_MAX, nullptr);
			new_index++;
		}

		if (new_index == capacity_index) {
			return;
		}

		if (elements == nullptr) {
			capacity_index = new_index;
			return; // Unallocated yet.
		}
		_resize_and_rehash(new_index);
	}

	_FORCE_INLINE_ Element *front() {
		return head_element;
	}
	_FORCE_INLINE_ Element *back() {
		return tail_element;
	}

	_FORCE_INLINE_ const Element *front() const {
		return head_element;
	}
	_FORCE_INLINE_ const Element *back() const {
		return tail_element;
	}

	/**
	 * Get the next key to p_key, and the first key if p_key is null.
	 * Returns a pointer to the next key if found, NULL otherwise.
	 * Adding/Removing elements while iterating will, of course, have unexpected results, don't do it.
	 *
	 * Example:
	 *
	 * 	const TKey *k=NULL;
	 *
	 * 	while( (k=table.next(k)) ) {
	 *
	 * 		print( *k );
	 * 	}
	 *
	 * This is for backwards compatibility. Use this syntax instead for new code:
	 *
	 * for (const HashMap<K, V>::Element *E = map.front(); E; E = E->next) {
	 *     ...
	 * }
	 *
	 */
	const TKey *next(const TKey *p_key) const {
		if (unlikely(!elements)) {
			return nullptr;
		}

		if (!p_key) { /* get the first key */

			if (unlikely(!front())) {
				return nullptr;
			}

			return &front()->data.key;

		} else { /* get the next key */

			const Element *e = get_element(*p_key);
			ERR_FAIL_COND_V_MSG(!e, nullptr, "Invalid key supplied.");
			if (e->next) {
				/* if there is a "next" in the list, return that */
				return &e->next->data.key;
			}

			/* nothing found, was at end */
		}

		return nullptr; /* nothing found */
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		CRASH_COND(!exists);
		return elements[pos]->data.value;
	}

	TValue &operator[](const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		if (!exists) {
			return _insert(p_key, TValue())->data.value;
		} else {
			return elements[pos]->data.value;
		}
	}

	/* Insert */

	Element *insert(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		return _insert(p_key, p_value, p_front_insert);
	}

	Element *set(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		return _insert(p_key, p_value, p_front_insert);
	}

	/* Helpers */

	void get_key_list(List<TKey> *p_keys) const {
		if (unlikely(!elements)) {
			return;
		}

		for (const Element *E = front(); E; E = E->next) {
			p_keys->push_back(E->data.key);
		}
	}

	/* Constructors */

	HashMap(const HashMap &p_other) {
		reserve(hash_table_size_primes[p_other.capacity_index]);

		if (p_other.num_elements == 0) {
			return;
		}

		for (const Element *E = p_other.front(); E; E = E->next) {
			insert(E->data.key, E->data.value);
		}
	}

	void operator=(const HashMap &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}
		if (num_elements != 0) {
			clear();
		}

		reserve(hash_table_size_primes[p_other.capacity_index]);

		if (p_other.elements == nullptr) {
			return; // Nothing to copy.
		}

		for (const Element *E = p_other.front(); E; E = E->next) {
			insert(E->data.key, E->data.value);
		}
	}

	HashMap(uint32_t p_initial_capacity) {
		// Capacity can't be 0.
		capacity_index = 0;
		reserve(p_initial_capacity);
	}
	HashMap() {
		capacity_index = MIN_CAPACITY_INDEX;
	}

	uint32_t debug_get_hash(uint32_t p_index) {
		if (num_elements == 0) {
			return 0;
		}
		ERR_FAIL_INDEX_V(p_index, get_capacity(), 0);
		return hashes[p_index];
	}
	Element *debug_get_element(uint32_t p_index) {
		if (num_elements == 0) {
			return NULL;
		}

		ERR_FAIL_INDEX_V(p_index, get_capacity(), NULL);

		return elements[p_index];
	}

	~HashMap() {
		clear();

		if (elements != nullptr) {
			Memory::free_static(elements);
			Memory::free_static(hashes);
		}
	}

private:
	Element **elements = nullptr;
	uint32_t *hashes = nullptr;
	Element *head_element = nullptr;
	Element *tail_element = nullptr;

	uint32_t capacity_index = 0;
	uint32_t num_elements = 0;

	_FORCE_INLINE_ uint32_t _hash(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);

		if (unlikely(hash == EMPTY_HASH)) {
			hash = EMPTY_HASH + 1;
		}

		return hash;
	}

	static _FORCE_INLINE_ uint32_t _get_probe_length(const uint32_t p_pos, const uint32_t p_hash, const uint32_t p_capacity, const uint64_t p_capacity_inv) {
		const uint32_t original_pos = fastmod(p_hash, p_capacity_inv, p_capacity);
		return fastmod(p_pos - original_pos + p_capacity, p_capacity_inv, p_capacity);
	}

	bool _lookup_pos(const TKey &p_key, uint32_t &r_pos) const {
		if (elements == nullptr || num_elements == 0) {
			return false; // Failed lookups, no elements
		}

		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t hash = _hash(p_key);
		uint32_t pos = fastmod(hash, capacity_inv, capacity);
		uint32_t distance = 0;

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				return false;
			}

			if (distance > _get_probe_length(pos, hashes[pos], capacity, capacity_inv)) {
				return false;
			}

			if (hashes[pos] == hash && Comparator::compare(elements[pos]->data.key, p_key)) {
				r_pos = pos;
				return true;
			}

			pos = fastmod((pos + 1), capacity_inv, capacity);
			distance++;
		}
	}

	void _insert_with_hash(uint32_t p_hash, Element *p_value) {
		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t hash = p_hash;
		Element *value = p_value;
		uint32_t distance = 0;
		uint32_t pos = fastmod(hash, capacity_inv, capacity);

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				elements[pos] = value;
				hashes[pos] = hash;

				num_elements++;

				return;
			}

			// Not an empty slot, let's check the probing length of the existing one.
			uint32_t existing_probe_len = _get_probe_length(pos, hashes[pos], capacity, capacity_inv);
			if (existing_probe_len < distance) {
				SWAP(hash, hashes[pos]);
				SWAP(value, elements[pos]);
				distance = existing_probe_len;
			}

			pos = fastmod((pos + 1), capacity_inv, capacity);
			distance++;
		}
	}

	void _resize_and_rehash(uint32_t p_new_capacity_index) {
		uint32_t old_capacity = hash_table_size_primes[capacity_index];

		// Capacity can't be 0.
		capacity_index = MAX((uint32_t)MIN_CAPACITY_INDEX, p_new_capacity_index);

		uint32_t capacity = hash_table_size_primes[capacity_index];

		Element **old_elements = elements;
		uint32_t *old_hashes = hashes;

		num_elements = 0;
		hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
		elements = reinterpret_cast<Element **>(Memory::alloc_static(sizeof(Element *) * capacity));

		for (uint32_t i = 0; i < capacity; i++) {
			hashes[i] = 0;
			elements[i] = nullptr;
		}

		if (old_capacity == 0) {
			// Nothing to do.
			return;
		}

		for (uint32_t i = 0; i < old_capacity; i++) {
			if (old_hashes[i] == EMPTY_HASH) {
				continue;
			}

			_insert_with_hash(old_hashes[i], old_elements[i]);
		}

		Memory::free_static(old_elements);
		Memory::free_static(old_hashes);
	}

	_FORCE_INLINE_ Element *_insert(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		uint32_t capacity = hash_table_size_primes[capacity_index];
		if (unlikely(elements == nullptr)) {
			// Allocate on demand to save memory.

			hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
			elements = reinterpret_cast<Element **>(Memory::alloc_static(sizeof(Element *) * capacity));

			for (uint32_t i = 0; i < capacity; i++) {
				hashes[i] = EMPTY_HASH;
				elements[i] = nullptr;
			}
		}

		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			elements[pos]->data.value = p_value;
			return elements[pos];
		} else {
			if (num_elements + 1 > MAX_OCCUPANCY * capacity) {
				ERR_FAIL_COND_V_MSG(capacity_index + 1 == HASH_TABLE_SIZE_MAX, nullptr, "Hash table maximum capacity reached, aborting insertion.");
				_resize_and_rehash(capacity_index + 1);
			}

			Element *elem = memnew(Element(p_key, p_value));

			if (tail_element == nullptr) {
				head_element = elem;
				tail_element = elem;
			} else if (p_front_insert) {
				head_element->prev = elem;
				elem->next = head_element;
				head_element = elem;
			} else {
				tail_element->next = elem;
				elem->prev = tail_element;
				tail_element = elem;
			}

			uint32_t hash = _hash(p_key);
			_insert_with_hash(hash, elem);
			return elem;
		}
	}
};

#line 0

#line 1 "sfwl/core/hash_set.h"

/*************************************************************************/
/*  hash_set.h                                                           */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
 * Implementation of Set using a bidi indexed hash map.
 * Use RBSet instead of this only if the following conditions are met:
 *
 * - You need to keep an iterator or const pointer to Key and you intend to add/remove elements in the meantime.
 * - Iteration order does matter (via operator<)
 *
 */

template <class TKey,
		class Hasher = HashMapHasherDefault,
		class Comparator = HashMapComparatorDefault<TKey>>
class HashSet {
public:
	static constexpr uint32_t MIN_CAPACITY_INDEX = 2; // Use a prime.
	static constexpr float MAX_OCCUPANCY = 0.75;
	static constexpr uint32_t EMPTY_HASH = 0;

private:
	TKey *keys = nullptr;
	uint32_t *hash_to_key = nullptr;
	uint32_t *key_to_hash = nullptr;
	uint32_t *hashes = nullptr;

	uint32_t capacity_index = 0;
	uint32_t num_elements = 0;

	_FORCE_INLINE_ uint32_t _hash(const TKey &p_key) const {
		uint32_t hash = Hasher::hash(p_key);

		if (unlikely(hash == EMPTY_HASH)) {
			hash = EMPTY_HASH + 1;
		}

		return hash;
	}

	static _FORCE_INLINE_ uint32_t _get_probe_length(const uint32_t p_pos, const uint32_t p_hash, const uint32_t p_capacity, const uint64_t p_capacity_inv) {
		const uint32_t original_pos = fastmod(p_hash, p_capacity_inv, p_capacity);
		return fastmod(p_pos - original_pos + p_capacity, p_capacity_inv, p_capacity);
	}

	bool _lookup_pos(const TKey &p_key, uint32_t &r_pos) const {
		if (keys == nullptr || num_elements == 0) {
			return false; // Failed lookups, no elements
		}

		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t hash = _hash(p_key);
		uint32_t pos = fastmod(hash, capacity_inv, capacity);
		uint32_t distance = 0;

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				return false;
			}

			if (distance > _get_probe_length(pos, hashes[pos], capacity, capacity_inv)) {
				return false;
			}

			if (hashes[pos] == hash && Comparator::compare(keys[hash_to_key[pos]], p_key)) {
				r_pos = hash_to_key[pos];
				return true;
			}

			pos = fastmod(pos + 1, capacity_inv, capacity);
			distance++;
		}
	}

	uint32_t _insert_with_hash(uint32_t p_hash, uint32_t p_index) {
		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t hash = p_hash;
		uint32_t index = p_index;
		uint32_t distance = 0;
		uint32_t pos = fastmod(hash, capacity_inv, capacity);

		while (true) {
			if (hashes[pos] == EMPTY_HASH) {
				hashes[pos] = hash;
				key_to_hash[index] = pos;
				hash_to_key[pos] = index;
				return pos;
			}

			// Not an empty slot, let's check the probing length of the existing one.
			uint32_t existing_probe_len = _get_probe_length(pos, hashes[pos], capacity, capacity_inv);
			if (existing_probe_len < distance) {
				key_to_hash[index] = pos;
				SWAP(hash, hashes[pos]);
				SWAP(index, hash_to_key[pos]);
				distance = existing_probe_len;
			}

			pos = fastmod(pos + 1, capacity_inv, capacity);
			distance++;
		}
	}

	void _resize_and_rehash(uint32_t p_new_capacity_index) {
		// Capacity can't be 0.
		capacity_index = MAX((uint32_t)MIN_CAPACITY_INDEX, p_new_capacity_index);

		uint32_t capacity = hash_table_size_primes[capacity_index];

		uint32_t *old_hashes = hashes;
		uint32_t *old_key_to_hash = key_to_hash;

		hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
		keys = reinterpret_cast<TKey *>(Memory::realloc_static(keys, sizeof(TKey) * capacity));
		key_to_hash = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
		hash_to_key = reinterpret_cast<uint32_t *>(Memory::realloc_static(hash_to_key, sizeof(uint32_t) * capacity));

		for (uint32_t i = 0; i < capacity; i++) {
			hashes[i] = EMPTY_HASH;
		}

		for (uint32_t i = 0; i < num_elements; i++) {
			uint32_t h = old_hashes[old_key_to_hash[i]];
			_insert_with_hash(h, i);
		}

		Memory::free_static(old_hashes);
		Memory::free_static(old_key_to_hash);
	}

	_FORCE_INLINE_ int32_t _insert(const TKey &p_key) {
		uint32_t capacity = hash_table_size_primes[capacity_index];
		if (unlikely(keys == nullptr)) {
			// Allocate on demand to save memory.

			hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
			keys = reinterpret_cast<TKey *>(Memory::alloc_static(sizeof(TKey) * capacity));
			key_to_hash = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
			hash_to_key = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));

			for (uint32_t i = 0; i < capacity; i++) {
				hashes[i] = EMPTY_HASH;
			}
		}

		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (exists) {
			return pos;
		} else {
			if (num_elements + 1 > MAX_OCCUPANCY * capacity) {
				ERR_FAIL_COND_V_MSG(capacity_index + 1 == HASH_TABLE_SIZE_MAX, -1, "Hash table maximum capacity reached, aborting insertion.");
				_resize_and_rehash(capacity_index + 1);
			}

			uint32_t hash = _hash(p_key);
			memnew_placement(&keys[num_elements], TKey(p_key));
			_insert_with_hash(hash, num_elements);
			num_elements++;
			return num_elements - 1;
		}
	}

	void _init_from(const HashSet &p_other) {
		capacity_index = p_other.capacity_index;
		num_elements = p_other.num_elements;

		if (p_other.num_elements == 0) {
			return;
		}

		uint32_t capacity = hash_table_size_primes[capacity_index];

		hashes = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
		keys = reinterpret_cast<TKey *>(Memory::alloc_static(sizeof(TKey) * capacity));
		key_to_hash = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));
		hash_to_key = reinterpret_cast<uint32_t *>(Memory::alloc_static(sizeof(uint32_t) * capacity));

		for (uint32_t i = 0; i < num_elements; i++) {
			memnew_placement(&keys[i], TKey(p_other.keys[i]));
			key_to_hash[i] = p_other.key_to_hash[i];
		}

		for (uint32_t i = 0; i < capacity; i++) {
			hashes[i] = p_other.hashes[i];
			hash_to_key[i] = p_other.hash_to_key[i];
		}
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return hash_table_size_primes[capacity_index]; }
	_FORCE_INLINE_ uint32_t size() const { return num_elements; }

	/* Standard Godot Container API */

	bool is_empty() const {
		return num_elements == 0;
	}

	void clear() {
		if (keys == nullptr || num_elements == 0) {
			return;
		}
		uint32_t capacity = hash_table_size_primes[capacity_index];
		for (uint32_t i = 0; i < capacity; i++) {
			hashes[i] = EMPTY_HASH;
		}
		for (uint32_t i = 0; i < num_elements; i++) {
			keys[i].~TKey();
		}

		num_elements = 0;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) const {
		uint32_t _pos = 0;
		return _lookup_pos(p_key, _pos);
	}

	bool erase(const TKey &p_key) {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);

		if (!exists) {
			return false;
		}

		uint32_t key_pos = pos;
		pos = key_to_hash[pos]; //make hash pos

		const uint32_t capacity = hash_table_size_primes[capacity_index];
		const uint64_t capacity_inv = hash_table_size_primes_inv[capacity_index];
		uint32_t next_pos = fastmod(pos + 1, capacity_inv, capacity);
		while (hashes[next_pos] != EMPTY_HASH && _get_probe_length(next_pos, hashes[next_pos], capacity, capacity_inv) != 0) {
			uint32_t kpos = hash_to_key[pos];
			uint32_t kpos_next = hash_to_key[next_pos];
			SWAP(key_to_hash[kpos], key_to_hash[kpos_next]);
			SWAP(hashes[next_pos], hashes[pos]);
			SWAP(hash_to_key[next_pos], hash_to_key[pos]);

			pos = next_pos;
			next_pos = fastmod(pos + 1, capacity_inv, capacity);
		}

		hashes[pos] = EMPTY_HASH;
		keys[key_pos].~TKey();
		num_elements--;
		if (key_pos < num_elements) {
			// Not the last key, move the last one here to keep keys lineal
			memnew_placement(&keys[key_pos], TKey(keys[num_elements]));
			keys[num_elements].~TKey();
			key_to_hash[key_pos] = key_to_hash[num_elements];
			hash_to_key[key_to_hash[num_elements]] = key_pos;
		}

		return true;
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		uint32_t new_index = capacity_index;

		while (hash_table_size_primes[new_index] < p_new_capacity) {
			ERR_FAIL_COND_MSG(new_index + 1 == (uint32_t)HASH_TABLE_SIZE_MAX, nullptr);
			new_index++;
		}

		if (new_index == capacity_index) {
			return;
		}

		if (keys == nullptr) {
			capacity_index = new_index;
			return; // Unallocated yet.
		}
		_resize_and_rehash(new_index);
	}

	/** Iterator API **/

	struct Iterator {
		_FORCE_INLINE_ const TKey &operator*() const {
			return keys[index];
		}
		_FORCE_INLINE_ const TKey *operator->() const {
			return &keys[index];
		}
		_FORCE_INLINE_ Iterator &operator++() {
			index++;
			if (index >= (int32_t)num_keys) {
				index = -1;
				keys = nullptr;
				num_keys = 0;
			}
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			index--;
			if (index < 0) {
				index = -1;
				keys = nullptr;
				num_keys = 0;
			}
			return *this;
		}

		_FORCE_INLINE_ const TKey &key() const {
			return keys[index];
		}
		_FORCE_INLINE_ const TKey *key_ptr() const {
			return &keys[index];
		}

		_FORCE_INLINE_ Iterator &next() {
			index++;
			if (index >= (int32_t)num_keys) {
				index = -1;
				keys = nullptr;
				num_keys = 0;
			}
			return *this;
		}
		_FORCE_INLINE_ Iterator &prev() {
			index--;
			if (index < 0) {
				index = -1;
				keys = nullptr;
				num_keys = 0;
			}
			return *this;
		}

		_FORCE_INLINE_ bool valid() const {
			return keys != nullptr;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return keys == b.keys && index == b.index; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return keys != b.keys || index != b.index; }

		_FORCE_INLINE_ explicit operator bool() const {
			return keys != nullptr;
		}

		_FORCE_INLINE_ Iterator(const TKey *p_keys, uint32_t p_num_keys, int32_t p_index = -1) {
			keys = p_keys;
			num_keys = p_num_keys;
			index = p_index;
		}
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) {
			keys = p_it.keys;
			num_keys = p_it.num_keys;
			index = p_it.index;
		}
		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			keys = p_it.keys;
			num_keys = p_it.num_keys;
			index = p_it.index;
		}

	private:
		const TKey *keys = nullptr;
		uint32_t num_keys = 0;
		int32_t index = -1;
	};

	_FORCE_INLINE_ Iterator begin() const {
		return num_elements ? Iterator(keys, num_elements, 0) : Iterator();
	}
	_FORCE_INLINE_ Iterator end() const {
		return Iterator();
	}
	_FORCE_INLINE_ Iterator last() const {
		if (num_elements == 0) {
			return Iterator();
		}
		return Iterator(keys, num_elements, num_elements - 1);
	}

	_FORCE_INLINE_ Iterator find(const TKey &p_key) const {
		uint32_t pos = 0;
		bool exists = _lookup_pos(p_key, pos);
		if (!exists) {
			return end();
		}
		return Iterator(keys, num_elements, pos);
	}

	_FORCE_INLINE_ void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(*p_iter);
		}
	}

	/* Insert */

	Iterator insert(const TKey &p_key) {
		uint32_t pos = _insert(p_key);
		return Iterator(keys, num_elements, pos);
	}

	/* Constructors */

	HashSet(const HashSet &p_other) {
		_init_from(p_other);
	}

	void operator=(const HashSet &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}

		clear();

		if (keys != nullptr) {
			Memory::free_static(keys);
			Memory::free_static(key_to_hash);
			Memory::free_static(hash_to_key);
			Memory::free_static(hashes);
			keys = nullptr;
			hashes = nullptr;
			hash_to_key = nullptr;
			key_to_hash = nullptr;
		}

		_init_from(p_other);
	}

	HashSet(uint32_t p_initial_capacity) {
		// Capacity can't be 0.
		capacity_index = 0;
		reserve(p_initial_capacity);
	}
	HashSet() {
		capacity_index = MIN_CAPACITY_INDEX;
	}

	void reset() {
		clear();

		if (keys != nullptr) {
			Memory::free_static(keys);
			Memory::free_static(key_to_hash);
			Memory::free_static(hash_to_key);
			Memory::free_static(hashes);
			keys = nullptr;
			hashes = nullptr;
			hash_to_key = nullptr;
			key_to_hash = nullptr;
		}
		capacity_index = MIN_CAPACITY_INDEX;
	}

	~HashSet() {
		clear();

		if (keys != nullptr) {
			Memory::free_static(keys);
			Memory::free_static(key_to_hash);
			Memory::free_static(hash_to_key);
			Memory::free_static(hashes);
		}
	}
};

#line 0

#line 1 "sfwl/core/file_access.h"

/*************************************************************************/
/*  file_access.h                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(_WIN64) || defined(_WIN32)

// Mingw
struct _iobuf;
typedef struct _iobuf FILE;

#elif defined(__APPLE__)

struct __sFILE;
typedef struct __sFILE FILE;

typedef void (*FileCloseNotificationFunc)(const String &p_file, int p_flags);

#else
struct _IO_FILE;
typedef struct _IO_FILE FILE;

typedef void (*FileCloseNotificationFunc)(const String &p_file, int p_flags);
#endif

class FileAccess {
public:
	typedef void (*FileCloseFailNotify)(const String &);

	bool endian_swap;
	bool real_is_double;

	virtual uint32_t _get_unix_permissions(const String &p_file);
	virtual Error _set_unix_permissions(const String &p_file, uint32_t p_permissions);

protected:
	String fix_path(const String &p_path) const;
	virtual Error _open(const String &p_path, int p_mode_flags); ///< open a file
	virtual uint64_t _get_modified_time(const String &p_file);

	static FileCloseFailNotify close_fail_notify;

private:
	static bool backup_save;

public:
	static void set_file_close_fail_notify_callback(FileCloseFailNotify p_cbk) { close_fail_notify = p_cbk; }

	enum ModeFlags {
		READ = 1,
		WRITE = 2,
		READ_WRITE = 3,
		WRITE_READ = 7,
	};

	virtual void close(); ///< close a file
	virtual bool is_open() const; ///< true when file is open

	virtual String get_path() const; /// returns the path for the current open file
	virtual String get_path_absolute() const; /// returns the absolute path for the current open file

	virtual void seek(uint64_t p_position); ///< seek to a given position
	virtual void seek_end(int64_t p_position = 0); ///< seek from the end of file with negative offset
	virtual uint64_t get_position() const; ///< get position in the file
	virtual uint64_t get_len() const; ///< get size of the file

	virtual bool eof_reached() const; ///< reading passed EOF

	virtual uint8_t get_8() const; ///< get a byte
	virtual uint16_t get_16() const; ///< get 16 bits uint
	virtual uint32_t get_32() const; ///< get 32 bits uint
	virtual uint64_t get_64() const; ///< get 64 bits uint

	virtual float get_float() const;
	virtual double get_double() const;
	virtual real_t get_real() const;

	virtual uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const; ///< get an array of bytes
	virtual String get_line() const;
	virtual String get_token() const;
	virtual Vector<String> get_csv_line(const String &p_delim = ",") const;
	virtual String get_as_utf8_string(bool p_skip_cr = true) const; // Skip CR by default for compat.

	/**< use this for files WRITTEN in _big_ endian machines (ie, amiga/mac)
	 * It's not about the current CPU type but file formats.
	 * this flags get reset to false (little endian) on each open
	 */

	virtual void set_endian_swap(bool p_swap) { endian_swap = p_swap; }
	inline bool get_endian_swap() const { return endian_swap; }

	virtual Error get_error() const; ///< get last error

	virtual void flush();
	virtual void store_8(uint8_t p_dest); ///< store a byte
	virtual void store_16(uint16_t p_dest); ///< store 16 bits uint
	virtual void store_32(uint32_t p_dest); ///< store 32 bits uint
	virtual void store_64(uint64_t p_dest); ///< store 64 bits uint

	virtual void store_float(float p_dest);
	virtual void store_double(double p_dest);
	virtual void store_real(real_t p_real);

	virtual void store_string(const String &p_string);
	virtual void store_line(const String &p_line);
	virtual void store_csv_line(const Vector<String> &p_values, const String &p_delim = ",");

	virtual void store_pascal_string(const String &p_string);
	virtual String get_pascal_string();

	void store_buffer_vec(const Vector<uint8_t> &data); ///< store an array of bytes
	virtual void store_buffer(const uint8_t *p_src, uint64_t p_length); ///< store an array of bytes

	virtual bool file_exists(const String &p_name); ///< return true if a file exists

	virtual Error reopen(const String &p_path, int p_mode_flags); ///< does not change the AccessType

	Error open(const String &p_path, int p_mode_flags);

	static FileAccess *create(); /// Helper that Creates a file access
	static FileAccess *create_and_open(const String &p_path, int p_mode_flags, Error *r_error = nullptr);
	static bool exists(const String &p_name); ///< return true if a file exists
	static uint64_t get_modified_time(const String &p_file);
	static uint32_t get_unix_permissions(const String &p_file);
	static Error set_unix_permissions(const String &p_file, uint32_t p_permissions);

	static void set_backup_save(bool p_enable) { backup_save = p_enable; };
	static bool is_backup_save_enabled() { return backup_save; };

	static Vector<uint8_t> get_file_as_array(const String &p_path, Error *r_error = nullptr);
	static String get_file_as_string(const String &p_path, Error *r_error = nullptr);

	static void write_file(const String &p_path, const String &data, Error *r_error = nullptr);
	static void write_file_buffer(const String &p_path, const Vector<uint8_t> &data, Error *r_error = nullptr);

	FileAccess();
	virtual ~FileAccess();

#if defined(_WIN64) || defined(_WIN32)
#else
	static FileCloseNotificationFunc close_notification_func;
#endif

protected:
#if defined(_WIN64) || defined(_WIN32)
	void check_errors() const;

	FILE *f;
	int flags;
	mutable int prev_op;
	mutable Error last_error;
	String path;
	String path_src;
	String save_path;
#else
	void check_errors() const;

	FILE *f;
	int flags;

	mutable Error last_error;
	String save_path;
	String path;
	String path_src;
#endif
};

struct FileAccessRef {
	FileAccess *f;

	_FORCE_INLINE_ bool is_null() const { return f == nullptr; }
	_FORCE_INLINE_ bool is_valid() const { return f != nullptr; }

	_FORCE_INLINE_ operator bool() const { return f != nullptr; }
	_FORCE_INLINE_ operator FileAccess *() { return f; }

	_FORCE_INLINE_ FileAccess *operator->() {
		return f;
	}

	FileAccessRef(FileAccess *fa) { f = fa; }
	FileAccessRef(FileAccessRef &&other) {
		f = other.f;
		other.f = nullptr;
	}
	~FileAccessRef() {
		if (f) {
			memdelete(f);
		}
	}
};

#line 0

#line 1 "sfwl/core/dir_access.h"

/*************************************************************************/
/*  dir_access.h                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(_WIN64) || defined(_WIN32)
struct DirAccessWindowsPrivate;

#elif defined(__APPLE__)

#include <dirent.h>

#else
struct __dirstream;
typedef struct __dirstream DIR;
#endif

class DirAccess {
	Error _copy_dir(DirAccess *p_target_da, String p_to, int p_chmod_flags, bool p_copy_links);

public:
	virtual Error list_dir_begin(bool skip_specials = false); ///< This starts dir listing
	virtual String get_next();
	virtual bool current_is_dir() const;
	virtual bool current_is_hidden() const;
	virtual bool current_is_file() const;
	virtual bool current_is_special_dir() const;

	virtual void list_dir_end(); ///<

	virtual int get_drive_count();
	virtual String get_drive(int p_drive);
	virtual int get_current_drive();
	virtual bool drives_are_shortcuts();

	virtual Error change_dir(String p_dir); ///< can be relative or absolute, return false on success
	virtual String get_current_dir(); ///< return current dir location
	virtual String get_current_dir_without_drive();
	virtual Error make_dir(String p_dir);
	virtual Error make_dir_recursive(String p_dir);
	virtual Error erase_contents_recursive(); //super dangerous, use with care!

	virtual bool file_exists(String p_file);
	virtual bool dir_exists(String p_dir);
	static bool exists(String p_dir);
	virtual uint64_t get_space_left();

	Error copy_dir(String p_from, String p_to, int p_chmod_flags = -1, bool p_copy_links = false);
	virtual Error copy(String p_from, String p_to, int p_chmod_flags = -1);
	virtual Error rename(String p_from, String p_to);
	virtual Error remove(String p_name);

	virtual bool is_link(String p_file);
	virtual String read_link(String p_file);
	virtual Error create_link(String p_source, String p_target);

	virtual uint64_t get_modified_time(String p_file);

	virtual String get_filesystem_type() const;
	static String get_full_path(const String &p_path);

	static DirAccess *create_for_path(const String &p_path);
	static DirAccess *create();

	Error open(const String &p_path);

	static String get_filesystem_abspath_for(String p_path);

	static bool is_special(const String &p_path);

	DirAccess();
	virtual ~DirAccess();

protected:
#if defined(_WIN64) || defined(_WIN32)
#else
	virtual String fix_unicode_name(const char *p_name) const { return String::utf8(p_name); }
	virtual bool is_hidden(const String &p_name);
#endif

	bool next_is_dir;
	bool _skip_specials;

#if defined(_WIN64) || defined(_WIN32)
	enum {
		MAX_DRIVES = 26
	};

	DirAccessWindowsPrivate *p;
	/* Windows stuff */

	char drives[MAX_DRIVES]; // a-z:
	int drive_count;

	String current_dir;

	bool _cisdir;
	bool _cishidden;
	bool _cisspecial;
#else
	String current_dir;
	DIR *dir_stream;

	bool _cisdir;
	bool _cishidden;
	bool _cisspecial;
#endif
};

struct DirAccessRef {
	DirAccess *f;

	_FORCE_INLINE_ bool is_null() const { return f == nullptr; }
	_FORCE_INLINE_ bool is_valid() const { return f != nullptr; }

	_FORCE_INLINE_ operator bool() const { return f != nullptr; }
	_FORCE_INLINE_ operator DirAccess *() { return f; }

	_FORCE_INLINE_ DirAccess *operator->() {
		return f;
	}

	DirAccessRef(DirAccess *fa) { f = fa; }
	DirAccessRef(DirAccessRef &&other) {
		f = other.f;
		other.f = nullptr;
	}
	~DirAccessRef() {
		if (f) {
			memdelete(f);
		}
	}
};

#line 0

#line 1 "sfwl/core/inet_address.h"
//Based on:

// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)
//
// This is a public header file, it must only include public header files.

// Taken from Muduo and modified
// Copyright 2016, Tao An.  All rights reserved.
// https://github.com/an-tao/trantor
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Tao An

//On windows link to ws2_32

#if defined(_WIN64) || defined(_WIN32)
using sa_family_t = unsigned short;
using in_addr_t = uint32_t;
using uint16_t = unsigned short;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

struct sockaddr_in;
struct sockaddr_in6;

class InetAddress {
public:
	struct InetAddressData;

	sa_family_t family() const;

	String to_ip() const;
	String to_ip_port() const;

	uint16_t to_port() const;

	bool is_ip_v6() const;
	bool is_intranet_ip() const;
	bool is_loopback_ip() const;

	const struct sockaddr *get_sock_addr() const;

	void set_sock_addr_inet6(const struct sockaddr_in6 &addr6);

	uint32_t ip_net_endian() const;
	const uint32_t *ip6_net_endian() const;

	uint16_t port_net_endian() const;

	void set_port_net_endian(uint16_t port);

	inline bool is_unspecified() const;

	InetAddress(uint16_t port = 0, bool loopbackOnly = false, bool ipv6 = false);
	InetAddress(const String &ip, uint16_t port, bool ipv6 = false);

	explicit InetAddress(const struct sockaddr_in &addr);
	explicit InetAddress(const struct sockaddr_in6 &addr);

	~InetAddress();

private:
	InetAddressData *_data;

	bool _is_ip_v6;
	bool _is_unspecified;
};

#line 0

#line 1 "sfwl/core/socket.h"

//Based on:

/**
 *
 *  Socket.h
 *  An Tao
 *
 *  Public header file in trantor lib.
 *
 *  Copyright 2018, An Tao.  All rights reserved.
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the License file.
 *
 *
 */

class Socket {
public:
	void create_net_socket();
	void create(int family);
	void close_socket();

	int connect(const InetAddress &address);
	int bind_address(const InetAddress &address);
	int listen();
	int accept(Socket *sock);

	int close_write();

	int read(char *buffer, uint64_t len);
	int send(const char *buffer, uint64_t len);

	bool is_self_connect();

	void set_tcp_nodelay(bool on);
	void set_reuse_addr(bool on);
	int set_reuse_port(bool on);
	void set_keep_alive(bool on);

	int set_non_block();
	int set_close_on_exit();

	int get_error();

	struct sockaddr_in6 get_local_addr(int *r_err = NULL);
	struct sockaddr_in6 get_peer_addr(int *r_err = NULL);

	static int global_init();

	Socket();
	Socket(int socketFD, const InetAddress &address);
	~Socket();

	int _socket;
	InetAddress _address;
};

#line 0

#line 1 "sfwl/core/sub_process.h"

/*************************************************************************/
/*  sub_process.h                                                        */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

#if defined(_WIN64) || defined(_WIN32)

#endif

/**
 * Multi-Platform abstraction for running and communicating with sub processes
 */

class SubProcess {
public:
	typedef int64_t ProcessID;

#if defined(_WIN64) || defined(_WIN32)
	struct SubProcessWindowsData;
#endif

	static SubProcess *create();

	String get_executable_path() const;
	void set_executable_path(const String &p_executable_path);

	Vector<String> get_arguments() const;
	void set_arguments(const Vector<String> &p_arguments);

	bool get_blocking() const;
	void set_blocking(const bool p_value);

	bool get_read_output() const;
	void set_read_output(const bool p_value);

	bool get_read_std() const;
	void set_read_std(const bool p_value);

	bool get_read_std_err() const;
	void set_read_std_err(const bool p_value);

	bool get_use_pipe_mutex() const;
	void set_use_pipe_mutex(const bool p_value);

	bool get_open_console() const;
	void set_open_console(const bool p_value);

	String get_data() const {
		return _pipe;
	}

	int get_process_id() const {
		return _process_id;
	}

	int get_exitcode() const {
		return _exitcode;
	}

	virtual Error start();
	virtual Error stop();
	virtual Error poll();
	virtual Error send_signal(const int p_signal);
	virtual Error send_data(const String &p_data);
	virtual bool is_process_running() const;

	Error run(const String &p_executable_path, const Vector<String> &p_arguments = Vector<String>(), bool p_output = true, bool p_blocking = true, bool p_read_std_err = false, bool p_use_pipe_mutex = false, bool p_open_console = false);

	SubProcess();
	virtual ~SubProcess();

protected:
	void _setup_pipe_mutex();

	String _executable_path;
	Vector<String> _arguments;

	bool _blocking;

	bool _read_output;

	bool _read_std;
	bool _read_std_err;

	String _pipe;

	bool _use_pipe_mutex;

	Mutex *_pipe_mutex;

	bool _open_console;

	ProcessID _process_id;
	int _exitcode;

#if defined(_WIN64) || defined(_WIN32)

	String _quote_command_line_argument(const String &p_text) const;
	void _append_to_pipe(char *p_bytes, int p_size);

	bool _process_started;
	LocalVector<char> _bytes;

	SubProcessWindowsData *_data;

#else
	FILE *_process_fp;
	char _process_buf[65535];
#endif
};

struct SubProcessRef {
	SubProcess *f;

	_FORCE_INLINE_ bool is_null() const { return f == nullptr; }
	_FORCE_INLINE_ bool is_valid() const { return f != nullptr; }

	_FORCE_INLINE_ operator bool() const { return f != nullptr; }
	_FORCE_INLINE_ operator SubProcess *() { return f; }

	_FORCE_INLINE_ SubProcess *operator->() {
		return f;
	}

	SubProcessRef(SubProcess *fa) { f = fa; }
	SubProcessRef(SubProcessRef &&other) {
		f = other.f;
		other.f = nullptr;
	}
	~SubProcessRef() {
		if (f) {
			memdelete(f);
		}
	}
};

#line 0

#line 1 "sfwl/core/sfw_core.h"

class SFWCore {
public:
	static void setup();
	static void cleanup();

protected:
	static bool _initialized;
};

#line 0

//===================  OBJECT SECTION  ===================

#line 1 "sfwl/object/ref_ptr.h"

/*************************************************************************/
/*  ref_ptr.h                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

/**
	@author Juan Linietsky <reduzio@gmail.com>
 * This class exists to workaround a limitation in C++ but keep the design OK.
 * It's basically an opaque container of a Reference reference, so Variant can use it.
*/

class RefPtr {
	enum {
		DATASIZE = sizeof(void *) //*4 -ref was shrunk
	};

	mutable char data[DATASIZE]; // too much probably, virtual class + pointer
public:
	bool is_null() const;
	void operator=(const RefPtr &p_other);
	bool operator==(const RefPtr &p_other) const;
	bool operator!=(const RefPtr &p_other) const;
	void unref();

	_FORCE_INLINE_ void *get_data() const { return data; }
	RefPtr(const RefPtr &p_other);
	RefPtr();
	~RefPtr();
};

#line 0

#line 1 "sfwl/object/object_id.h"

/*************************************************************************/
/*  object_id.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

typedef uint64_t ObjectID;

#line 0

#line 1 "sfwl/object/core_string_names.h"

/*************************************************************************/
/*  core_string_names.h                                                  */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class CoreStringNames {
public:
	static void create() { singleton = memnew(CoreStringNames); }
	static void free() {
		memdelete(singleton);
		singleton = nullptr;
	}

	CoreStringNames();

	_FORCE_INLINE_ static CoreStringNames *get_singleton() { return singleton; }

	static CoreStringNames *singleton;

	StringName _free;
	StringName changed;
	StringName _meta;
	StringName _script;
	StringName script_changed;
	StringName ___pdcdata;
	StringName __getvar;
	StringName _iter_init;
	StringName _iter_next;
	StringName _iter_get;
	StringName get_rid;
	StringName _to_string;
#ifdef TOOLS_ENABLED
	StringName _sections_unfolded;
#endif
	StringName _custom_features;

	StringName x;
	StringName y;
	StringName z;
	StringName w;
	StringName r;
	StringName g;
	StringName b;
	StringName a;
	StringName position;
	StringName size;
	StringName end;
	StringName basis;
	StringName origin;
	StringName normal;
	StringName d;
	StringName h;
	StringName s;
	StringName v;
	StringName r8;
	StringName g8;
	StringName b8;
	StringName a8;
};

#line 0

#line 1 "sfwl/object/array.h"

/*************************************************************************/
/*  array.h                                                              */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class Variant;
class ArrayPrivate;
class Object;
class StringName;
class String;

class Array {
	mutable ArrayPrivate *_p;
	void _ref(const Array &p_from) const;
	void _unref() const;

	inline int _clamp_slice_index(int p_index) const;

public:
	Variant &operator[](int p_idx);
	const Variant &operator[](int p_idx) const;

	void set(int p_idx, const Variant &p_value);
	const Variant &get(int p_idx) const;

	int size() const;
	bool empty() const;
	void clear();

	bool deep_equal(const Array &p_array, int p_recursion_count = 0) const;
	bool operator==(const Array &p_array) const;

	uint32_t hash() const;
	uint32_t recursive_hash(int p_recursion_count) const;
	void operator=(const Array &p_array);

	void push_back(const Variant &p_value);
	_FORCE_INLINE_ void append(const Variant &p_value) { push_back(p_value); } //for python compatibility
	void append_array(const Array &p_array);
	Error resize(int p_new_size);

	void insert(int p_pos, const Variant &p_value);
	void remove(int p_pos);
	void fill(const Variant &p_value);

	Variant front() const;
	Variant back() const;

	Array &sort();
	Array &sort_custom(Object *p_obj);
	void shuffle();
	int bsearch(const Variant &p_value, bool p_before = true);
	int bsearch_custom(const Variant &p_value, Object *p_obj, const StringName &p_function, bool p_before = true);
	Array &invert();

	int find(const Variant &p_value, int p_from = 0) const;
	int rfind(const Variant &p_value, int p_from = -1) const;
	int find_last(const Variant &p_value) const;
	int count(const Variant &p_value) const;
	bool has(const Variant &p_value) const;

	void erase(const Variant &p_value);

	void push_front(const Variant &p_value);
	Variant pop_back();
	Variant pop_front();
	Variant pop_at(int p_pos);

	Array duplicate(bool p_deep = false) const;

	Array slice(int p_begin, int p_end, int p_step = 1, bool p_deep = false) const;

	Variant min() const;
	Variant max() const;

	const void *id() const;

	String sprintf(const String &p_format, bool *error) const;

	Array(const Array &p_from);
	Array();
	~Array();
};

#line 0

#line 1 "sfwl/object/dictionary.h"

/*************************************************************************/
/*  dictionary.h                                                         */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class Variant;

struct DictionaryPrivate;

class Dictionary {
	mutable DictionaryPrivate *_p;

	void _ref(const Dictionary &p_from) const;
	void _unref() const;

public:
	void get_key_list(List<Variant> *p_keys) const;
	Variant get_key_at_index(int p_index) const;
	Variant get_value_at_index(int p_index) const;

	Variant &operator[](const Variant &p_key);
	const Variant &operator[](const Variant &p_key) const;

	const Variant *getptr(const Variant &p_key) const;
	Variant *getptr(const Variant &p_key);

	Variant get_valid(const Variant &p_key) const;
	Variant get(const Variant &p_key, const Variant &p_default) const;

	int size() const;
	bool empty() const;
	void clear();
	void merge(const Dictionary &p_dictionary, bool p_overwrite = false);

	bool has(const Variant &p_key) const;
	bool has_all(const Array &p_keys) const;
	Variant find_key(const Variant &p_value) const;

	bool erase(const Variant &p_key);

	bool deep_equal(const Dictionary &p_dictionary, int p_recursion_count = 0) const;
	bool operator==(const Dictionary &p_dictionary) const;
	bool operator!=(const Dictionary &p_dictionary) const;

	uint32_t hash() const;
	uint32_t recursive_hash(int p_recursion_count) const;
	void operator=(const Dictionary &p_dictionary);

	const Variant *next(const Variant *p_key = nullptr) const;

	Array keys() const;
	Array values() const;

	Dictionary duplicate(bool p_deep = false) const;

	const void *id() const;

	Dictionary(const Dictionary &p_from);
	Dictionary();
	~Dictionary();
};

#line 0

#line 1 "sfwl/object/variant.h"

/*************************************************************************/
/*  variant.h                                                            */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class Object;
class ObjectRC;

typedef PoolVector<uint8_t> PoolByteArray;
typedef PoolVector<int> PoolIntArray;
typedef PoolVector<real_t> PoolRealArray;
typedef PoolVector<String> PoolStringArray;
typedef PoolVector<Vector2i> PoolVector2iArray;
typedef PoolVector<Color> PoolColorArray;

// Temporary workaround until c++11 alignas()
#ifdef __GNUC__
#define GCC_ALIGNED_8 __attribute__((aligned(8)))
#else
#define GCC_ALIGNED_8
#endif

#define _REF_OBJ_PTR(m_variant) (reinterpret_cast<Ref<Reference> *>((m_variant)._get_obj().ref.get_data())->ptr())
#define _OBJ_PTR(m_variant) ((m_variant)._get_obj().rc ? (m_variant)._get_obj().rc->get_ptr() : _REF_OBJ_PTR(m_variant))
// _UNSAFE_OBJ_PROXY_PTR is needed for comparing an object Variant against NIL or compare two object Variants.
// It's guaranteed to be unique per object, in contrast to the pointer stored in the RC structure,
// which is set to null when the object is destroyed.
#define _UNSAFE_OBJ_PROXY_PTR(m_variant) ((m_variant)._get_obj().rc ? reinterpret_cast<uint8_t *>((m_variant)._get_obj().rc) : reinterpret_cast<uint8_t *>(_REF_OBJ_PTR(m_variant)))

class Variant {
public:
	// If this changes the table in variant_op must be updated
	enum Type {

		NIL,

		// atomic types
		BOOL,
		INT,
		REAL,
		STRING,

		// math types
		RECT2I,
		VECTOR2I,

		// misc types
		COLOR,
		OBJECT,
		STRING_NAME,
		DICTIONARY,
		ARRAY,

		// arrays
		POOL_BYTE_ARRAY,
		POOL_INT_ARRAY,
		POOL_REAL_ARRAY,
		POOL_STRING_ARRAY,
		POOL_VECTOR2I_ARRAY,
		POOL_COLOR_ARRAY,

		VARIANT_MAX

	};

	enum {
		// Maximum recursion depth allowed when serializing variants.
		MAX_RECURSION_DEPTH = 1024,
	};

private:
	// Variant takes 20 bytes when real_t is float, and 36 if double
	// it only allocates extra memory for aabb/matrix.

	Type type;

	struct ObjData {
		// Will be null for every type deriving from Reference as they have their
		// own reference count mechanism
		ObjectRC *rc;
		// Always initialized, but will be null if the Ref<> assigned was null
		// or this Variant is not even holding a Reference-derived object
		RefPtr ref;
	};

	_FORCE_INLINE_ ObjData &_get_obj();
	_FORCE_INLINE_ const ObjData &_get_obj() const;

	union {
		bool _bool;
		int64_t _int;
		double _real;
		void *_ptr; //generic pointer
		uint8_t _mem[sizeof(ObjData) > (sizeof(real_t) * 4) ? sizeof(ObjData) : (sizeof(real_t) * 4)];
	} _data GCC_ALIGNED_8;

	void reference(const Variant &p_variant);
	void clear();

public:
	_FORCE_INLINE_ Type get_type() const { return type; }
	static String get_type_name(Variant::Type p_type);
	static bool can_convert(Type p_type_from, Type p_type_to);
	static bool can_convert_strict(Type p_type_from, Type p_type_to);

	bool is_ref() const;
	_FORCE_INLINE_ bool is_num() const { return type == INT || type == REAL; };
	_FORCE_INLINE_ bool is_array() const { return type >= ARRAY; };
	_FORCE_INLINE_ bool is_null() const { return type == NIL; };
	bool is_shared() const;
	bool is_zero() const;
	bool is_one() const;

	ObjectID get_object_instance_id() const;
	bool is_invalid_object() const;

	operator bool() const;
	operator signed int() const;
	operator unsigned int() const; // this is the real one
	operator signed short() const;
	operator unsigned short() const;
	operator signed char() const;
	operator unsigned char() const;
	//operator long unsigned int() const;
	operator int64_t() const;
	operator uint64_t() const;
#ifdef NEED_LONG_INT
	operator signed long() const;
	operator unsigned long() const;
#endif

	operator CharType() const;
	operator float() const;
	operator double() const;
	operator String() const;
	operator StringName() const;
	operator Rect2i() const;
	operator Vector2i() const;

	operator Color() const;
	operator RefPtr() const;

	operator Object *() const;

	operator Dictionary() const;
	operator Array() const;

	operator PoolVector<uint8_t>() const;
	operator PoolVector<int>() const;
	operator PoolVector<real_t>() const;
	operator PoolVector<String>() const;
	operator PoolVector<Vector2i>() const;
	operator PoolVector<Color>() const;

	operator Vector<Variant>() const;
	operator Vector<uint8_t>() const;
	operator Vector<int>() const;
	operator Vector<real_t>() const;
	operator Vector<String>() const;
	operator Vector<StringName>() const;
	operator Vector<Color>() const;
	operator Vector<Vector2i>() const;

	// some core type enums to convert to
	operator Margin() const;
	operator Side() const;
	operator Orientation() const;

	Variant(bool p_bool);
	Variant(signed int p_int); // real one
	Variant(unsigned int p_int);
#ifdef NEED_LONG_INT
	Variant(signed long p_long); // real one
	Variant(unsigned long p_long);
//Variant(long unsigned int p_long);
#endif
	Variant(signed short p_short); // real one
	Variant(unsigned short p_short);
	Variant(signed char p_char); // real one
	Variant(unsigned char p_char);
	Variant(int64_t p_int); // real one
	Variant(uint64_t p_int);
	Variant(float p_float);
	Variant(double p_double);
	Variant(const String &p_string);
	Variant(const StringName &p_string);
	Variant(const char *const p_cstring);
	Variant(const CharType *p_wstring);
	Variant(const Vector2i &p_vector2);
	Variant(const Rect2i &p_rect2);
	Variant(const Color &p_color);
	Variant(const RefPtr &p_resource);
	Variant(const Object *p_object);
	Variant(const Dictionary &p_dictionary);

	Variant(const Array &p_array);
	Variant(const PoolVector<uint8_t> &p_raw_array);
	Variant(const PoolVector<int> &p_int_array);
	Variant(const PoolVector<real_t> &p_real_array);
	Variant(const PoolVector<String> &p_string_array);
	Variant(const PoolVector<Color> &p_color_array);
	Variant(const PoolVector<Vector2i> &p_vector2_array);

	Variant(const Vector<Variant> &p_array);
	Variant(const Vector<uint8_t> &p_array);
	Variant(const Vector<int> &p_array);
	Variant(const Vector<real_t> &p_array);
	Variant(const Vector<String> &p_array);
	Variant(const Vector<StringName> &p_array);
	Variant(const Vector<Color> &p_array);
	Variant(const Vector<Vector2i> &p_array);

	// If this changes the table in variant_op must be updated
	enum Operator {
		//comparison
		OP_EQUAL,
		OP_NOT_EQUAL,
		OP_LESS,
		OP_LESS_EQUAL,
		OP_GREATER,
		OP_GREATER_EQUAL,
		//mathematic
		OP_ADD,
		OP_SUBTRACT,
		OP_MULTIPLY,
		OP_DIVIDE,
		OP_NEGATE,
		OP_POSITIVE,
		OP_MODULE,
		OP_STRING_CONCAT,
		//bitwise
		OP_SHIFT_LEFT,
		OP_SHIFT_RIGHT,
		OP_BIT_AND,
		OP_BIT_OR,
		OP_BIT_XOR,
		OP_BIT_NEGATE,
		//logic
		OP_AND,
		OP_OR,
		OP_XOR,
		OP_NOT,
		//containment
		OP_IN,
		OP_MAX
	};

	//Maybe add helper methods that use these
	//Like max()
	//add()
	//maybe operators like with add
	//?

	static String get_operator_name(Operator p_op);
	static void evaluate(const Operator &p_op, const Variant &p_a, const Variant &p_b, Variant &r_ret, bool &r_valid);
	static _FORCE_INLINE_ Variant evaluate(const Operator &p_op, const Variant &p_a, const Variant &p_b) {
		bool valid = true;
		Variant res;
		evaluate(p_op, p_a, p_b, res, valid);
		return res;
	}

	void zero();
	Variant duplicate(bool deep = false) const;
	static void blend(const Variant &a, const Variant &b, float c, Variant &r_dst);
	static void interpolate(const Variant &a, const Variant &b, float c, Variant &r_dst);
	static void sub(const Variant &a, const Variant &b, Variant &r_dst);

	//Maybe add helper methods that call this? like get_x(), set_x() etc Or just x(), x(value)?
	//Check if operator[] can take strings and ints that might work too
	void set_named(const StringName &p_index, const Variant &p_value, bool *r_valid = nullptr);
	Variant get_named(const StringName &p_index, bool *r_valid = nullptr) const;

	//Implement operator[]s that use these?
	void set(const Variant &p_index, const Variant &p_value, bool *r_valid = nullptr);
	Variant get(const Variant &p_index, bool *r_valid = nullptr) const;
	bool in(const Variant &p_index, bool *r_valid = nullptr) const;

	bool iter_init(Variant &r_iter, bool &r_valid) const;
	bool iter_next(Variant &r_iter, bool &r_valid) const;
	Variant iter_get(const Variant &r_iter, bool &r_valid) const;

	bool deep_equal(const Variant &p_variant, int p_recursion_count = 0) const;
	bool operator==(const Variant &p_variant) const;
	bool operator!=(const Variant &p_variant) const;
	bool operator<(const Variant &p_variant) const;
	uint32_t hash() const;
	uint32_t recursive_hash(int p_recursion_count) const;

	bool hash_compare(const Variant &p_variant) const;
	bool booleanize() const;
	String stringify(List<const void *> &stack) const;

	void operator=(const Variant &p_variant); // only this is enough for all the other types
	Variant(const Variant &p_variant);
	_FORCE_INLINE_ Variant() {
		type = NIL;
	}
	_FORCE_INLINE_ ~Variant() {
		if (type != Variant::NIL) {
			clear();
		}
	}
};

//typedef Dictionary Dictionary; no
//typedef Array Array;

Vector<Variant> varray();
Vector<Variant> varray(const Variant &p_arg1);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4);
Vector<Variant> varray(const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5);

struct VariantHasher {
	static _FORCE_INLINE_ uint32_t hash(const Variant &p_variant) { return p_variant.hash(); }
};

struct VariantComparator {
	static _FORCE_INLINE_ bool compare(const Variant &p_lhs, const Variant &p_rhs) { return p_lhs.hash_compare(p_rhs); }
};

Variant::ObjData &Variant::_get_obj() {
	return *reinterpret_cast<ObjData *>(&_data._mem[0]);
}

const Variant::ObjData &Variant::_get_obj() const {
	return *reinterpret_cast<const ObjData *>(&_data._mem[0]);
}

String vformat(const String &p_text, const Variant &p1 = Variant(), const Variant &p2 = Variant(), const Variant &p3 = Variant(), const Variant &p4 = Variant(), const Variant &p5 = Variant());

#line 0

#line 1 "sfwl/object/psignal.h"

class Signal {
public:
	Object *emitter;
	Vector<Variant> params;
	Vector<Variant> static_data;

	template <class T>
	void connect(T *obj, void (*func)(T *, Signal *));
	template <class T>
	void disconnect(T *obj, void (*func)(T *, Signal *));
	template <class T>
	bool is_connected(T *obj, void (*func)(T *, Signal *));

	void connect_static(void (*func)(Signal *));
	void disconnect_static(void (*func)(Signal *));
	bool is_connected_static(void (*func)(Signal *));

	void emit(Object *p_emitter);
	void emit(Object *p_emitter, const Variant &p1);
	void emit(Object *p_emitter, const Variant &p1, const Variant &p2);
	void emit(Object *p_emitter, const Variant &p1, const Variant &p2, const Variant &p3);
	void emit(Object *p_emitter, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4);
	void emit(Object *p_emitter, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4, const Variant &p5);

	Signal();
	~Signal();

protected:
	enum SignalEntryType {
		SIGNAL_ENTRY_TYPE_NONE = 0,
		SIGNAL_ENTRY_TYPE_STATIC = 1,
		SIGNAL_ENTRY_TYPE_CLASS = 2,
	};

	struct SignalEntry {
		SignalEntryType type;

		virtual void call(Signal *s) {
		}

		SignalEntry() {
			type = SIGNAL_ENTRY_TYPE_NONE;
		}
	};

	struct StaticSignalEntry : public SignalEntry {
		void (*func)(Signal *);

		virtual void call(Signal *s) {
			func(s);
		}

		StaticSignalEntry() {
			type = SIGNAL_ENTRY_TYPE_STATIC;
			func = nullptr;
		}
	};

	struct ClassSignalEntry : public SignalEntry {
		virtual void *get_obj_ptr() {
			return nullptr;
		}

		virtual void *get_func_ptr() {
			return nullptr;
		}

		ClassSignalEntry() {
			type = SIGNAL_ENTRY_TYPE_CLASS;
		}
	};

	template <typename T>
	struct ClassSignalEntrySpec : public ClassSignalEntry {
		union {
			T *obj;
			void *obj_ptr;
		};
		union {
			void (*func)(T *, Signal *);
			void *func_ptr;
		};

		virtual void call(Signal *s) {
			func(obj, s);
		}

		void *get_obj_ptr() {
			return obj_ptr;
		}

		void *get_func_ptr() {
			return func_ptr;
		}

		ClassSignalEntrySpec() {
			obj = nullptr;
			func = nullptr;
		}
	};

protected:
	Vector<SignalEntry *> entries;
};

template <typename T>
void Signal::connect(T *obj, void (*func)(T *, Signal *)) {
	ClassSignalEntrySpec<T> *ce = new ClassSignalEntrySpec<T>();
	ce->obj = obj;
	ce->func = func;

	entries.push_back(ce);
}

template <typename T>
void Signal::disconnect(T *obj, void (*func)(T *, Signal *)) {
	ClassSignalEntrySpec<T> t;
	t.obj = obj;
	t.func = func;

	void *obj_ptr = t.obj_ptr;
	void *func_ptr = t.func_ptr;

	for (int i = 0; i < entries.size(); ++i) {
		SignalEntry *e = entries[i];

		if (e->type == SIGNAL_ENTRY_TYPE_CLASS) {
			ClassSignalEntry *se = static_cast<ClassSignalEntry *>(e);

			if (se->get_obj_ptr() == obj_ptr && se->get_func_ptr() == func_ptr) {
				entries.remove(i);
				return;
			}
		}
	}
}

template <typename T>
bool Signal::is_connected(T *obj, void (*func)(T *, Signal *)) {
	ClassSignalEntrySpec<T> t;
	t.obj = obj;
	t.func = func;

	void *obj_ptr = t.obj_ptr;
	void *func_ptr = t.func_ptr;

	for (int i = 0; i < entries.size(); ++i) {
		SignalEntry *e = entries[i];

		if (e->type == SIGNAL_ENTRY_TYPE_CLASS) {
			ClassSignalEntry *se = static_cast<ClassSignalEntry *>(e);

			if (se->get_obj_ptr() == obj_ptr && se->get_func_ptr() == func_ptr) {
				return true;
			}
		}
	}

	return false;
}

#line 0

#line 1 "sfwl/object/object.h"

/*************************************************************************/
/*  object.h                                                             */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class ObjectRC;

#define SFW_OBJECT(m_class, m_inherits)                                                                    \
private:                                                                                                   \
	void operator=(const m_class &p_rval) {}                                                               \
                                                                                                           \
public:                                                                                                    \
	virtual String get_class() const {                                                                     \
		return String(#m_class);                                                                           \
	}                                                                                                      \
	virtual const StringName *_get_class_namev() const {                                                   \
		if (!_class_name)                                                                                  \
			_class_name = get_class_static();                                                              \
		return &_class_name;                                                                               \
	}                                                                                                      \
	static void *get_class_ptr_static() {                                                                  \
		static int ptr;                                                                                    \
		return &ptr;                                                                                       \
	}                                                                                                      \
	static String get_class_static() {                                                                     \
		return String(#m_class);                                                                           \
	}                                                                                                      \
	static String get_parent_class_static() {                                                              \
		return m_inherits::get_class_static();                                                             \
	}                                                                                                      \
	static void get_inheritance_list_static(Vector<String> *p_inheritance_list) {                          \
		m_inherits::get_inheritance_list_static(p_inheritance_list);                                       \
		p_inheritance_list->push_back(String(#m_class));                                                   \
	}                                                                                                      \
	static String inherits_static() {                                                                      \
		return String(#m_inherits);                                                                        \
	}                                                                                                      \
	virtual bool is_class(const String &p_class) const {                                                   \
		return (p_class == (#m_class)) ? true : m_inherits::is_class(p_class);                             \
	}                                                                                                      \
	virtual bool is_class_ptr(void *p_ptr) const {                                                         \
		return (p_ptr == get_class_ptr_static()) ? true : m_inherits::is_class_ptr(p_ptr);                 \
	}                                                                                                      \
	static void get_valid_parents_static(Vector<String> *p_parents) {                                      \
		if (m_class::_get_valid_parents_static != m_inherits::_get_valid_parents_static) {                 \
			m_class::_get_valid_parents_static(p_parents);                                                 \
		}                                                                                                  \
		m_inherits::get_valid_parents_static(p_parents);                                                   \
	}                                                                                                      \
	_FORCE_INLINE_ void (Object::*_get_notification() const)(int) {                                        \
		return (void(Object::*)(int)) & m_class::_notification;                                            \
	}                                                                                                      \
	virtual void _notificationv(int p_notification, bool p_reversed) {                                     \
		if (!p_reversed)                                                                                   \
			m_inherits::_notificationv(p_notification, p_reversed);                                        \
		if (m_class::_get_notification() != m_inherits::_get_notification()) {                             \
			_notification(p_notification);                                                                 \
		}                                                                                                  \
		if (p_reversed)                                                                                    \
			m_inherits::_notificationv(p_notification, p_reversed);                                        \
	}                                                                                                      \
	_FORCE_INLINE_ bool (Object::*_get_get() const)(const StringName &p_name, Variant &) const {           \
		return (bool(Object::*)(const StringName &, Variant &) const) & m_class::_get;                     \
	}                                                                                                      \
	virtual bool _getv(const StringName &p_name, Variant &r_ret) const {                                   \
		if (m_class::_get_get() != m_inherits::_get_get()) {                                               \
			if (_get(p_name, r_ret))                                                                       \
				return true;                                                                               \
		}                                                                                                  \
		return m_inherits::_getv(p_name, r_ret);                                                           \
	}                                                                                                      \
	_FORCE_INLINE_ bool (Object::*_get_set() const)(const StringName &p_name, const Variant &p_property) { \
		return (bool(Object::*)(const StringName &, const Variant &)) & m_class::_set;                     \
	}                                                                                                      \
	virtual bool _setv(const StringName &p_name, const Variant &p_property) {                              \
		if (m_inherits::_setv(p_name, p_property))                                                         \
			return true;                                                                                   \
		if (m_class::_get_set() != m_inherits::_get_set()) {                                               \
			return _set(p_name, p_property);                                                               \
		}                                                                                                  \
		return false;                                                                                      \
	}                                                                                                      \
                                                                                                           \
private:

class Object {
public:
	enum {
		NOTIFICATION_POSTINITIALIZE = 0,
		NOTIFICATION_PREDELETE = 1
	};

	void set(const StringName &p_name, const Variant &p_value, bool *r_valid = nullptr);
	Variant get(const StringName &p_name, bool *r_valid = nullptr) const;

	//less than. < "operator" used for cutrom sorting Arrays.
	virtual bool lt(const Variant &p_value_l, const Variant &p_value_r);

	virtual String get_class() const { return "Object"; }
	static void *get_class_ptr_static() {
		static int ptr;
		return &ptr;
	}

	static String get_class_static() { return "Object"; }
	static String get_parent_class_static() { return String(); }

	static void get_inheritance_list_static(Vector<String> *p_inheritance_list) { p_inheritance_list->push_back("Object"); }

	virtual bool is_class(const String &p_class) const { return (p_class == "Object"); }
	virtual bool is_class_ptr(void *p_ptr) const { return get_class_ptr_static() == p_ptr; }

	static void get_valid_parents_static(Vector<String> *p_parents) {}
	static void _get_valid_parents_static(Vector<String> *p_parents) {}

	virtual const StringName *_get_class_namev() const {
		if (!_class_name) {
			_class_name = get_class_static();
		}
		return &_class_name;
	}

	_FORCE_INLINE_ const StringName &get_class_name() const {
		if (!_class_ptr) {
			return *_get_class_namev();
		} else {
			return *_class_ptr;
		}
	}

	ObjectRC *_use_rc();

	_FORCE_INLINE_ ObjectID get_instance_id() const {
		return _instance_id;
	}

	void notification(int p_notification, bool p_reversed = false);
	virtual String to_string();

	bool _is_queued_for_deletion;
	bool is_queued_for_deletion() const {
		return _is_queued_for_deletion;
	}

	void cancel_free();

	bool has_meta(const String &p_name) const;
	void set_meta(const String &p_name, const Variant &p_value);
	void remove_meta(const String &p_name);
	Variant get_meta(const String &p_name, const Variant &p_default = Variant()) const;
	void get_meta_list(List<String> *p_list) const;

	Object();
	virtual ~Object();

	template <class T>
	static T *cast_to(Object *p_object) {
		if (!p_object)
			return NULL;
		if (p_object->is_class_ptr(T::get_class_ptr_static()))
			return static_cast<T *>(p_object);
		else
			return NULL;
	}

	template <class T>
	static const T *cast_to(const Object *p_object) {
		if (!p_object)
			return NULL;
		if (p_object->is_class_ptr(T::get_class_ptr_static()))
			return static_cast<const T *>(p_object);
		else
			return NULL;
	}

protected:
	_FORCE_INLINE_ void (Object::*_get_notification() const)(int) {
		return &Object::_notification;
	}

	bool _set(const StringName &p_name, const Variant &p_property) {
		return false;
	};
	bool _get(const StringName &p_name, Variant &r_property) const {
		return false;
	};

	virtual bool _setv(const StringName &p_name, const Variant &p_property) {
		return false;
	};
	virtual bool _getv(const StringName &p_name, Variant &r_property) const {
		return false;
	};

	_FORCE_INLINE_ bool (Object::*_get_get() const)(const StringName &p_name, Variant &r_ret) const {
		return &Object::_get;
	}
	_FORCE_INLINE_ bool (Object::*_get_set() const)(const StringName &p_name, const Variant &p_property) {
		return &Object::_set;
	}

	virtual void _notificationv(int p_notification, bool p_reversed){};
	void _notification(int p_notification){};

	friend bool predelete_handler(Object *);
	friend void postinitialize_handler(Object *);

	int _predelete_ok;
	bool _predelete();
	void _postinitialize();

	mutable StringName _class_name;
	mutable const StringName *_class_ptr;

	ObjectID _instance_id;
	std::atomic<ObjectRC *> _rc;

	Dictionary metadata;
};

bool predelete_handler(Object *p_object);
void postinitialize_handler(Object *p_object);

class ObjectDB {
	struct ObjectPtrHash {
		static _FORCE_INLINE_ uint32_t hash(const Object *p_obj) {
			union {
				const Object *p;
				unsigned long i;
			} u;
			u.p = p_obj;
			return HashMapHasherDefault::hash((uint64_t)u.i);
		}
	};

	static HashMap<ObjectID, Object *> instances;
	static HashMap<Object *, ObjectID, ObjectPtrHash> instance_checks;

	static ObjectID instance_counter;
	friend class Object;
	friend void unregister_core_types();

	static RWLock rw_lock;
	static void cleanup();
	static ObjectID add_instance(Object *p_object);
	static void remove_instance(Object *p_object);
	friend void register_core_types();

public:
	typedef void (*DebugFunc)(Object *p_obj);

	static Object *get_instance(ObjectID p_instance_id);
	static void debug_objects(DebugFunc p_func);
	static int get_object_count();

	// This one may give false positives because a new object may be allocated at the same memory of a previously freed one
	_FORCE_INLINE_ static bool instance_validate(Object *p_ptr) {
		rw_lock.read_lock();

		bool exists = instance_checks.has(p_ptr);

		rw_lock.read_unlock();

		return exists;
	}
};

#line 0

#line 1 "sfwl/object/object_rc.h"

/*************************************************************************/
/*  object_rc.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class Object;

// Used to track Variants pointing to a non-Reference Object
class ObjectRC {
	std::atomic<Object *> _ptr;
	std::atomic<uint32_t> _users;

public:
	// This is for allowing debug builds to check for instance ID validity,
	// so warnings are shown in debug builds when a stray Variant (one pointing
	// to a released Object) would have happened.
	const ObjectID instance_id;

	_FORCE_INLINE_ void increment() {
		_users.fetch_add(1, std::memory_order_relaxed);
	}

	_FORCE_INLINE_ bool decrement() {
		return _users.fetch_sub(1, std::memory_order_relaxed) == 1;
	}

	_FORCE_INLINE_ bool invalidate() {
		_ptr.store(nullptr, std::memory_order_release);
		return decrement();
	}

	_FORCE_INLINE_ Object *get_ptr() {
		return _ptr.load(std::memory_order_acquire);
	}

	_FORCE_INLINE_ ObjectRC(Object *p_object) :
			instance_id(p_object->get_instance_id()) {
		// 1 (the Object) + 1 (the first user)
		_users.store(2, std::memory_order_relaxed);
		_ptr.store(p_object, std::memory_order_release);
	}
};

#line 0

#line 1 "sfwl/object/reference.h"

/*************************************************************************/
/*  reference.h                                                          */
/*  From https://github.com/Relintai/pandemonium_engine (MIT)            */
/*************************************************************************/

class Reference : public Object {
	SFW_OBJECT(Reference, Object);

public:
	_FORCE_INLINE_ bool is_referenced() const { return refcount_init.get() != 1; }
	bool init_ref();
	bool reference(); // returns false if refcount is at zero and didn't get increased
	bool unreference();
	int reference_get_count() const;

	Reference();
	virtual ~Reference();

private:
	SafeRefCount refcount;
	SafeRefCount refcount_init;
};

template <class T>
class Ref {
	T *reference;

	void ref(const Ref &p_from) {
		if (p_from.reference == reference) {
			return;
		}

		unref();

		reference = p_from.reference;
		if (reference) {
			reference->reference();
		}
	}

	void ref_pointer(T *p_ref) {
		//ERR_FAIL_COND(!p_ref);

		if (p_ref->init_ref()) {
			reference = p_ref;
		}
	}

	//virtual Reference * get_reference() const { return reference; }
public:
	_FORCE_INLINE_ bool operator==(const T *p_ptr) const {
		return reference == p_ptr;
	}
	_FORCE_INLINE_ bool operator!=(const T *p_ptr) const {
		return reference != p_ptr;
	}

	_FORCE_INLINE_ bool operator<(const Ref<T> &p_r) const {
		return reference < p_r.reference;
	}
	_FORCE_INLINE_ bool operator==(const Ref<T> &p_r) const {
		return reference == p_r.reference;
	}
	_FORCE_INLINE_ bool operator!=(const Ref<T> &p_r) const {
		return reference != p_r.reference;
	}

	_FORCE_INLINE_ T *operator->() {
		return reference;
	}

	_FORCE_INLINE_ T *operator*() {
		return reference;
	}

	_FORCE_INLINE_ const T *operator->() const {
		return reference;
	}

	_FORCE_INLINE_ const T *ptr() const {
		return reference;
	}
	_FORCE_INLINE_ T *ptr() {
		return reference;
	}

	_FORCE_INLINE_ const T *operator*() const {
		return reference;
	}

	void operator=(const Ref &p_from) {
		ref(p_from);
	}

	template <class T_Other>
	void operator=(const Ref<T_Other> &p_from) {
		Reference *refb = const_cast<Reference *>(static_cast<const Reference *>(p_from.ptr()));
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	template <class T_Other>
	void reference_ptr(T_Other *p_ptr) {
		if (reference == p_ptr) {
			return;
		}
		unref();

		T *r = Object::cast_to<T>(p_ptr);
		if (r) {
			ref_pointer(r);
		}
	}

	Ref(const Ref &p_from) {
		reference = nullptr;
		ref(p_from);
	}

	template <class T_Other>
	Ref(const Ref<T_Other> &p_from) {
		reference = nullptr;
		Reference *refb = const_cast<Reference *>(static_cast<const Reference *>(p_from.ptr()));
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	Ref(T *p_reference) {
		reference = nullptr;
		if (p_reference) {
			ref_pointer(p_reference);
		}
	}

	inline bool is_valid() const { return reference != nullptr; }
	inline bool is_null() const { return reference == nullptr; }

	void unref() {
		//TODO this should be moved to mutexes, since this engine does not really
		// do a lot of referencing on references and stuff
		// mutexes will avoid more crashes?

		if (reference && reference->unreference()) {
			memdelete(reference);
		}
		reference = nullptr;
	}

	void instance() {
		ref(memnew(T));
	}

	RefPtr get_ref_ptr() const {
		RefPtr refptr;
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(refptr.get_data());
		*irr = *this;
		return refptr;
	};

	operator Variant() const {
		return Variant(get_ref_ptr());
	}

	void operator=(const RefPtr &p_refptr) {
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(p_refptr.get_data());
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	void operator=(const Variant &p_variant) {
		RefPtr refptr = p_variant;
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(refptr.get_data());
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	Ref(const Variant &p_variant) {
		RefPtr refptr = p_variant;
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(refptr.get_data());
		reference = nullptr;
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	Ref(const RefPtr &p_refptr) {
		Ref<Reference> *irr = reinterpret_cast<Ref<Reference> *>(p_refptr.get_data());
		reference = nullptr;
		Reference *refb = irr->ptr();
		if (!refb) {
			unref();
			return;
		}
		Ref r;
		r.reference = Object::cast_to<T>(refb);
		ref(r);
		r.reference = nullptr;
	}

	Ref() {
		reference = nullptr;
	}

	~Ref() {
		unref();
	}
};

typedef Ref<Reference> REF;

/*
class WeakRef : public Reference {
	SFW_OBJECT(WeakRef, Reference);

	ObjectID ref;

protected:
	static void _bind_methods();

public:
	void set_obj(Object *p_object);
	void set_ref(const REF &p_ref);

	WeakRef();
};
*/

#line 0

#line 1 "sfwl/object/resource.h"

class Resource : public Reference {
	SFW_OBJECT(Resource, Reference);

public:
	Signal changed;

	void emit_changed();

	virtual Error load(const String &path);
	virtual Error save(const String &path);

	Resource();
	virtual ~Resource();
};

#line 0

#endif
