#ifndef ERROR_MACROS_H
#define ERROR_MACROS_H

#include "logger.h"
#include "typedefs.h"

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

#define CRASH_MSG(msg)                                                      \
	RLogger::log_error(__FUNCTION__, __FILE__, __LINE__, msg); \
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

#endif
