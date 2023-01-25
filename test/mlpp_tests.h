#ifndef MLPP_TESTS_H
#define MLPP_TESTS_H

// TODO port this class to use the test module once it's working
// Also don't forget to remove it's bindings

#include "core/containers/vector.h"

#include "core/object/reference.h"

#include "core/string/ustring.h"

class MLPPTests : public Reference {
	GDCLASS(MLPPTests, Reference);

public:
	void test_statistics();

	void is_approx_equalsd(double a, double b, const String &str);
	void is_approx_equals_dvec(const Vector<double> &a, const Vector<double> &b, const String &str);

	MLPPTests();
	~MLPPTests();

protected:
	static void _bind_methods();
};

#endif
