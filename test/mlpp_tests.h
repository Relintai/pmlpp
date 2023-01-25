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
	void test_linear_algebra();
	void test_univariate_linear_regression();

	void test_multivariate_linear_regression_gradient_descent(bool ui = false);
	void test_multivariate_linear_regression_sgd(bool ui = false);
	void test_multivariate_linear_regression_mbgd(bool ui = false);
	void test_multivariate_linear_regression_normal_equation(bool ui = false);
	void test_multivariate_linear_regression_adam();
	void test_multivariate_linear_regression_score_sgd_adam(bool ui = false);
	void test_multivariate_linear_regression_epochs_gradient_descent(bool ui = false);
	void test_multivariate_linear_regression_newton_raphson(bool ui = false);

	void is_approx_equalsd(double a, double b, const String &str);
	void is_approx_equals_dvec(const Vector<double> &a, const Vector<double> &b, const String &str);
	void is_approx_equals_dmat(const Vector<Vector<double>> &a, const Vector<Vector<double>> &b, const String &str);

	MLPPTests();
	~MLPPTests();

protected:
	static void _bind_methods();

	String _load_fires_and_crime_data_path;
	String _load_california_housing_data_path;
};

#endif
