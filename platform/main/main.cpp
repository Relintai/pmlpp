
#include "sfw.h"

#include "test/mlpp_tests.h"
#include "test/mlpp_matrix_tests.h"

int main() {
	SFWCore::setup();

	Ref<MLPPTests> tests;
	tests.instance();

	tests->test_statistics();

	//tests->test_multivariate_linear_regression_gradient_descent(true);

	SFWCore::cleanup();

	return 0;
}
