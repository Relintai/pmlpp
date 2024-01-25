
#include "sfw.h"

#include "test/mlpp_tests.h"
#include "test/mlpp_matrix_tests.h"

int main() {
	SFWCore::setup();

	Ref<MLPPTests> tests;
	tests.instance();

	tests->_breast_cancer_data_path = "bin/datasets/BreastCancer.csv";
	tests->_breast_cancer_svm_data_path = "bin/datasets/BreastCancerSVM.csv";
	tests->_california_housing_data_path = "bin/datasets/CaliforniaHousing.csv";
	tests->_fires_and_crime_data_path = "bin/datasets/FiresAndCrime.csv";
	tests->_iris_data_path = "bin/datasets/Iris.csv";
	tests->_mnist_test_data_path = "bin/datasets/MnistTest.csv";
	tests->_mnist_train_data_path = "bin/datasets/MnistTrain.csv";
	tests->_wine_data_path = "bin/datasets/Wine.csv";

	tests->test_statistics();
	tests->test_multivariate_linear_regression_gradient_descent(false);
	tests->test_softmax_regression(false);

	SFWCore::cleanup();

	return 0;
}
