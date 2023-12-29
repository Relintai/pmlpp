#ifndef MLPP_TESTS_H
#define MLPP_TESTS_H



#include "core/math/math_defs.h"

#include "core/containers/vector.h"

#include "core/object/reference.h"

#include "core/string/ustring.h"

// TODO port this class to use the test module once it's working
// Also don't forget to remove it's bindings

class MLPPMatrix;
class MLPPVector;

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
	void test_multivariate_linear_regression_adam(bool ui = false);
	void test_multivariate_linear_regression_score_sgd_adam(bool ui = false);
	void test_multivariate_linear_regression_epochs_gradient_descent(bool ui = false);
	void test_multivariate_linear_regression_newton_raphson(bool ui = false);

	void test_logistic_regression(bool ui = false);
	void test_probit_regression(bool ui = false);
	void test_c_log_log_regression(bool ui = false);
	void test_exp_reg_regression(bool ui = false);
	void test_tanh_regression(bool ui = false);
	void test_softmax_regression(bool ui = false);
	void test_support_vector_classification(bool ui = false);

	void test_mlp(bool ui = false);
	void test_soft_max_network(bool ui = false);
	void test_autoencoder(bool ui = false);
	void test_dynamically_sized_ann(bool ui = false);
	void test_wgan_old(bool ui = false);
	void test_wgan(bool ui = false);
	void test_ann(bool ui = false);
	void test_dynamically_sized_mann(bool ui = false);
	void test_train_test_split_mann(bool ui = false);

	void test_naive_bayes();
	void test_k_means(bool ui = false);
	void test_knn(bool ui = false);

	void test_convolution_tensors_etc();
	void test_pca_svd_eigenvalues_eigenvectors(bool ui = false);

	void test_nlp_and_data(bool ui = false);
	void test_outlier_finder(bool ui = false);
	void test_new_math_functions();
	void test_positive_definiteness_checker();
	void test_numerical_analysis();
	void test_support_vector_classification_kernel(bool ui = false);

	void test_mlpp_vector();

	void is_approx_equalsd(real_t a, real_t b, const String &str);
	void is_approx_equals_dvec(const Vector<real_t> &a, const Vector<real_t> &b, const String &str);
	void is_approx_equals_dmat(const Vector<Vector<real_t>> &a, const Vector<Vector<real_t>> &b, const String &str);

	void is_approx_equals_mat(Ref<MLPPMatrix> a, Ref<MLPPMatrix> b, const String &str);
	void is_approx_equals_vec(Ref<MLPPVector> a, Ref<MLPPVector> b, const String &str);

	MLPPTests();
	~MLPPTests();

protected:
	static void _bind_methods();

	String _breast_cancer_data_path;
	String _breast_cancer_svm_data_path;
	String _california_housing_data_path;
	String _fires_and_crime_data_path;
	String _iris_data_path;
	String _mnist_test_data_path;
	String _mnist_train_data_path;
	String _wine_data_path;
};

#endif
