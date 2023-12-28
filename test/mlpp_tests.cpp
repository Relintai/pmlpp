
#include "mlpp_tests.h"

#include "core/math/math_funcs.h"

#include "core/log/logger.h"

//TODO remove
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include "../mlpp/lin_alg/mlpp_matrix.h"
#include "../mlpp/lin_alg/mlpp_vector.h"

#include "../mlpp/activation/activation.h"
#include "../mlpp/ann/ann.h"
#include "../mlpp/auto_encoder/auto_encoder.h"
#include "../mlpp/bernoulli_nb/bernoulli_nb.h"
#include "../mlpp/c_log_log_reg/c_log_log_reg.h"
#include "../mlpp/convolutions/convolutions.h"
#include "../mlpp/cost/cost.h"
#include "../mlpp/data/data.h"
#include "../mlpp/dual_svc/dual_svc.h"
#include "../mlpp/exp_reg/exp_reg.h"
#include "../mlpp/gan/gan.h"
#include "../mlpp/gaussian_nb/gaussian_nb.h"
#include "../mlpp/kmeans/kmeans.h"
#include "../mlpp/knn/knn.h"
#include "../mlpp/lin_alg/lin_alg.h"
#include "../mlpp/lin_reg/lin_reg.h"
#include "../mlpp/log_reg/log_reg.h"
#include "../mlpp/mann/mann.h"
#include "../mlpp/mlp/mlp.h"
#include "../mlpp/multinomial_nb/multinomial_nb.h"
#include "../mlpp/numerical_analysis/numerical_analysis.h"
#include "../mlpp/outlier_finder/outlier_finder.h"
#include "../mlpp/pca/pca.h"
#include "../mlpp/probit_reg/probit_reg.h"
#include "../mlpp/softmax_net/softmax_net.h"
#include "../mlpp/softmax_reg/softmax_reg.h"
#include "../mlpp/stat/stat.h"
#include "../mlpp/svc/svc.h"
#include "../mlpp/tanh_reg/tanh_reg.h"
#include "../mlpp/transforms/transforms.h"
#include "../mlpp/uni_lin_reg/uni_lin_reg.h"
#include "../mlpp/wgan/wgan.h"

Vector<real_t> dstd_vec_to_vec(const std::vector<real_t> &in) {
	Vector<real_t> r;

	r.resize(static_cast<int>(in.size()));
	real_t *darr = r.ptrw();

	for (uint32_t i = 0; i < in.size(); ++i) {
		darr[i] = in[i];
	}

	return r;
}

Vector<Vector<real_t>> dstd_mat_to_mat(const std::vector<std::vector<real_t>> &in) {
	Vector<Vector<real_t>> r;

	for (uint32_t i = 0; i < in.size(); ++i) {
		r.push_back(dstd_vec_to_vec(in[i]));
	}

	return r;
}

void MLPPTests::test_statistics() {
	MLPPStat stat;
	MLPPConvolutions conv;

	// STATISTICS
	const real_t x_arr[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	const real_t y_arr[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
	const real_t w_arr[] = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };

	Ref<MLPPVector> x = memnew(MLPPVector(x_arr, 10));
	Ref<MLPPVector> y = memnew(MLPPVector(y_arr, 10));
	Ref<MLPPVector> w = memnew(MLPPVector(w_arr, 10));

	is_approx_equalsd(stat.meanv(x), 5.5, "Arithmetic Mean");
	is_approx_equalsd(stat.meanv(x), 5.5, "Median");

	is_approx_equals_vec(stat.mode(x), x, "stat.mode(x)");

	is_approx_equalsd(stat.range(x), 9, "Range");
	is_approx_equalsd(stat.midrange(x), 4.5, "Midrange");
	is_approx_equalsd(stat.abs_avg_deviation(x), 2.5, "Absolute Average Deviation");
	is_approx_equalsd(stat.standard_deviationv(x), 3.02765, "Standard Deviation");
	is_approx_equalsd(stat.variancev(x), 9.16667, "Variance");
	is_approx_equalsd(stat.covariancev(x, y), -9.16667, "Covariance");
	is_approx_equalsd(stat.correlation(x, y), -1, "Correlation");
	is_approx_equalsd(stat.r2(x, y), 1, "R^2");

	// Returns 1 - (1/k^2)
	is_approx_equalsd(stat.chebyshev_ineq(2), 0.75, "Chebyshev Inequality");
	is_approx_equalsd(stat.weighted_mean(x, w), 5.5, "Weighted Mean");
	is_approx_equalsd(stat.geometric_mean(x), 4.52873, "Geometric Mean");
	is_approx_equalsd(stat.harmonic_mean(x), 3.41417, "Harmonic Mean");
	is_approx_equalsd(stat.rms(x), 6.20484, "Root Mean Square (Quadratic mean)");
	is_approx_equalsd(stat.power_mean(x, 5), 7.39281, "Power Mean (p = 5)");
	is_approx_equalsd(stat.lehmer_mean(x, 5), 8.71689, "Lehmer Mean (p = 5)");
	is_approx_equalsd(stat.weighted_lehmer_mean(x, w, 5), 8.71689, "Weighted Lehmer Mean (p = 5)");
	is_approx_equalsd(stat.contra_harmonic_mean(x), 7, "Contraharmonic Mean");
	is_approx_equalsd(stat.heronian_mean(1, 10), 4.72076, "Hernonian Mean");
	is_approx_equalsd(stat.heinz_mean(1, 10, 1), 5.5, "Heinz Mean (x = 1)");
	is_approx_equalsd(stat.neuman_sandor_mean(1, 10), 3.36061, "Neuman-Sandor Mean");
	is_approx_equalsd(stat.stolarsky_mean(1, 10, 5), 6.86587, "Stolarsky Mean (p = 5)");
	is_approx_equalsd(stat.identric_mean(1, 10), 4.75135, "Identric Mean");
	is_approx_equalsd(stat.log_mean(1, 10), 3.90865, "Logarithmic Mean");
	is_approx_equalsd(stat.abs_avg_deviation(x), 2.5, "Absolute Average Deviation");
}

void MLPPTests::test_linear_algebra() {
	MLPPLinAlg alg;

	const real_t square_arr[] = {
		1, 1, //
		-1, 1, //
		1, -1, //
		-1, -1, //
	};

	const real_t square_rot_res_arr[] = {
		1.41421, 1.11022e-16, //
		-1.11022e-16, 1.41421, //
		1.11022e-16, -1.41421, //
		-1.41421, -1.11022e-16, //
	};

	Ref<MLPPMatrix> square(memnew(MLPPMatrix(square_arr, 4, 2)));
	Ref<MLPPMatrix> square_rot(memnew(MLPPMatrix(square_rot_res_arr, 4, 2)));

	is_approx_equals_mat(square->rotaten(Math_PI / 4), square_rot, "square->rotaten(Math_PI / 4)");

	const real_t A_arr[] = {
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, //
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, //
	};
	const real_t a_arr[] = { 4, 3, 1, 3 };
	const real_t b_arr[] = { 3, 5, 6, 1 };

	const real_t mmtr_res_arr[] = {
		2, 4, 6, 8, 10, 12, 14, 16, 18, 20, //
		4, 8, 12, 16, 20, 24, 28, 32, 36, 40, //
		6, 12, 18, 24, 30, 36, 42, 48, 54, 60, //
		8, 16, 24, 32, 40, 48, 56, 64, 72, 80, //
		10, 20, 30, 40, 50, 60, 70, 80, 90, 100, //
		12, 24, 36, 48, 60, 72, 84, 96, 108, 120, //
		14, 28, 42, 56, 70, 84, 98, 112, 126, 140, //
		16, 32, 48, 64, 80, 96, 112, 128, 144, 160, //
		18, 36, 54, 72, 90, 108, 126, 144, 162, 180, //
		20, 40, 60, 80, 100, 120, 140, 160, 180, 200 //
	};

	Ref<MLPPMatrix> A(memnew(MLPPMatrix(A_arr, 2, 10)));
	Ref<MLPPVector> a(memnew(MLPPVector(a_arr, 4)));
	Ref<MLPPVector> b(memnew(MLPPVector(b_arr, 4)));
	Ref<MLPPMatrix> mmtr_res(memnew(MLPPMatrix(mmtr_res_arr, 10, 10)));

	is_approx_equals_mat(alg.matmultnm(alg.transposenm(A), A), mmtr_res, "alg.matmultnm(alg.transposenm(A), A)");

	is_approx_equalsd(alg.dotnv(a, b), 36, "alg.dotnv(a, b)");

	const real_t had_prod_res_arr[] = {
		1, 4, 9, 16, 25, 36, 49, 64, 81, 100, //
		1, 4, 9, 16, 25, 36, 49, 64, 81, 100 //
	};

	Ref<MLPPMatrix> had_prod_res(memnew(MLPPMatrix(had_prod_res_arr, 2, 10)));

	is_approx_equals_mat(alg.hadamard_productnm(A, A), had_prod_res, "alg.hadamard_productnm(A, A)");

	const real_t id_10_res_arr[] = {
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0, //
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0, //
		0, 0, 0, 1, 0, 0, 0, 0, 0, 0, //
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0, //
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, //
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0, //
		0, 0, 0, 0, 0, 0, 0, 1, 0, 0, //
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, //
		0, 0, 0, 0, 0, 0, 0, 0, 0, 1, //
	};

	Ref<MLPPMatrix> id_10_res(memnew(MLPPMatrix(id_10_res_arr, 10, 10)));

	is_approx_equals_mat(alg.identitym(10), id_10_res, "alg.identitym(10)");
}

void MLPPTests::test_univariate_linear_regression() {
	const real_t slr_res_n_arr[] = {
		24.109467, 28.482935, 29.808228, 26.097408, 27.290173, 61.085152, 30.470875, 25.037172, 25.567291, //
		35.904579, 54.458687, 18.808294, 23.446819, 18.543236, 19.205883, 21.193821, 23.049232, 18.808294, //
		25.434761, 35.904579, 37.759987, 40.278046, 63.868271, 68.50679, 40.410576, 46.77198, 32.061226, //
		23.314291, 44.784042, 44.518982, 27.82029, 20.663704, 22.519115, 53.796036, 38.952751, //
		30.868464, 20.398645 //
	};

	Ref<MLPPVector> slr_res_v(memnew(MLPPVector(slr_res_n_arr, 37)));

	// Univariate, simple linear regression, case where k = 1
	MLPPData data;
	Ref<MLPPDataESimple> ds = data.load_fires_and_crime(_fires_and_crime_data_path);
	MLPPUniLinReg model(ds->get_input(), ds->get_output());

	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	is_approx_equals_vec(res, slr_res_v, "test_univariate_linear_regression()");
}

void MLPPTests::test_multivariate_linear_regression_gradient_descent(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);

	MLPPLinReg model(ds->get_input(), ds->get_output()); // Can use Lasso, Ridge, ElasticNet Reg
	model.gradient_descent(0.0000001, 30, ui);
	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	MLPPCost mlpp_cost;

	int rmse = (int)mlpp_cost.rmsev(ds->get_output(), res);

	//Lose the bottom 14 bits (This should allow for 16384 difference.)
	rmse = rmse >> 14;
	rmse = rmse << 14;

	is_approx_equalsd(rmse, 163840, "test_multivariate_linear_regression_gradient_descent() RMSE");
}

void MLPPTests::test_multivariate_linear_regression_sgd(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);

	MLPPLinReg model(ds->get_input(), ds->get_output()); // Can use Lasso, Ridge, ElasticNet Reg
	model.sgd(0.00000001, 300000, ui);
	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	MLPPCost mlpp_cost;

	int rmse = (int)mlpp_cost.rmsev(ds->get_output(), res);

	//Lose the bottom X bits (This should allow for 2^X difference.)
	rmse = rmse >> 15;
	rmse = rmse << 15;

	is_approx_equalsd(rmse, 98304, "test_multivariate_linear_regression_sgd() RMSE");
}

void MLPPTests::test_multivariate_linear_regression_mbgd(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);

	MLPPLinReg model(ds->get_input(), ds->get_output()); // Can use Lasso, Ridge, ElasticNet Reg
	model.mbgd(0.00000001, 30, 2, ui);
	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	MLPPCost mlpp_cost;

	int rmse = (int)mlpp_cost.rmsev(ds->get_output(), res);

	//Lose the bottom X bits (This should allow for 2^X difference.)
	rmse = rmse >> 10;
	rmse = rmse << 10;

	is_approx_equalsd(rmse, 230400, "test_multivariate_linear_regression_mbgd() RMSE");
}

void MLPPTests::test_multivariate_linear_regression_normal_equation(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);
	ds->get_input()->resize(Size2i(8, 10));
	ds->get_output()->resize(10);

	MLPPLinReg model(ds->get_input(), ds->get_output()); // Can use Lasso, Ridge, ElasticNet Reg
	model.normal_equation();
	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	MLPPCost mlpp_cost;

	int rmse = (int)mlpp_cost.rmsev(ds->get_output(), res);

	//Lose the bottom X bits (This should allow for 2^X difference.)
	rmse = rmse >> 10;
	rmse = rmse << 10;

	is_approx_equalsd(rmse, 319488, "test_multivariate_linear_regression_normal_equation() RMSE");
}

void MLPPTests::test_multivariate_linear_regression_adam(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);

	MLPPLinReg model(ds->get_input(), ds->get_output());

	model.adam(0.0001, 30, 10, 0.9, 0.999, 1e-8, ui);

	//real_t score = 100 * model.score();

	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	MLPPCost mlpp_cost;

	int rmse = (int)mlpp_cost.rmsev(ds->get_output(), res);

	//Lose the bottom X bits (This should allow for 2^X difference.)
	rmse = rmse >> 10;
	rmse = rmse << 10;

	is_approx_equalsd(rmse, 156672, "test_multivariate_linear_regression_adam() RMSE");
	//is_approx_equalsd(score, 319488, "test_multivariate_linear_regression_adam() score");
}

void MLPPTests::test_multivariate_linear_regression_score_sgd_adam(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;
	MLPPLinAlg algn;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);

	const int TRIAL_NUM = 10;

	real_t scoreSGD = 0;
	real_t scoreADAM = 0;
	for (int i = 0; i < TRIAL_NUM; i++) {
		MLPPLinReg modelf(ds->get_input(), ds->get_output());
		modelf.mbgd(0.001, 5, 1, ui);
		scoreSGD += modelf.score();

		MLPPLinReg adamModelf(ds->get_input(), ds->get_output());
		adamModelf.adam(0.1, 5, 1, 0.9, 0.999, 1e-8, ui); // Change batch size = sgd, bgd
		scoreADAM += adamModelf.score();
	}

	is_approx_equalsd((int)(100 * scoreSGD / TRIAL_NUM), 0, "test_multivariate_linear_regression_score_sgd_adam() ACCURACY, AVG, SGD");
	is_approx_equalsd((int)(100 * scoreADAM / TRIAL_NUM), 0, "test_multivariate_linear_regression_score_sgd_adam() ACCURACY, AVG, ADAM");
}

void MLPPTests::test_multivariate_linear_regression_epochs_gradient_descent(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;
	MLPPLinAlg algn;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);

	MLPPLinReg model(ds->get_input(), ds->get_output()); // Can use Lasso, Ridge, ElasticNet Reg
	model.gradient_descent(0.0000001, 300, ui);

	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	MLPPCost mlpp_cost;

	int rmse = (int)mlpp_cost.rmsev(ds->get_output(), res);

	//Lose the bottom X bits (This should allow for 2^X difference.)
	rmse = rmse >> 16;
	rmse = rmse << 16;

	is_approx_equalsd(rmse, 131072, "test_multivariate_linear_regression_epochs_gradient_descent() RMSE");
}

void MLPPTests::test_multivariate_linear_regression_newton_raphson(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;
	MLPPLinAlg algn;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_california_housing_data_path);

	MLPPLinReg model(ds->get_input(), ds->get_output());
	model.newton_raphson(1.5, 300, ui);
	Ref<MLPPVector> res = model.model_set_test(ds->get_input());

	MLPPCost mlpp_cost;

	//int rmse = (int)mlpp_cost.rmsev(ds->get_output(), res);

	//Lose the bottom X bits (This should allow for 2^X difference.)
	//rmse = rmse >> 15;
	//rmse = rmse << 15;

	//is_approx_equalsd(rmse, 98304, "test_multivariate_linear_regression_newton_raphson() RMSE");
}

void MLPPTests::test_logistic_regression(bool ui) {
	MLPPLinAlg alg;
	MLPPData data;

	Ref<MLPPDataSimple> dt = data.load_breast_cancer(_breast_cancer_data_path);

	// LOGISTIC REGRESSION

	MLPPLogReg model(dt->get_input(), dt->get_output());
	model.sgd(0.001, 100000, ui);
	PLOG_MSG(model.model_set_test(dt->get_input())->to_string());
	std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_probit_regression(bool ui) {
	MLPPLinAlg alg;
	MLPPData data;

	// PROBIT REGRESSION
	Ref<MLPPDataSimple> dt = data.load_breast_cancer(_breast_cancer_data_path);

	MLPPProbitReg model(dt->get_input(), dt->get_output());
	model.train_sgd(0.001, 10000, ui);
	PLOG_MSG(model.model_set_test(dt->get_input())->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * model.score()) + "%");
}
void MLPPTests::test_c_log_log_regression(bool ui) {
	MLPPLinAlg alg;
	MLPPLinAlg algn;

	// CLOGLOG REGRESSION
	std::vector<std::vector<real_t>> inputSet = { { 1, 2, 3, 4, 5, 6, 7, 8 }, { 0, 0, 0, 0, 1, 1, 1, 1 } };
	std::vector<real_t> outputSet = { 0, 0, 0, 0, 1, 1, 1, 1 };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);
	input_set = input_set->transposen();

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	MLPPCLogLogReg model(algn.transposenm(input_set), output_set);
	model.sgd(0.1, 10000, ui);
	PLOG_MSG(model.model_set_test(algn.transposenm(input_set))->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * model.score()) + "%");
}
void MLPPTests::test_exp_reg_regression(bool ui) {
	MLPPLinAlg alg;

	// EXPREG REGRESSION
	std::vector<std::vector<real_t>> inputSet = { { 0, 1, 2, 3, 4 } };
	std::vector<real_t> outputSet = { 1, 2, 4, 8, 16 };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	MLPPExpReg model(alg.transposenm(input_set), output_set);
	model.sgd(0.001, 10000, ui);
	PLOG_MSG(model.model_set_test(alg.transposenm(input_set))->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * model.score()) + "%");
}
void MLPPTests::test_tanh_regression(bool ui) {
	MLPPLinAlg alg;

	// TANH REGRESSION
	std::vector<std::vector<real_t>> inputSet = { { 4, 3, 0, -3, -4 }, { 0, 0, 0, 1, 1 } };
	std::vector<real_t> outputSet = { 1, 1, 0, -1, -1 };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	MLPPTanhReg model(alg.transposenm(input_set), output_set);
	model.train_sgd(0.1, 10000, ui);
	//PLOG_MSG(model.model_set_test(alg.transposenm(input_set))->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * model.score()) + "%");
}
void MLPPTests::test_softmax_regression(bool ui) {
	MLPPLinAlg alg;
	MLPPData data;

	Ref<MLPPDataComplex> dt = data.load_iris(_iris_data_path);

	// SOFTMAX REGRESSION
	MLPPSoftmaxReg model(dt->get_input(), dt->get_output());
	model.train_sgd(0.1, 10000, ui);
	//PLOG_MSG(model.model_set_test(dt->get_input())->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * model.score()) + "%");
}
void MLPPTests::test_support_vector_classification(bool ui) {
	//MLPPStat stat;
	MLPPLinAlg alg;
	//MLPPActivation avn;
	//MLPPCost cost;
	MLPPData data;
	//MLPPConvolutions conv;

	// SUPPORT VECTOR CLASSIFICATION
	Ref<MLPPDataSimple> dt = data.load_breast_cancer_svc(_breast_cancer_svm_data_path);

	MLPPSVC model(dt->get_input(), dt->get_output(), ui);
	model.train_sgd(0.00001, 100000, ui);
	PLOG_MSG((model.model_set_test(dt->get_input())->to_string()));
	PLOG_MSG("ACCURACY: " + String::num(100 * model.score()) + "%");
}

void MLPPTests::test_mlp(bool ui) {
	MLPPLinAlg alg;

	// MLP
	std::vector<std::vector<real_t>> inputSet = {
		{ 0, 0 },
		{ 1, 1 },
		{ 0, 1 },
		{ 1, 0 }
	};
	std::vector<real_t> outputSet = { 0, 1, 1, 0 };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	MLPPMLP model_new(input_set, output_set, 2);
	model_new.gradient_descent(0.1, 10000, ui);
	String res = model_new.model_set_test(input_set)->to_string();
	res += "\nACCURACY (gradient_descent): " + String::num(100.0 * model_new.score()) + "%";

	PLOG_MSG(res);

	MLPPMLP model_new2(input_set, output_set, 2);
	model_new2.sgd(0.01, 10000, ui);
	res = model_new2.model_set_test(input_set)->to_string();
	res += "\nACCURACY (sgd): " + String::num(100.0 * model_new2.score()) + "%";

	PLOG_MSG(res);

	MLPPMLP model_new3(input_set, output_set, 2);
	model_new3.mbgd(0.01, 10000, 2, ui);
	res = model_new3.model_set_test(input_set)->to_string();
	res += "\nACCURACY (mbgd): " + String::num(100.0 * model_new3.score()) + "%";

	PLOG_MSG(res);
}
void MLPPTests::test_soft_max_network(bool ui) {
	MLPPLinAlg alg;
	MLPPData data;

	// SOFTMAX NETWORK
	Ref<MLPPDataComplex> dt = data.load_wine(_wine_data_path);

	MLPPSoftmaxNet model(dt->get_input(), dt->get_output(), 1);
	model.train_gradient_descent(0.000001, 300, ui);
	PLOG_MSG(model.model_set_test(dt->get_input())->to_string());
	std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_autoencoder(bool ui) {
	MLPPLinAlg alg;
	MLPPLinAlg algn;

	std::vector<std::vector<real_t>> inputSet = { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 3, 5, 9, 12, 15, 18, 21, 24, 27, 30 } };

	// AUTOENCODER
	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	MLPPAutoEncoder model(algn.transposenm(input_set), 5);
	model.sgd(0.001, 300000, ui);
	PLOG_MSG(model.model_set_test(algn.transposenm(input_set))->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * model.score()) + "%");
}
void MLPPTests::test_dynamically_sized_ann(bool ui) {
	MLPPLinAlg alg;
	MLPPLinAlg algn;

	// DYNAMICALLY SIZED ANN
	// Possible Weight Init Methods: Default, Uniform, HeNormal, HeUniform, XavierNormal, XavierUniform
	// Possible Activations: Linear, Sigmoid, Swish, Softplus, Softsign, CLogLog, Ar{Sinh, Cosh, Tanh, Csch, Sech, Coth},  GaussianCDF, GELU, UnitStep
	// Possible Loss Functions: MSE, RMSE, MBE, LogLoss, CrossEntropy, HingeLoss
	std::vector<std::vector<real_t>> inputSet = { { 0, 0, 1, 1 }, { 0, 1, 0, 1 } };
	std::vector<real_t> outputSet = { 0, 1, 1, 0 };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	MLPPANN ann(algn.transposenm(input_set), output_set);

	ann.add_layer(2, MLPPActivation::ACTIVATION_FUNCTION_COSH);
	ann.add_output_layer(MLPPActivation::ACTIVATION_FUNCTION_SIGMOID, MLPPCost::COST_TYPE_LOGISTIC_LOSS);

	ann.amsgrad(0.1, 10000, 1, 0.9, 0.999, 0.000001, ui);
	ann.adadelta(1, 1000, 2, 0.9, 0.000001, ui);
	ann.momentum(0.1, 8000, 2, 0.9, true, ui);
	ann.set_learning_rate_scheduler_drop(MLPPANN::SCHEDULER_TYPE_STEP, 0.5, 1000);
	ann.gradient_descent(0.01, 30000);

	PLOG_MSG(ann.model_set_test(algn.transposenm(input_set))->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * ann.score()) + "%");
}
void MLPPTests::test_wgan_old(bool ui) {
}
void MLPPTests::test_wgan(bool ui) {
	//MLPPStat stat;
	MLPPLinAlg alg;
	//MLPPActivation avn;
	//MLPPCost cost;
	//MLPPData data;
	//MLPPConvolutions conv;

	std::vector<std::vector<real_t>> outputSet = {
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
		{ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40 }
	};

	Ref<MLPPMatrix> output_set;
	output_set.instance();
	output_set->set_from_std_vectors(outputSet);
	output_set = output_set->transposen();

	MLPPWGAN gan(2, output_set); // our gan is a wasserstein gan (wgan)
	gan.create_layer(5, MLPPActivation::ACTIVATION_FUNCTION_SIGMOID);
	gan.create_layer(2, MLPPActivation::ACTIVATION_FUNCTION_RELU);
	gan.create_layer(5, MLPPActivation::ACTIVATION_FUNCTION_SIGMOID);
	gan.add_output_layer(); // User can specify weight init- if necessary.
	gan.gradient_descent(0.1, 55000, ui);

	String str = "GENERATED INPUT: (Gaussian-sampled noise):\n";
	str += gan.generate_example(100)->to_string();
	PLOG_MSG(str);
}
void MLPPTests::test_ann(bool ui) {
	MLPPLinAlg alg;

	std::vector<std::vector<real_t>> inputSet = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }; // XOR
	std::vector<real_t> outputSet = { 0, 1, 1, 0 };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	MLPPANN ann(input_set, output_set);
	ann.add_layer(5, MLPPActivation::ACTIVATION_FUNCTION_SIGMOID);
	ann.add_layer(8, MLPPActivation::ACTIVATION_FUNCTION_SIGMOID); // Add more layers as needed.
	ann.add_output_layer(MLPPActivation::ACTIVATION_FUNCTION_SIGMOID, MLPPCost::COST_TYPE_LOGISTIC_LOSS);
	ann.gradient_descent(1, 20000, ui);

	Ref<MLPPVector> predictions = ann.model_set_test(input_set);
	PLOG_MSG(predictions->to_string()); // Testing out the model's preds for train set.
	PLOG_MSG("ACCURACY: " + String::num(100 * ann.score()) + "%"); // Accuracy.
}
void MLPPTests::test_dynamically_sized_mann(bool ui) {
	MLPPLinAlg alg;
	MLPPData data;

	// DYNAMICALLY SIZED MANN (Multidimensional Output ANN)
	std::vector<std::vector<real_t>> inputSet = { { 1, 2, 3 }, { 2, 4, 6 }, { 3, 6, 9 }, { 4, 8, 12 } };
	std::vector<std::vector<real_t>> outputSet = { { 1, 5 }, { 2, 10 }, { 3, 15 }, { 4, 20 } };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	Ref<MLPPMatrix> output_set;
	output_set.instance();
	output_set->set_from_std_vectors(outputSet);

	MLPPMANN mann(input_set, output_set);
	mann.add_output_layer(MLPPActivation::ACTIVATION_FUNCTION_LINEAR, MLPPCost::COST_TYPE_MSE);
	mann.gradient_descent(0.001, 80000, false);
	PLOG_MSG(mann.model_set_test(input_set)->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * mann.score()) + "%");
}
void MLPPTests::test_train_test_split_mann(bool ui) {
	MLPPLinAlg alg;
	MLPPLinAlg algn;
	MLPPData data;

	// TRAIN TEST SPLIT CHECK
	std::vector<std::vector<real_t>> inputSet1 = { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 3, 5, 9, 12, 15, 18, 21, 24, 27, 30 } };
	std::vector<std::vector<real_t>> outputSet1 = { { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 } };

	Ref<MLPPMatrix> input_set_1;
	input_set_1.instance();
	input_set_1->set_from_std_vectors(inputSet1);

	Ref<MLPPMatrix> output_set_1;
	output_set_1.instance();
	output_set_1->set_from_std_vectors(outputSet1);

	Ref<MLPPDataComplex> d;
	d.instance();

	d->set_input(algn.transposenm(input_set_1));
	d->set_output(algn.transposenm(output_set_1));

	MLPPData::SplitComplexData split_data = data.train_test_split(d, 0.2);

	PLOG_MSG(split_data.train->get_input()->to_string());
	PLOG_MSG(split_data.train->get_output()->to_string());
	PLOG_MSG(split_data.test->get_input()->to_string());
	PLOG_MSG(split_data.test->get_output()->to_string());

	MLPPMANN mann(split_data.train->get_input(), split_data.train->get_output());
	mann.add_layer(100, MLPPActivation::ACTIVATION_FUNCTION_RELU, MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_XAVIER_NORMAL);
	mann.add_output_layer(MLPPActivation::ACTIVATION_FUNCTION_SOFTMAX, MLPPCost::COST_TYPE_CROSS_ENTROPY, MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_XAVIER_NORMAL);
	mann.gradient_descent(0.1, 80000, ui);
	PLOG_MSG(mann.model_set_test(split_data.test->get_input())->to_string());
	PLOG_MSG("ACCURACY: " + String::num(100 * mann.score()) + "%");
}

void MLPPTests::test_naive_bayes() {
	MLPPLinAlg alg;
	MLPPLinAlg algn;

	// NAIVE BAYES
	std::vector<std::vector<real_t>> inputSet = { { 1, 1, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 0, 1 } };
	std::vector<real_t> outputSet = { 0, 1, 0, 1, 1 };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);
	input_set = input_set->transposen();

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	ERR_PRINT("MLPPMultinomialNB");

	MLPPMultinomialNB MNB(input_set, output_set, 2);
	PLOG_MSG(MNB.model_set_test(input_set)->to_string());

	ERR_PRINT("MLPPBernoulliNB");

	MLPPBernoulliNB BNB(input_set, output_set);
	PLOG_MSG(BNB.model_set_test(input_set)->to_string());

	ERR_PRINT("MLPPGaussianNB");

	MLPPGaussianNB GNB(input_set, output_set, 2);
	PLOG_MSG(GNB.model_set_test(input_set)->to_string());
}
void MLPPTests::test_k_means(bool ui) {
	// KMeans
	std::vector<std::vector<real_t>> inputSet = { { 32, 0, 7 }, { 2, 28, 17 }, { 0, 9, 23 } };

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	Ref<MLPPKMeans> kmeans;
	kmeans.instance();
	kmeans->set_input_set(input_set);
	kmeans->set_k(3);
	kmeans->set_mean_type(MLPPKMeans::MEAN_TYPE_KMEANSPP);

	kmeans->train(3, ui);

	PLOG_MSG(kmeans->model_set_test(input_set)->to_string());
	PLOG_MSG(kmeans->silhouette_scores()->to_string());
}
void MLPPTests::test_knn(bool ui) {
	MLPPLinAlg alg;

	// kNN
	std::vector<std::vector<real_t>> inputSet = {
		{ 1, 2, 3, 4, 5, 6, 7, 8 },
		{ 0, 0, 0, 0, 1, 1, 1, 1 }
	};
	std::vector<real_t> outputSet = { 0, 0, 0, 0, 1, 1, 1, 1 };

	Ref<MLPPMatrix> ism;
	ism.instance();
	ism->set_from_std_vectors(inputSet);
	ism = ism->transposen();

	//ERR_PRINT(ism->to_string());

	Ref<MLPPVector> osm;
	osm.instance();
	osm->set_from_std_vector(outputSet);

	//ERR_PRINT(osm->to_string());

	Ref<MLPPKNN> knn;
	knn.instance();

	knn->set_k(7);
	knn->set_input_set(ism);
	knn->set_output_set(osm);

	PoolIntArray res = knn->model_set_test(ism);

	ERR_PRINT(String(Variant(res)));
	ERR_PRINT("ACCURACY: " + itos(100 * knn->score()) + "%");

	//(alg.transpose(inputSet), outputSet, 8);
	//alg.printVector(knn.modelSetTest(alg.transpose(inputSet)));
	//std::cout << "ACCURACY: " << 100 * knn.score() << "%" << std::endl;
}

void MLPPTests::test_convolution_tensors_etc() {
	MLPPLinAlg alg;
	MLPPLinAlg algn;
	MLPPData data;
	MLPPConvolutions conv;
	MLPPTransforms trans;

	// CONVOLUTION, POOLING, ETC..
	const real_t input_arr[] = {
		1,
	};

	Ref<MLPPMatrix> input = Ref<MLPPMatrix>(memnew(MLPPMatrix(input_arr, 1, 1)));

	Ref<MLPPTensor3> tensor_set;
	tensor_set.instance();
	tensor_set->resize(Size3i(1, 1, 0));
	tensor_set->z_slice_add_mlpp_matrix(input);
	tensor_set->z_slice_add_mlpp_matrix(input);
	tensor_set->z_slice_add_mlpp_matrix(input);

	ERR_PRINT("TODO data.rgb2xyz(tensor_set)");
	//ERR_PRINT(data.rgb2xyz(tensor_set)->to_string());

	const real_t input2_arr[] = {
		62, 55, 55, 54, 49, 48, 47, 55, //
		62, 57, 54, 52, 48, 47, 48, 53, //
		61, 60, 52, 49, 48, 47, 49, 54, //
		63, 61, 60, 60, 63, 65, 68, 65, //
		67, 67, 70, 74, 79, 85, 91, 92, //
		82, 95, 101, 106, 114, 115, 112, 117, //
		96, 111, 115, 119, 128, 128, 130, 127, //
		109, 121, 127, 133, 139, 141, 140, 133, //
	};

	Ref<MLPPMatrix> input2 = Ref<MLPPMatrix>(memnew(MLPPMatrix(input2_arr, 8, 8)));

	ERR_PRINT(trans.discrete_cosine_transform(input2)->to_string());

	ERR_PRINT(conv.convolve_2d(input2, conv.get_prewitt_vertical(), 1)->to_string()); // Can use padding
	ERR_PRINT(conv.pool_2d(input2, 4, 4, MLPPConvolutions::POOL_TYPE_MAX)->to_string()); // Can use Max, Min, or Average pooling.

	Ref<MLPPTensor3> tensor_set2;
	tensor_set2.instance();
	tensor_set2->resize(Size3i(8, 8, 0));
	tensor_set2->z_slice_add_mlpp_matrix(input2);
	tensor_set2->z_slice_add_mlpp_matrix(input2);

	ERR_PRINT(conv.global_pool_3d(tensor_set2, MLPPConvolutions::POOL_TYPE_AVERAGE)->to_string()); // Can use Max, Min, or Average global pooling.

	const real_t laplacian_arr[] = {
		1, 1, 1, //
		1, -4, 1, //
		1, 1, 1 //
	};

	Ref<MLPPMatrix> laplacian = Ref<MLPPMatrix>(memnew(MLPPMatrix(laplacian_arr, 3, 3)));

	ERR_PRINT(conv.convolve_2d(conv.gaussian_filter_2d(5, 1), laplacian, 1)->to_string());
}
void MLPPTests::test_pca_svd_eigenvalues_eigenvectors(bool ui) {
	MLPPLinAlg alg;

	const real_t input_set_arr[] = {
		1, 1, //
		1, 1 //
	};

	Ref<MLPPMatrix> input_set = Ref<MLPPMatrix>(memnew(MLPPMatrix(input_set_arr, 2, 2)));

	// eigenvalues & eigenvectors

	MLPPLinAlg::EigenResult eigen = alg.eigen(input_set);

	PLOG_MSG("== Eigen ==");

	PLOG_MSG("Eigenvectors:");
	PLOG_MSG(eigen.eigen_vectors->to_string());
	PLOG_MSG("Eigenvalues:");
	PLOG_MSG(eigen.eigen_values->to_string());

	// SVD

	PLOG_MSG("== SVD ==");

	String str_svd;

	MLPPLinAlg::SVDResult svd = alg.svd(input_set);

	str_svd += "U:\n";
	str_svd += svd.U->to_string();
	str_svd += "\nS:\n";
	str_svd += svd.S->to_string();
	str_svd += "\nVt:\n";
	str_svd += svd.Vt->to_string();
	str_svd += "\n";

	PLOG_MSG(str_svd);

	// PCA

	PLOG_MSG("== PCA ==");

	// PCA done using Jacobi's method to approximate eigenvalues and eigenvectors.
	MLPPPCA dr(input_set, 1); // 1 dimensional representation.

	String str = "\nDimensionally reduced representation:\n";
	str += dr.principal_components()->to_string();
	str += "\nSCORE: " + String::num(dr.score()) + "\n";
	PLOG_MSG(str);
}

void MLPPTests::test_nlp_and_data(bool ui) {
	MLPPLinAlg alg;
	MLPPData data;

	// NLP/DATA
	String verb_text = "I am appearing and thinking, as well as conducting.";

	data.load_default_suffixes();
	data.load_default_stop_words();

	PLOG_MSG("Stemming Example:");
	PLOG_MSG(data.stemming(verb_text));

	Vector<String> sentences = String("He is a good boy|She is a good girl|The boy and girl are good").split("|");

	PLOG_MSG("Bag of Words Example (BAG_OF_WORDS_TYPE_DEFAULT):");
	PLOG_MSG(data.bag_of_words(sentences, MLPPData::BAG_OF_WORDS_TYPE_DEFAULT)->to_string());

	PLOG_MSG("Bag of Words Example (BAG_OF_WORDS_TYPE_BINARY):");
	PLOG_MSG(data.bag_of_words(sentences, MLPPData::BAG_OF_WORDS_TYPE_BINARY)->to_string());

	PLOG_MSG("TFIDF Example:");
	PLOG_MSG(data.tfidf(sentences)->to_string());

	PLOG_MSG("Tokenization:");
	PLOG_MSG(String(Variant(data.tokenize(verb_text))));

	String text_archive = "He is a good boy. She is a good girl. The boy and girl are good.";
	Vector<String> corpus = data.split_sentences(text_archive);

	PLOG_MSG("Word2Vec (WORD_TO_VEC_TYPE_CBOW):");

	MLPPData::WordsToVecResult wtvres = data.word_to_vec(corpus, MLPPData::WORD_TO_VEC_TYPE_CBOW, 2, 2, 0.1, 10000); // Can use either CBOW or Skip-n-gram.
	PLOG_MSG(wtvres.word_embeddings->to_string());

	PLOG_MSG("Word2Vec (WORD_TO_VEC_TYPE_SKIPGRAM):");

	MLPPData::WordsToVecResult wtvres2 = data.word_to_vec(corpus, MLPPData::WORD_TO_VEC_TYPE_SKIPGRAM, 2, 2, 0.1, 10000); // Can use either CBOW or Skip-n-gram.
	PLOG_MSG(wtvres2.word_embeddings->to_string());

	Vector<String> text_archive2 = String("pizza|pizza hamburger cookie|hamburger|ramen|sushi|ramen sushi").split("|");

	PLOG_MSG("LSA:");
	PLOG_MSG(data.lsa(text_archive2, 2)->to_string());

	std::vector<std::vector<real_t>> input_set_vec = { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 } };

	Ref<MLPPMatrix> input_set = Ref<MLPPMatrix>(memnew(MLPPMatrix(input_set_vec)));

	PLOG_MSG("Feature Scaling Example:");
	PLOG_MSG(data.feature_scaling(input_set)->to_string());

	PLOG_MSG("Mean Centering Example:");
	PLOG_MSG(data.mean_centering(input_set)->to_string());

	PLOG_MSG("Mean Normalization Example:");
	PLOG_MSG(data.mean_normalization(input_set)->to_string());
}
void MLPPTests::test_outlier_finder(bool ui) {
	MLPPLinAlg alg;

	// Outlier Finder
	//std::vector<real_t> inputSet = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 23554332523523 };
	std::vector<real_t> inputSet = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 23554332 };

	Ref<MLPPVector> input_set;
	input_set.instance();
	input_set->set_from_std_vector(inputSet);

	MLPPOutlierFinder outlier_finder(2); // Any datapoint outside of 2 stds from the mean is marked as an outlier.
	PLOG_MSG(Variant(outlier_finder.model_test(input_set)));
}
void MLPPTests::test_new_math_functions() {
	/*
	MLPPLinAlg alg;
	MLPPActivationOld avn;
	MLPPData data;

	// Testing new Functions
	real_t z_s = 0.001;
	std::cout << avn.logit(z_s) << std::endl;
	std::cout << avn.logit(z_s, true) << std::endl;

	std::vector<real_t> z_v = { 0.001 };
	alg.printVector(avn.logit(z_v));
	alg.printVector(avn.logit(z_v, true));

	std::vector<std::vector<real_t>> Z_m = { { 0.001 } };
	alg.printMatrix(avn.logit(Z_m));
	alg.printMatrix(avn.logit(Z_m, true));

	std::cout << alg.trace({ { 1, 2 }, { 3, 4 } }) << std::endl;
	alg.printMatrix(alg.pinverse({ { 1, 2 }, { 3, 4 } }));
	alg.printMatrix(alg.diag({ 1, 2, 3, 4, 5 }));
	alg.printMatrix(alg.kronecker_product({ { 1, 2, 3, 4, 5 } }, { { 6, 7, 8, 9, 10 } }));
	alg.printMatrix(alg.matrixPower({ { 5, 5 }, { 5, 5 } }, 2));
	alg.printVector(alg.solve({ { 1, 1 }, { 1.5, 4.0 } }, { 2200, 5050 }));

	std::vector<std::vector<real_t>> matrixOfCubes = { { 1, 2, 64, 27 } };
	std::vector<real_t> vectorOfCubes = { 1, 2, 64, 27 };
	alg.printMatrix(alg.cbrt(matrixOfCubes));
	alg.printVector(alg.cbrt(vectorOfCubes));
	std::cout << alg.max({ { 1, 2, 3, 4, 5 }, { 6, 5, 3, 4, 1 }, { 9, 9, 9, 9, 9 } }) << std::endl;
	std::cout << alg.min({ { 1, 2, 3, 4, 5 }, { 6, 5, 3, 4, 1 }, { 9, 9, 9, 9, 9 } }) << std::endl;

	//std::vector<real_t> chicken;
	//data.getImage("../../Data/apple.jpeg", chicken);
	//alg.printVector(chicken);

	std::vector<std::vector<real_t>> P = { { 12, -51, 4 }, { 6, 167, -68 }, { -4, 24, -41 } };
	alg.printMatrix(P);

	alg.printMatrix(alg.gramSchmidtProcess(P));

	//MLPPLinAlg::QRDResult qrd_result = alg.qrd(P); // It works!
	//alg.printMatrix(qrd_result.Q);
	//alg.printMatrix(qrd_result.R);
	*/
}
void MLPPTests::test_positive_definiteness_checker() {
	/*
	//MLPPStat stat;
	MLPPLinAlg alg;
	//MLPPActivation avn;
	//MLPPCost cost;
	//MLPPData data;
	//MLPPConvolutions conv;

	// Checking positive-definiteness checker. For Cholesky Decomp.
	std::vector<std::vector<real_t>> A = {
		{ 1, -1, -1, -1 },
		{ -1, 2, 2, 2 },
		{ -1, 2, 3, 1 },
		{ -1, 2, 1, 4 }
	};

	std::cout << std::boolalpha << alg.positiveDefiniteChecker(A) << std::endl;
	MLPPLinAlg::CholeskyResult chres = alg.cholesky(A); // works.
	alg.printMatrix(chres.L);
	alg.printMatrix(chres.Lt);
	*/
}

// real_t f(real_t x){
//     return x*x*x + 2*x - 2;
// }

real_t f(real_t x) {
	return sin(x);
}

real_t f_prime(real_t x) {
	return 2 * x;
}

real_t f_prime_2var(std::vector<real_t> x) {
	return 2 * x[0] + x[1];
}
/*
	y = x^3 + 2x - 2
	y' = 3x^2 + 2
	y'' = 6x
	y''(2) = 12
*/

// real_t f_mv(std::vector<real_t> x){
//     return x[0] * x[0] + x[0] * x[1] * x[1] + x[1] + 5;
// }

/*
	Where x, y = x[0], x[1], this function is defined as:
	f(x, y) = x^2 + xy^2 + y + 5
	∂f/∂x = 2x + 2y
	∂^2f/∂x∂y = 2
*/

real_t f_mv(std::vector<real_t> x) {
	return x[0] * x[0] * x[0] + x[0] + x[1] * x[1] * x[1] * x[0] + x[2] * x[2] * x[1];
}

/*
	Where x, y = x[0], x[1], this function is defined as:
	f(x, y) = x^3 + x + xy^3 + yz^2

	fy = 3xy^2 + 2yz
	fyy = 6xy + 2z
	fyyz = 2

	∂^2f/∂y^2 = 6xy + 2z
	∂^3f/∂y^3 = 6x

	∂f/∂z = 2zy
	∂^2f/∂z^2 = 2y
	∂^3f/∂z^3 = 0

	∂f/∂x = 3x^2 + 1 + y^3
	∂^2f/∂x^2 = 6x
	∂^3f/∂x^3 = 6

	∂f/∂z = 2zy
	∂^2f/∂z^2 = 2z

	∂f/∂y = 3xy^2
	∂^2f/∂y∂x = 3y^2

*/

void MLPPTests::test_numerical_analysis() {
	/*
	MLPPLinAlg alg;
	MLPPConvolutionsOld conv;

	// Checks for numerical analysis class.
	MLPPNumericalAnalysisOld numAn;

	std::cout << numAn.quadraticApproximation(f, 0, 1) << std::endl;

	std::cout << numAn.cubicApproximation(f, 0, 1.001) << std::endl;

	std::cout << f(1.001) << std::endl;

	std::cout << numAn.quadraticApproximation(f_mv, { 0, 0, 0 }, { 1, 1, 1 }) << std::endl;

	std::cout << numAn.numDiff(&f, 1) << std::endl;
	std::cout << numAn.newtonRaphsonMethod(&f, 1, 1000) << std::endl;
	std::cout << numAn.invQuadraticInterpolation(&f, { 100, 2, 1.5 }, 10) << std::endl;

	std::cout << numAn.numDiff(&f_mv, { 1, 1 }, 1) << std::endl; // Derivative w.r.t. x.

	alg.printVector(numAn.jacobian(&f_mv, { 1, 1 }));

	std::cout << numAn.numDiff_2(&f, 2) << std::endl;

	std::cout << numAn.numDiff_3(&f, 2) << std::endl;

	std::cout << numAn.numDiff_2(&f_mv, { 2, 2, 500 }, 2, 2) << std::endl;
	std::cout << numAn.numDiff_3(&f_mv, { 2, 1000, 130 }, 0, 0, 0) << std::endl;

	alg.printTensor(numAn.thirdOrderTensor(&f_mv, { 1, 1, 1 }));
	std::cout << "Our Hessian." << std::endl;
	alg.printMatrix(numAn.hessian(&f_mv, { 2, 2, 500 }));

	std::cout << numAn.laplacian(f_mv, { 1, 1, 1 }) << std::endl;

	std::vector<std::vector<std::vector<real_t>>> tensor;
	tensor.push_back({ { 1, 2 }, { 1, 2 }, { 1, 2 } });
	tensor.push_back({ { 1, 2 }, { 1, 2 }, { 1, 2 } });

	alg.printTensor(tensor);

	alg.printMatrix(alg.tensor_vec_mult(tensor, { 1, 2 }));

	std::cout << numAn.cubicApproximation(f_mv, { 0, 0, 0 }, { 1, 1, 1 }) << std::endl;
	std::cout << numAn.eulerianMethod(f_prime, { 1, 1 }, 1.5, 0.000001) << std::endl;
	std::cout << numAn.eulerianMethod(f_prime_2var, { 2, 3 }, 2.5, 0.00000001) << std::endl;

	std::vector<std::vector<real_t>> A = {
		{ 1, 0, 0, 0 },
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 1 }
	};

	alg.printMatrix(conv.dx(A));
	alg.printMatrix(conv.dy(A));

	alg.printMatrix(conv.grad_orientation(A));

	std::vector<std::vector<std::string>> h = conv.harris_corner_detection(A);

	for (uint32_t i = 0; i < h.size(); i++) {
		for (uint32_t j = 0; j < h[i].size(); j++) {
			std::cout << h[i][j] << " ";
		}
		std::cout << std::endl;
	} // Harris detector works. Life is good!

	std::vector<real_t> a = { 3, 4, 4 };
	std::vector<real_t> b = { 4, 4, 4 };
	alg.printVector(alg.cross(a, b));
	*/
}
void MLPPTests::test_support_vector_classification_kernel(bool ui) {
	MLPPLinAlg alg;
	MLPPData data;

	//SUPPORT VECTOR CLASSIFICATION (kernel method)
	Ref<MLPPDataSimple> dt = data.load_breast_cancer_svc(_breast_cancer_svm_data_path);

	MLPPDualSVC kernelSVM(dt->get_input(), dt->get_output(), 1000);
	kernelSVM.gradient_descent(0.0001, 20, ui);
	PLOG_MSG("SCORE: " + String::num(kernelSVM.score()));

	/*
	std::vector<std::vector<real_t>> linearlyIndependentMat = {
		{ 1, 2, 3, 4 },
		{ 2345384, 4444, 6111, 55 }
	};

	std::cout << "True of false: linearly independent?: " << std::boolalpha << alg.linearIndependenceChecker(linearlyIndependentMat) << std::endl;
	*/
}

void MLPPTests::test_mlpp_vector() {
	std::vector<real_t> a = { 4, 3, 1, 3 };

	Ref<MLPPVector> rv;
	rv.instance();
	rv->set_from_std_vector(a);

	Ref<MLPPVector> rv2;
	rv2.instance();
	rv2->set_from_std_vector(a);

	is_approx_equals_vec(rv, rv2, "set_from_std_vectors test.");

	rv2->set_from_std_vector(a);

	is_approx_equals_vec(rv, rv2, "re-set_from_std_vectors test.");
}

void MLPPTests::is_approx_equalsd(real_t a, real_t b, const String &str) {
	if (!Math::is_equal_approx(a, b)) {
		PLOG_ERR("TEST FAILED: " + str + " Got: " + String::num(a) + " Should be: " + String::num(b));
	} else {
		PLOG_TRACE("TEST PASSED: " + str);
	}
}

void MLPPTests::is_approx_equals_dvec(const Vector<real_t> &a, const Vector<real_t> &b, const String &str) {
	if (a.size() != b.size()) {
		goto IAEDVEC_FAILED;
	}

	for (int i = 0; i < a.size(); ++i) {
		if (!Math::is_equal_approx(a[i], b[i])) {
			goto IAEDVEC_FAILED;
		}
	}

	PLOG_TRACE("TEST PASSED: " + str);

	return;

IAEDVEC_FAILED:

	String fail_str = "TEST FAILED: ";
	fail_str += str;
	fail_str += " Got: [ ";

	for (int i = 0; i < a.size(); ++i) {
		fail_str += String::num(a[i]);
		fail_str += " ";
	}

	fail_str += "] Should be: [ ";

	for (int i = 0; i < b.size(); ++i) {
		fail_str += String::num(b[i]);
		fail_str += " ";
	}

	fail_str += "].";

	PLOG_ERR(fail_str);
}

String vmat_to_str(const Vector<Vector<real_t>> &a) {
	String str;

	str += "[ \n";

	for (int i = 0; i < a.size(); ++i) {
		str += "  [ ";

		const Vector<real_t> &aa = a[i];

		for (int j = 0; j < aa.size(); ++j) {
			str += String::num(aa[j]);
			str += " ";
		}

		str += "]\n";
	}

	str += "]\n";

	return str;
}

void MLPPTests::is_approx_equals_dmat(const Vector<Vector<real_t>> &a, const Vector<Vector<real_t>> &b, const String &str) {
	if (a.size() != b.size()) {
		goto IAEDMAT_FAILED;
	}

	for (int i = 0; i < a.size(); ++i) {
		const Vector<real_t> &aa = a[i];
		const Vector<real_t> &bb = b[i];

		if (aa.size() != bb.size()) {
			goto IAEDMAT_FAILED;
		}

		for (int j = 0; j < aa.size(); ++j) {
			if (!Math::is_equal_approx(aa[j], bb[j])) {
				goto IAEDMAT_FAILED;
			}
		}
	}

	PLOG_TRACE("TEST PASSED: " + str);

	return;

IAEDMAT_FAILED:

	String fail_str = "TEST FAILED: ";
	fail_str += str;
	fail_str += "\nGot:\n";
	fail_str += vmat_to_str(a);
	fail_str += "Should be:\n";
	fail_str += vmat_to_str(b);

	PLOG_ERR(fail_str);
}

void MLPPTests::is_approx_equals_mat(Ref<MLPPMatrix> a, Ref<MLPPMatrix> b, const String &str) {
	ERR_FAIL_COND(!a.is_valid());
	ERR_FAIL_COND(!b.is_valid());

	int ds = a->data_size();

	const real_t *aa = a->ptr();
	const real_t *bb = b->ptr();

	if (a->size() != b->size()) {
		goto IAEMAT_FAILED;
	}

	ERR_FAIL_COND(!aa);
	ERR_FAIL_COND(!bb);

	for (int i = 0; i < ds; ++i) {
		if (!Math::is_equal_approx(aa[i], bb[i])) {
			goto IAEMAT_FAILED;
		}
	}

	PLOG_TRACE("TEST PASSED: " + str);

	return;

IAEMAT_FAILED:

	String fail_str = "TEST FAILED: ";
	fail_str += str;
	fail_str += "\nGot:\n";
	fail_str += a->to_string();
	fail_str += "\nShould be:\n";
	fail_str += b->to_string();

	PLOG_ERR(fail_str);
}
void MLPPTests::is_approx_equals_vec(Ref<MLPPVector> a, Ref<MLPPVector> b, const String &str) {
	ERR_FAIL_COND(!a.is_valid());
	ERR_FAIL_COND(!b.is_valid());

	if (a->size() != b->size()) {
		goto IAEDVEC_FAILED;
	}

	for (int i = 0; i < a->size(); ++i) {
		if (!Math::is_equal_approx(a->element_get(i), b->element_get(i))) {
			goto IAEDVEC_FAILED;
		}
	}

	PLOG_TRACE("TEST PASSED: " + str);

	return;

IAEDVEC_FAILED:

	String fail_str = "TEST FAILED: ";
	fail_str += str;
	fail_str += "\nGot:\n";
	fail_str += a->to_string();
	fail_str += "\nShould be:\n";
	fail_str += b->to_string();
	fail_str += "\n.";

	PLOG_ERR(fail_str);
}

MLPPTests::MLPPTests() {
	_breast_cancer_data_path = "res://datasets/BreastCancer.csv";
	_breast_cancer_svm_data_path = "res://datasets/BreastCancerSVM.csv";
	_california_housing_data_path = "res://datasets/CaliforniaHousing.csv";
	_fires_and_crime_data_path = "res://datasets/FiresAndCrime.csv";
	_iris_data_path = "res://datasets/Iris.csv";
	_mnist_test_data_path = "res://datasets/MnistTest.csv";
	_mnist_train_data_path = "res://datasets/MnistTrain.csv";
	_wine_data_path = "res://datasets/Wine.csv";
}

MLPPTests::~MLPPTests() {
}

void MLPPTests::_bind_methods() {
	ClassDB::bind_method(D_METHOD("test_statistics"), &MLPPTests::test_statistics);
	ClassDB::bind_method(D_METHOD("test_linear_algebra"), &MLPPTests::test_linear_algebra);
	ClassDB::bind_method(D_METHOD("test_univariate_linear_regression"), &MLPPTests::test_univariate_linear_regression);

	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_gradient_descent", "ui"), &MLPPTests::test_multivariate_linear_regression_gradient_descent, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_sgd", "ui"), &MLPPTests::test_multivariate_linear_regression_sgd, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_mbgd", "ui"), &MLPPTests::test_multivariate_linear_regression_mbgd, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_normal_equation", "ui"), &MLPPTests::test_multivariate_linear_regression_normal_equation, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_adam"), &MLPPTests::test_multivariate_linear_regression_adam, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_score_sgd_adam", "ui"), &MLPPTests::test_multivariate_linear_regression_score_sgd_adam, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_epochs_gradient_descent", "ui"), &MLPPTests::test_multivariate_linear_regression_epochs_gradient_descent, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_newton_raphson", "ui"), &MLPPTests::test_multivariate_linear_regression_newton_raphson, false);

	ClassDB::bind_method(D_METHOD("test_logistic_regression", "ui"), &MLPPTests::test_logistic_regression, false);
	ClassDB::bind_method(D_METHOD("test_probit_regression", "ui"), &MLPPTests::test_probit_regression, false);
	ClassDB::bind_method(D_METHOD("test_c_log_log_regression", "ui"), &MLPPTests::test_c_log_log_regression, false);
	ClassDB::bind_method(D_METHOD("test_exp_reg_regression", "ui"), &MLPPTests::test_exp_reg_regression, false);
	ClassDB::bind_method(D_METHOD("test_tanh_regression", "ui"), &MLPPTests::test_tanh_regression, false);
	ClassDB::bind_method(D_METHOD("test_softmax_regression", "ui"), &MLPPTests::test_softmax_regression, false);
	ClassDB::bind_method(D_METHOD("test_support_vector_classification", "ui"), &MLPPTests::test_support_vector_classification, false);

	ClassDB::bind_method(D_METHOD("test_mlp", "ui"), &MLPPTests::test_mlp, false);
	ClassDB::bind_method(D_METHOD("test_soft_max_network", "ui"), &MLPPTests::test_soft_max_network, false);
	ClassDB::bind_method(D_METHOD("test_autoencoder", "ui"), &MLPPTests::test_autoencoder, false);
	ClassDB::bind_method(D_METHOD("test_dynamically_sized_ann", "ui"), &MLPPTests::test_dynamically_sized_ann, false);
	ClassDB::bind_method(D_METHOD("test_wgan_old", "ui"), &MLPPTests::test_wgan_old, false);
	ClassDB::bind_method(D_METHOD("test_wgan", "ui"), &MLPPTests::test_wgan, false);
	ClassDB::bind_method(D_METHOD("test_ann", "ui"), &MLPPTests::test_ann, false);
	ClassDB::bind_method(D_METHOD("test_dynamically_sized_mann", "ui"), &MLPPTests::test_dynamically_sized_mann, false);
	ClassDB::bind_method(D_METHOD("test_train_test_split_mann", "ui"), &MLPPTests::test_train_test_split_mann, false);

	ClassDB::bind_method(D_METHOD("test_naive_bayes"), &MLPPTests::test_naive_bayes);
	ClassDB::bind_method(D_METHOD("test_k_means", "ui"), &MLPPTests::test_k_means, false);
	ClassDB::bind_method(D_METHOD("test_knn", "ui"), &MLPPTests::test_knn, false);

	ClassDB::bind_method(D_METHOD("test_convolution_tensors_etc"), &MLPPTests::test_convolution_tensors_etc);
	ClassDB::bind_method(D_METHOD("test_pca_svd_eigenvalues_eigenvectors", "ui"), &MLPPTests::test_pca_svd_eigenvalues_eigenvectors, false);

	ClassDB::bind_method(D_METHOD("test_nlp_and_data", "ui"), &MLPPTests::test_nlp_and_data, false);
	ClassDB::bind_method(D_METHOD("test_outlier_finder", "ui"), &MLPPTests::test_outlier_finder, false);

	ClassDB::bind_method(D_METHOD("test_new_math_functions"), &MLPPTests::test_new_math_functions);
	ClassDB::bind_method(D_METHOD("test_positive_definiteness_checker"), &MLPPTests::test_positive_definiteness_checker);
	ClassDB::bind_method(D_METHOD("test_numerical_analysis"), &MLPPTests::test_numerical_analysis);

	ClassDB::bind_method(D_METHOD("test_support_vector_classification_kernel", "ui"), &MLPPTests::test_support_vector_classification_kernel, false);

	ClassDB::bind_method(D_METHOD("test_mlpp_vector"), &MLPPTests::test_mlpp_vector);
}
