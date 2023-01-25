
#include "mlpp_tests.h"

#include "core/math/math_funcs.h"

//TODO remove
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

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

Vector<double> dstd_vec_to_vec(const std::vector<double> &in) {
	Vector<double> r;

	r.resize(static_cast<int>(in.size()));
	double *darr = r.ptrw();

	for (uint32_t i = 0; i < in.size(); ++i) {
		darr[i] = in[i];
	}

	return r;
}

Vector<Vector<double>> dstd_mat_to_mat(const std::vector<std::vector<double>> &in) {
	Vector<Vector<double>> r;

	for (uint32_t i = 0; i < in.size(); ++i) {
		r.push_back(dstd_vec_to_vec(in[i]));
	}

	return r;
}

void MLPPTests::test_statistics() {
	ERR_PRINT("MLPPTests::test_statistics() Started!");

	MLPPStat stat;
	MLPPConvolutions conv;

	// STATISTICS
	std::vector<double> x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	std::vector<double> y = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
	std::vector<double> w = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };

	is_approx_equalsd(stat.mean(x), 5.5, "Arithmetic Mean");
	is_approx_equalsd(stat.mean(x), 5.5, "Median");

	is_approx_equals_dvec(dstd_vec_to_vec(stat.mode(x)), dstd_vec_to_vec(x), "stat.mode(x)");

	is_approx_equalsd(stat.range(x), 9, "Range");
	is_approx_equalsd(stat.midrange(x), 4.5, "Midrange");
	is_approx_equalsd(stat.absAvgDeviation(x), 2.5, "Absolute Average Deviation");
	is_approx_equalsd(stat.standardDeviation(x), 3.02765, "Standard Deviation");
	is_approx_equalsd(stat.variance(x), 9.16667, "Variance");
	is_approx_equalsd(stat.covariance(x, y), -9.16667, "Covariance");
	is_approx_equalsd(stat.correlation(x, y), -1, "Correlation");
	is_approx_equalsd(stat.R2(x, y), 1, "R^2");

	// Returns 1 - (1/k^2)
	is_approx_equalsd(stat.chebyshevIneq(2), 0.75, "Chebyshev Inequality");
	is_approx_equalsd(stat.weightedMean(x, w), 5.5, "Weighted Mean");
	is_approx_equalsd(stat.geometricMean(x), 4.52873, "Geometric Mean");
	is_approx_equalsd(stat.harmonicMean(x), 3.41417, "Harmonic Mean");
	is_approx_equalsd(stat.RMS(x), 6.20484, "Root Mean Square (Quadratic mean)");
	is_approx_equalsd(stat.powerMean(x, 5), 7.39281, "Power Mean (p = 5)");
	is_approx_equalsd(stat.lehmerMean(x, 5), 8.71689, "Lehmer Mean (p = 5)");
	is_approx_equalsd(stat.weightedLehmerMean(x, w, 5), 8.71689, "Weighted Lehmer Mean (p = 5)");
	is_approx_equalsd(stat.contraHarmonicMean(x), 7, "Contraharmonic Mean");
	is_approx_equalsd(stat.heronianMean(1, 10), 4.72076, "Hernonian Mean");
	is_approx_equalsd(stat.heinzMean(1, 10, 1), 5.5, "Heinz Mean (x = 1)");
	is_approx_equalsd(stat.neumanSandorMean(1, 10), 3.36061, "Neuman-Sandor Mean");
	is_approx_equalsd(stat.stolarskyMean(1, 10, 5), 6.86587, "Stolarsky Mean (p = 5)");
	is_approx_equalsd(stat.identricMean(1, 10), 4.75135, "Identric Mean");
	is_approx_equalsd(stat.logMean(1, 10), 3.90865, "Logarithmic Mean");
	is_approx_equalsd(stat.absAvgDeviation(x), 2.5, "Absolute Average Deviation");

	ERR_PRINT("MLPPTests::test_statistics() Finished!");
}

void MLPPTests::test_linear_algebra() {
	MLPPLinAlg alg;

	std::vector<std::vector<double>> square = { { 1, 1 }, { -1, 1 }, { 1, -1 }, { -1, -1 } };
	std::vector<std::vector<double>> square_rot_res = { { 1.41421, 1.11022e-16 }, { -1.11022e-16, 1.41421 }, { 1.11022e-16, -1.41421 }, { -1.41421, -1.11022e-16 } };

	is_approx_equals_dmat(dstd_mat_to_mat(alg.rotate(square, M_PI / 4)), dstd_mat_to_mat(square_rot_res), "alg.rotate(square, M_PI / 4)");

	std::vector<std::vector<double>> A = {
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
	};
	std::vector<double> a = { 4, 3, 1, 3 };
	std::vector<double> b = { 3, 5, 6, 1 };

	std::vector<std::vector<double>> mmtr_res = {
		{ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 },
		{ 4, 8, 12, 16, 20, 24, 28, 32, 36, 40 },
		{ 6, 12, 18, 24, 30, 36, 42, 48, 54, 60 },
		{ 8, 16, 24, 32, 40, 48, 56, 64, 72, 80 },
		{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 },
		{ 12, 24, 36, 48, 60, 72, 84, 96, 108, 120 },
		{ 14, 28, 42, 56, 70, 84, 98, 112, 126, 140 },
		{ 16, 32, 48, 64, 80, 96, 112, 128, 144, 160 },
		{ 18, 36, 54, 72, 90, 108, 126, 144, 162, 180 },
		{ 20, 40, 60, 80, 100, 120, 140, 160, 180, 200 }
	};

	is_approx_equals_dmat(dstd_mat_to_mat(alg.matmult(alg.transpose(A), A)), dstd_mat_to_mat(mmtr_res), "alg.matmult(alg.transpose(A), A)");

	is_approx_equalsd(alg.dot(a, b), 36, "alg.dot(a, b)");

	std::vector<std::vector<double>> had_prod_res = {
		{ 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 },
		{ 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 }
	};

	is_approx_equals_dmat(dstd_mat_to_mat(alg.hadamard_product(A, A)), dstd_mat_to_mat(had_prod_res), "alg.hadamard_product(A, A)");

	std::vector<std::vector<double>> id_10_res = {
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
	};

	is_approx_equals_dmat(dstd_mat_to_mat(alg.identity(10)), dstd_mat_to_mat(id_10_res), "alg.identity(10)");
}

void MLPPTests::test_univariate_linear_regression() {
	// Univariate, simple linear regression, case where k = 1
	MLPPData data;

	Ref<MLPPDataESimple> ds = data.load_fires_and_crime(_load_fires_and_crime_data_path);

	MLPPUniLinReg model(ds->input, ds->output);

	std::vector<double> slr_res = {
		24.1095, 28.4829, 29.8082, 26.0974, 27.2902, 61.0851, 30.4709, 25.0372, 25.5673, 35.9046,
		54.4587, 18.8083, 23.4468, 18.5432, 19.2059, 21.1938, 23.0492, 18.8083, 25.4348, 35.9046,
		37.76, 40.278, 63.8683, 68.5068, 40.4106, 46.772, 32.0612, 23.3143, 44.784, 44.519,
		27.8203, 20.6637, 22.5191, 53.796, 38.9527, 30.8685, 20.3986
	};

	is_approx_equals_dvec(dstd_vec_to_vec(model.modelSetTest(ds->input)), dstd_vec_to_vec(slr_res), "stat.mode(x)");
}

void MLPPTests::test_multivariate_linear_regression_gradient_descent(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	MLPPLinReg model(ds->input, ds->output); // Can use Lasso, Ridge, ElasticNet Reg

	model.gradientDescent(0.001, 30, ui);
	alg.printVector(model.modelSetTest(ds->input));
}

void MLPPTests::test_multivariate_linear_regression_sgd(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	MLPPLinReg model(ds->input, ds->output); // Can use Lasso, Ridge, ElasticNet Reg

	model.SGD(0.00000001, 300000, ui);
	alg.printVector(model.modelSetTest(ds->input));
}

void MLPPTests::test_multivariate_linear_regression_mbgd(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	MLPPLinReg model(ds->input, ds->output); // Can use Lasso, Ridge, ElasticNet Reg

	model.MBGD(0.001, 10000, 2, ui);
	alg.printVector(model.modelSetTest(ds->input));
}

void MLPPTests::test_multivariate_linear_regression_normal_equation(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	MLPPLinReg model(ds->input, ds->output); // Can use Lasso, Ridge, ElasticNet Reg

	model.normalEquation();
	alg.printVector(model.modelSetTest(ds->input));
}

void MLPPTests::test_multivariate_linear_regression_adam() {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	MLPPLinReg adamModel(alg.transpose(ds->input), ds->output);
	alg.printVector(adamModel.modelSetTest(ds->input));
	std::cout << "ACCURACY: " << 100 * adamModel.score() << "%" << std::endl;
}

void MLPPTests::test_multivariate_linear_regression_score_sgd_adam(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	const int TRIAL_NUM = 1000;

	double scoreSGD = 0;
	double scoreADAM = 0;
	for (int i = 0; i < TRIAL_NUM; i++) {
		MLPPLinReg modelf(alg.transpose(ds->input), ds->output);
		modelf.MBGD(0.001, 5, 1, ui);
		scoreSGD += modelf.score();

		MLPPLinReg adamModelf(alg.transpose(ds->input), ds->output);
		adamModelf.Adam(0.1, 5, 1, 0.9, 0.999, 1e-8, ui); // Change batch size = sgd, bgd
		scoreADAM += adamModelf.score();
	}

	std::cout << "ACCURACY, AVG, SGD: " << 100 * scoreSGD / TRIAL_NUM << "%" << std::endl;
	std::cout << std::endl;
	std::cout << "ACCURACY, AVG, ADAM: " << 100 * scoreADAM / TRIAL_NUM << "%" << std::endl;
}

void MLPPTests::test_multivariate_linear_regression_epochs_gradient_descent(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	std::cout << "Total epoch num: 300" << std::endl;
	std::cout << "Method: 1st Order w/ Jacobians" << std::endl;

	MLPPLinReg model3(alg.transpose(ds->input), ds->output); // Can use Lasso, Ridge, ElasticNet Reg
	model3.gradientDescent(0.001, 300, ui);
	alg.printVector(model3.modelSetTest(ds->input));
}

void MLPPTests::test_multivariate_linear_regression_newton_raphson(bool ui) {
	MLPPData data;
	MLPPLinAlg alg;

	Ref<MLPPDataSimple> ds = data.load_california_housing(_load_california_housing_data_path);

	std::cout << "--------------------------------------------" << std::endl;
	std::cout << "Total epoch num: 300" << std::endl;
	std::cout << "Method: Newtonian 2nd Order w/ Hessians" << std::endl;
	MLPPLinReg model2(alg.transpose(ds->input), ds->output);

	model2.NewtonRaphson(1.5, 300, ui);
	alg.printVector(model2.modelSetTest(ds->input));
}

void MLPPTests::test_logistic_regression(bool ui) {
	//MLPPStat stat;
	// MLPPLinAlg alg;
	//MLPPActivation avn;
	// MLPPCost cost;
	// MLPPData data;
	// MLPPConvolutions conv;

	// // LOGISTIC REGRESSION
	// auto [inputSet, outputSet] = data.load rastCancer();
	// LogReg model(inputSet, outputSet);
	// model.SGD(0.001, 100000, 0);
	// alg.printVector(model.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_probit_regression(bool ui) {
	// // PROBIT REGRESSION
	// std::vector<std::vector<double>> inputSet;
	// std::vector<double> outputSet;
	// data.setData(30, "/Users/marcmelikyan/Desktop/Data/BreastCancer.csv", inputSet, outputSet);
	// ProbitReg model(inputSet, outputSet);
	// model.SGD(0.001, 10000, 1);
	// alg.printVector(model.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_c_log_log_regression(bool ui) {
	// // CLOGLOG REGRESSION
	// std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8}, {0,0,0,0,1,1,1,1}};
	// std::vector<double> outputSet = {0,0,0,0,1,1,1,1};
	// CLogLogReg model(alg.transpose(inputSet), outputSet);
	// model.SGD(0.1, 10000, 0);
	// alg.printVector(model.modelSetTest(alg.transpose(inputSet)));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_exp_reg_regression(bool ui) {
	// // EXPREG REGRESSION
	// std::vector<std::vector<double>> inputSet = {{0,1,2,3,4}};
	// std::vector<double> outputSet = {1,2,4,8,16};
	// ExpReg model(alg.transpose(inputSet), outputSet);
	// model.SGD(0.001, 10000, 0);
	// alg.printVector(model.modelSetTest(alg.transpose(inputSet)));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_tanh_regression(bool ui) {
	// // TANH REGRESSION
	// std::vector<std::vector<double>> inputSet = {{4,3,0,-3,-4}, {0,0,0,1,1}};
	// std::vector<double> outputSet = {1,1,0,-1,-1};
	// TanhReg model(alg.transpose(inputSet), outputSet);
	// model.SGD(0.1, 10000, 0);
	// alg.printVector(model.modelSetTest(alg.transpose(inputSet)));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_softmax_regression(bool ui) {
	// // SOFTMAX REGRESSION
	// auto [inputSet, outputSet] = data.loadIris();
	// SoftmaxReg model(inputSet, outputSet);
	// model.SGD(0.1, 10000, 1);
	// alg.printMatrix(model.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_support_vector_classification(bool ui) {
	// // SUPPORT VECTOR CLASSIFICATION
	// auto [inputSet, outputSet] = data.loadBreastCancerSVC();
	// SVC model(inputSet, outputSet, 1);
	// model.SGD(0.00001, 100000, 1);
	// alg.printVector(model.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

	// SoftmaxReg model(inputSet, outputSet);
	// model.SGD(0.001, 20000, 0);
	// alg.printMatrix(model.modelSetTest(inputSet));
}

void MLPPTests::test_mlp(bool ui) {
	// // MLP
	// std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
	// inputSet = alg.transpose(inputSet);
	// std::vector<double> outputSet = {0,1,1,0};

	// MLP model(inputSet, outputSet, 2);
	// model.gradientDescent(0.1, 10000, 0);
	// alg.printVector(model.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_soft_max_network(bool ui) {
	// // SOFTMAX NETWORK
	// auto [inputSet, outputSet] = data.loadWine();
	// SoftmaxNet model(inputSet, outputSet, 1);
	// model.gradientDescent(0.01, 100000, 1);
	// alg.printMatrix(model.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_autoencoder(bool ui) {
	// // AUTOENCODER
	// std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8,9,10}, {3,5,9,12,15,18,21,24,27,30}};
	// AutoEncoder model(alg.transpose(inputSet), 5);
	// model.SGD(0.001, 300000, 0);
	// alg.printMatrix(model.modelSetTest(alg.transpose(inputSet)));
	// std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTests::test_dynamically_sized_ann(bool ui) {
	// DYNAMICALLY SIZED ANN
	// Possible Weight Init Methods: Default, Uniform, HeNormal, HeUniform, XavierNormal, XavierUniform
	// Possible Activations: Linear, Sigmoid, Swish, Softplus, Softsign, CLogLog, Ar{Sinh, Cosh, Tanh, Csch, Sech, Coth},  GaussianCDF, GELU, UnitStep
	// Possible Loss Functions: MSE, RMSE, MBE, LogLoss, CrossEntropy, HingeLoss
	// std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
	// std::vector<double> outputSet = {0,1,1,0};
	// ANN ann(alg.transpose(inputSet), outputSet);
	// ann.addLayer(2, "Cosh");
	// ann.addOutputLayer("Sigmoid", "LogLoss");

	// ann.AMSGrad(0.1, 10000, 1, 0.9, 0.999, 0.000001, 1);
	// ann.Adadelta(1, 1000, 2, 0.9, 0.000001, 1);
	// ann.Momentum(0.1, 8000, 2, 0.9, true, 1);

	//ann.setLearningRateScheduler("Step", 0.5, 1000);
	// ann.gradientDescent(0.01, 30000);
	// alg.printVector(ann.modelSetTest(alg.transpose(inputSet)));
	// std::cout << "ACCURACY: " << 100 * ann.score() << "%" << std::endl;
}
void MLPPTests::test_wgan(bool ui) {
	/*
		std::vector<std::vector<double>> outputSet = {{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},
											   {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40}};

	WGAN gan(2, alg.transpose(outputSet)); // our gan is a wasserstein gan (wgan)
	gan.addLayer(5, "Sigmoid");
	gan.addLayer(2, "RELU");
	gan.addLayer(5, "Sigmoid");
	gan.addOutputLayer(); // User can specify weight init- if necessary.
	gan.gradientDescent(0.1, 55000, 0);
	std::cout << "GENERATED INPUT: (Gaussian-sampled noise):" << std::endl;
	alg.printMatrix(gan.generateExample(100));
	*/
}
void MLPPTests::test_ann(bool ui) {
	// typedef std::vector<std::vector<double>> Matrix;
	// typedef std::vector<double> Vector;

	// Matrix inputSet = {{0,0}, {0,1}, {1,0}, {1,1}}; // XOR
	// Vector outputSet = {0,1,1,0};

	// ANN ann(inputSet, outputSet);
	// ann.addLayer(5, "Sigmoid");
	// ann.addLayer(8, "Sigmoid"); // Add more layers as needed.
	// ann.addOutputLayer("Sigmoid", "LogLoss");
	// ann.gradientDescent(1, 20000, 1);

	// Vector predictions = ann.modelSetTest(inputSet);
	// alg.printVector(predictions); // Testing out the model's preds for train set.
	// std::cout << "ACCURACY: " << 100 * ann.score() << "%" << std::endl; // Accuracy.
}
void MLPPTests::test_dynamically_sized_mann(bool ui) {
	// // DYNAMICALLY SIZED MANN (Multidimensional Output ANN)
	// std::vector<std::vector<double>> inputSet = {{1,2,3},{2,4,6},{3,6,9},{4,8,12}};
	// std::vector<std::vector<double>> outputSet = {{1,5}, {2,10}, {3,15}, {4,20}};

	// MANN mann(inputSet, outputSet);
	// mann.addOutputLayer("Linear", "MSE");
	// mann.gradientDescent(0.001, 80000, 0);
	// alg.printMatrix(mann.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * mann.score() << "%" << std::endl;

	// std::vector<std::vector<double>> inputSet;
	// std::vector<double> tempOutputSet;
	// data.setData(4, "/Users/marcmelikyan/Desktop/Data/Iris.csv", inputSet, tempOutputSet);
	// std::vector<std::vector<double>> outputSet = data.oneHotRep(tempOutputSet, 3);
}
void MLPPTests::test_train_test_split_mann(bool ui) {
	// TRAIN TEST SPLIT CHECK
	// std::vector<std::vector<double>> inputSet1 = {{1,2,3,4,5,6,7,8,9,10}, {3,5,9,12,15,18,21,24,27,30}};
	// std::vector<std::vector<double>> outputSet1 = {{2,4,6,8,10,12,14,16,18,20}};
	// auto [inputSet, outputSet, inputTestSet, outputTestSet] = data.trainTestSplit(alg.transpose(inputSet1), alg.transpose(outputSet1), 0.2);
	// alg.printMatrix(inputSet);
	// alg.printMatrix(outputSet);
	// alg.printMatrix(inputTestSet);
	// alg.printMatrix(outputTestSet);

	// alg.printMatrix(inputSet);
	// alg.printMatrix(outputSet);

	// MANN mann(inputSet, outputSet);
	// mann.addLayer(100, "RELU", "XavierNormal");
	// mann.addOutputLayer("Softmax", "CrossEntropy", "XavierNormal");
	// mann.gradientDescent(0.1, 80000, 1);
	// alg.printMatrix(mann.modelSetTest(inputSet));
	// std::cout << "ACCURACY: " << 100 * mann.score() << "%" << std::endl;
}

void MLPPTests::test_naive_bayes(bool ui) {
	// // NAIVE BAYES
	// std::vector<std::vector<double>> inputSet = {{1,1,1,1,1}, {0,0,1,1,1}, {0,0,1,0,1}};
	// std::vector<double> outputSet = {0,1,0,1,1};

	// MultinomialNB MNB(alg.transpose(inputSet), outputSet, 2);
	// alg.printVector(MNB.modelSetTest(alg.transpose(inputSet)));

	// BernoulliNB BNB(alg.transpose(inputSet), outputSet);
	// alg.printVector(BNB.modelSetTest(alg.transpose(inputSet)));

	// GaussianNB GNB(alg.transpose(inputSet), outputSet, 2);
	// alg.printVector(GNB.modelSetTest(alg.transpose(inputSet)));
}
void MLPPTests::test_k_means(bool ui) {
	// // KMeans
	// std::vector<std::vector<double>> inputSet = {{32, 0, 7}, {2, 28, 17}, {0, 9, 23}};
	// KMeans kmeans(inputSet, 3, "KMeans++");
	// kmeans.train(3, 1);
	// std::cout << std::endl;
	// alg.printMatrix(kmeans.modelSetTest(inputSet)); // Returns the assigned centroids to each of the respective training examples
	// std::cout << std::endl;
	// alg.printVector(kmeans.silhouette_scores());
}
void MLPPTests::test_knn(bool ui) {
	// // kNN
	// std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8}, {0,0,0,0,1,1,1,1}};
	// std::vector<double> outputSet = {0,0,0,0,1,1,1,1};
	// kNN knn(alg.transpose(inputSet), outputSet, 8);
	// alg.printVector(knn.modelSetTest(alg.transpose(inputSet)));
	// std::cout << "ACCURACY: " << 100 * knn.score() << "%" << std::endl;
}

void MLPPTests::test_convolution_tensors_etc() {
	// // CONVOLUTION, POOLING, ETC..
	// std::vector<std::vector<double>> input = {
	//     {1},
	// };

	// std::vector<std::vector<std::vector<double>>> tensorSet;
	// tensorSet.push_back(input);
	// tensorSet.push_back(input);
	// tensorSet.push_back(input);

	// alg.printTensor(data.rgb2xyz(tensorSet));

	// std::vector<std::vector<double>> input = {
	//     {62,55,55,54,49,48,47,55},
	//     {62,57,54,52,48,47,48,53},
	//     {61,60,52,49,48,47,49,54},
	//     {63,61,60,60,63,65,68,65},
	//     {67,67,70,74,79,85,91,92},
	//     {82,95,101,106,114,115,112,117},
	//     {96,111,115,119,128,128,130,127},
	//     {109,121,127,133,139,141,140,133},
	// };

	// Transforms trans;

	// alg.printMatrix(trans.discreteCosineTransform(input));

	// alg.printMatrix(conv.convolve(input, conv.getPrewittVertical(), 1)); // Can use padding
	// alg.printMatrix(conv.pool(input, 4, 4, "Max")); // Can use Max, Min, or Average pooling.

	// std::vector<std::vector<std::vector<double>>> tensorSet;
	// tensorSet.push_back(input);
	// tensorSet.push_back(input);
	// alg.printVector(conv.globalPool(tensorSet, "Average")); // Can use Max, Min, or Average global pooling.

	// std::vector<std::vector<double>> laplacian = {{1, 1, 1}, {1, -4, 1}, {1, 1, 1}};
	// alg.printMatrix(conv.convolve(conv.gaussianFilter2D(5, 1), laplacian, 1));
}
void MLPPTests::test_pca_svd_eigenvalues_eigenvectors(bool ui) {
	// // PCA, SVD, eigenvalues & eigenvectors
	// std::vector<std::vector<double>> inputSet = {{1,1}, {1,1}};
	// auto [Eigenvectors, Eigenvalues] = alg.eig(inputSet);
	// std::cout << "Eigenvectors:" << std::endl;
	// alg.printMatrix(Eigenvectors);
	// std::cout << std::endl;
	// std::cout << "Eigenvalues:" << std::endl;
	// alg.printMatrix(Eigenvalues);

	// auto [U, S, Vt] = alg.SVD(inputSet);

	// // PCA done using Jacobi's method to approximate eigenvalues and eigenvectors.
	// PCA dr(inputSet, 1); // 1 dimensional representation.
	// std::cout << std::endl;
	// std::cout << "Dimensionally reduced representation:" << std::endl;
	// alg.printMatrix(dr.principalComponents());
	// std::cout << "SCORE: " << dr.score() << std::endl;
}

void MLPPTests::test_nlp_and_data(bool ui) {
	// // NLP/DATA
	// std::string verbText = "I am appearing and thinking, as well as conducting.";
	// std::cout << "Stemming Example:" << std::endl;
	// std::cout << data.stemming(verbText) << std::endl;
	// std::cout << std::endl;

	// std::vector<std::string> sentences = {"He is a good boy", "She is a good girl", "The boy and girl are good"};
	// std::cout << "Bag of Words Example:" << std::endl;
	// alg.printMatrix(data.BOW(sentences, "Default"));
	// std::cout << std::endl;
	// std::cout << "TFIDF Example:" << std::endl;
	// alg.printMatrix(data.TFIDF(sentences));
	// std::cout << std::endl;

	// std::cout << "Tokenization:" << std::endl;
	// alg.printVector(data.tokenize(verbText));
	// std::cout << std::endl;

	// std::cout << "Word2Vec:" << std::endl;
	// std::string textArchive = {"He is a good boy. She is a good girl. The boy and girl are good."};
	// std::vector<std::string> corpus = data.splitSentences(textArchive);
	// auto [wordEmbeddings, wordList] = data.word2Vec(corpus, "CBOW", 2, 2, 0.1, 10000); // Can use either CBOW or Skip-n-gram.
	// alg.printMatrix(wordEmbeddings);
	// std::cout << std::endl;

	// std::vector<std::string> textArchive = {"pizza", "pizza hamburger cookie", "hamburger", "ramen", "sushi", "ramen sushi"};

	// alg.printMatrix(data.LSA(textArchive, 2));
	// //alg.printMatrix(data.BOW(textArchive, "Default"));
	// std::cout << std::endl;

	// std::vector<std::vector<double>> inputSet = {{1,2},{2,3},{3,4},{4,5},{5,6}};
	// std::cout << "Feature Scaling Example:" << std::endl;
	// alg.printMatrix(data.featureScaling(inputSet));
	// std::cout << std::endl;

	// std::cout << "Mean Centering Example:" << std::endl;
	// alg.printMatrix(data.meanCentering(inputSet));
	// std::cout << std::endl;

	// std::cout << "Mean Normalization Example:" << std::endl;
	// alg.printMatrix(data.meanNormalization(inputSet));
	// std::cout << std::endl;
}
void MLPPTests::test_outlier_finder(bool ui) {
	// // Outlier Finder
	// std::vector<double> inputSet = {1,2,3,4,5,6,7,8,9,23554332523523};
	// OutlierFinder outlierFinder(2); // Any datapoint outside of 2 stds from the mean is marked as an outlier.
	// alg.printVector(outlierFinder.modelTest(inputSet));
}
void MLPPTests::test_new_math_functions() {
	// // Testing new Functions
	// double z_s = 0.001;
	// std::cout << avn.logit(z_s) << std::endl;
	// std::cout << avn.logit(z_s, 1) << std::endl;

	// std::vector<double> z_v = {0.001};
	// alg.printVector(avn.logit(z_v));
	// alg.printVector(avn.logit(z_v, 1));

	// std::vector<std::vector<double>> Z_m = {{0.001}};
	// alg.printMatrix(avn.logit(Z_m));
	// alg.printMatrix(avn.logit(Z_m, 1));

	// std::cout << alg.trace({{1,2}, {3,4}}) << std::endl;
	// alg.printMatrix(alg.pinverse({{1,2}, {3,4}}));
	// alg.printMatrix(alg.diag({1,2,3,4,5}));
	// alg.printMatrix(alg.kronecker_product({{1,2,3,4,5}}, {{6,7,8,9,10}}));
	// alg.printMatrix(alg.matrixPower({{5,5},{5,5}}, 2));
	// alg.printVector(alg.solve({{1,1}, {1.5, 4.0}}, {2200, 5050}));

	// std::vector<std::vector<double>> matrixOfCubes = {{1,2,64,27}};
	// std::vector<double> vectorOfCubes = {1,2,64,27};
	// alg.printMatrix(alg.cbrt(matrixOfCubes));
	// alg.printVector(alg.cbrt(vectorOfCubes));
	// std::cout << alg.max({{1,2,3,4,5}, {6,5,3,4,1}, {9,9,9,9,9}}) << std::endl;
	// std::cout << alg.min({{1,2,3,4,5}, {6,5,3,4,1}, {9,9,9,9,9}}) << std::endl;

	// std::vector<double> chicken;
	// data.getImage("../../Data/apple.jpeg", chicken);
	// alg.printVector(chicken);

	// std::vector<std::vector<double>> P = {{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}};
	// alg.printMatrix(P);

	// alg.printMatrix(alg.gramSchmidtProcess(P));

	// auto [Q, R] = alg.QRD(P); // It works!

	//  alg.printMatrix(Q);

	//  alg.printMatrix(R);
}
void MLPPTests::test_positive_definiteness_checker() {
	// // Checking positive-definiteness checker. For Cholesky Decomp.
	// std::vector<std::vector<double>> A =
	// {
	//     {1,-1,-1,-1},
	//     {-1,2,2,2},
	//     {-1,2,3,1},
	//     {-1,2,1,4}
	// };

	// std::cout << std::boolalpha << alg.positiveDefiniteChecker(A) << std::endl;
	// auto [L, Lt] = alg.chol(A); // works.
	// alg.printMatrix(L);
	// alg.printMatrix(Lt);
}
void MLPPTests::test_numerical_analysis() {
	// Checks for numerical analysis class.
	//NumericalAnalysis numAn;

	//std::cout << numAn.quadraticApproximation(f, 0, 1) << std::endl;

	// std::cout << numAn.cubicApproximation(f, 0, 1.001) << std::endl;

	// std::cout << f(1.001) << std::endl;

	// std::cout << numAn.quadraticApproximation(f_mv, {0, 0, 0}, {1, 1, 1}) << std::endl;

	// std::cout << numAn.numDiff(&f, 1) << std::endl;
	// std::cout << numAn.newtonRaphsonMethod(&f, 1, 1000) << std::endl;
	//std::cout << numAn.invQuadraticInterpolation(&f, {100, 2,1.5}, 10) << std::endl;

	// std::cout << numAn.numDiff(&f_mv, {1, 1}, 1) << std::endl; // Derivative w.r.t. x.

	// alg.printVector(numAn.jacobian(&f_mv, {1, 1}));

	//std::cout << numAn.numDiff_2(&f, 2) << std::endl;

	//std::cout << numAn.numDiff_3(&f, 2) << std::endl;

	// std::cout << numAn.numDiff_2(&f_mv, {2, 2, 500}, 2, 2) << std::endl;
	//std::cout << numAn.numDiff_3(&f_mv, {2, 1000, 130}, 0, 0, 0) << std::endl;

	// alg.printTensor(numAn.thirdOrderTensor(&f_mv, {1, 1, 1}));
	// std::cout << "Our Hessian." << std::endl;
	// alg.printMatrix(numAn.hessian(&f_mv, {2, 2, 500}));

	// std::cout << numAn.laplacian(f_mv, {1,1,1}) << std::endl;

	// std::vector<std::vector<std::vector<double>>> tensor;
	// tensor.push_back({{1,2}, {1,2}, {1,2}});
	// tensor.push_back({{1,2}, {1,2}, {1,2}});

	// alg.printTensor(tensor);

	// alg.printMatrix(alg.tensor_vec_mult(tensor, {1,2}));

	// std::cout << numAn.cubicApproximation(f_mv, {0, 0, 0}, {1, 1, 1}) << std::endl;

	// std::cout << numAn.eulerianMethod(f_prime, {1, 1}, 1.5, 0.000001) << std::endl;

	// std::cout << numAn.eulerianMethod(f_prime_2var, {2, 3}, 2.5, 0.00000001) << std::endl;

	// alg.printMatrix(conv.dx(A));
	// alg.printMatrix(conv.dy(A));

	// alg.printMatrix(conv.gradOrientation(A));

	// std::vector<std::vector<double>> A =
	// {
	//     {1,0,0,0},
	//     {0,0,0,0},
	//     {0,0,0,0},
	//     {0,0,0,1}
	// };

	// std::vector<std::vector<std::string>> h = conv.harrisCornerDetection(A);

	// for(int i = 0; i < h.size(); i++){
	//     for(int j = 0; j < h[i].size(); j++){
	//         std::cout << h[i][j] << " ";
	//     }
	//     std::cout << std::endl;
	// } // Harris detector works. Life is good!

	// std::vector<double> a = {3,4,4};
	// std::vector<double> b = {4,4,4};
	// alg.printVector(alg.cross(a,b));
}
void MLPPTests::test_support_vector_classification_kernel(bool ui) {
	//SUPPORT VECTOR CLASSIFICATION (kernel method)
	// std::vector<std::vector<double>> inputSet;
	// std::vector<double> outputSet;
	// data.setData(30, "/Users/marcmelikyan/Desktop/Data/BreastCancerSVM.csv", inputSet, outputSet);

	// std::vector<std::vector<double>> inputSet;
	// std::vector<double> outputSet;
	// data.setData(4, "/Users/marcmelikyan/Desktop/Data/IrisSVM.csv", inputSet, outputSet);

	// DualSVC kernelSVM(inputSet, outputSet, 1000);
	// kernelSVM.gradientDescent(0.0001, 20, 1);

	// std::vector<std::vector<double>> linearlyIndependentMat =

	// {
	//     {1,2,3,4},
	//     {234538495,4444,6111,55}
	// };

	// std::cout << "True of false: linearly independent?: " << std::boolalpha << alg.linearIndependenceChecker(linearlyIndependentMat) << std::endl;
}

void MLPPTests::is_approx_equalsd(double a, double b, const String &str) {
	if (!Math::is_equal_approx(a, b)) {
		ERR_PRINT("TEST FAILED: " + str + " Got: " + String::num(a) + " Should be: " + String::num(b));
	}
}

void MLPPTests::is_approx_equals_dvec(const Vector<double> &a, const Vector<double> &b, const String &str) {
	if (a.size() != b.size()) {
		goto IAEDVEC_FAILED;
	}

	for (int i = 0; i < a.size(); ++i) {
		if (!Math::is_equal_approx(a[i], b[i])) {
			goto IAEDVEC_FAILED;
		}
	}

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

	ERR_PRINT(fail_str);
}

String vmat_to_str(const Vector<Vector<double>> &a) {
	String str;

	str += "[ \n";

	for (int i = 0; i < a.size(); ++i) {
		str += "  [ ";

		const Vector<double> &aa = a[i];

		for (int j = 0; j < aa.size(); ++j) {
			str += String::num(aa[j]);
			str += " ";
		}

		str += "]\n";
	}

	str += "]\n";

	return str;
}

void MLPPTests::is_approx_equals_dmat(const Vector<Vector<double>> &a, const Vector<Vector<double>> &b, const String &str) {
	if (a.size() != b.size()) {
		goto IAEDMAT_FAILED;
	}

	for (int i = 0; i < a.size(); ++i) {
		const Vector<double> &aa = a[i];
		const Vector<double> &bb = b[i];

		if (aa.size() != bb.size()) {
			goto IAEDMAT_FAILED;
		}

		for (int j = 0; j < aa.size(); ++j) {
			if (!Math::is_equal_approx(aa[j], bb[j])) {
				goto IAEDMAT_FAILED;
			}
		}
	}

	return;

IAEDMAT_FAILED:

	String fail_str = "TEST FAILED: ";
	fail_str += str;
	fail_str += "\nGot:\n";
	fail_str += vmat_to_str(a);
	fail_str += "Should be:\n";
	fail_str += vmat_to_str(b);

	ERR_PRINT(fail_str);
}

MLPPTests::MLPPTests() {
	_load_fires_and_crime_data_path = "res://datasets/FiresAndCrime.csv";
	_load_california_housing_data_path = "res://datasets/CaliforniaHousing.csv";
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
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_adam"), &MLPPTests::test_multivariate_linear_regression_adam);
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
	ClassDB::bind_method(D_METHOD("test_logistic_regression", "ui"), &MLPPTests::test_logistic_regression, false);

	ClassDB::bind_method(D_METHOD("test_mlp", "ui"), &MLPPTests::test_mlp, false);
	ClassDB::bind_method(D_METHOD("test_soft_max_network", "ui"), &MLPPTests::test_soft_max_network, false);
	ClassDB::bind_method(D_METHOD("test_autoencoder", "ui"), &MLPPTests::test_autoencoder, false);
	ClassDB::bind_method(D_METHOD("test_dynamically_sized_ann", "ui"), &MLPPTests::test_dynamically_sized_ann, false);
	ClassDB::bind_method(D_METHOD("test_wgan", "ui"), &MLPPTests::test_wgan, false);
	ClassDB::bind_method(D_METHOD("test_ann", "ui"), &MLPPTests::test_ann, false);
	ClassDB::bind_method(D_METHOD("test_dynamically_sized_mann", "ui"), &MLPPTests::test_dynamically_sized_mann, false);
	ClassDB::bind_method(D_METHOD("test_train_test_split_mann", "ui"), &MLPPTests::test_train_test_split_mann, false);

	ClassDB::bind_method(D_METHOD("test_naive_bayes", "ui"), &MLPPTests::test_naive_bayes, false);
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
}
