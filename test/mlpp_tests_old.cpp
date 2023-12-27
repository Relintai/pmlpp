
#include "mlpp_tests_old.h"

#include "core/math/math_funcs.h"

#include "core/log/logger.h"

//TODO remove
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include "../mlpp/data/data.h"

#include "../mlpp/activation/activation_old.h"
#include "../mlpp/ann/ann_old.h"
#include "../mlpp/auto_encoder/auto_encoder_old.h"
#include "../mlpp/bernoulli_nb/bernoulli_nb_old.h"
#include "../mlpp/c_log_log_reg/c_log_log_reg_old.h"
#include "../mlpp/convolutions/convolutions_old.h"
#include "../mlpp/cost/cost_old.h"
#include "../mlpp/data/data_old.h"
#include "../mlpp/dual_svc/dual_svc_old.h"
#include "../mlpp/exp_reg/exp_reg_old.h"
#include "../mlpp/gan/gan_old.h"
#include "../mlpp/gaussian_nb/gaussian_nb_old.h"
#include "../mlpp/hidden_layer/hidden_layer_old.h"
#include "../mlpp/lin_alg/lin_alg_old.h"
#include "../mlpp/lin_reg/lin_reg_old.h"
#include "../mlpp/log_reg/log_reg_old.h"
#include "../mlpp/mann/mann_old.h"
#include "../mlpp/mlp/mlp_old.h"
#include "../mlpp/multi_output_layer/multi_output_layer_old.h"
#include "../mlpp/multinomial_nb/multinomial_nb_old.h"
#include "../mlpp/numerical_analysis/numerical_analysis_old.h"
#include "../mlpp/outlier_finder/outlier_finder_old.h"
#include "../mlpp/output_layer/output_layer_old.h"
#include "../mlpp/pca/pca_old.h"
#include "../mlpp/probit_reg/probit_reg_old.h"
#include "../mlpp/softmax_net/softmax_net_old.h"
#include "../mlpp/softmax_reg/softmax_reg_old.h"
#include "../mlpp/stat/stat_old.h"
#include "../mlpp/svc/svc_old.h"
#include "../mlpp/tanh_reg/tanh_reg_old.h"
#include "../mlpp/transforms/transforms_old.h"
#include "../mlpp/uni_lin_reg/uni_lin_reg_old.h"
#include "../mlpp/wgan/wgan_old.h"

Vector<real_t> dstd_vec_to_vec_old(const std::vector<real_t> &in) {
	Vector<real_t> r;

	r.resize(static_cast<int>(in.size()));
	real_t *darr = r.ptrw();

	for (uint32_t i = 0; i < in.size(); ++i) {
		darr[i] = in[i];
	}

	return r;
}

Vector<Vector<real_t>> dstd_mat_to_mat_old(const std::vector<std::vector<real_t>> &in) {
	Vector<Vector<real_t>> r;

	for (uint32_t i = 0; i < in.size(); ++i) {
		r.push_back(dstd_vec_to_vec_old(in[i]));
	}

	return r;
}

void MLPPTestsOld::test_statistics() {
}

void MLPPTestsOld::test_linear_algebra() {
}

void MLPPTestsOld::test_univariate_linear_regression() {
}

void MLPPTestsOld::test_multivariate_linear_regression_gradient_descent(bool ui) {
}

void MLPPTestsOld::test_multivariate_linear_regression_sgd(bool ui) {
}

void MLPPTestsOld::test_multivariate_linear_regression_mbgd(bool ui) {
}

void MLPPTestsOld::test_multivariate_linear_regression_normal_equation(bool ui) {
}

void MLPPTestsOld::test_multivariate_linear_regression_adam() {
}

void MLPPTestsOld::test_multivariate_linear_regression_score_sgd_adam(bool ui) {
}

void MLPPTestsOld::test_multivariate_linear_regression_epochs_gradient_descent(bool ui) {
}

void MLPPTestsOld::test_multivariate_linear_regression_newton_raphson(bool ui) {
}

void MLPPTestsOld::test_logistic_regression(bool ui) {
}
void MLPPTestsOld::test_probit_regression(bool ui) {
}
void MLPPTestsOld::test_c_log_log_regression(bool ui) {
}
void MLPPTestsOld::test_exp_reg_regression(bool ui) {
}
void MLPPTestsOld::test_tanh_regression(bool ui) {
}
void MLPPTestsOld::test_softmax_regression(bool ui) {
}
void MLPPTestsOld::test_support_vector_classification(bool ui) {
}

void MLPPTestsOld::test_mlp(bool ui) {
	MLPPLinAlgOld alg;

	// MLP
	std::vector<std::vector<real_t>> inputSet = {
		{ 0, 0 },
		{ 1, 1 },
		{ 0, 1 },
		{ 1, 0 }
	};
	std::vector<real_t> outputSet = { 0, 1, 1, 0 };

	MLPPMLPOld model(inputSet, outputSet, 2);
	model.gradientDescent(0.1, 10000, ui);
	alg.printVector(model.modelSetTest(inputSet));
	std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;
}
void MLPPTestsOld::test_soft_max_network(bool ui) {
	MLPPLinAlgOld alg;
	MLPPData data;

	// SOFTMAX NETWORK
	Ref<MLPPDataComplex> dt = data.load_wine(_wine_data_path);

	MLPPSoftmaxNetOld model_old(dt->get_input()->to_std_vector(), dt->get_output()->to_std_vector(), 1);
	model_old.gradientDescent(0.01, 100000, ui);
	alg.printMatrix(model_old.modelSetTest(dt->get_input()->to_std_vector()));
	std::cout << "ACCURACY: " << 100 * model_old.score() << "%" << std::endl;
}
void MLPPTestsOld::test_autoencoder(bool ui) {
	MLPPLinAlgOld alg;

	std::vector<std::vector<real_t>> inputSet = { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, { 3, 5, 9, 12, 15, 18, 21, 24, 27, 30 } };

	// AUTOENCODER
	MLPPAutoEncoderOld model_old(alg.transpose(inputSet), 5);
	model_old.SGD(0.001, 300000, ui);
	alg.printMatrix(model_old.modelSetTest(alg.transpose(inputSet)));
	std::cout << "ACCURACY (Old): " << 100 * model_old.score() << "%" << std::endl;

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);
}
void MLPPTestsOld::test_dynamically_sized_ann(bool ui) {
	MLPPLinAlgOld alg;

	// DYNAMICALLY SIZED ANN
	// Possible Weight Init Methods: Default, Uniform, HeNormal, HeUniform, XavierNormal, XavierUniform
	// Possible Activations: Linear, Sigmoid, Swish, Softplus, Softsign, CLogLog, Ar{Sinh, Cosh, Tanh, Csch, Sech, Coth},  GaussianCDF, GELU, UnitStep
	// Possible Loss Functions: MSE, RMSE, MBE, LogLoss, CrossEntropy, HingeLoss
	std::vector<std::vector<real_t>> inputSet = { { 0, 0, 1, 1 }, { 0, 1, 0, 1 } };
	std::vector<real_t> outputSet = { 0, 1, 1, 0 };

	MLPPANNOld ann_old(alg.transpose(inputSet), outputSet);
	ann_old.addLayer(2, "Cosh");
	ann_old.addOutputLayer("Sigmoid", "LogLoss");

	ann_old.AMSGrad(0.1, 10000, 1, 0.9, 0.999, 0.000001, ui);
	ann_old.Adadelta(1, 1000, 2, 0.9, 0.000001, ui);
	ann_old.Momentum(0.1, 8000, 2, 0.9, true, ui);

	ann_old.setLearningRateScheduler("Step", 0.5, 1000);
	ann_old.gradientDescent(0.01, 30000);
	alg.printVector(ann_old.modelSetTest(alg.transpose(inputSet)));
	std::cout << "ACCURACY: " << 100 * ann_old.score() << "%" << std::endl;
}
void MLPPTestsOld::test_wgan_old(bool ui) {
	//MLPPStat stat;
	MLPPLinAlgOld alg;
	//MLPPActivation avn;
	//MLPPCost cost;
	//MLPPData data;
	//MLPPConvolutions conv;

	std::vector<std::vector<real_t>> outputSet = {
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
		{ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40 }
	};

	MLPPWGANOld gan_old(2, alg.transpose(outputSet)); // our gan is a wasserstein gan (wgan)
	gan_old.addLayer(5, "Sigmoid");
	gan_old.addLayer(2, "RELU");
	gan_old.addLayer(5, "Sigmoid");
	gan_old.addOutputLayer(); // User can specify weight init- if necessary.
	gan_old.gradientDescent(0.1, 55000, ui);
	std::cout << "GENERATED INPUT: (Gaussian-sampled noise):" << std::endl;
	alg.printMatrix(gan_old.generateExample(100));
}
void MLPPTestsOld::test_wgan(bool ui) {
}
void MLPPTestsOld::test_ann(bool ui) {
	MLPPLinAlgOld alg;

	std::vector<std::vector<real_t>> inputSet = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }; // XOR
	std::vector<real_t> outputSet = { 0, 1, 1, 0 };

	MLPPANNOld ann_old(inputSet, outputSet);
	ann_old.addLayer(5, "Sigmoid");
	ann_old.addLayer(8, "Sigmoid"); // Add more layers as needed.
	ann_old.addOutputLayer("Sigmoid", "LogLoss");
	ann_old.gradientDescent(1, 20000, ui);

	std::vector<real_t> predictions_old = ann_old.modelSetTest(inputSet);
	alg.printVector(predictions_old); // Testing out the model's preds for train set.
	std::cout << "ACCURACY: " << 100 * ann_old.score() << "%" << std::endl; // Accuracy.
}
void MLPPTestsOld::test_dynamically_sized_mann(bool ui) {
	MLPPLinAlgOld alg;
	MLPPData data;

	// DYNAMICALLY SIZED MANN (Multidimensional Output ANN)
	std::vector<std::vector<real_t>> inputSet = { { 1, 2, 3 }, { 2, 4, 6 }, { 3, 6, 9 }, { 4, 8, 12 } };
	std::vector<std::vector<real_t>> outputSet = { { 1, 5 }, { 2, 10 }, { 3, 15 }, { 4, 20 } };

	MLPPMANNOld mann_old(inputSet, outputSet);
	mann_old.addOutputLayer("Linear", "MSE");
	mann_old.gradientDescent(0.001, 80000, false);
	alg.printMatrix(mann_old.modelSetTest(inputSet));
	std::cout << "ACCURACY (old): " << 100 * mann_old.score() << "%" << std::endl;
}
void MLPPTestsOld::test_train_test_split_mann(bool ui) {
	MLPPLinAlgOld alg;
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

	d->set_input(input_set_1->transposen());
	d->set_output(output_set_1->transposen());

	MLPPData::SplitComplexData split_data = data.train_test_split(d, 0.2);

	PLOG_MSG(split_data.train->get_input()->to_string());
	PLOG_MSG(split_data.train->get_output()->to_string());
	PLOG_MSG(split_data.test->get_input()->to_string());
	PLOG_MSG(split_data.test->get_output()->to_string());

	MLPPMANNOld mann_old(split_data.train->get_input()->to_std_vector(), split_data.train->get_output()->to_std_vector());
	mann_old.addLayer(100, "RELU", "XavierNormal");
	mann_old.addOutputLayer("Softmax", "CrossEntropy", "XavierNormal");
	mann_old.gradientDescent(0.1, 80000, ui);
	alg.printMatrix(mann_old.modelSetTest(split_data.test->get_input()->to_std_vector()));
	std::cout << "ACCURACY (old): " << 100 * mann_old.score() << "%" << std::endl;
}

void MLPPTestsOld::test_naive_bayes() {
	MLPPLinAlgOld alg;

	// NAIVE BAYES
	std::vector<std::vector<real_t>> inputSet = { { 1, 1, 1, 1, 1 }, { 0, 0, 1, 1, 1 }, { 0, 0, 1, 0, 1 } };
	std::vector<real_t> outputSet = { 0, 1, 0, 1, 1 };

	MLPPMultinomialNBOld MNB_old(alg.transpose(inputSet), outputSet, 2);
	alg.printVector(MNB_old.modelSetTest(alg.transpose(inputSet)));

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(alg.transpose(inputSet));

	Ref<MLPPVector> output_set;
	output_set.instance();
	output_set->set_from_std_vector(outputSet);

	MLPPBernoulliNBOld BNBOld(alg.transpose(inputSet), outputSet);
	alg.printVector(BNBOld.modelSetTest(alg.transpose(inputSet)));

	MLPPGaussianNBOld GNBOld(alg.transpose(inputSet), outputSet, 2);
	alg.printVector(GNBOld.modelSetTest(alg.transpose(inputSet)));
}
void MLPPTestsOld::test_k_means(bool ui) {
}
void MLPPTestsOld::test_knn(bool ui) {
}

void MLPPTestsOld::test_convolution_tensors_etc() {
	MLPPLinAlgOld alg;
	MLPPData data;
	MLPPConvolutionsOld conv;

	// CONVOLUTION, POOLING, ETC..
	std::vector<std::vector<real_t>> input = {
		{ 1 },
	};

	std::vector<std::vector<std::vector<real_t>>> tensorSet;
	tensorSet.push_back(input);
	tensorSet.push_back(input);
	tensorSet.push_back(input);

	alg.printTensor(data.rgb2xyz(tensorSet));

	std::vector<std::vector<real_t>> input2 = {
		{ 62, 55, 55, 54, 49, 48, 47, 55 },
		{ 62, 57, 54, 52, 48, 47, 48, 53 },
		{ 61, 60, 52, 49, 48, 47, 49, 54 },
		{ 63, 61, 60, 60, 63, 65, 68, 65 },
		{ 67, 67, 70, 74, 79, 85, 91, 92 },
		{ 82, 95, 101, 106, 114, 115, 112, 117 },
		{ 96, 111, 115, 119, 128, 128, 130, 127 },
		{ 109, 121, 127, 133, 139, 141, 140, 133 },
	};

	MLPPTransformsOld trans;

	alg.printMatrix(trans.discreteCosineTransform(input2));

	alg.printMatrix(conv.convolve_2d(input2, conv.get_prewitt_vertical(), 1)); // Can use padding
	alg.printMatrix(conv.pool_2d(input2, 4, 4, "Max")); // Can use Max, Min, or Average pooling.

	std::vector<std::vector<std::vector<real_t>>> tensorSet2;
	tensorSet2.push_back(input2);
	tensorSet2.push_back(input2);
	alg.printVector(conv.global_pool_3d(tensorSet2, "Average")); // Can use Max, Min, or Average global pooling.

	std::vector<std::vector<real_t>> laplacian = { { 1, 1, 1 }, { 1, -4, 1 }, { 1, 1, 1 } };
	alg.printMatrix(conv.convolve_2d(conv.gaussian_filter_2d(5, 1), laplacian, 1));
}
void MLPPTestsOld::test_pca_svd_eigenvalues_eigenvectors(bool ui) {
	MLPPLinAlgOld alg;

	// PCA, SVD, eigenvalues & eigenvectors
	std::vector<std::vector<real_t>> inputSet = { { 1, 1 }, { 1, 1 } };

	MLPPLinAlgOld::EigenResultOld eigen = alg.eigen_old(inputSet);

	std::cout << "Eigenvectors:" << std::endl;
	alg.printMatrix(eigen.eigen_vectors);
	std::cout << std::endl;
	std::cout << "Eigenvalues:" << std::endl;
	alg.printMatrix(eigen.eigen_values);

	std::cout << "SVD OLD START" << std::endl;

	MLPPLinAlgOld::SVDResultOld svd_old = alg.SVD(inputSet);

	std::cout << "U:" << std::endl;
	alg.printMatrix(svd_old.U);
	std::cout << "S:" << std::endl;
	alg.printMatrix(svd_old.S);
	std::cout << "Vt:" << std::endl;
	alg.printMatrix(svd_old.Vt);

	std::cout << "SVD OLD FIN" << std::endl;

	Ref<MLPPMatrix> input_set;
	input_set.instance();
	input_set->set_from_std_vectors(inputSet);

	/*
	String str_svd = "SVD\n";

	MLPPLinAlgOld::SVDResult svd = alg.svd(input_set);

	str_svd += "U:\n";
	str_svd += svd.U->to_string();
	str_svd += "\nS:\n";
	str_svd += svd.S->to_string();
	str_svd += "\nVt:\n";
	str_svd += svd.Vt->to_string();
	str_svd += "\n";

	PLOG_MSG(str_svd);
	*/

	std::cout << "PCA" << std::endl;

	// PCA done using Jacobi's method to approximate eigenvalues and eigenvectors.
	MLPPPCAOld dr_old(inputSet, 1); // 1 dimensional representation.
	std::cout << std::endl;
	std::cout << "OLD Dimensionally reduced representation:" << std::endl;
	alg.printMatrix(dr_old.principalComponents());
	std::cout << "SCORE: " << dr_old.score() << std::endl;
}

void MLPPTestsOld::test_nlp_and_data(bool ui) {
	MLPPLinAlgOld alg;
	MLPPData data;

	// NLP/DATA
	std::string verbText = "I am appearing and thinking, as well as conducting.";
	std::cout << "Stemming Example:" << std::endl;
	std::cout << data.stemming(verbText) << std::endl;
	std::cout << std::endl;

	std::vector<std::string> sentences = { "He is a good boy", "She is a good girl", "The boy and girl are good" };
	std::cout << "Bag of Words Example:" << std::endl;
	alg.printMatrix(data.BOW(sentences, "Default"));
	std::cout << std::endl;
	std::cout << "TFIDF Example:" << std::endl;
	alg.printMatrix(data.TFIDF(sentences));
	std::cout << std::endl;

	std::cout << "Tokenization:" << std::endl;
	alg.printVector(data.tokenize(verbText));
	std::cout << std::endl;

	std::cout << "Word2Vec:" << std::endl;
	std::string textArchive = { "He is a good boy. She is a good girl. The boy and girl are good." };
	std::vector<std::string> corpus = data.splitSentences(textArchive);

	MLPPData::WordsToVecResult wtvres = data.word_to_vec(corpus, "CBOW", 2, 2, 0.1, 10000); // Can use either CBOW or Skip-n-gram.

	alg.printMatrix(wtvres.word_embeddings);
	std::cout << std::endl;

	std::vector<std::string> textArchive2 = { "pizza", "pizza hamburger cookie", "hamburger", "ramen", "sushi", "ramen sushi" };

	alg.printMatrix(data.LSA(textArchive2, 2));
	//alg.printMatrix(data.BOW(textArchive, "Default"));
	std::cout << std::endl;

	std::vector<std::vector<real_t>> inputSet = { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 } };
	std::cout << "Feature Scaling Example:" << std::endl;
	alg.printMatrix(data.featureScaling(inputSet));
	std::cout << std::endl;

	std::cout << "Mean Centering Example:" << std::endl;
	alg.printMatrix(data.meanCentering(inputSet));
	std::cout << std::endl;

	std::cout << "Mean Normalization Example:" << std::endl;
	alg.printMatrix(data.meanNormalization(inputSet));
	std::cout << std::endl;
}
void MLPPTestsOld::test_outlier_finder(bool ui) {
	MLPPLinAlgOld alg;

	// Outlier Finder
	//std::vector<real_t> inputSet = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 23554332523523 };
	std::vector<real_t> inputSet = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 23554332 };
	MLPPOutlierFinderOld outlierFinderOld(2); // Any datapoint outside of 2 stds from the mean is marked as an outlier.
	alg.printVector(outlierFinderOld.modelTest(inputSet));
}
void MLPPTestsOld::test_new_math_functions() {
	MLPPLinAlgOld alg;
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

	//MLPPLinAlgOld::QRDResult qrd_result = alg.qrd(P); // It works!
	//alg.printMatrix(qrd_result.Q);
	//alg.printMatrix(qrd_result.R);
}
void MLPPTestsOld::test_positive_definiteness_checker() {
	//MLPPStat stat;
	MLPPLinAlgOld alg;
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
	MLPPLinAlgOld::CholeskyResult chres = alg.cholesky(A); // works.
	alg.printMatrix(chres.L);
	alg.printMatrix(chres.Lt);
}

// real_t f(real_t x){
//     return x*x*x + 2*x - 2;
// }

real_t f_old(real_t x) {
	return sin(x);
}

real_t f_prime_old(real_t x) {
	return 2 * x;
}

real_t f_prime_2var_old(std::vector<real_t> x) {
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

real_t f_mv_old(std::vector<real_t> x) {
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

void MLPPTestsOld::test_numerical_analysis() {
	MLPPLinAlgOld alg;
	MLPPConvolutionsOld conv;

	// Checks for numerical analysis class.
	MLPPNumericalAnalysisOld numAn;

	std::cout << numAn.quadraticApproximation(f_old, 0, 1) << std::endl;

	std::cout << numAn.cubicApproximation(f_old, 0, 1.001) << std::endl;

	std::cout << f_old(1.001) << std::endl;

	std::cout << numAn.quadraticApproximation(f_mv_old, { 0, 0, 0 }, { 1, 1, 1 }) << std::endl;

	std::cout << numAn.numDiff(&f_old, 1) << std::endl;
	std::cout << numAn.newtonRaphsonMethod(&f_old, 1, 1000) << std::endl;
	std::cout << numAn.invQuadraticInterpolation(&f_old, { 100, 2, 1.5 }, 10) << std::endl;

	std::cout << numAn.numDiff(&f_mv_old, { 1, 1 }, 1) << std::endl; // Derivative w.r.t. x.

	alg.printVector(numAn.jacobian(&f_mv_old, { 1, 1 }));

	std::cout << numAn.numDiff_2(&f_old, 2) << std::endl;

	std::cout << numAn.numDiff_3(&f_old, 2) << std::endl;

	std::cout << numAn.numDiff_2(&f_mv_old, { 2, 2, 500 }, 2, 2) << std::endl;
	std::cout << numAn.numDiff_3(&f_mv_old, { 2, 1000, 130 }, 0, 0, 0) << std::endl;

	alg.printTensor(numAn.thirdOrderTensor(&f_mv_old, { 1, 1, 1 }));
	std::cout << "Our Hessian." << std::endl;
	alg.printMatrix(numAn.hessian(&f_mv_old, { 2, 2, 500 }));

	std::cout << numAn.laplacian(f_mv_old, { 1, 1, 1 }) << std::endl;

	std::vector<std::vector<std::vector<real_t>>> tensor;
	tensor.push_back({ { 1, 2 }, { 1, 2 }, { 1, 2 } });
	tensor.push_back({ { 1, 2 }, { 1, 2 }, { 1, 2 } });

	alg.printTensor(tensor);

	alg.printMatrix(alg.tensor_vec_mult(tensor, { 1, 2 }));

	std::cout << numAn.cubicApproximation(f_mv_old, { 0, 0, 0 }, { 1, 1, 1 }) << std::endl;
	std::cout << numAn.eulerianMethod(f_prime_old, { 1, 1 }, 1.5, 0.000001) << std::endl;
	std::cout << numAn.eulerianMethod(f_prime_2var_old, { 2, 3 }, 2.5, 0.00000001) << std::endl;

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
}
void MLPPTestsOld::test_support_vector_classification_kernel(bool ui) {
	MLPPLinAlgOld alg;
	MLPPData data;

	//SUPPORT VECTOR CLASSIFICATION (kernel method)
	Ref<MLPPDataSimple> dt = data.load_breast_cancer_svc(_breast_cancer_svm_data_path);

	MLPPDualSVCOld kernelSVMOld(dt->get_input()->to_std_vector(), dt->get_output()->to_std_vector(), 1000);
	kernelSVMOld.gradientDescent(0.0001, 20, ui);
	std::cout << "SCORE: " << kernelSVMOld.score() << std::endl;

	std::vector<std::vector<real_t>> linearlyIndependentMat = {
		{ 1, 2, 3, 4 },
		{ 2345384, 4444, 6111, 55 }
	};

	std::cout << "True of false: linearly independent?: " << std::boolalpha << alg.linearIndependenceChecker(linearlyIndependentMat) << std::endl;
}

void MLPPTestsOld::test_mlpp_vector() {
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

void MLPPTestsOld::is_approx_equalsd(real_t a, real_t b, const String &str) {
	if (!Math::is_equal_approx(a, b)) {
		PLOG_ERR("TEST FAILED: " + str + " Got: " + String::num(a) + " Should be: " + String::num(b));
	} else {
		PLOG_MSG("TEST PASSED: " + str);
	}
}

void MLPPTestsOld::is_approx_equals_dvec(const Vector<real_t> &a, const Vector<real_t> &b, const String &str) {
	if (a.size() != b.size()) {
		goto IAEDVEC_FAILED;
	}

	for (int i = 0; i < a.size(); ++i) {
		if (!Math::is_equal_approx(a[i], b[i])) {
			goto IAEDVEC_FAILED;
		}
	}

	PLOG_MSG("TEST PASSED: " + str);

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

String vmat_to_str_old(const Vector<Vector<real_t>> &a) {
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

void MLPPTestsOld::is_approx_equals_dmat(const Vector<Vector<real_t>> &a, const Vector<Vector<real_t>> &b, const String &str) {
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

	PLOG_MSG("TEST PASSED: " + str);

	return;

IAEDMAT_FAILED:

	String fail_str = "TEST FAILED: ";
	fail_str += str;
	fail_str += "\nGot:\n";
	fail_str += vmat_to_str_old(a);
	fail_str += "Should be:\n";
	fail_str += vmat_to_str_old(b);

	PLOG_ERR(fail_str);
}

void MLPPTestsOld::is_approx_equals_mat(Ref<MLPPMatrix> a, Ref<MLPPMatrix> b, const String &str) {
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

	PLOG_MSG("TEST PASSED: " + str);

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
void MLPPTestsOld::is_approx_equals_vec(Ref<MLPPVector> a, Ref<MLPPVector> b, const String &str) {
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

	PLOG_MSG("TEST PASSED: " + str);

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

MLPPTestsOld::MLPPTestsOld() {
	_breast_cancer_data_path = "res://datasets/BreastCancer.csv";
	_breast_cancer_svm_data_path = "res://datasets/BreastCancerSVM.csv";
	_california_housing_data_path = "res://datasets/CaliforniaHousing.csv";
	_fires_and_crime_data_path = "res://datasets/FiresAndCrime.csv";
	_iris_data_path = "res://datasets/Iris.csv";
	_mnist_test_data_path = "res://datasets/MnistTest.csv";
	_mnist_train_data_path = "res://datasets/MnistTrain.csv";
	_wine_data_path = "res://datasets/Wine.csv";
}

MLPPTestsOld::~MLPPTestsOld() {
}

void MLPPTestsOld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("test_statistics"), &MLPPTestsOld::test_statistics);
	ClassDB::bind_method(D_METHOD("test_linear_algebra"), &MLPPTestsOld::test_linear_algebra);
	ClassDB::bind_method(D_METHOD("test_univariate_linear_regression"), &MLPPTestsOld::test_univariate_linear_regression);

	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_gradient_descent", "ui"), &MLPPTestsOld::test_multivariate_linear_regression_gradient_descent, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_sgd", "ui"), &MLPPTestsOld::test_multivariate_linear_regression_sgd, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_mbgd", "ui"), &MLPPTestsOld::test_multivariate_linear_regression_mbgd, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_normal_equation", "ui"), &MLPPTestsOld::test_multivariate_linear_regression_normal_equation, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_adam"), &MLPPTestsOld::test_multivariate_linear_regression_adam);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_score_sgd_adam", "ui"), &MLPPTestsOld::test_multivariate_linear_regression_score_sgd_adam, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_epochs_gradient_descent", "ui"), &MLPPTestsOld::test_multivariate_linear_regression_epochs_gradient_descent, false);
	ClassDB::bind_method(D_METHOD("test_multivariate_linear_regression_newton_raphson", "ui"), &MLPPTestsOld::test_multivariate_linear_regression_newton_raphson, false);

	ClassDB::bind_method(D_METHOD("test_logistic_regression", "ui"), &MLPPTestsOld::test_logistic_regression, false);
	ClassDB::bind_method(D_METHOD("test_probit_regression", "ui"), &MLPPTestsOld::test_probit_regression, false);
	ClassDB::bind_method(D_METHOD("test_c_log_log_regression", "ui"), &MLPPTestsOld::test_c_log_log_regression, false);
	ClassDB::bind_method(D_METHOD("test_exp_reg_regression", "ui"), &MLPPTestsOld::test_exp_reg_regression, false);
	ClassDB::bind_method(D_METHOD("test_tanh_regression", "ui"), &MLPPTestsOld::test_tanh_regression, false);
	ClassDB::bind_method(D_METHOD("test_softmax_regression", "ui"), &MLPPTestsOld::test_softmax_regression, false);
	ClassDB::bind_method(D_METHOD("test_support_vector_classification", "ui"), &MLPPTestsOld::test_support_vector_classification, false);

	ClassDB::bind_method(D_METHOD("test_mlp", "ui"), &MLPPTestsOld::test_mlp, false);
	ClassDB::bind_method(D_METHOD("test_soft_max_network", "ui"), &MLPPTestsOld::test_soft_max_network, false);
	ClassDB::bind_method(D_METHOD("test_autoencoder", "ui"), &MLPPTestsOld::test_autoencoder, false);
	ClassDB::bind_method(D_METHOD("test_dynamically_sized_ann", "ui"), &MLPPTestsOld::test_dynamically_sized_ann, false);
	ClassDB::bind_method(D_METHOD("test_wgan_old", "ui"), &MLPPTestsOld::test_wgan_old, false);
	ClassDB::bind_method(D_METHOD("test_wgan", "ui"), &MLPPTestsOld::test_wgan, false);
	ClassDB::bind_method(D_METHOD("test_ann", "ui"), &MLPPTestsOld::test_ann, false);
	ClassDB::bind_method(D_METHOD("test_dynamically_sized_mann", "ui"), &MLPPTestsOld::test_dynamically_sized_mann, false);
	ClassDB::bind_method(D_METHOD("test_train_test_split_mann", "ui"), &MLPPTestsOld::test_train_test_split_mann, false);

	ClassDB::bind_method(D_METHOD("test_naive_bayes"), &MLPPTestsOld::test_naive_bayes);
	ClassDB::bind_method(D_METHOD("test_k_means", "ui"), &MLPPTestsOld::test_k_means, false);
	ClassDB::bind_method(D_METHOD("test_knn", "ui"), &MLPPTestsOld::test_knn, false);

	ClassDB::bind_method(D_METHOD("test_convolution_tensors_etc"), &MLPPTestsOld::test_convolution_tensors_etc);
	ClassDB::bind_method(D_METHOD("test_pca_svd_eigenvalues_eigenvectors", "ui"), &MLPPTestsOld::test_pca_svd_eigenvalues_eigenvectors, false);

	ClassDB::bind_method(D_METHOD("test_nlp_and_data", "ui"), &MLPPTestsOld::test_nlp_and_data, false);
	ClassDB::bind_method(D_METHOD("test_outlier_finder", "ui"), &MLPPTestsOld::test_outlier_finder, false);

	ClassDB::bind_method(D_METHOD("test_new_math_functions"), &MLPPTestsOld::test_new_math_functions);
	ClassDB::bind_method(D_METHOD("test_positive_definiteness_checker"), &MLPPTestsOld::test_positive_definiteness_checker);
	ClassDB::bind_method(D_METHOD("test_numerical_analysis"), &MLPPTestsOld::test_numerical_analysis);

	ClassDB::bind_method(D_METHOD("test_support_vector_classification_kernel", "ui"), &MLPPTestsOld::test_support_vector_classification_kernel, false);

	ClassDB::bind_method(D_METHOD("test_mlpp_vector"), &MLPPTestsOld::test_mlpp_vector);
}
