/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
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

#include "register_types.h"

#include "data/data.h"
#include "lin_alg/mlpp_matrix.h"
#include "lin_alg/mlpp_tensor3.h"
#include "lin_alg/mlpp_vector.h"

#include "activation/activation.h"
#include "convolutions/convolutions.h"
#include "cost/cost.h"
#include "gauss_markov_checker/gauss_markov_checker.h"
#include "hypothesis_testing/hypothesis_testing.h"
#include "lin_alg/lin_alg.h"
#include "numerical_analysis/numerical_analysis.h"
#include "regularization/reg.h"
#include "stat/stat.h"
#include "transforms/transforms.h"
#include "utilities/utilities.h"

#include "hidden_layer/hidden_layer.h"
#include "multi_output_layer/multi_output_layer.h"
#include "output_layer/output_layer.h"

#include "ann/ann.h"
#include "auto_encoder/auto_encoder.h"
#include "bernoulli_nb/bernoulli_nb.h"
#include "c_log_log_reg/c_log_log_reg.h"
#include "dual_svc/dual_svc.h"
#include "exp_reg/exp_reg.h"
#include "gan/gan.h"
#include "gaussian_nb/gaussian_nb.h"
#include "kmeans/kmeans.h"
#include "knn/knn.h"
#include "lin_reg/lin_reg.h"
#include "log_reg/log_reg.h"
#include "mann/mann.h"
#include "mlp/mlp.h"
#include "multinomial_nb/multinomial_nb.h"
#include "outlier_finder/outlier_finder.h"
#include "pca/pca.h"
#include "probit_reg/probit_reg.h"
#include "softmax_net/softmax_net.h"
#include "softmax_reg/softmax_reg.h"
#include "svc/svc.h"
#include "tanh_reg/tanh_reg.h"
#include "uni_lin_reg/uni_lin_reg.h"
#include "wgan/wgan.h"

#ifdef TESTS_ENABLED
#include "test/mlpp_matrix_tests.h"
#include "test/mlpp_tests.h"
#endif

void register_pmlpp_types(ModuleRegistrationLevel p_level) {
	if (p_level == MODULE_REGISTRATION_LEVEL_SCENE) {
		ClassDB::register_class<MLPPVector>();
		ClassDB::register_class<MLPPMatrix>();
		ClassDB::register_class<MLPPTensor3>();

		ClassDB::register_class<MLPPUtilities>();
		ClassDB::register_class<MLPPReg>();
		ClassDB::register_class<MLPPActivation>();
		ClassDB::register_class<MLPPCost>();
		ClassDB::register_class<MLPPTransforms>();
		ClassDB::register_class<MLPPStat>();
		ClassDB::register_class<MLPPNumericalAnalysis>();
		ClassDB::register_class<MLPPHypothesisTesting>();
		ClassDB::register_class<MLPPGaussMarkovChecker>();
		ClassDB::register_class<MLPPConvolutions>();
		ClassDB::register_class<MLPPLinAlg>();

		ClassDB::register_class<MLPPHiddenLayer>();
		ClassDB::register_class<MLPPOutputLayer>();
		ClassDB::register_class<MLPPMultiOutputLayer>();

		ClassDB::register_class<MLPPKNN>();
		ClassDB::register_class<MLPPKMeans>();

		ClassDB::register_class<MLPPMLP>();
		ClassDB::register_class<MLPPWGAN>();
		ClassDB::register_class<MLPPPCA>();
		ClassDB::register_class<MLPPUniLinReg>();
		ClassDB::register_class<MLPPOutlierFinder>();
		ClassDB::register_class<MLPPProbitReg>();
		ClassDB::register_class<MLPPSVC>();
		ClassDB::register_class<MLPPSoftmaxReg>();
		ClassDB::register_class<MLPPAutoEncoder>();
		ClassDB::register_class<MLPPTanhReg>();
		ClassDB::register_class<MLPPSoftmaxNet>();
		ClassDB::register_class<MLPPMultinomialNB>();
		ClassDB::register_class<MLPPMANN>();
		ClassDB::register_class<MLPPLogReg>();
		ClassDB::register_class<MLPPLinReg>();
		ClassDB::register_class<MLPPGaussianNB>();
		ClassDB::register_class<MLPPGAN>();
		ClassDB::register_class<MLPPExpReg>();
		ClassDB::register_class<MLPPDualSVC>();
		ClassDB::register_class<MLPPCLogLogReg>();
		ClassDB::register_class<MLPPBernoulliNB>();
		ClassDB::register_class<MLPPANN>();

		ClassDB::register_class<MLPPDataESimple>();
		ClassDB::register_class<MLPPDataSimple>();
		ClassDB::register_class<MLPPDataComplex>();
		ClassDB::register_class<MLPPData>();

#ifdef TESTS_ENABLED
		ClassDB::register_class<MLPPTests>();
		ClassDB::register_class<MLPPMatrixTests>();
#endif
	}
}

void unregister_pmlpp_types(ModuleRegistrationLevel p_level) {
}
