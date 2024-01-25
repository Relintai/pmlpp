#ifndef MLPP_UTILITIES_H
#define MLPP_UTILITIES_H

/*************************************************************************/
/*  utilities.h                                                          */
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

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/containers/vector.h"
#include "core/math/math_defs.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

#include "core/object/reference.h"
#endif

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPUtilities : public Reference {
	GDCLASS(MLPPUtilities, Reference);

public:
	// Weight Init
	static std::vector<real_t> weightInitialization(int n, std::string type = "Default");
	static real_t biasInitialization();

	static std::vector<std::vector<real_t>> weightInitialization(int n, int m, std::string type = "Default");
	static std::vector<real_t> biasInitialization(int n);

	enum WeightDistributionType {
		WEIGHT_DISTRIBUTION_TYPE_DEFAULT = 0,
		WEIGHT_DISTRIBUTION_TYPE_XAVIER_NORMAL,
		WEIGHT_DISTRIBUTION_TYPE_XAVIER_UNIFORM,
		WEIGHT_DISTRIBUTION_TYPE_HE_NORMAL,
		WEIGHT_DISTRIBUTION_TYPE_HE_UNIFORM,
		WEIGHT_DISTRIBUTION_TYPE_LE_CUN_NORMAL,
		WEIGHT_DISTRIBUTION_TYPE_LE_CUN_UNIFORM,
		WEIGHT_DISTRIBUTION_TYPE_UNIFORM,
	};

	void weight_initializationv(Ref<MLPPVector> weights, WeightDistributionType type = WEIGHT_DISTRIBUTION_TYPE_DEFAULT);
	void weight_initializationm(Ref<MLPPMatrix> weights, WeightDistributionType type = WEIGHT_DISTRIBUTION_TYPE_DEFAULT);
	real_t bias_initializationr();
	void bias_initializationv(Ref<MLPPVector> z);

	// Cost/Performance related Functions
	real_t performance(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t performance(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t performance_vec(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set);
	real_t performance_mat(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);
	real_t performance_pool_int_array_vec(PoolIntArray y_hat, const Ref<MLPPVector> &output_set);

	// Parameter Saving Functions
	void saveParameters(std::string fileName, std::vector<real_t> weights, real_t bias, bool app = false, int layer = -1);
	void saveParameters(std::string fileName, std::vector<real_t> weights, std::vector<real_t> initial, real_t bias, bool app = false, int layer = -1);
	void saveParameters(std::string fileName, std::vector<std::vector<real_t>> weights, std::vector<real_t> bias, bool app = false, int layer = -1);

	// Gradient Descent related
	static void UI(std::vector<real_t> weights, real_t bias);
	static void UI(std::vector<real_t> weights, std::vector<real_t> initial, real_t bias);
	static void UI(std::vector<std::vector<real_t>> weights, std::vector<real_t> bias);

	static void print_ui_vb(Ref<MLPPVector> weights, real_t bias);
	static void print_ui_vib(Ref<MLPPVector> weights, Ref<MLPPVector> initial, real_t bias);
	static void print_ui_mb(Ref<MLPPMatrix> weights, Ref<MLPPVector> bias);

	static void CostInfo(int epoch, real_t cost_prev, real_t Cost);
	static void cost_info(int epoch, real_t cost_prev, real_t cost);

	static std::vector<std::vector<std::vector<real_t>>> createMiniBatches(std::vector<std::vector<real_t>> inputSet, int n_mini_batch);
	static std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<std::vector<real_t>>> createMiniBatches(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int n_mini_batch);
	static std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<std::vector<std::vector<real_t>>>> createMiniBatches(std::vector<std::vector<real_t>> inputSet, std::vector<std::vector<real_t>> outputSet, int n_mini_batch);

	struct CreateMiniBatchMVBatch {
		Vector<Ref<MLPPMatrix>> input_sets;
		Vector<Ref<MLPPVector>> output_sets;
	};

	struct CreateMiniBatchMMBatch {
		Vector<Ref<MLPPMatrix>> input_sets;
		Vector<Ref<MLPPMatrix>> output_sets;
	};

	static Vector<Ref<MLPPMatrix>> create_mini_batchesm(const Ref<MLPPMatrix> &input_set, int n_mini_batch);
	static CreateMiniBatchMVBatch create_mini_batchesmv(const Ref<MLPPMatrix> &input_set, const Ref<MLPPVector> &output_set, int n_mini_batch);
	static CreateMiniBatchMMBatch create_mini_batchesmm(const Ref<MLPPMatrix> &input_set, const Ref<MLPPMatrix> &output_set, int n_mini_batch);

	Array create_mini_batchesm_bind(const Ref<MLPPMatrix> &input_set, int n_mini_batch);
	Array create_mini_batchesmv_bind(const Ref<MLPPMatrix> &input_set, const Ref<MLPPVector> &output_set, int n_mini_batch);
	Array create_mini_batchesmm_bind(const Ref<MLPPMatrix> &input_set, const Ref<MLPPMatrix> &output_set, int n_mini_batch);

	// F1 score, Precision/Recall, TP, FP, TN, FN, etc.
	std::tuple<real_t, real_t, real_t, real_t> TF_PN(std::vector<real_t> y_hat, std::vector<real_t> y); //TF_PN = "True", "False", "Positive", "Negative"
	real_t recall(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t precision(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t accuracy(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t f1_score(std::vector<real_t> y_hat, std::vector<real_t> y);

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(MLPPUtilities::WeightDistributionType);

#endif /* Utilities_hpp */
