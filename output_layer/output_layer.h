#ifndef MLPP_OUTPUT_LAYER_H
#define MLPP_OUTPUT_LAYER_H

/*************************************************************************/
/*  output_layer.h                                                       */
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
#include "core/math/math_defs.h"
#include "core/string/ustring.h"

#include "core/object/reference.h"
#endif

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPOutputLayer : public Reference {
	GDCLASS(MLPPOutputLayer, Reference);

public:
	int get_n_hidden();
	void set_n_hidden(const int val);

	MLPPActivation::ActivationFunction get_activation();
	void set_activation(const MLPPActivation::ActivationFunction val);

	MLPPCost::CostTypes get_cost();
	void set_cost(const MLPPCost::CostTypes val);

	Ref<MLPPMatrix> get_input();
	void set_input(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_weights();
	void set_weights(const Ref<MLPPVector> &val);

	real_t get_bias();
	void set_bias(const real_t val);

	Ref<MLPPVector> get_z();
	void set_z(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_a();
	void set_a(const Ref<MLPPVector> &val);

	real_t get_z_test();
	void set_z_test(const real_t val);

	real_t get_a_test();
	void set_a_test(const real_t val);

	Ref<MLPPVector> get_delta();
	void set_delta(const Ref<MLPPVector> &val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);

	MLPPUtilities::WeightDistributionType get_weight_init();
	void set_weight_init(const MLPPUtilities::WeightDistributionType val);

	bool is_initialized();
	void initialize();

	void forward_pass();
	void test(const Ref<MLPPVector> &x);

	MLPPOutputLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, MLPPCost::CostTypes p_cost, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha);

	MLPPOutputLayer();
	~MLPPOutputLayer();

protected:
	static void _bind_methods();

	int _n_hidden;
	MLPPActivation::ActivationFunction _activation;
	MLPPCost::CostTypes _cost;

	Ref<MLPPMatrix> _input;

	Ref<MLPPVector> _weights;
	real_t _bias;

	Ref<MLPPVector> _z;
	Ref<MLPPVector> _a;

	real_t _z_test;
	real_t _a_test;

	Ref<MLPPVector> _delta;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda; /* Regularization Parameter */
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	MLPPUtilities::WeightDistributionType _weight_init;

	bool _initialized;
};

#endif /* OutputLayer_hpp */
