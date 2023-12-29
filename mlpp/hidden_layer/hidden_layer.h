#ifndef MLPP_HIDDEN_LAYER_H
#define MLPP_HIDDEN_LAYER_H

/*************************************************************************/
/*  hidden_layer.h                                                       */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2022-present Péter Magyar.                              */
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

#include "core/math/math_defs.h"
#include "core/string/ustring.h"

#include "core/object/reference.h"

#include "../activation/activation.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <map>
#include <string>
#include <vector>

class MLPPHiddenLayer : public Reference {
	GDCLASS(MLPPHiddenLayer, Reference);

public:
	int get_n_hidden() const;
	void set_n_hidden(const int val);

	MLPPActivation::ActivationFunction get_activation() const;
	void set_activation(const MLPPActivation::ActivationFunction val);

	Ref<MLPPMatrix> get_input();
	void set_input(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_weights();
	void set_weights(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_bias();
	void set_bias(const Ref<MLPPVector> &val);

	Ref<MLPPMatrix> get_z();
	void set_z(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_a();
	void set_a(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_z_test();
	void set_z_test(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_a_test();
	void set_a_test(const Ref<MLPPVector> &val);

	Ref<MLPPMatrix> get_delta();
	void set_delta(const Ref<MLPPMatrix> &val);

	MLPPReg::RegularizationType get_reg() const;
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda() const;
	void set_lambda(const real_t val);

	real_t get_alpha() const;
	void set_alpha(const real_t val);

	MLPPUtilities::WeightDistributionType get_weight_init() const;
	void set_weight_init(const MLPPUtilities::WeightDistributionType val);

	bool is_initialized();
	void initialize();

	void forward_pass();
	void test(const Ref<MLPPVector> &x);

	MLPPHiddenLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha);

	MLPPHiddenLayer();
	~MLPPHiddenLayer();

protected:
	static void _bind_methods();

	int _n_hidden;
	MLPPActivation::ActivationFunction _activation;

	Ref<MLPPMatrix> _input;

	Ref<MLPPMatrix> _weights;
	Ref<MLPPVector> _bias;

	Ref<MLPPMatrix> _z;
	Ref<MLPPMatrix> _a;

	Ref<MLPPVector> _z_test;
	Ref<MLPPVector> _a_test;

	Ref<MLPPMatrix> _delta;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda; /* Regularization Parameter */
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	MLPPUtilities::WeightDistributionType _weight_init;

	bool _initialized;
};

#endif /* HiddenLayer_hpp */