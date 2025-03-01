#ifndef MLPP_MULTINOMIAL_NB_H
#define MLPP_MULTINOMIAL_NB_H

/*************************************************************************/
/*  multinomial_nb.h                                                     */
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
#include "core/containers/hash_map.h"
#include "core/containers/vector.h"
#include "core/math/math_defs.h"

#include "core/object/reference.h"
#endif

#include "../core/mlpp_matrix.h"
#include "../core/mlpp_vector.h"

class MLPPMultinomialNB : public Reference {
	GDCLASS(MLPPMultinomialNB, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	real_t get_class_num();
	void set_class_num(const real_t val);

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	real_t score();

	bool is_initialized();
	void initialize();

	MLPPMultinomialNB(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int class_num);

	MLPPMultinomialNB();
	~MLPPMultinomialNB();

protected:
	void compute_theta();
	void evaluate();

	static void _bind_methods();

	// Model Params
	Ref<MLPPVector> _priors;

	Vector<HashMap<real_t, int>> _theta;
	Ref<MLPPVector> _vocab;
	int _class_num;

	// Datasets
	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	Ref<MLPPVector> _y_hat;

	bool _initialized;
};

#endif /* MultinomialNB_hpp */
