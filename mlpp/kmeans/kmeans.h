#ifndef MLPP_K_MEANS_H
#define MLPP_K_MEANS_H

/*************************************************************************/
/*  kmeans.h                                                             */
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

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPKMeans : public Reference {
	GDCLASS(MLPPKMeans, Reference);

public:
	enum MeanType {
		MEAN_TYPE_CENTROID = 0,
		MEAN_TYPE_KMEANSPP,
	};

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	int get_k();
	void set_k(const int val);

	MeanType get_mean_type();
	void set_mean_type(const MeanType val);

	void initialize();

	Ref<MLPPMatrix> model_set_test(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> model_test(const Ref<MLPPVector> &x);
	void train(int epoch_num, bool UI = false);
	real_t score();
	Ref<MLPPVector> silhouette_scores();

	MLPPKMeans();
	~MLPPKMeans();

protected:
	void _evaluate();
	void _compute_mu();

	void _centroid_initialization();
	void _kmeanspp_initialization();
	real_t _cost();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPMatrix> _mu;
	Ref<MLPPMatrix> _r;

	real_t _accuracy_threshold;
	int _k;
	bool _initialized;

	MeanType _mean_type;
};

VARIANT_ENUM_CAST(MLPPKMeans::MeanType);

#endif /* KMeans_hpp */
