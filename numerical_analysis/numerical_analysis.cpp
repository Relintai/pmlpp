/*************************************************************************/
/*  numerical_analysis.cpp                                               */
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

#include "numerical_analysis.h"

#include "../lin_alg/lin_alg.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_tensor3.h"
#include "../lin_alg/mlpp_vector.h"

real_t MLPPNumericalAnalysis::num_diffr(real_t (*function)(real_t), real_t x) {
	real_t eps = 1e-10;
	return (function(x + eps) - function(x)) / eps; // This is just the formal def. of the derivative.
}

real_t MLPPNumericalAnalysis::num_diff_2r(real_t (*function)(real_t), real_t x) {
	real_t eps = 1e-5;
	return (function(x + 2 * eps) - 2 * function(x + eps) + function(x)) / (eps * eps);
}

real_t MLPPNumericalAnalysis::num_diff_3r(real_t (*function)(real_t), real_t x) {
	real_t eps = 1e-5;
	real_t t1 = function(x + 3 * eps) - 2 * function(x + 2 * eps) + function(x + eps);
	real_t t2 = function(x + 2 * eps) - 2 * function(x + eps) + function(x);
	return (t1 - t2) / (eps * eps * eps);
}

real_t MLPPNumericalAnalysis::constant_approximationr(real_t (*function)(real_t), real_t c) {
	return function(c);
}

real_t MLPPNumericalAnalysis::linear_approximationr(real_t (*function)(real_t), real_t c, real_t x) {
	return constant_approximationr(function, c) + num_diffr(function, c) * (x - c);
}

real_t MLPPNumericalAnalysis::quadratic_approximationr(real_t (*function)(real_t), real_t c, real_t x) {
	return linear_approximationr(function, c, x) + 0.5 * num_diff_2r(function, c) * (x - c) * (x - c);
}

real_t MLPPNumericalAnalysis::cubic_approximationr(real_t (*function)(real_t), real_t c, real_t x) {
	return quadratic_approximationr(function, c, x) + (1 / 6) * num_diff_3r(function, c) * (x - c) * (x - c) * (x - c);
}

real_t MLPPNumericalAnalysis::num_diffv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x, int axis) {
	// For multivariable function analysis.
	// This will be used for calculating Jacobian vectors.
	// Diffrentiate with respect to indicated axis. (0, 1, 2 ...)
	real_t eps = 1e-10;
	Ref<MLPPVector> x_eps = x->duplicate_fast();
	x_eps->element_get_ref(axis) += eps;

	return (function(x_eps) - function(x)) / eps;
}

real_t MLPPNumericalAnalysis::num_diff_2v(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x, int axis1, int axis2) {
	//For Hessians.
	real_t eps = 1e-5;

	Ref<MLPPVector> x_pp = x->duplicate_fast();
	x_pp->element_get_ref(axis1) += eps;
	x_pp->element_get_ref(axis2) += eps;

	Ref<MLPPVector> x_np = x->duplicate_fast();
	x_np->element_get_ref(axis2) += eps;

	Ref<MLPPVector> x_pn = x->duplicate_fast();
	x_pn->element_get_ref(axis1) += eps;

	return (function(x_pp) - function(x_np) - function(x_pn) + function(x)) / (eps * eps);
}

real_t MLPPNumericalAnalysis::num_diff_3v(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x, int axis1, int axis2, int axis3) {
	// For third order derivative tensors.
	// NOTE: Approximations do not appear to be accurate for sinusodial functions...
	// Should revisit this later.
	real_t eps = 1e-5;

	Ref<MLPPVector> x_ppp = x->duplicate_fast();
	x_ppp->element_get_ref(axis1) += eps;
	x_ppp->element_get_ref(axis2) += eps;
	x_ppp->element_get_ref(axis3) += eps;

	Ref<MLPPVector> x_npp = x->duplicate_fast();
	x_npp->element_get_ref(axis2) += eps;
	x_npp->element_get_ref(axis3) += eps;

	Ref<MLPPVector> x_pnp = x->duplicate_fast();
	x_pnp->element_get_ref(axis1) += eps;
	x_pnp->element_get_ref(axis3) += eps;

	Ref<MLPPVector> x_nnp = x->duplicate_fast();
	x_nnp->element_get_ref(axis3) += eps;

	Ref<MLPPVector> x_ppn = x->duplicate_fast();
	x_ppn->element_get_ref(axis1) += eps;
	x_ppn->element_get_ref(axis2) += eps;

	Ref<MLPPVector> x_npn = x->duplicate_fast();
	x_npn->element_get_ref(axis2) += eps;

	Ref<MLPPVector> x_pnn = x->duplicate_fast();
	x_pnn->element_get_ref(axis1) += eps;

	real_t thirdAxis = function(x_ppp) - function(x_npp) - function(x_pnp) + function(x_nnp);
	real_t noThirdAxis = function(x_ppn) - function(x_npn) - function(x_pnn) + function(x);
	return (thirdAxis - noThirdAxis) / (eps * eps * eps);
}

real_t MLPPNumericalAnalysis::newton_raphson_method(real_t (*function)(real_t), real_t x_0, real_t epoch_num) {
	real_t x = x_0;
	for (int i = 0; i < epoch_num; i++) {
		x -= function(x) / num_diffr(function, x);
	}
	return x;
}

real_t MLPPNumericalAnalysis::halley_method(real_t (*function)(real_t), real_t x_0, real_t epoch_num) {
	real_t x = x_0;
	for (int i = 0; i < epoch_num; i++) {
		x -= ((2 * function(x) * num_diffr(function, x)) / (2 * num_diffr(function, x) * num_diffr(function, x) - function(x) * num_diff_2r(function, x)));
	}
	return x;
}

real_t MLPPNumericalAnalysis::inv_quadratic_interpolation(real_t (*function)(real_t), const Ref<MLPPVector> &x_0, int epoch_num) {
	real_t x = 0;
	Ref<MLPPVector> ct = x_0->duplicate_fast();
	MLPPVector &current_three = *(ct.ptr());

	for (int i = 0; i < epoch_num; i++) {
		real_t t1 = ((function(current_three[1]) * function(current_three[2])) / ((function(current_three[0]) - function(current_three[1])) * (function(current_three[0]) - function(current_three[2])))) * current_three[0];
		real_t t2 = ((function(current_three[0]) * function(current_three[2])) / ((function(current_three[1]) - function(current_three[0])) * (function(current_three[1]) - function(current_three[2])))) * current_three[1];
		real_t t3 = ((function(current_three[0]) * function(current_three[1])) / ((function(current_three[2]) - function(current_three[0])) * (function(current_three[2]) - function(current_three[1])))) * current_three[2];
		x = t1 + t2 + t3;

		current_three.remove(0);
		current_three.push_back(x);
	}
	return x;
}

real_t MLPPNumericalAnalysis::eulerian_methodr(real_t (*derivative)(real_t), real_t q_0, real_t q_1, real_t p, real_t h) {
	int max_epoch = static_cast<int>((p - q_0) / h);
	real_t x = q_0;
	real_t y = q_1;
	for (int i = 0; i < max_epoch; i++) {
		y = y + h * derivative(x);
		x += h;
	}
	return y;
}

real_t MLPPNumericalAnalysis::eulerian_methodv(real_t (*derivative)(const Ref<MLPPVector> &), real_t q_0, real_t q_1, real_t p, real_t h) {
	int max_epoch = static_cast<int>((p - q_0) / h);

	Ref<MLPPVector> v;
	v.instance();
	v->resize(2);

	real_t x = q_0;
	real_t y = q_1;
	for (int i = 0; i < max_epoch; i++) {
		v->element_set(0, x);
		v->element_set(1, y);
		y = y + h * derivative(v);
		x += h;
	}
	return y;
}

real_t MLPPNumericalAnalysis::growth_method(real_t C, real_t k, real_t t) {
	//dP/dt = kP
	//dP/P = kdt
	//integral(1/P)dP = integral(k) dt
	//ln|P| = kt + C_initial
	//|P| = e^(kt + C_initial)
	//|P| = e^(C_initial) * e^(kt)
	//P = +/- e^(C_initial) * e^(kt)
	//P = C * e^(kt)

	// auto growthFunction = [&C, &k](real_t t) { return C * exp(k * t); };
	return C * Math::exp(k * t);
}

Ref<MLPPVector> MLPPNumericalAnalysis::jacobian(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x) {
	Ref<MLPPVector> jacobian;
	jacobian.instance();
	jacobian->resize(x->size());

	int jacobian_size = jacobian->size();

	for (int i = 0; i < jacobian_size; ++i) {
		jacobian->element_set(i, num_diffv(function, x, i)); // Derivative w.r.t axis i evaluated at x. For all x_i.
	}

	return jacobian;
}

Ref<MLPPMatrix> MLPPNumericalAnalysis::hessian(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x) {
	Ref<MLPPMatrix> hessian;
	hessian.instance();
	hessian->resize(Size2i(x->size(), x->size()));

	Size2i hessian_size = hessian->size();

	for (int i = 0; i < hessian_size.y; i++) {
		for (int j = 0; j < hessian_size.x; j++) {
			hessian->element_set(i, j, num_diff_2v(function, x, i, j));
		}
	}

	return hessian;
}

Ref<MLPPTensor3> MLPPNumericalAnalysis::third_order_tensor(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x) {
	Ref<MLPPTensor3> tensor;
	tensor.instance();
	tensor->resize(Size3i(x->size(), x->size(), x->size()));

	Size3i tensor_size = tensor->size();

	for (int i = 0; i < tensor_size.z; i++) { // O(n^3) time complexity :(
		for (int j = 0; j < tensor_size.y; j++) {
			for (int k = 0; k < tensor_size.x; k++) {
				tensor->element_set(i, j, k, num_diff_3v(function, x, i, j, k));
			}
		}
	}

	return tensor;
}

Vector<Ref<MLPPMatrix>> MLPPNumericalAnalysis::third_order_tensorvt(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x) {
	int x_size = x->size();

	Vector<Ref<MLPPMatrix>> tensor;
	tensor.resize(x_size);

	for (int i = 0; i < tensor.size(); i++) {
		Ref<MLPPMatrix> m;
		m.instance();
		m->resize(Size2i(x_size, x_size));

		tensor.write[i] = m;
	}

	for (int i = 0; i < tensor.size(); i++) { // O(n^3) time complexity :(
		Ref<MLPPMatrix> m = tensor[i];

		for (int j = 0; j < x_size; j++) {
			for (int k = 0; k < x_size; k++) {
				m->element_set(j, k, num_diff_3v(function, x, i, j, k));
			}
		}
	}

	return tensor;
}

real_t MLPPNumericalAnalysis::constant_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c) {
	return function(c);
}

real_t MLPPNumericalAnalysis::linear_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c, const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;

	Ref<MLPPVector> j = jacobian(function, c);
	Ref<MLPPMatrix> mj;
	mj.instance();
	mj->row_add_mlpp_vector(j);

	Ref<MLPPVector> xsc = x->subn(c);
	Ref<MLPPMatrix> mxsc;
	mxsc.instance();
	mxsc->row_add_mlpp_vector(xsc);

	Ref<MLPPMatrix> m = mj->transposen()->multn(mxsc);

	return constant_approximationv(function, c) + m->element_get(0, 0);
}

real_t MLPPNumericalAnalysis::quadratic_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c, const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> h = hessian(function, c);

	Ref<MLPPVector> xsc = x->subn(c);
	Ref<MLPPMatrix> mxsc;
	mxsc.instance();
	mxsc->row_add_mlpp_vector(xsc);

	Ref<MLPPMatrix> r = mxsc->multn(h->multn(mxsc->transposen()));

	return linear_approximationv(function, c, x) + 0.5 * r->element_get(0, 0);
}

real_t MLPPNumericalAnalysis::cubic_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c, const Ref<MLPPVector> &x) {
	//Not completely sure as the literature seldom discusses the third order taylor approximation,
	//in particular for multivariate cases, but ostensibly, the matrix/tensor/vector multiplies
	//should look something like this:

	//(N x N x N) (N x 1) [tensor vector mult] => (N x N x 1) => (N x N)
	//Perform remaining multiplies as done for the 2nd order approximation.
	//Result is a scalar.

	MLPPLinAlg alg;

	Ref<MLPPVector> xsc = x->subn(c);
	Ref<MLPPMatrix> mxsc;
	mxsc.instance();
	mxsc->row_add_mlpp_vector(xsc);

	Ref<MLPPMatrix> result_mat = third_order_tensor(function, c)->tensor_vec_mult(xsc);
	real_t result_scalar = mxsc->multn(result_mat->multn(mxsc->transposen()))->element_get(0, 0);

	return quadratic_approximationv(function, c, x) + (1 / 6) * result_scalar;
}

real_t MLPPNumericalAnalysis::laplacian(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x) {
	Ref<MLPPMatrix> hessian_matrix = hessian(function, x);
	real_t laplacian = 0;

	Size2i hessian_matrix_size = hessian_matrix->size();

	for (int i = 0; i < hessian_matrix_size.y; i++) {
		laplacian += hessian_matrix->element_get(i, i); // homogenous 2nd derivs w.r.t i, then i
	}

	return laplacian;
}

String MLPPNumericalAnalysis::second_partial_derivative_test(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	Ref<MLPPMatrix> hessian_matrix = hessian(function, x);

	Size2i hessian_matrix_size = hessian_matrix->size();

	// The reason we do this is because the 2nd partial derivative test is less conclusive for functions of variables greater than
	// 2, and the calculations specific to the bivariate case are less computationally intensive.

	if (hessian_matrix_size.y == 2) {
		real_t det = hessian_matrix->det();
		real_t secondDerivative = num_diff_2v(function, x, 0, 0);
		if (secondDerivative > 0 && det > 0) {
			return "min";
		} else if (secondDerivative < 0 && det > 0) {
			return "max";
		} else if (det < 0) {
			return "saddle";
		} else {
			return "test was inconclusive";
		}
	} else {
		if (alg.positive_definite_checker(hessian_matrix)) {
			return "min";
		} else if (alg.negative_definite_checker(hessian_matrix)) {
			return "max";
		} else if (!alg.zero_eigenvalue(hessian_matrix)) {
			return "saddle";
		} else {
			return "test was inconclusive";
		}
	}
}

void MLPPNumericalAnalysis::_bind_methods() {
}
