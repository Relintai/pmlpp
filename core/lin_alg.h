#ifndef MLPP_LIN_ALG_H
#define MLPP_LIN_ALG_H

/*************************************************************************/
/*  lin_alg.h                                                            */
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

//TODO Methods here should probably use error macros in a way where they get disabled in non-tools(?) (maybe release?) builds

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/math/math_defs.h"

#include "core/object/reference.h"
#endif

#include "../core/mlpp_matrix.h"
#include "../core/mlpp_vector.h"

#include <tuple>
#include <vector>

class MLPPLinAlg : public Reference {
	GDCLASS(MLPPLinAlg, Reference);

public:
	// MATRIX FUNCTIONS

	Ref<MLPPMatrix> gram_matrix(const Ref<MLPPMatrix> &A);
	bool linear_independence_checker(const Ref<MLPPMatrix> &A);

	Ref<MLPPMatrix> gaussian_noise(int n, int m);

	Ref<MLPPMatrix> additionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> subtractionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> matmultnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	Ref<MLPPMatrix> hadamard_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> kronecker_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> division_element_wisenvnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	Ref<MLPPMatrix> transposenm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> scalar_multiplynm(real_t scalar, const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> scalar_addnm(real_t scalar, const Ref<MLPPMatrix> &A);

	Ref<MLPPMatrix> lognm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> log10nm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> expnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> erfnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> exponentiatenm(const Ref<MLPPMatrix> &A, real_t p);
	Ref<MLPPMatrix> sqrtnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> cbrtnm(const Ref<MLPPMatrix> &A);

	//std::vector<std::vector<real_t>> matrixPower(std::vector<std::vector<real_t>> A, int n);

	Ref<MLPPMatrix> absnm(const Ref<MLPPMatrix> &A);

	real_t detm(const Ref<MLPPMatrix> &A, int d);

	//real_t trace(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> cofactornm(const Ref<MLPPMatrix> &A, int n, int i, int j);
	Ref<MLPPMatrix> adjointnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> inversenm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> pinversenm(const Ref<MLPPMatrix> &A);

	Ref<MLPPMatrix> zeromatnm(int n, int m);
	Ref<MLPPMatrix> onematnm(int n, int m);
	Ref<MLPPMatrix> fullnm(int n, int m, int k);

	Ref<MLPPMatrix> sinnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> cosnm(const Ref<MLPPMatrix> &A);

	Ref<MLPPMatrix> maxnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	//real_t max(std::vector<std::vector<real_t>> A);
	//real_t min(std::vector<std::vector<real_t>> A);

	//std::vector<std::vector<real_t>> round(std::vector<std::vector<real_t>> A);

	//real_t norm_2(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> identitym(int d);

	Ref<MLPPMatrix> covnm(const Ref<MLPPMatrix> &A);

	struct EigenResult {
		Ref<MLPPMatrix> eigen_vectors;
		Ref<MLPPMatrix> eigen_values;
	};

	EigenResult eigen(Ref<MLPPMatrix> A);

	struct SVDResult {
		Ref<MLPPMatrix> U;
		Ref<MLPPMatrix> S;
		Ref<MLPPMatrix> Vt;
	};

	SVDResult svd(const Ref<MLPPMatrix> &A);

	Ref<MLPPVector> vector_projection(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	Ref<MLPPMatrix> gram_schmidt_process(const Ref<MLPPMatrix> &A);

	struct QRDResult {
		Ref<MLPPMatrix> Q;
		Ref<MLPPMatrix> R;
	};

	QRDResult qrd(const Ref<MLPPMatrix> &A);

	struct CholeskyResult {
		Ref<MLPPMatrix> L;
		Ref<MLPPMatrix> Lt;
	};

	CholeskyResult cholesky(const Ref<MLPPMatrix> &A);

	//real_t sum_elements(std::vector<std::vector<real_t>> A);

	Ref<MLPPVector> flattenvvnv(const Ref<MLPPMatrix> &A);
	Ref<MLPPVector> solve(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);

	bool positive_definite_checker(const Ref<MLPPMatrix> &A);
	bool negative_definite_checker(const Ref<MLPPMatrix> &A);

	bool zero_eigenvalue(const Ref<MLPPMatrix> &A);

	// VECTOR FUNCTIONS

	Ref<MLPPVector> flattenmnv(const Vector<Ref<MLPPVector>> &A);

	Ref<MLPPVector> hadamard_productnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void hadamard_productv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	Ref<MLPPVector> division_element_wisenv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	Ref<MLPPVector> scalar_multiplynv(real_t scalar, const Ref<MLPPVector> &a);
	void scalar_multiplyv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out);

	Ref<MLPPVector> scalar_addnv(real_t scalar, const Ref<MLPPVector> &a);
	void scalar_addv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out);

	Ref<MLPPVector> additionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void additionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	Ref<MLPPVector> subtractionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void subtractionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	Ref<MLPPVector> lognv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> log10nv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> expnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> erfnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> exponentiatenv(const Ref<MLPPVector> &a, real_t p);
	Ref<MLPPVector> sqrtnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> cbrtnv(const Ref<MLPPVector> &a);

	real_t dotnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	//std::vector<real_t> cross(std::vector<real_t> a, std::vector<real_t> b);

	Ref<MLPPVector> absv(const Ref<MLPPVector> &a);

	Ref<MLPPVector> zerovecnv(int n);
	Ref<MLPPVector> onevecnv(int n);
	Ref<MLPPVector> fullnv(int n, int k);

	Ref<MLPPVector> sinnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> cosnv(const Ref<MLPPVector> &a);

	Ref<MLPPVector> maxnvv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	real_t maxvr(const Ref<MLPPVector> &a);
	real_t minvr(const Ref<MLPPVector> &a);

	//std::vector<real_t> round(std::vector<real_t> a);

	real_t euclidean_distance(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	real_t euclidean_distance_squared(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	/*
	real_t norm_2(std::vector<real_t> a);
	*/

	real_t norm_sqv(const Ref<MLPPVector> &a);

	real_t sum_elementsv(const Ref<MLPPVector> &a);

	//real_t cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b);

	// MATRIX-VECTOR FUNCTIONS
	Ref<MLPPVector> mat_vec_multnv(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);
	Ref<MLPPVector> subtract_matrix_rowsnv(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B);

	Ref<MLPPMatrix> outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b); // This multiplies a, bT
	Ref<MLPPMatrix> mat_vec_addnm(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);
	Ref<MLPPMatrix> diagnm(const Ref<MLPPVector> &a);

	// TENSOR FUNCTIONS
	Vector<Ref<MLPPMatrix>> additionnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	void division_element_wisevt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);
	Vector<Ref<MLPPMatrix>> division_element_wisenvnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	Vector<Ref<MLPPMatrix>> sqrtnvt(const Vector<Ref<MLPPMatrix>> &A);

	Vector<Ref<MLPPMatrix>> exponentiatenvt(const Vector<Ref<MLPPMatrix>> &A, real_t p);

	//std::vector<std::vector<real_t>> tensor_vec_mult(std::vector<std::vector<std::vector<real_t>>> A, std::vector<real_t> b);

	//std::vector<real_t> flatten(std::vector<std::vector<std::vector<real_t>>> A);

	Vector<Ref<MLPPMatrix>> scalar_multiplynvt(real_t scalar, Vector<Ref<MLPPMatrix>> A);
	Vector<Ref<MLPPMatrix>> scalar_addnvt(real_t scalar, Vector<Ref<MLPPMatrix>> A);

	void resizevt(Vector<Ref<MLPPMatrix>> &r_target, const Vector<Ref<MLPPMatrix>> &A);
	Vector<Ref<MLPPMatrix>> resizencvt(const Vector<Ref<MLPPMatrix>> &A);

	//std::vector<std::vector<std::vector<real_t>>> hadamard_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	Vector<Ref<MLPPMatrix>> maxnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);
	Vector<Ref<MLPPMatrix>> absnvt(const Vector<Ref<MLPPMatrix>> &A);

	//real_t norm_2(std::vector<std::vector<std::vector<real_t>>> A);

	//std::vector<std::vector<std::vector<real_t>>> vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B);

protected:
	static void _bind_methods();
};

#endif /* LinAlg_hpp */