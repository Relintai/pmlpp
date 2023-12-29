//
//  LinAlg.cpp
//
//  Created by Marc Melikyan on 1/8/21.
//

#include "lin_alg.h"

#include "core/math/math_funcs.h"

#include "../stat/stat.h"
#include <cmath>
#include <iostream>
#include <map>
#include <random>

/*
std::vector<std::vector<real_t>> MLPPLinAlg::gramMatrix(std::vector<std::vector<real_t>> A) {
	return matmult(transpose(A), A); // AtA
}
*/

/*
bool MLPPLinAlg::linearIndependenceChecker(std::vector<std::vector<real_t>> A) {
	if (det(gramMatrix(A), A.size()) == 0) {
		return false;
	}
	return true;
}
*/

Ref<MLPPMatrix> MLPPLinAlg::gaussian_noise(int n, int m) {
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<real_t> distribution(0, 1); // Standard normal distribution. Mean of 0, std of 1.

	Ref<MLPPMatrix> A;
	A.instance();
	A->resize(Size2i(m, n));

	int a_data_size = A->data_size();
	real_t *a_ptr = A->ptrw();

	for (int i = 0; i < a_data_size; ++i) {
		a_ptr[i] = distribution(generator);
	}

	return A;
}

Ref<MLPPMatrix> MLPPLinAlg::additionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND_V(!A.is_valid() || !B.is_valid(), Ref<MLPPMatrix>());
	Size2i a_size = A->size();
	ERR_FAIL_COND_V(a_size != B->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(a_size);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = a_ptr[i] + b_ptr[i];
	}

	return C;
}
Ref<MLPPMatrix> MLPPLinAlg::subtractionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND_V(!A.is_valid() || !B.is_valid(), Ref<MLPPMatrix>());
	Size2i a_size = A->size();
	ERR_FAIL_COND_V(a_size != B->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(a_size);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = a_ptr[i] - b_ptr[i];
	}

	return C;
}
Ref<MLPPMatrix> MLPPLinAlg::matmultnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND_V(!A.is_valid() || !B.is_valid(), Ref<MLPPMatrix>());

	Size2i a_size = A->size();
	Size2i b_size = B->size();

	ERR_FAIL_COND_V(a_size.x != b_size.y, Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(Size2i(b_size.x, a_size.y));
	C->fill(0);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < a_size.y; i++) {
		for (int k = 0; k < b_size.y; k++) {
			int ind_i_k = A->calculate_index(i, k);

			for (int j = 0; j < b_size.x; j++) {
				int ind_i_j = C->calculate_index(i, j);
				int ind_k_j = B->calculate_index(k, j);

				c_ptr[ind_i_j] += a_ptr[ind_i_k] * b_ptr[ind_k_j];

				//C->element_set(i, j, C->element_get(i, j) + A->element_get(i, k) * B->element_get(k, j
			}
		}
	}

	return C;
}

Ref<MLPPMatrix> MLPPLinAlg::hadamard_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND_V(!A.is_valid() || !B.is_valid(), Ref<MLPPMatrix>());
	Size2i a_size = A->size();
	ERR_FAIL_COND_V(a_size != B->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(a_size);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < a_size.y; i++) {
		for (int j = 0; j < a_size.x; j++) {
			int ind_i_j = A->calculate_index(i, j);
			c_ptr[ind_i_j] = a_ptr[ind_i_j] * b_ptr[ind_i_j];
		}
	}

	return C;
}
Ref<MLPPMatrix> MLPPLinAlg::kronecker_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	// [1,1,1,1]   [1,2,3,4,5]
	// [1,1,1,1]   [1,2,3,4,5]
	//             [1,2,3,4,5]

	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]

	// Resulting matrix: A.size() * B.size()
	//                   A[0].size() * B[0].size()

	ERR_FAIL_COND_V(!A.is_valid() || !B.is_valid(), Ref<MLPPMatrix>());
	Size2i a_size = A->size();
	Size2i b_size = B->size();

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(Size2i(b_size.x * a_size.x, b_size.y * a_size.y));

	const real_t *a_ptr = A->ptr();

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(b_size.x);

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < b_size.y; ++j) {
			B->row_get_into_mlpp_vector(j, row_tmp);

			Vector<Ref<MLPPVector>> row;
			for (int k = 0; k < a_size.x; ++k) {
				row.push_back(scalar_multiplynv(a_ptr[A->calculate_index(i, k)], row_tmp));
			}

			Ref<MLPPVector> flattened_row = flattenmnv(row);

			C->row_set_mlpp_vector(i * b_size.y + j, flattened_row);
		}
	}

	return C;
}
Ref<MLPPMatrix> MLPPLinAlg::division_element_wisenvnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND_V(!A.is_valid() || !B.is_valid(), Ref<MLPPMatrix>());
	Size2i a_size = A->size();
	ERR_FAIL_COND_V(a_size != B->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(a_size);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < a_size.y; i++) {
		for (int j = 0; j < a_size.x; j++) {
			int ind_i_j = A->calculate_index(i, j);
			c_ptr[ind_i_j] = a_ptr[ind_i_j] / b_ptr[ind_i_j];
		}
	}

	return C;
}

Ref<MLPPMatrix> MLPPLinAlg::transposenm(const Ref<MLPPMatrix> &A) {
	Size2i a_size = A->size();

	Ref<MLPPMatrix> AT;
	AT.instance();
	AT->resize(Size2i(a_size.y, a_size.x));

	const real_t *a_ptr = A->ptr();
	real_t *at_ptr = AT->ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			at_ptr[AT->calculate_index(j, i)] = a_ptr[A->calculate_index(i, j)];
		}
	}

	return AT;
}
Ref<MLPPMatrix> MLPPLinAlg::scalar_multiplynm(real_t scalar, const Ref<MLPPMatrix> &A) {
	Ref<MLPPMatrix> AN = A->duplicate_fast();
	Size2i a_size = AN->size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			an_ptr[AN->calculate_index(i, j)] *= scalar;
		}
	}

	return AN;
}

Ref<MLPPMatrix> MLPPLinAlg::scalar_addnm(real_t scalar, const Ref<MLPPMatrix> &A) {
	Ref<MLPPMatrix> AN = A->duplicate_fast();
	Size2i a_size = AN->size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			an_ptr[AN->calculate_index(i, j)] += scalar;
		}
	}

	return AN;
}

Ref<MLPPMatrix> MLPPLinAlg::lognm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::log(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::log10nm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::log10(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::expnm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::exp(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::erfnm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::erf(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::exponentiatenm(const Ref<MLPPMatrix> &A, real_t p) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::sqrtnm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::cbrtnm(const Ref<MLPPMatrix> &A) {
	return exponentiatenm(A, real_t(1) / real_t(3));
}

/*
std::vector<std::vector<real_t>> MLPPLinAlg::matrixPower(std::vector<std::vector<real_t>> A, int n) {
	std::vector<std::vector<real_t>> B = identity(A.size());
	if (n == 0) {
		return identity(A.size());
	} else if (n < 0) {
		A = inverse(A);
	}
	for (int i = 0; i < std::abs(n); i++) {
		B = matmult(B, A);
	}
	return B;
}
*/

Ref<MLPPMatrix> MLPPLinAlg::absnm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}

	return out;
}

real_t MLPPLinAlg::detm(const Ref<MLPPMatrix> &A, int d) {
	ERR_FAIL_COND_V(!A.is_valid(), 0);

	real_t deter = 0;
	Ref<MLPPMatrix> B;
	B.instance();
	B->resize(Size2i(d, d));
	B->fill(0);

	/* This is the base case in which the input is a 2x2 square matrix.
	Recursion is performed unless and until we reach this base case,
	such that we recieve a scalar as the result. */
	if (d == 2) {
		return A->element_get(0, 0) * A->element_get(1, 1) - A->element_get(0, 1) * A->element_get(1, 0);
	} else {
		for (int i = 0; i < d; i++) {
			int sub_i = 0;
			for (int j = 1; j < d; j++) {
				int sub_j = 0;
				for (int k = 0; k < d; k++) {
					if (k == i) {
						continue;
					}

					B->element_set(sub_i, sub_j, A->element_get(j, k));
					sub_j++;
				}
				sub_i++;
			}

			deter += Math::pow(static_cast<real_t>(-1), static_cast<real_t>(i)) * A->element_get(0, i) * detm(B, d - 1);
		}
	}

	return deter;
}

/*
real_t MLPPLinAlg::trace(std::vector<std::vector<real_t>> A) {
	real_t trace = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		trace += A[i][i];
	}
	return trace;
}
*/

Ref<MLPPMatrix> MLPPLinAlg::cofactornm(const Ref<MLPPMatrix> &A, int n, int i, int j) {
	Ref<MLPPMatrix> cof;
	cof.instance();
	cof->resize(A->size());

	int sub_i = 0;
	int sub_j = 0;

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			if (row != i && col != j) {
				cof->element_set(sub_i, sub_j++, A->element_get(row, col));

				if (sub_j == n - 1) {
					sub_j = 0;
					sub_i++;
				}
			}
		}
	}

	return cof;
}
Ref<MLPPMatrix> MLPPLinAlg::adjointnm(const Ref<MLPPMatrix> &A) {
	Ref<MLPPMatrix> adj;

	ERR_FAIL_COND_V(!A.is_valid(), adj);

	Size2i a_size = A->size();

	ERR_FAIL_COND_V(a_size.x != a_size.y, adj);

	//Resizing the initial adjoint matrix

	adj.instance();
	adj->resize(a_size);

	// Checking for the case where the given N x N matrix is a scalar
	if (a_size.y == 1) {
		adj->element_set(0, 0, 1);
		return adj;
	}

	if (a_size.y == 2) {
		adj->element_set(0, 0, A->element_get(1, 1));
		adj->element_set(1, 1, A->element_get(0, 0));

		adj->element_set(0, 1, -A->element_get(0, 1));
		adj->element_set(1, 0, -A->element_get(1, 0));

		return adj;
	}

	for (int i = 0; i < a_size.y; i++) {
		for (int j = 0; j < a_size.x; j++) {
			Ref<MLPPMatrix> cof = cofactornm(A, a_size.y, i, j);
			// 1 if even, -1 if odd
			int sign = (i + j) % 2 == 0 ? 1 : -1;
			adj->element_set(j, i, sign * detm(cof, int(a_size.y) - 1));
		}
	}
	return adj;
}
Ref<MLPPMatrix> MLPPLinAlg::inversenm(const Ref<MLPPMatrix> &A) {
	return scalar_multiplynm(1 / detm(A, int(A->size().y)), adjointnm(A));
}
Ref<MLPPMatrix> MLPPLinAlg::pinversenm(const Ref<MLPPMatrix> &A) {
	return matmultnm(inversenm(matmultnm(transposenm(A), A)), transposenm(A));
}
Ref<MLPPMatrix> MLPPLinAlg::zeromatnm(int n, int m) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(0);

	return mat;
}
Ref<MLPPMatrix> MLPPLinAlg::onematnm(int n, int m) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(1);

	return mat;
}
Ref<MLPPMatrix> MLPPLinAlg::fullnm(int n, int m, int k) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(k);

	return mat;
}

Ref<MLPPMatrix> MLPPLinAlg::sinnm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::sin(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::cosnm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = Math::cos(a_ptr[i]);
	}

	return out;
}

Ref<MLPPVector> MLPPLinAlg::maxnvv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	Ref<MLPPVector> ret;
	ret.instance();

	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), ret);

	int a_size = a->size();

	ERR_FAIL_COND_V(a_size != b->size(), ret);

	ret->resize(a_size);

	const real_t *aa = a->ptr();
	const real_t *ba = b->ptr();
	real_t *ret_ptr = ret->ptrw();

	for (int i = 0; i < a_size; i++) {
		real_t aa_i = aa[i];
		real_t bb_i = ba[i];

		if (aa_i > bb_i) {
			ret_ptr[i] = aa_i;
		} else {
			ret_ptr[i] = bb_i;
		}
	}

	return ret;
}

/*
real_t MLPPLinAlg::max(std::vector<std::vector<real_t>> A) {
	return max(flatten(A));
}

real_t MLPPLinAlg::min(std::vector<std::vector<real_t>> A) {
	return min(flatten(A));
}

std::vector<std::vector<real_t>> MLPPLinAlg::round(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = Math::round(A[i][j]);
		}
	}
	return B;
}
*/

/*
real_t MLPPLinAlg::norm_2(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j] * A[i][j];
		}
	}
	return Math::sqrt(sum);
}
*/

Ref<MLPPMatrix> MLPPLinAlg::identitym(int d) {
	Ref<MLPPMatrix> identity_mat;
	identity_mat.instance();
	identity_mat->resize(Size2i(d, d));
	identity_mat->fill(0);

	real_t *im_ptr = identity_mat->ptrw();

	for (int i = 0; i < d; i++) {
		im_ptr[identity_mat->calculate_index(i, i)] = 1;
	}

	return identity_mat;
}

Ref<MLPPMatrix> MLPPLinAlg::covnm(const Ref<MLPPMatrix> &A) {
	MLPPStat stat;

	Ref<MLPPMatrix> cov_mat;
	cov_mat.instance();

	Size2i a_size = A->size();

	cov_mat->resize(a_size);

	Ref<MLPPVector> a_i_row_tmp;
	a_i_row_tmp.instance();
	a_i_row_tmp->resize(a_size.x);

	Ref<MLPPVector> a_j_row_tmp;
	a_j_row_tmp.instance();
	a_j_row_tmp->resize(a_size.x);

	for (int i = 0; i < a_size.y; ++i) {
		A->row_get_into_mlpp_vector(i, a_i_row_tmp);

		for (int j = 0; j < a_size.x; ++j) {
			A->row_get_into_mlpp_vector(j, a_j_row_tmp);

			cov_mat->element_set(i, j, stat.covariancev(a_i_row_tmp, a_j_row_tmp));
		}
	}

	return cov_mat;
}

MLPPLinAlg::EigenResult MLPPLinAlg::eigen(Ref<MLPPMatrix> A) {
	EigenResult res;

	ERR_FAIL_COND_V(!A.is_valid(), res);

	/*
	A (the entered parameter) in most use cases will be X'X, XX', etc. and must be symmetric.
	That simply means that 1) X' = X and 2) X is a square matrix. This function that computes the
	eigenvalues of a matrix is utilizing Jacobi's method.
	*/

	real_t diagonal = true; // Perform the iterative Jacobi algorithm unless and until we reach a diagonal matrix which yields us the eigenvals.

	HashMap<int, int> val_to_vec;
	Ref<MLPPMatrix> a_new;
	Ref<MLPPMatrix> eigenvectors = identitym(A->size().y);
	Size2i a_size = A->size();

	do {
		real_t a_ij = A->element_get(0, 1);
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (int i = 0; i < a_size.y; ++i) {
			for (int j = 0; j < a_size.x; ++j) {
				real_t ca_ij = A->element_get(i, j);
				real_t abs_ca_ij = ABS(ca_ij);

				if (i != j && abs_ca_ij > a_ij) {
					a_ij = ca_ij;
					sub_i = i;
					sub_j = j;
				} else if (i != j && abs_ca_ij == a_ij) {
					if (i < sub_i) {
						a_ij = ca_ij;
						sub_i = i;
						sub_j = j;
					}
				}
			}
		}

		real_t a_ii = A->element_get(sub_i, sub_i);
		real_t a_jj = A->element_get(sub_j, sub_j);
		//real_t a_ji = A->element_get(sub_j, sub_i);
		real_t theta;

		if (a_ii == a_jj) {
			theta = Math_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		Ref<MLPPMatrix> P = identitym(A->size().y);
		P->element_set(sub_i, sub_j, -Math::sin(theta));
		P->element_set(sub_i, sub_i, Math::cos(theta));
		P->element_set(sub_j, sub_j, Math::cos(theta));
		P->element_set(sub_j, sub_i, Math::sin(theta));

		a_new = matmultnm(matmultnm(inversenm(P), A), P);

		Size2i a_new_size = a_new->size();

		for (int i = 0; i < a_new_size.y; ++i) {
			for (int j = 0; j < a_new_size.x; ++j) {
				if (i != j && Math::is_zero_approx(Math::round(a_new->element_get(i, j)))) {
					a_new->element_set(i, j, 0);
				}
			}
		}

		bool non_zero = false;
		for (int i = 0; i < a_new_size.y; ++i) {
			for (int j = 0; j < a_new_size.x; ++j) {
				if (i != j && Math::is_zero_approx(Math::round(a_new->element_get(i, j)))) {
					non_zero = true;
				}
			}
		}

		if (non_zero) {
			diagonal = false;
		} else {
			diagonal = true;
		}

		if (a_new->is_equal_approx(A)) {
			diagonal = true;
			for (int i = 0; i < a_new_size.y; ++i) {
				for (int j = 0; j < a_new_size.x; ++j) {
					if (i != j) {
						a_new->element_set(i, j, 0);
					}
				}
			}
		}

		eigenvectors = matmultnm(eigenvectors, P);
		A = a_new;

	} while (!diagonal);

	Ref<MLPPMatrix> a_new_prior = a_new->duplicate_fast();

	Size2i a_new_size = a_new->size();

	// Bubble Sort. Should change this later.
	for (int i = 0; i < a_new_size.y - 1; ++i) {
		for (int j = 0; j < a_new_size.x - 1 - i; ++j) {
			if (a_new->element_get(j, j) < a_new->element_get(j + 1, j + 1)) {
				real_t temp = a_new->element_get(j + 1, j + 1);
				a_new->element_set(j + 1, j + 1, a_new->element_get(j, j));
				a_new->element_set(j, j, temp);
			}
		}
	}

	for (int i = 0; i < a_new_size.y; ++i) {
		for (int j = 0; j < a_new_size.x; ++j) {
			if (a_new->element_get(i, i) == a_new_prior->element_get(j, j)) {
				val_to_vec[i] = j;
			}
		}
	}

	Ref<MLPPMatrix> eigen_temp = eigenvectors->duplicate_fast();

	Size2i eigenvectors_size = eigenvectors->size();

	for (int i = 0; i < eigenvectors_size.y; ++i) {
		for (int j = 0; j < eigenvectors_size.x; ++j) {
			eigenvectors->element_set(i, j, eigen_temp->element_get(i, val_to_vec[j]));
		}
	}

	res.eigen_vectors = eigenvectors;
	res.eigen_values = a_new;

	return res;
}

MLPPLinAlg::SVDResult MLPPLinAlg::svd(const Ref<MLPPMatrix> &A) {
	SVDResult res;

	ERR_FAIL_COND_V(!A.is_valid(), res);

	Size2i a_size = A->size();

	EigenResult left_eigen = eigen(matmultnm(A, transposenm(A)));
	EigenResult right_eigen = eigen(matmultnm(transposenm(A), A));

	Ref<MLPPMatrix> singularvals = sqrtnm(left_eigen.eigen_values);
	Ref<MLPPMatrix> sigma = zeromatnm(a_size.y, a_size.x);

	Size2i sigma_size = sigma->size();

	for (int i = 0; i < sigma_size.y; ++i) {
		for (int j = 0; j < sigma_size.x; ++j) {
			sigma->element_set(i, j, singularvals->element_get(i, j));
		}
	}

	res.U = left_eigen.eigen_vectors;
	res.S = sigma;
	res.Vt = right_eigen.eigen_vectors;

	return res;
}

Ref<MLPPVector> MLPPLinAlg::vector_projection(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	real_t product = a->dot(b) / a->dot(a);

	return a->scalar_multiplyn(product); // Projection of vector a onto b. Denotated as proj_a(b).
}

Ref<MLPPMatrix> MLPPLinAlg::gram_schmidt_process(const Ref<MLPPMatrix> &p_A) {
	Ref<MLPPMatrix> A = p_A->transposen();
	Size2i a_size = A->size();

	Ref<MLPPMatrix> B;
	B.instance();
	B->resize(a_size);
	B->fill(0);

	Ref<MLPPVector> b_i_row_tmp;
	b_i_row_tmp.instance();
	b_i_row_tmp->resize(a_size.x);

	A->row_get_into_mlpp_vector(0, b_i_row_tmp);
	b_i_row_tmp->scalar_multiply((real_t)1 / b_i_row_tmp->norm_2());
	B->row_set_mlpp_vector(0, b_i_row_tmp);

	Ref<MLPPVector> a_i_row_tmp;
	a_i_row_tmp.instance();
	a_i_row_tmp->resize(a_size.x);

	Ref<MLPPVector> b_j_row_tmp;
	b_j_row_tmp.instance();
	b_j_row_tmp->resize(a_size.x);

	for (int i = 1; i < a_size.y; ++i) {
		A->row_get_into_mlpp_vector(i, b_i_row_tmp);
		B->row_set_mlpp_vector(i, b_i_row_tmp);

		for (int j = i - 1; j >= 0; j--) {
			A->row_get_into_mlpp_vector(i, a_i_row_tmp);
			B->row_get_into_mlpp_vector(j, b_j_row_tmp);
			B->row_get_into_mlpp_vector(i, b_i_row_tmp);

			b_i_row_tmp->sub(vector_projection(b_j_row_tmp, a_i_row_tmp));

			B->row_set_mlpp_vector(i, b_i_row_tmp);
		}

		// Very simply multiply all elements of vec B[i] by 1/||B[i]||_2
		B->row_get_into_mlpp_vector(i, b_i_row_tmp);
		b_i_row_tmp->scalar_multiply((real_t)1 / b_i_row_tmp->norm_2());
		B->row_set_mlpp_vector(i, b_i_row_tmp);
	}

	return B->transposen(); // We re-transpose the marix.
}

MLPPLinAlg::QRDResult MLPPLinAlg::qrd(const Ref<MLPPMatrix> &A) {
	QRDResult res;

	res.Q = gram_schmidt_process(A);
	res.R = res.Q->transposen()->multn(A);

	return res;
}

MLPPLinAlg::CholeskyResult MLPPLinAlg::cholesky(const Ref<MLPPMatrix> &A) {
	Size2i a_size = A->size();

	CholeskyResult res;

	ERR_FAIL_COND_V(a_size.x != a_size.y, res);

	Ref<MLPPMatrix> L = zeromatnm(a_size.y, a_size.x);

	for (int j = 0; j < a_size.y; ++j) { // Matrices entered must be square. No problem here.
		for (int i = j; i < a_size.y; ++i) {
			if (i == j) {
				real_t sum = 0;

				for (int k = 0; k < j; k++) {
					real_t lik = L->element_get(i, k);

					sum += lik * lik;
				}

				L->element_set(i, j, Math::sqrt(A->element_get(i, j) - sum));
			} else { // That is, i!=j
				real_t sum = 0;

				for (int k = 0; k < j; k++) {
					sum += L->element_get(i, k) * L->element_get(j, k);
				}

				L->element_set(i, j, (A->element_get(i, j) - sum) / L->element_get(j, j));
			}
		}
	}

	res.L = L;
	res.Lt = L->transposen(); // Indeed, L.T is our upper triangular matrix.

	return res;
}

/*
real_t MLPPLinAlg::sum_elements(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j];
		}
	}
	return sum;
}
*/

Ref<MLPPVector> MLPPLinAlg::flattenvvnv(const Ref<MLPPMatrix> &A) {
	int data_size = A->data_size();

	Ref<MLPPVector> res;
	res.instance();
	res->resize(data_size);

	real_t *res_ptr = res->ptrw();
	const real_t *a_ptr = A->ptr();

	for (int i = 0; i < data_size; ++i) {
		res_ptr[i] = a_ptr[i];
	}

	return res;
}

Ref<MLPPVector> MLPPLinAlg::solve(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b) {
	return A->inverse()->mult_vec(b);
}

bool MLPPLinAlg::positive_definite_checker(const Ref<MLPPMatrix> &A) {
	EigenResult eig_result = eigen(A);

	Ref<MLPPMatrix> eigenvals = eig_result.eigen_values;
	Size2i eigenvals_size = eigenvals->size();

	for (int i = 0; i < eigenvals_size.y; ++i) {
		if (eigenvals->element_get(i, i) <= 0) { // Simply check to ensure all eigenvalues are positive.
			return false;
		}
	}

	return true;
}

bool MLPPLinAlg::negative_definite_checker(const Ref<MLPPMatrix> &A) {
	EigenResult eig_result = eigen(A);

	Ref<MLPPMatrix> eigenvals = eig_result.eigen_values;
	Size2i eigenvals_size = eigenvals->size();

	for (int i = 0; i < eigenvals_size.y; ++i) {
		if (eigenvals->element_get(i, i) >= 0) { // Simply check to ensure all eigenvalues are negative.
			return false;
		}
	}

	return true;
}

bool MLPPLinAlg::zero_eigenvalue(const Ref<MLPPMatrix> &A) {
	EigenResult eig_result = eigen(A);

	Ref<MLPPMatrix> eigenvals = eig_result.eigen_values;
	Size2i eigenvals_size = eigenvals->size();

	for (int i = 0; i < eigenvals_size.y; ++i) {
		if (eigenvals->element_get(i, i) == 0) { // TODO should it use is_equal_approx?
			return false;
		}
	}

	return true;
}

Ref<MLPPVector> MLPPLinAlg::flattenmnv(const Vector<Ref<MLPPVector>> &A) {
	Ref<MLPPVector> a;
	a.instance();

	int vsize = 0;
	for (int i = 0; i < A.size(); ++i) {
		vsize += A[i]->size();
	}

	a->resize(vsize);

	int a_index = 0;
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < A.size(); ++i) {
		const Ref<MLPPVector> &r = A[i];

		int r_size = r->size();
		const real_t *r_ptr = r->ptr();

		for (int j = 0; j < r_size; ++j) {
			a_ptr[a_index] = r_ptr[j];
			++a_index;
		}
	}

	return a;
}

Ref<MLPPVector> MLPPLinAlg::hadamard_productnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();

	ERR_FAIL_COND_V(size != b->size(), Ref<MLPPVector>());

	out->resize(size);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] * b_ptr[i];
	}

	return out;
}
void MLPPLinAlg::hadamard_productv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid() || !out.is_valid());

	int size = a->size();

	ERR_FAIL_COND(size != b->size());

	if (unlikely(out->size() != size)) {
		out->resize(size);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] * b_ptr[i];
	}
}

Ref<MLPPVector> MLPPLinAlg::division_element_wisenv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();

	ERR_FAIL_COND_V(size != b->size(), Ref<MLPPVector>());

	out->resize(size);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] / b_ptr[i];
	}

	return out;
}

Ref<MLPPVector> MLPPLinAlg::scalar_multiplynv(real_t scalar, const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();

	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] * scalar;
	}

	return out;
}
void MLPPLinAlg::scalar_multiplyv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out) {
	ERR_FAIL_COND(!a.is_valid() || !out.is_valid());

	int size = a->size();

	if (unlikely(out->size() != size)) {
		out->resize(size);
	}

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] * scalar;
	}
}

Ref<MLPPVector> MLPPLinAlg::scalar_addnv(real_t scalar, const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();

	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] + scalar;
	}

	return out;
}
void MLPPLinAlg::scalar_addv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out) {
	ERR_FAIL_COND(!a.is_valid() || !out.is_valid());

	int size = a->size();

	if (unlikely(out->size() != size)) {
		out->resize(size);
	}

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] + scalar;
	}
}

Ref<MLPPVector> MLPPLinAlg::additionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), Ref<MLPPVector>());

	int size = a->size();

	ERR_FAIL_COND_V(size != b->size(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] + b_ptr[i];
	}

	return out;
}
void MLPPLinAlg::additionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid() || !out.is_valid());

	int size = a->size();

	ERR_FAIL_COND(size != b->size());

	if (unlikely(out->size() != size)) {
		out->resize(size);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] + b_ptr[i];
	}
}

Ref<MLPPVector> MLPPLinAlg::subtractionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), Ref<MLPPVector>());

	int size = a->size();

	ERR_FAIL_COND_V(size != b->size(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	if (unlikely(size == 0)) {
		return out;
	}

	out->resize(size);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] - b_ptr[i];
	}

	return out;
}
void MLPPLinAlg::subtractionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid() || !out.is_valid());

	int size = a->size();

	ERR_FAIL_COND(size != b->size());

	if (unlikely(out->size() != size)) {
		out->resize(size);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = a_ptr[i] - b_ptr[i];
	}
}

Ref<MLPPVector> MLPPLinAlg::lognv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::log(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::log10nv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::log10(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::expnv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::exp(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::erfnv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::erf(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::exponentiatenv(const Ref<MLPPVector> &a, real_t p) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::sqrtnv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::cbrtnv(const Ref<MLPPVector> &a) {
	return exponentiatenv(a, static_cast<real_t>(1) / static_cast<real_t>(3));
}

real_t MLPPLinAlg::dotnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	int a_size = a->size();

	ERR_FAIL_COND_V(a_size != b->size(), 0);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();

	real_t c = 0;
	for (int i = 0; i < a_size; ++i) {
		c += a_ptr[i] * b_ptr[i];
	}
	return c;
}

/*
std::vector<real_t> MLPPLinAlg::cross(std::vector<real_t> a, std::vector<real_t> b) {
	// Cross products exist in R^7 also. Though, I will limit it to R^3 as Wolfram does this.
	std::vector<std::vector<real_t>> mat = { onevec(3), a, b };

	real_t det1 = det({ { a[1], a[2] }, { b[1], b[2] } }, 2);
	real_t det2 = -det({ { a[0], a[2] }, { b[0], b[2] } }, 2);
	real_t det3 = det({ { a[0], a[1] }, { b[0], b[1] } }, 2);

	return { det1, det2, det3 };
}
*/

Ref<MLPPVector> MLPPLinAlg::absv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}

	return out;
}

Ref<MLPPVector> MLPPLinAlg::zerovecnv(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(0);

	return vec;
}
Ref<MLPPVector> MLPPLinAlg::onevecnv(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(1);

	return vec;
}
Ref<MLPPVector> MLPPLinAlg::fullnv(int n, int k) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(k);

	return vec;
}

Ref<MLPPVector> MLPPLinAlg::sinnv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::sin(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::cosnv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = Math::cos(a_ptr[i]);
	}

	return out;
}

Ref<MLPPMatrix> MLPPLinAlg::maxnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(A->size());

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int size = A->data_size();

	for (int i = 0; i < size; i++) {
		c_ptr[i] = MAX(a_ptr[i], b_ptr[i]);
	}

	return C;
}

real_t MLPPLinAlg::maxvr(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), -Math_INF);

	int a_size = a->size();

	const real_t *aa = a->ptr();

	real_t max_element = -Math_INF;

	for (int i = 0; i < a_size; i++) {
		real_t current_element = aa[i];

		if (current_element > max_element) {
			max_element = current_element;
		}
	}

	return max_element;
}
real_t MLPPLinAlg::minvr(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Math_INF);

	int a_size = a->size();

	const real_t *aa = a->ptr();

	real_t min_element = Math_INF;

	for (int i = 0; i < a_size; i++) {
		real_t current_element = aa[i];

		if (current_element < min_element) {
			min_element = current_element;
		}
	}

	return min_element;
}

/*
std::vector<real_t> MLPPLinAlg::round(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = Math::round(a[i]);
	}
	return b;
}
*/

// Multidimensional Euclidean Distance

real_t MLPPLinAlg::euclidean_distance(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), 0);

	int a_size = a->size();

	ERR_FAIL_COND_V(a_size != b->size(), 0);

	const real_t *aa = a->ptr();
	const real_t *ba = b->ptr();

	real_t dist = 0;

	for (int i = 0; i < a_size; i++) {
		dist += (aa[i] - ba[i]) * (aa[i] - ba[i]);
	}

	return Math::sqrt(dist);
}
real_t MLPPLinAlg::euclidean_distance_squared(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), 0);

	int a_size = a->size();

	ERR_FAIL_COND_V(a_size != b->size(), 0);

	const real_t *aa = a->ptr();
	const real_t *ba = b->ptr();

	real_t dist = 0;

	for (int i = 0; i < a_size; i++) {
		dist += (aa[i] - ba[i]) * (aa[i] - ba[i]);
	}

	return dist;
}

/*
real_t MLPPLinAlg::norm_2(std::vector<real_t> a) {
	return Math::sqrt(norm_sq(a));
}
*/

real_t MLPPLinAlg::norm_sqv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), 0);

	int size = a->size();
	const real_t *a_ptr = a->ptr();

	real_t n_sq = 0;
	for (int i = 0; i < size; ++i) {
		n_sq += a_ptr[i] * a_ptr[i];
	}
	return n_sq;
}

real_t MLPPLinAlg::sum_elementsv(const Ref<MLPPVector> &a) {
	int a_size = a->size();

	const real_t *a_ptr = a->ptr();

	real_t sum = 0;
	for (int i = 0; i < a_size; ++i) {
		sum += a_ptr[i];
	}
	return sum;
}

/*
real_t MLPPLinAlg::cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b) {
	return dot(a, b) / (norm_2(a) * norm_2(b));
}
*/

Ref<MLPPVector> MLPPLinAlg::mat_vec_multnv(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!A.is_valid() || !b.is_valid(), Ref<MLPPMatrix>());

	Size2i a_size = A->size();
	int b_size = b->size();

	ERR_FAIL_COND_V(a_size.x < b->size(), Ref<MLPPMatrix>());

	Ref<MLPPVector> c;
	c.instance();
	c->resize(a_size.y);
	c->fill(0);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *c_ptr = c->ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int k = 0; k < b_size; ++k) {
			int mat_index = A->calculate_index(i, k);

			c_ptr[i] += a_ptr[mat_index] * b_ptr[k];
		}
	}

	return c;
}

Ref<MLPPVector> MLPPLinAlg::subtract_matrix_rowsnv(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B) {
	Ref<MLPPVector> c = a->duplicate_fast();

	Size2i b_size = B->size();

	ERR_FAIL_COND_V(b_size.x != c->size(), c);

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = c->ptrw();

	for (int i = 0; i < b_size.y; ++i) {
		for (int j = 0; j < b_size.x; ++j) {
			c_ptr[j] -= b_ptr[B->calculate_index(i, j)];
		}
	}

	return c;
}

Ref<MLPPMatrix> MLPPLinAlg::outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	Ref<MLPPMatrix> C;
	C.instance();
	Size2i size = Size2i(b->size(), a->size());
	C->resize(size);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();

	for (int i = 0; i < size.y; ++i) {
		real_t curr_a = a_ptr[i];

		for (int j = 0; j < size.x; ++j) {
			C->element_set(i, j, curr_a * b_ptr[j]);
		}
	}

	return C;
}

Ref<MLPPMatrix> MLPPLinAlg::mat_vec_addnm(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!A.is_valid() || !b.is_valid(), Ref<MLPPMatrix>());

	Size2i a_size = A->size();

	ERR_FAIL_COND_V(a_size.x != b->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> ret;
	ret.instance();
	ret->resize(a_size);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *ret_ptr = ret->ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			int mat_index = A->calculate_index(i, j);

			ret_ptr[mat_index] = a_ptr[mat_index] + b_ptr[j];
		}
	}

	return ret;
}

Ref<MLPPMatrix> MLPPLinAlg::diagnm(const Ref<MLPPVector> &a) {
	int a_size = a->size();

	Ref<MLPPMatrix> B;
	B.instance();

	B->resize(Size2i(a_size, a_size));
	B->fill(0);

	const real_t *a_ptr = a->ptr();
	real_t *b_ptr = B->ptrw();

	for (int i = 0; i < a_size; ++i) {
		b_ptr[B->calculate_index(i, i)] = a_ptr[i];
	}

	return B;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::additionnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < res.size(); i++) {
		res.write[i] = additionnm(A[i], B[i]);
	}

	return res;
}

void MLPPLinAlg::division_element_wisevt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	for (int i = 0; i < A.size(); i++) {
		Ref<MLPPMatrix> m = A[i];

		m->division_element_wise(B[i]);
	}
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::division_element_wisenvnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = division_element_wisenvnm(A[i], B[i]);
	}

	return res;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::sqrtnvt(const Vector<Ref<MLPPMatrix>> &A) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = sqrtnm(A[i]);
	}

	return res;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::exponentiatenvt(const Vector<Ref<MLPPMatrix>> &A, real_t p) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = exponentiatenm(A[i], p);
	}

	return res;
}

/*
std::vector<std::vector<real_t>> MLPPLinAlg::tensor_vec_mult(std::vector<std::vector<std::vector<real_t>>> A, std::vector<real_t> b) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < C.size(); i++) {
		for (uint32_t j = 0; j < C[i].size(); j++) {
			C[i][j] = dot(A[i][j], b);
		}
	}
	return C;
}
*/

/*
std::vector<real_t> MLPPLinAlg::flatten(std::vector<std::vector<std::vector<real_t>>> A) {
	std::vector<real_t> c;
	for (uint32_t i = 0; i < A.size(); i++) {
		std::vector<real_t> flattenedVec = flatten(A[i]);
		c.insert(c.end(), flattenedVec.begin(), flattenedVec.end());
	}
	return c;
}
*/

Vector<Ref<MLPPMatrix>> MLPPLinAlg::scalar_multiplynvt(real_t scalar, Vector<Ref<MLPPMatrix>> A) {
	for (int i = 0; i < A.size(); i++) {
		A.write[i] = scalar_multiplynm(scalar, A[i]);
	}
	return A;
}
Vector<Ref<MLPPMatrix>> MLPPLinAlg::scalar_addnvt(real_t scalar, Vector<Ref<MLPPMatrix>> A) {
	for (int i = 0; i < A.size(); i++) {
		A.write[i] = scalar_addnm(scalar, A[i]);
	}
	return A;
}

void MLPPLinAlg::resizevt(Vector<Ref<MLPPMatrix>> &r_target, const Vector<Ref<MLPPMatrix>> &A) {
	r_target.resize(A.size());

	for (int i = 0; i < r_target.size(); i++) {
		Ref<MLPPMatrix> m;
		m.instance();
		m->resize(A[i]->size());

		r_target.write[i] = m;
	}
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::resizencvt(const Vector<Ref<MLPPMatrix>> &A) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < res.size(); i++) {
		Ref<MLPPMatrix> m;
		m.instance();
		m->resize(A[i]->size());

		res.write[i] = m;
	}

	return res;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::maxnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = maxnm(A[i], B[i]);
	}

	return res;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::absnvt(const Vector<Ref<MLPPMatrix>> &A) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = absnm(A[i]);
	}

	return A;
}

/*
real_t MLPPLinAlg::norm_2(std::vector<std::vector<std::vector<real_t>>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			for (uint32_t k = 0; k < A[i][j].size(); k++) {
				sum += A[i][j][k] * A[i][j][k];
			}
		}
	}
	return Math::sqrt(sum);
}
*/

/*
// Bad implementation. Change this later.
std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<std::vector<real_t>>> C;
	C = resize(C, A);
	for (uint32_t i = 0; i < A[0].size(); i++) {
		for (uint32_t j = 0; j < A[0][i].size(); j++) {
			std::vector<real_t> currentVector;
			currentVector.resize(A.size());

			for (uint32_t k = 0; k < C.size(); k++) {
				currentVector[k] = A[k][i][j];
			}

			currentVector = mat_vec_mult(B, currentVector);

			for (uint32_t k = 0; k < C.size(); k++) {
				C[k][i][j] = currentVector[k];
			}
		}
	}
	return C;
}
*/

void MLPPLinAlg::_bind_methods() {
}
