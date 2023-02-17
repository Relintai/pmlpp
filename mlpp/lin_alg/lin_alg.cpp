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

std::vector<std::vector<real_t>> MLPPLinAlg::gramMatrix(std::vector<std::vector<real_t>> A) {
	return matmult(transpose(A), A); // AtA
}

bool MLPPLinAlg::linearIndependenceChecker(std::vector<std::vector<real_t>> A) {
	if (det(gramMatrix(A), A.size()) == 0) {
		return false;
	}
	return true;
}

std::vector<std::vector<real_t>> MLPPLinAlg::gaussianNoise(int n, int m) {
	std::random_device rd;
	std::default_random_engine generator(rd());

	std::vector<std::vector<real_t>> A;
	A.resize(n);
	for (int i = 0; i < n; i++) {
		A[i].resize(m);
		for (int j = 0; j < m; j++) {
			std::normal_distribution<real_t> distribution(0, 1); // Standard normal distribution. Mean of 0, std of 1.
			A[i][j] = distribution(generator);
		}
	}
	return A;
}

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

std::vector<std::vector<real_t>> MLPPLinAlg::addition(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[0].size(); j++) {
			C[i][j] = A[i][j] + B[i][j];
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlg::subtraction(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[0].size(); j++) {
			C[i][j] = A[i][j] - B[i][j];
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlg::matmult(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(B[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t k = 0; k < B.size(); k++) {
			for (uint32_t j = 0; j < B[0].size(); j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return C;
}

Ref<MLPPMatrix> MLPPLinAlg::additionm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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
Ref<MLPPMatrix> MLPPLinAlg::subtractionm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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
Ref<MLPPMatrix> MLPPLinAlg::matmultm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

				//C->set_element(i, j, C->get_element(i, j) + A->get_element(i, k) * B->get_element(k, j
			}
		}
	}

	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlg::hadamard_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[0].size(); j++) {
			C[i][j] = A[i][j] * B[i][j];
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlg::kronecker_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;

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

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < B.size(); j++) {
			std::vector<std::vector<real_t>> row;
			for (uint32_t k = 0; k < A[0].size(); k++) {
				row.push_back(scalarMultiply(A[i][k], B[j]));
			}
			C.push_back(flatten(row));
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlg::elementWiseDivision(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			C[i][j] = A[i][j] / B[i][j];
		}
	}
	return C;
}

Ref<MLPPMatrix> MLPPLinAlg::hadamard_productm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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
Ref<MLPPMatrix> MLPPLinAlg::kronecker_productm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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
			B->get_row_into_mlpp_vector(j, row_tmp);

			Vector<Ref<MLPPVector>> row;
			for (int k = 0; k < a_size.x; ++k) {
				row.push_back(scalar_multiplynv(a_ptr[A->calculate_index(i, k)], row_tmp));
			}

			Ref<MLPPVector> flattened_row = flattenvv(row);

			C->set_row_mlpp_vector(i * b_size.y + j, flattened_row);
		}
	}

	return C;
}
Ref<MLPPMatrix> MLPPLinAlg::element_wise_divisionm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

std::vector<std::vector<real_t>> MLPPLinAlg::transpose(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> AT;
	AT.resize(A[0].size());
	for (uint32_t i = 0; i < AT.size(); i++) {
		AT[i].resize(A.size());
	}

	for (uint32_t i = 0; i < A[0].size(); i++) {
		for (uint32_t j = 0; j < A.size(); j++) {
			AT[i][j] = A[j][i];
		}
	}
	return AT;
}

std::vector<std::vector<real_t>> MLPPLinAlg::scalarMultiply(real_t scalar, std::vector<std::vector<real_t>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] *= scalar;
		}
	}
	return A;
}

std::vector<std::vector<real_t>> MLPPLinAlg::scalarAdd(real_t scalar, std::vector<std::vector<real_t>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] += scalar;
		}
	}
	return A;
}

Ref<MLPPMatrix> MLPPLinAlg::transposem(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPLinAlg::scalar_multiplym(real_t scalar, const Ref<MLPPMatrix> &A) {
	Ref<MLPPMatrix> AN = A->duplicate();
	Size2i a_size = AN->size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			an_ptr[AN->calculate_index(i, j)] *= scalar;
		}
	}

	return AN;
}

Ref<MLPPMatrix> MLPPLinAlg::scalar_addm(real_t scalar, const Ref<MLPPMatrix> &A) {
	Ref<MLPPMatrix> AN = A->duplicate();
	Size2i a_size = AN->size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			an_ptr[AN->calculate_index(i, j)] += scalar;
		}
	}

	return AN;
}

std::vector<std::vector<real_t>> MLPPLinAlg::log(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::log(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlg::log10(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::log10(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlg::exp(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::exp(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlg::erf(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::erf(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlg::exponentiate(std::vector<std::vector<real_t>> A, real_t p) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] = std::pow(A[i][j], p);
		}
	}
	return A;
}

std::vector<std::vector<real_t>> MLPPLinAlg::sqrt(std::vector<std::vector<real_t>> A) {
	return exponentiate(A, 0.5);
}

std::vector<std::vector<real_t>> MLPPLinAlg::cbrt(std::vector<std::vector<real_t>> A) {
	return exponentiate(A, real_t(1) / real_t(3));
}

Ref<MLPPMatrix> MLPPLinAlg::logm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPLinAlg::log10m(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = std::log10(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::expm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPLinAlg::erfm(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND_V(!A.is_valid(), Ref<MLPPVector>());

	Ref<MLPPMatrix> out;
	out.instance();

	int data_size = A->data_size();
	out->resize(A->size());

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < data_size; ++i) {
		out_ptr[i] = std::erf(a_ptr[i]);
	}

	return out;
}
Ref<MLPPMatrix> MLPPLinAlg::exponentiatem(const Ref<MLPPMatrix> &A, real_t p) {
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
Ref<MLPPMatrix> MLPPLinAlg::sqrtm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPLinAlg::cbrtm(const Ref<MLPPMatrix> &A) {
	return exponentiatem(A, real_t(1) / real_t(3));
}

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

std::vector<std::vector<real_t>> MLPPLinAlg::abs(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < B.size(); i++) {
		for (uint32_t j = 0; j < B[i].size(); j++) {
			B[i][j] = std::abs(A[i][j]);
		}
	}
	return B;
}

Ref<MLPPMatrix> MLPPLinAlg::absm(const Ref<MLPPMatrix> &A) {
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

real_t MLPPLinAlg::det(std::vector<std::vector<real_t>> A, int d) {
	real_t deter = 0;
	std::vector<std::vector<real_t>> B;
	B.resize(d);
	for (int i = 0; i < d; i++) {
		B[i].resize(d);
	}

	/* This is the base case in which the input is a 2x2 square matrix.
	Recursion is performed unless and until we reach this base case,
	such that we recieve a scalar as the result. */
	if (d == 2) {
		return A[0][0] * A[1][1] - A[0][1] * A[1][0];
	}

	else {
		for (int i = 0; i < d; i++) {
			int sub_i = 0;
			for (int j = 1; j < d; j++) {
				int sub_j = 0;
				for (int k = 0; k < d; k++) {
					if (k == i) {
						continue;
					}
					B[sub_i][sub_j] = A[j][k];
					sub_j++;
				}
				sub_i++;
			}
			deter += std::pow(-1, i) * A[0][i] * det(B, d - 1);
		}
	}
	return deter;
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
		return A->get_element(0, 0) * A->get_element(1, 1) - A->get_element(0, 1) * A->get_element(1, 0);
	} else {
		for (int i = 0; i < d; i++) {
			int sub_i = 0;
			for (int j = 1; j < d; j++) {
				int sub_j = 0;
				for (int k = 0; k < d; k++) {
					if (k == i) {
						continue;
					}

					B->set_element(sub_i, sub_j, A->get_element(j, k));
					sub_j++;
				}
				sub_i++;
			}

			deter += Math::pow(static_cast<real_t>(-1), static_cast<real_t>(i)) * A->get_element(0, i) * detm(B, d - 1);
		}
	}

	return deter;
}

real_t MLPPLinAlg::trace(std::vector<std::vector<real_t>> A) {
	real_t trace = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		trace += A[i][i];
	}
	return trace;
}

std::vector<std::vector<real_t>> MLPPLinAlg::cofactor(std::vector<std::vector<real_t>> A, int n, int i, int j) {
	std::vector<std::vector<real_t>> cof;
	cof.resize(A.size());
	for (uint32_t ii = 0; ii < cof.size(); ii++) {
		cof[ii].resize(A.size());
	}
	int sub_i = 0, sub_j = 0;

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			if (row != i && col != j) {
				cof[sub_i][sub_j++] = A[row][col];

				if (sub_j == n - 1) {
					sub_j = 0;
					sub_i++;
				}
			}
		}
	}
	return cof;
}

std::vector<std::vector<real_t>> MLPPLinAlg::adjoint(std::vector<std::vector<real_t>> A) {
	//Resizing the initial adjoint matrix
	std::vector<std::vector<real_t>> adj;
	adj.resize(A.size());
	for (uint32_t i = 0; i < adj.size(); i++) {
		adj[i].resize(A.size());
	}

	// Checking for the case where the given N x N matrix is a scalar
	if (A.size() == 1) {
		adj[0][0] = 1;
		return adj;
	}

	if (A.size() == 2) {
		adj[0][0] = A[1][1];
		adj[1][1] = A[0][0];

		adj[0][1] = -A[0][1];
		adj[1][0] = -A[1][0];
		return adj;
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A.size(); j++) {
			std::vector<std::vector<real_t>> cof = cofactor(A, int(A.size()), i, j);
			// 1 if even, -1 if odd
			int sign = (i + j) % 2 == 0 ? 1 : -1;
			adj[j][i] = sign * det(cof, int(A.size()) - 1);
		}
	}
	return adj;
}

// The inverse can be computed as (1 / determinant(A)) * adjoint(A)
std::vector<std::vector<real_t>> MLPPLinAlg::inverse(std::vector<std::vector<real_t>> A) {
	return scalarMultiply(1 / det(A, int(A.size())), adjoint(A));
}

// This is simply the Moore-Penrose least squares approximation of the inverse.
std::vector<std::vector<real_t>> MLPPLinAlg::pinverse(std::vector<std::vector<real_t>> A) {
	return matmult(inverse(matmult(transpose(A), A)), transpose(A));
}

Ref<MLPPMatrix> MLPPLinAlg::cofactorm(const Ref<MLPPMatrix> &A, int n, int i, int j) {
	Ref<MLPPMatrix> cof;
	cof.instance();
	cof->resize(A->size());

	int sub_i = 0;
	int sub_j = 0;

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			if (row != i && col != j) {
				cof->set_element(sub_i, sub_j++, A->get_element(row, col));

				if (sub_j == n - 1) {
					sub_j = 0;
					sub_i++;
				}
			}
		}
	}

	return cof;
}
Ref<MLPPMatrix> MLPPLinAlg::adjointm(const Ref<MLPPMatrix> &A) {
	Ref<MLPPMatrix> adj;

	ERR_FAIL_COND_V(!A.is_valid(), adj);

	Size2i a_size = A->size();

	ERR_FAIL_COND_V(a_size.x != a_size.y, adj);

	//Resizing the initial adjoint matrix

	adj.instance();
	adj->resize(a_size);

	// Checking for the case where the given N x N matrix is a scalar
	if (a_size.y == 1) {
		adj->set_element(0, 0, 1);
		return adj;
	}

	if (a_size.y == 2) {
		adj->set_element(0, 0, A->get_element(1, 1));
		adj->set_element(1, 1, A->get_element(0, 0));

		adj->set_element(0, 1, -A->get_element(0, 1));
		adj->set_element(1, 0, -A->get_element(1, 0));

		return adj;
	}

	for (int i = 0; i < a_size.y; i++) {
		for (int j = 0; j < a_size.x; j++) {
			Ref<MLPPMatrix> cof = cofactorm(A, a_size.y, i, j);
			// 1 if even, -1 if odd
			int sign = (i + j) % 2 == 0 ? 1 : -1;
			adj->set_element(j, i, sign * detm(cof, int(a_size.y) - 1));
		}
	}
	return adj;
}
Ref<MLPPMatrix> MLPPLinAlg::inversem(const Ref<MLPPMatrix> &A) {
	return scalar_multiplym(1 / detm(A, int(A->size().y)), adjointm(A));
}
Ref<MLPPMatrix> MLPPLinAlg::pinversem(const Ref<MLPPMatrix> &A) {
	return matmultm(inversem(matmultm(transposem(A), A)), transposem(A));
}

std::vector<std::vector<real_t>> MLPPLinAlg::zeromat(int n, int m) {
	std::vector<std::vector<real_t>> zeromat;
	zeromat.resize(n);
	for (uint32_t i = 0; i < zeromat.size(); i++) {
		zeromat[i].resize(m);
	}
	return zeromat;
}

std::vector<std::vector<real_t>> MLPPLinAlg::onemat(int n, int m) {
	return full(n, m, 1);
}

Ref<MLPPMatrix> MLPPLinAlg::zeromatm(int n, int m) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(0);

	return mat;
}
Ref<MLPPMatrix> MLPPLinAlg::onematm(int n, int m) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(1);

	return mat;
}
Ref<MLPPMatrix> MLPPLinAlg::fullm(int n, int m, int k) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(k);

	return mat;
}

std::vector<std::vector<real_t>> MLPPLinAlg::full(int n, int m, int k) {
	std::vector<std::vector<real_t>> full;
	full.resize(n);
	for (uint32_t i = 0; i < full.size(); i++) {
		full[i].resize(m);
	}
	for (uint32_t i = 0; i < full.size(); i++) {
		for (uint32_t j = 0; j < full[i].size(); j++) {
			full[i][j] = k;
		}
	}
	return full;
}

std::vector<std::vector<real_t>> MLPPLinAlg::sin(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::sin(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlg::cos(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::cos(A[i][j]);
		}
	}
	return B;
}

Ref<MLPPMatrix> MLPPLinAlg::sinm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPLinAlg::cosm(const Ref<MLPPMatrix> &A) {
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

std::vector<real_t> MLPPLinAlg::max(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());
	for (uint32_t i = 0; i < c.size(); i++) {
		if (a[i] >= b[i]) {
			c[i] = a[i];
		} else {
			c[i] = b[i];
		}
	}
	return c;
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
			B[i][j] = std::round(A[i][j]);
		}
	}
	return B;
}

real_t MLPPLinAlg::norm_2(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j] * A[i][j];
		}
	}
	return std::sqrt(sum);
}

std::vector<std::vector<real_t>> MLPPLinAlg::identity(real_t d) {
	std::vector<std::vector<real_t>> identityMat;
	identityMat.resize(d);
	for (uint32_t i = 0; i < identityMat.size(); i++) {
		identityMat[i].resize(d);
	}
	for (uint32_t i = 0; i < identityMat.size(); i++) {
		for (uint32_t j = 0; j < identityMat.size(); j++) {
			if (i == j) {
				identityMat[i][j] = 1;
			} else {
				identityMat[i][j] = 0;
			}
		}
	}
	return identityMat;
}

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

std::vector<std::vector<real_t>> MLPPLinAlg::cov(std::vector<std::vector<real_t>> A) {
	MLPPStat stat;
	std::vector<std::vector<real_t>> covMat;
	covMat.resize(A.size());
	for (uint32_t i = 0; i < covMat.size(); i++) {
		covMat[i].resize(A.size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A.size(); j++) {
			covMat[i][j] = stat.covariance(A[i], A[j]);
		}
	}
	return covMat;
}

Ref<MLPPMatrix> MLPPLinAlg::covm(const Ref<MLPPMatrix> &A) {
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
		A->get_row_into_mlpp_vector(i, a_i_row_tmp);

		for (int j = 0; j < a_size.x; ++j) {
			A->get_row_into_mlpp_vector(j, a_j_row_tmp);

			cov_mat->set_element(i, j, stat.covariancev(a_i_row_tmp, a_j_row_tmp));
		}
	}

	return cov_mat;
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPLinAlg::eig(std::vector<std::vector<real_t>> A) {
	/*
	A (the entered parameter) in most use cases will be X'X, XX', etc. and must be symmetric.
	That simply means that 1) X' = X and 2) X is a square matrix. This function that computes the
	eigenvalues of a matrix is utilizing Jacobi's method.
	*/

	real_t diagonal = true; // Perform the iterative Jacobi algorithm unless and until we reach a diagonal matrix which yields us the eigenvals.

	std::map<int, int> val_to_vec;
	std::vector<std::vector<real_t>> a_new;
	std::vector<std::vector<real_t>> eigenvectors = identity(A.size());
	do {
		real_t a_ij = A[0][1];
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (uint32_t i = 0; i < A.size(); i++) {
			for (uint32_t j = 0; j < A[i].size(); j++) {
				if (i != j && std::abs(A[i][j]) > a_ij) {
					a_ij = A[i][j];
					sub_i = i;
					sub_j = j;
				} else if (i != j && std::abs(A[i][j]) == a_ij) {
					if (i < sub_i) {
						a_ij = A[i][j];
						sub_i = i;
						sub_j = j;
					}
				}
			}
		}

		real_t a_ii = A[sub_i][sub_i];
		real_t a_jj = A[sub_j][sub_j];
		//real_t a_ji = A[sub_j][sub_i];
		real_t theta;

		if (a_ii == a_jj) {
			theta = M_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		std::vector<std::vector<real_t>> P = identity(A.size());
		P[sub_i][sub_j] = -std::sin(theta);
		P[sub_i][sub_i] = std::cos(theta);
		P[sub_j][sub_j] = std::cos(theta);
		P[sub_j][sub_i] = std::sin(theta);

		a_new = matmult(matmult(inverse(P), A), P);

		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) == 0) {
					a_new[i][j] = 0;
				}
			}
		}

		bool non_zero = false;
		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) != 0) {
					non_zero = true;
				}
			}
		}

		if (non_zero) {
			diagonal = false;
		} else {
			diagonal = true;
		}

		if (a_new == A) {
			diagonal = true;
			for (uint32_t i = 0; i < a_new.size(); i++) {
				for (uint32_t j = 0; j < a_new[i].size(); j++) {
					if (i != j) {
						a_new[i][j] = 0;
					}
				}
			}
		}

		eigenvectors = matmult(eigenvectors, P);
		A = a_new;

	} while (!diagonal);

	std::vector<std::vector<real_t>> a_new_prior = a_new;

	// Bubble Sort. Should change this later.
	for (uint32_t i = 0; i < a_new.size() - 1; i++) {
		for (uint32_t j = 0; j < a_new.size() - 1 - i; j++) {
			if (a_new[j][j] < a_new[j + 1][j + 1]) {
				real_t temp = a_new[j + 1][j + 1];
				a_new[j + 1][j + 1] = a_new[j][j];
				a_new[j][j] = temp;
			}
		}
	}

	for (uint32_t i = 0; i < a_new.size(); i++) {
		for (uint32_t j = 0; j < a_new.size(); j++) {
			if (a_new[i][i] == a_new_prior[j][j]) {
				val_to_vec[i] = j;
			}
		}
	}

	std::vector<std::vector<real_t>> eigen_temp = eigenvectors;
	for (uint32_t i = 0; i < eigenvectors.size(); i++) {
		for (uint32_t j = 0; j < eigenvectors[i].size(); j++) {
			eigenvectors[i][j] = eigen_temp[i][val_to_vec[j]];
		}
	}
	return { eigenvectors, a_new };
}

MLPPLinAlg::EigenResultOld MLPPLinAlg::eigen_old(std::vector<std::vector<real_t>> A) {
	/*
	A (the entered parameter) in most use cases will be X'X, XX', etc. and must be symmetric.
	That simply means that 1) X' = X and 2) X is a square matrix. This function that computes the
	eigenvalues of a matrix is utilizing Jacobi's method.
	*/

	real_t diagonal = true; // Perform the iterative Jacobi algorithm unless and until we reach a diagonal matrix which yields us the eigenvals.

	std::map<int, int> val_to_vec;
	std::vector<std::vector<real_t>> a_new;
	std::vector<std::vector<real_t>> eigenvectors = identity(A.size());
	do {
		real_t a_ij = A[0][1];
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (uint32_t i = 0; i < A.size(); i++) {
			for (uint32_t j = 0; j < A[i].size(); j++) {
				if (i != j && std::abs(A[i][j]) > a_ij) {
					a_ij = A[i][j];
					sub_i = i;
					sub_j = j;
				} else if (i != j && std::abs(A[i][j]) == a_ij) {
					if (i < sub_i) {
						a_ij = A[i][j];
						sub_i = i;
						sub_j = j;
					}
				}
			}
		}

		real_t a_ii = A[sub_i][sub_i];
		real_t a_jj = A[sub_j][sub_j];
		//real_t a_ji = A[sub_j][sub_i];
		real_t theta;

		if (a_ii == a_jj) {
			theta = M_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		std::vector<std::vector<real_t>> P = identity(A.size());
		P[sub_i][sub_j] = -std::sin(theta);
		P[sub_i][sub_i] = std::cos(theta);
		P[sub_j][sub_j] = std::cos(theta);
		P[sub_j][sub_i] = std::sin(theta);

		a_new = matmult(matmult(inverse(P), A), P);

		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) == 0) {
					a_new[i][j] = 0;
				}
			}
		}

		bool non_zero = false;
		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) != 0) {
					non_zero = true;
				}
			}
		}

		if (non_zero) {
			diagonal = false;
		} else {
			diagonal = true;
		}

		if (a_new == A) {
			diagonal = true;
			for (uint32_t i = 0; i < a_new.size(); i++) {
				for (uint32_t j = 0; j < a_new[i].size(); j++) {
					if (i != j) {
						a_new[i][j] = 0;
					}
				}
			}
		}

		eigenvectors = matmult(eigenvectors, P);
		A = a_new;

	} while (!diagonal);

	std::vector<std::vector<real_t>> a_new_prior = a_new;

	// Bubble Sort. Should change this later.
	for (uint32_t i = 0; i < a_new.size() - 1; i++) {
		for (uint32_t j = 0; j < a_new.size() - 1 - i; j++) {
			if (a_new[j][j] < a_new[j + 1][j + 1]) {
				real_t temp = a_new[j + 1][j + 1];
				a_new[j + 1][j + 1] = a_new[j][j];
				a_new[j][j] = temp;
			}
		}
	}

	for (uint32_t i = 0; i < a_new.size(); i++) {
		for (uint32_t j = 0; j < a_new.size(); j++) {
			if (a_new[i][i] == a_new_prior[j][j]) {
				val_to_vec[i] = j;
			}
		}
	}

	std::vector<std::vector<real_t>> eigen_temp = eigenvectors;
	for (uint32_t i = 0; i < eigenvectors.size(); i++) {
		for (uint32_t j = 0; j < eigenvectors[i].size(); j++) {
			eigenvectors[i][j] = eigen_temp[i][val_to_vec[j]];
		}
	}

	EigenResultOld res;
	res.eigen_vectors = eigenvectors;
	res.eigen_values = a_new;

	return res;
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
		real_t a_ij = A->get_element(0, 1);
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (int i = 0; i < a_size.y; ++i) {
			for (int j = 0; j < a_size.x; ++j) {
				real_t ca_ij = A->get_element(i, j);
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

		real_t a_ii = A->get_element(sub_i, sub_i);
		real_t a_jj = A->get_element(sub_j, sub_j);
		//real_t a_ji = A->get_element(sub_j, sub_i);
		real_t theta;

		if (a_ii == a_jj) {
			theta = M_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		Ref<MLPPMatrix> P = identitym(A->size().y);
		P->set_element(sub_i, sub_j, -Math::sin(theta));
		P->set_element(sub_i, sub_i, Math::cos(theta));
		P->set_element(sub_j, sub_j, Math::cos(theta));
		P->set_element(sub_j, sub_i, Math::sin(theta));

		a_new = matmultm(matmultm(inversem(P), A), P);

		Size2i a_new_size = a_new->size();

		for (int i = 0; i < a_new_size.y; ++i) {
			for (int j = 0; j < a_new_size.x; ++j) {
				if (i != j && Math::is_zero_approx(Math::round(a_new->get_element(i, j)))) {
					a_new->set_element(i, j, 0);
				}
			}
		}

		bool non_zero = false;
		for (int i = 0; i < a_new_size.y; ++i) {
			for (int j = 0; j < a_new_size.x; ++j) {
				if (i != j && Math::is_zero_approx(Math::round(a_new->get_element(i, j)))) {
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
						a_new->set_element(i, j, 0);
					}
				}
			}
		}

		eigenvectors = matmultm(eigenvectors, P);
		A = a_new;

	} while (!diagonal);

	Ref<MLPPMatrix> a_new_prior = a_new->duplicate();

	Size2i a_new_size = a_new->size();

	// Bubble Sort. Should change this later.
	for (int i = 0; i < a_new_size.y - 1; ++i) {
		for (int j = 0; j < a_new_size.x - 1 - i; ++j) {
			if (a_new->get_element(j, j) < a_new->get_element(j + 1, j + 1)) {
				real_t temp = a_new->get_element(j + 1, j + 1);
				a_new->set_element(j + 1, j + 1, a_new->get_element(j, j));
				a_new->set_element(j, j, temp);
			}
		}
	}

	for (int i = 0; i < a_new_size.y; ++i) {
		for (int j = 0; j < a_new_size.x; ++j) {
			if (a_new->get_element(i, i) == a_new_prior->get_element(j, j)) {
				val_to_vec[i] = j;
			}
		}
	}

	Ref<MLPPMatrix> eigen_temp = eigenvectors->duplicate();

	Size2i eigenvectors_size = eigenvectors->size();

	for (int i = 0; i < eigenvectors_size.y; ++i) {
		for (int j = 0; j < eigenvectors_size.x; ++j) {
			eigenvectors->set_element(i, j, eigen_temp->get_element(i, val_to_vec[j]));
		}
	}

	res.eigen_vectors = eigenvectors;
	res.eigen_values = a_new;

	return res;
}

MLPPLinAlg::SVDResultOld MLPPLinAlg::SVD(std::vector<std::vector<real_t>> A) {
	EigenResultOld left_eigen = eigen_old(matmult(A, transpose(A)));
	EigenResultOld right_eigen = eigen_old(matmult(transpose(A), A));

	std::vector<std::vector<real_t>> singularvals = sqrt(left_eigen.eigen_values);
	std::vector<std::vector<real_t>> sigma = zeromat(A.size(), A[0].size());
	for (uint32_t i = 0; i < singularvals.size(); i++) {
		for (uint32_t j = 0; j < singularvals[i].size(); j++) {
			sigma[i][j] = singularvals[i][j];
		}
	}

	SVDResultOld res;
	res.U = left_eigen.eigen_vectors;
	res.S = sigma;
	res.Vt = right_eigen.eigen_vectors;

	return res;
}

MLPPLinAlg::SVDResult MLPPLinAlg::svd(const Ref<MLPPMatrix> &A) {
	SVDResult res;

	ERR_FAIL_COND_V(!A.is_valid(), res);

	Size2i a_size = A->size();

	EigenResult left_eigen = eigen(matmultm(A, transposem(A)));
	EigenResult right_eigen = eigen(matmultm(transposem(A), A));

	Ref<MLPPMatrix> singularvals = sqrtm(left_eigen.eigen_values);
	Ref<MLPPMatrix> sigma = zeromatm(a_size.y, a_size.x);

	Size2i singularvals_size = singularvals->size();

	for (int i = 0; i < singularvals_size.y; ++i) {
		for (int j = 0; j < singularvals_size.x; ++j) {
			sigma->set_element(i, j, singularvals->get_element(i, j));
		}
	}

	res.U = left_eigen.eigen_vectors;
	res.S = sigma;
	res.Vt = right_eigen.eigen_vectors;

	return res;
}

std::vector<real_t> MLPPLinAlg::vectorProjection(std::vector<real_t> a, std::vector<real_t> b) {
	real_t product = dot(a, b) / dot(a, a);
	return scalarMultiply(product, a); // Projection of vector a onto b. Denotated as proj_a(b).
}

std::vector<std::vector<real_t>> MLPPLinAlg::gramSchmidtProcess(std::vector<std::vector<real_t>> A) {
	A = transpose(A); // C++ vectors lack a mechanism to directly index columns. So, we transpose *a copy* of A for this purpose for ease of use.
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}

	B[0] = A[0]; // We set a_1 = b_1 as an initial condition.
	B[0] = scalarMultiply(1 / norm_2(B[0]), B[0]);
	for (uint32_t i = 1; i < B.size(); i++) {
		B[i] = A[i];
		for (int j = i - 1; j >= 0; j--) {
			B[i] = subtraction(B[i], vectorProjection(B[j], A[i]));
		}
		B[i] = scalarMultiply(1 / norm_2(B[i]), B[i]); // Very simply multiply all elements of vec B[i] by 1/||B[i]||_2
	}
	return transpose(B); // We re-transpose the marix.
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPLinAlg::QRD(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> Q = gramSchmidtProcess(A);
	std::vector<std::vector<real_t>> R = matmult(transpose(Q), A);
	return { Q, R };
}

MLPPLinAlg::QRDResult MLPPLinAlg::qrd(std::vector<std::vector<real_t>> A) {
	QRDResult res;

	res.Q = gramSchmidtProcess(A);
	res.R = matmult(transpose(res.Q), A);

	return res;
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPLinAlg::chol(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> L = zeromat(A.size(), A[0].size());
	for (uint32_t j = 0; j < L.size(); j++) { // Matrices entered must be square. No problem here.
		for (uint32_t i = j; i < L.size(); i++) {
			if (i == j) {
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[i][k];
				}
				L[i][j] = std::sqrt(A[i][j] - sum);
			} else { // That is, i!=j
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				L[i][j] = (A[i][j] - sum) / L[j][j];
			}
		}
	}
	return { L, transpose(L) }; // Indeed, L.T is our upper triangular matrix.
}

MLPPLinAlg::CholeskyResult MLPPLinAlg::cholesky(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> L = zeromat(A.size(), A[0].size());
	for (uint32_t j = 0; j < L.size(); j++) { // Matrices entered must be square. No problem here.
		for (uint32_t i = j; i < L.size(); i++) {
			if (i == j) {
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[i][k];
				}
				L[i][j] = std::sqrt(A[i][j] - sum);
			} else { // That is, i!=j
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				L[i][j] = (A[i][j] - sum) / L[j][j];
			}
		}
	}

	CholeskyResult res;
	res.L = L;
	res.Lt = transpose(L); // Indeed, L.T is our upper triangular matrix.

	return res;
}

real_t MLPPLinAlg::sum_elements(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j];
		}
	}
	return sum;
}

std::vector<real_t> MLPPLinAlg::flatten(std::vector<std::vector<real_t>> A) {
	std::vector<real_t> a;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			a.push_back(A[i][j]);
		}
	}
	return a;
}

Ref<MLPPVector> MLPPLinAlg::flattenvv(const Vector<Ref<MLPPVector>> &A) {
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

Ref<MLPPVector> MLPPLinAlg::flattenv(const Ref<MLPPMatrix> &A) {
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

std::vector<real_t> MLPPLinAlg::solve(std::vector<std::vector<real_t>> A, std::vector<real_t> b) {
	return mat_vec_mult(inverse(A), b);
}

bool MLPPLinAlg::positiveDefiniteChecker(std::vector<std::vector<real_t>> A) {
	auto eig_result = eig(A);
	auto eigenvectors = std::get<0>(eig_result);
	auto eigenvals = std::get<1>(eig_result);

	std::vector<real_t> eigenvals_vec;
	for (uint32_t i = 0; i < eigenvals.size(); i++) {
		eigenvals_vec.push_back(eigenvals[i][i]);
	}
	for (uint32_t i = 0; i < eigenvals_vec.size(); i++) {
		if (eigenvals_vec[i] <= 0) { // Simply check to ensure all eigenvalues are positive.
			return false;
		}
	}
	return true;
}

bool MLPPLinAlg::negativeDefiniteChecker(std::vector<std::vector<real_t>> A) {
	auto eig_result = eig(A);
	auto eigenvectors = std::get<0>(eig_result);
	auto eigenvals = std::get<1>(eig_result);

	std::vector<real_t> eigenvals_vec;
	for (uint32_t i = 0; i < eigenvals.size(); i++) {
		eigenvals_vec.push_back(eigenvals[i][i]);
	}
	for (uint32_t i = 0; i < eigenvals_vec.size(); i++) {
		if (eigenvals_vec[i] >= 0) { // Simply check to ensure all eigenvalues are negative.
			return false;
		}
	}
	return true;
}

bool MLPPLinAlg::zeroEigenvalue(std::vector<std::vector<real_t>> A) {
	auto eig_result = eig(A);
	auto eigenvectors = std::get<0>(eig_result);
	auto eigenvals = std::get<1>(eig_result);

	std::vector<real_t> eigenvals_vec;
	for (uint32_t i = 0; i < eigenvals.size(); i++) {
		eigenvals_vec.push_back(eigenvals[i][i]);
	}
	for (uint32_t i = 0; i < eigenvals_vec.size(); i++) {
		if (eigenvals_vec[i] == 0) {
			return true;
		}
	}
	return false;
}

void MLPPLinAlg::printMatrix(std::vector<std::vector<real_t>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			std::cout << A[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

std::vector<std::vector<real_t>> MLPPLinAlg::outerProduct(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<std::vector<real_t>> C;
	C.resize(a.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i] = scalarMultiply(a[i], b);
	}
	return C;
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
			C->set_element(i, j, curr_a * b_ptr[j]);
		}
	}

	return C;
}

std::vector<real_t> MLPPLinAlg::hadamard_product(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] * b[i];
	}

	return c;
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

std::vector<real_t> MLPPLinAlg::elementWiseDivision(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] / b[i];
	}
	return c;
}

Ref<MLPPVector> MLPPLinAlg::element_wise_division(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

std::vector<real_t> MLPPLinAlg::scalarMultiply(real_t scalar, std::vector<real_t> a) {
	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] *= scalar;
	}
	return a;
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

std::vector<real_t> MLPPLinAlg::scalarAdd(real_t scalar, std::vector<real_t> a) {
	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] += scalar;
	}
	return a;
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

std::vector<real_t> MLPPLinAlg::addition(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] + b[i];
	}
	return c;
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

std::vector<real_t> MLPPLinAlg::subtraction(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] - b[i];
	}
	return c;
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

std::vector<real_t> MLPPLinAlg::subtractMatrixRows(std::vector<real_t> a, std::vector<std::vector<real_t>> B) {
	for (uint32_t i = 0; i < B.size(); i++) {
		a = subtraction(a, B[i]);
	}
	return a;
}

Ref<MLPPVector> MLPPLinAlg::subtract_matrix_rows(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B) {
	Ref<MLPPVector> c = a->duplicate();

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

std::vector<real_t> MLPPLinAlg::log(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::log(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlg::log10(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::log10(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlg::exp(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::exp(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlg::erf(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::erf(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlg::exponentiate(std::vector<real_t> a, real_t p) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < b.size(); i++) {
		b[i] = std::pow(a[i], p);
	}
	return b;
}

std::vector<real_t> MLPPLinAlg::sqrt(std::vector<real_t> a) {
	return exponentiate(a, 0.5);
}

std::vector<real_t> MLPPLinAlg::cbrt(std::vector<real_t> a) {
	return exponentiate(a, real_t(1) / real_t(3));
}

Ref<MLPPVector> MLPPLinAlg::logv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPLinAlg::log10v(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = std::log10(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::expv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPLinAlg::erfv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	int size = a->size();
	out->resize(size);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < size; ++i) {
		out_ptr[i] = std::erf(a_ptr[i]);
	}

	return out;
}
Ref<MLPPVector> MLPPLinAlg::exponentiatev(const Ref<MLPPVector> &a, real_t p) {
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
Ref<MLPPVector> MLPPLinAlg::sqrtv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPLinAlg::cbrtv(const Ref<MLPPVector> &a) {
	return exponentiatev(a, static_cast<real_t>(1) / static_cast<real_t>(3));
}

real_t MLPPLinAlg::dot(std::vector<real_t> a, std::vector<real_t> b) {
	real_t c = 0;
	for (uint32_t i = 0; i < a.size(); i++) {
		c += a[i] * b[i];
	}
	return c;
}

real_t MLPPLinAlg::dotv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

std::vector<real_t> MLPPLinAlg::cross(std::vector<real_t> a, std::vector<real_t> b) {
	// Cross products exist in R^7 also. Though, I will limit it to R^3 as Wolfram does this.
	std::vector<std::vector<real_t>> mat = { onevec(3), a, b };

	real_t det1 = det({ { a[1], a[2] }, { b[1], b[2] } }, 2);
	real_t det2 = -det({ { a[0], a[2] }, { b[0], b[2] } }, 2);
	real_t det3 = det({ { a[0], a[1] }, { b[0], b[1] } }, 2);

	return { det1, det2, det3 };
}

std::vector<real_t> MLPPLinAlg::abs(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < b.size(); i++) {
		b[i] = std::abs(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlg::zerovec(int n) {
	std::vector<real_t> zerovec;
	zerovec.resize(n);
	return zerovec;
}

std::vector<real_t> MLPPLinAlg::onevec(int n) {
	return full(n, 1);
}

std::vector<std::vector<real_t>> MLPPLinAlg::diag(std::vector<real_t> a) {
	std::vector<std::vector<real_t>> B = zeromat(a.size(), a.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i][i] = a[i];
	}
	return B;
}

Ref<MLPPVector> MLPPLinAlg::diagm(const Ref<MLPPVector> &a) {
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

std::vector<real_t> MLPPLinAlg::full(int n, int k) {
	std::vector<real_t> full;
	full.resize(n);
	for (uint32_t i = 0; i < full.size(); i++) {
		full[i] = k;
	}
	return full;
}

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

Ref<MLPPVector> MLPPLinAlg::zerovecv(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(0);

	return vec;
}
Ref<MLPPVector> MLPPLinAlg::onevecv(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(1);

	return vec;
}
Ref<MLPPVector> MLPPLinAlg::fullv(int n, int k) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(k);

	return vec;
}

std::vector<real_t> MLPPLinAlg::sin(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::sin(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlg::cos(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::cos(a[i]);
	}
	return b;
}

Ref<MLPPVector> MLPPLinAlg::sinv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPLinAlg::cosv(const Ref<MLPPVector> &a) {
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

std::vector<std::vector<real_t>> MLPPLinAlg::rotate(std::vector<std::vector<real_t>> A, real_t theta, int axis) {
	std::vector<std::vector<real_t>> rotationMatrix = { { std::cos(theta), -std::sin(theta) }, { std::sin(theta), std::cos(theta) } };
	if (axis == 0) {
		rotationMatrix = { { 1, 0, 0 }, { 0, std::cos(theta), -std::sin(theta) }, { 0, std::sin(theta), std::cos(theta) } };
	} else if (axis == 1) {
		rotationMatrix = { { std::cos(theta), 0, std::sin(theta) }, { 0, 1, 0 }, { -std::sin(theta), 0, std::cos(theta) } };
	} else if (axis == 2) {
		rotationMatrix = { { std::cos(theta), -std::sin(theta), 0 }, { std::sin(theta), std::cos(theta), 0 }, { 1, 0, 0 } };
	}

	return matmult(A, rotationMatrix);
}

std::vector<std::vector<real_t>> MLPPLinAlg::max(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		C[i] = max(A[i], B[i]);
	}
	return C;
}

Ref<MLPPMatrix> MLPPLinAlg::max_nm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

real_t MLPPLinAlg::max(std::vector<real_t> a) {
	int max = a[0];
	for (uint32_t i = 0; i < a.size(); i++) {
		if (a[i] > max) {
			max = a[i];
		}
	}
	return max;
}

real_t MLPPLinAlg::min(std::vector<real_t> a) {
	int min = a[0];
	for (uint32_t i = 0; i < a.size(); i++) {
		if (a[i] < min) {
			min = a[i];
		}
	}
	return min;
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

		if (current_element > min_element) {
			min_element = current_element;
		}
	}

	return min_element;
}

std::vector<real_t> MLPPLinAlg::round(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::round(a[i]);
	}
	return b;
}

// Multidimensional Euclidean Distance
real_t MLPPLinAlg::euclideanDistance(std::vector<real_t> a, std::vector<real_t> b) {
	real_t dist = 0;
	for (uint32_t i = 0; i < a.size(); i++) {
		dist += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return std::sqrt(dist);
}

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

real_t MLPPLinAlg::norm_2(std::vector<real_t> a) {
	return std::sqrt(norm_sq(a));
}

real_t MLPPLinAlg::norm_sq(std::vector<real_t> a) {
	real_t n_sq = 0;
	for (uint32_t i = 0; i < a.size(); i++) {
		n_sq += a[i] * a[i];
	}
	return n_sq;
}
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

real_t MLPPLinAlg::sum_elements(std::vector<real_t> a) {
	real_t sum = 0;
	for (uint32_t i = 0; i < a.size(); i++) {
		sum += a[i];
	}
	return sum;
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

real_t MLPPLinAlg::cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b) {
	return dot(a, b) / (norm_2(a) * norm_2(b));
}

void MLPPLinAlg::printVector(std::vector<real_t> a) {
	for (uint32_t i = 0; i < a.size(); i++) {
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}

std::vector<std::vector<real_t>> MLPPLinAlg::mat_vec_add(std::vector<std::vector<real_t>> A, std::vector<real_t> b) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] += b[j];
		}
	}
	return A;
}

std::vector<real_t> MLPPLinAlg::mat_vec_mult(std::vector<std::vector<real_t>> A, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(A.size());

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t k = 0; k < b.size(); k++) {
			c[i] += A[i][k] * b[k];
		}
	}
	return c;
}

Ref<MLPPMatrix> MLPPLinAlg::mat_vec_addv(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b) {
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
Ref<MLPPVector> MLPPLinAlg::mat_vec_multv(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b) {
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

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::addition(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = addition(A[i], B[i]);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::addition_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < res.size(); i++) {
		res.write[i] = additionm(A[i], B[i]);
	}

	return res;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::elementWiseDivision(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = elementWiseDivision(A[i], B[i]);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::element_wise_division_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = element_wise_divisionm(A[i], B[i]);
	}

	return res;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::sqrt(std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = sqrt(A[i]);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::sqrt_vt(const Vector<Ref<MLPPMatrix>> &A) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = sqrtm(A[i]);
	}

	return res;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::exponentiate(std::vector<std::vector<std::vector<real_t>>> A, real_t p) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = exponentiate(A[i], p);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::exponentiate_vt(const Vector<Ref<MLPPMatrix>> &A, real_t p) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = exponentiatem(A[i], p);
	}

	return res;
}

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

std::vector<real_t> MLPPLinAlg::flatten(std::vector<std::vector<std::vector<real_t>>> A) {
	std::vector<real_t> c;
	for (uint32_t i = 0; i < A.size(); i++) {
		std::vector<real_t> flattenedVec = flatten(A[i]);
		c.insert(c.end(), flattenedVec.begin(), flattenedVec.end());
	}
	return c;
}

void MLPPLinAlg::printTensor(std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		printMatrix(A[i]);
		if (i != A.size() - 1) {
			std::cout << std::endl;
		}
	}
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::scalarMultiply(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = scalarMultiply(scalar, A[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::scalarAdd(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = scalarAdd(scalar, A[i]);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::scalar_multiply_vm(real_t scalar, Vector<Ref<MLPPMatrix>> A) {
	for (int i = 0; i < A.size(); i++) {
		A.write[i] = scalar_multiplym(scalar, A[i]);
	}
	return A;
}
Vector<Ref<MLPPMatrix>> MLPPLinAlg::scalar_add_vm(real_t scalar, Vector<Ref<MLPPMatrix>> A) {
	for (int i = 0; i < A.size(); i++) {
		A.write[i] = scalar_addm(scalar, A[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::resize(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	A.resize(B.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		A[i].resize(B[i].size());
		for (uint32_t j = 0; j < B[i].size(); j++) {
			A[i][j].resize(B[i][j].size());
		}
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::resize_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(B.size());

	for (int i = 0; i < res.size(); i++) {
		Ref<MLPPMatrix> m;
		m.instance();
		m->resize(B[i]->size());

		res.write[i] = m;
	}

	return res;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::max(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = max(A[i], B[i]);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::max_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = max_nm(A[i], B[i]);
	}

	return res;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlg::abs(std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = abs(A[i]);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPLinAlg::abs_vt(const Vector<Ref<MLPPMatrix>> &A) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = absm(A[i]);
	}

	return A;
}

real_t MLPPLinAlg::norm_2(std::vector<std::vector<std::vector<real_t>>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			for (uint32_t k = 0; k < A[i][j].size(); k++) {
				sum += A[i][j][k] * A[i][j][k];
			}
		}
	}
	return std::sqrt(sum);
}

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

void MLPPLinAlg::_bind_methods() {
}
