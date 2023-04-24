
#include "mlpp_matrix.h"

#include "../stat/stat.h"
#include <random>

Ref<MLPPVector> MLPPMatrix::scalar_multiplynv(real_t scalar, const Ref<MLPPVector> &a) {
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

Ref<MLPPVector> MLPPMatrix::flattenmnv(const Vector<Ref<MLPPVector>> &A) {
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

/*
std::vector<std::vector<real_t>> MLPPMatrix::gramMatrix(std::vector<std::vector<real_t>> A) {
	return matmult(transpose(A), A); // AtA
}
*/

/*
bool MLPPMatrix::linearIndependenceChecker(std::vector<std::vector<real_t>> A) {
	if (det(gramMatrix(A), A.size()) == 0) {
		return false;
	}
	return true;
}
*/

Ref<MLPPMatrix> MLPPMatrix::gaussian_noise(int n, int m) {
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

Ref<MLPPMatrix> MLPPMatrix::additionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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
Ref<MLPPMatrix> MLPPMatrix::subtractionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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
Ref<MLPPMatrix> MLPPMatrix::matmultnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

Ref<MLPPMatrix> MLPPMatrix::hadamard_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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
Ref<MLPPMatrix> MLPPMatrix::kronecker_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

			Ref<MLPPVector> flattened_row = flattenmnv(row);

			C->set_row_mlpp_vector(i * b_size.y + j, flattened_row);
		}
	}

	return C;
}
Ref<MLPPMatrix> MLPPMatrix::element_wise_divisionnvnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

Ref<MLPPMatrix> MLPPMatrix::transposenm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPMatrix::scalar_multiplynm(real_t scalar, const Ref<MLPPMatrix> &A) {
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

Ref<MLPPMatrix> MLPPMatrix::scalar_addnm(real_t scalar, const Ref<MLPPMatrix> &A) {
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

Ref<MLPPMatrix> MLPPMatrix::lognm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPMatrix::log10nm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPMatrix::expnm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPMatrix::erfnm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPMatrix::exponentiatenm(const Ref<MLPPMatrix> &A, real_t p) {
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
Ref<MLPPMatrix> MLPPMatrix::sqrtnm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPMatrix::cbrtnm(const Ref<MLPPMatrix> &A) {
	return exponentiatenm(A, real_t(1) / real_t(3));
}

/*
std::vector<std::vector<real_t>> MLPPMatrix::matrixPower(std::vector<std::vector<real_t>> A, int n) {
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

Ref<MLPPMatrix> MLPPMatrix::absnm(const Ref<MLPPMatrix> &A) {
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

real_t MLPPMatrix::detm(const Ref<MLPPMatrix> &A, int d) {
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

/*
real_t MLPPMatrix::trace(std::vector<std::vector<real_t>> A) {
	real_t trace = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		trace += A[i][i];
	}
	return trace;
}
*/

Ref<MLPPMatrix> MLPPMatrix::cofactornm(const Ref<MLPPMatrix> &A, int n, int i, int j) {
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
Ref<MLPPMatrix> MLPPMatrix::adjointnm(const Ref<MLPPMatrix> &A) {
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
			Ref<MLPPMatrix> cof = cofactornm(A, a_size.y, i, j);
			// 1 if even, -1 if odd
			int sign = (i + j) % 2 == 0 ? 1 : -1;
			adj->set_element(j, i, sign * detm(cof, int(a_size.y) - 1));
		}
	}
	return adj;
}
Ref<MLPPMatrix> MLPPMatrix::inversenm(const Ref<MLPPMatrix> &A) {
	return scalar_multiplynm(1 / detm(A, int(A->size().y)), adjointnm(A));
}
Ref<MLPPMatrix> MLPPMatrix::pinversenm(const Ref<MLPPMatrix> &A) {
	return matmultnm(inversenm(matmultnm(transposenm(A), A)), transposenm(A));
}
Ref<MLPPMatrix> MLPPMatrix::zeromatnm(int n, int m) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(0);

	return mat;
}
Ref<MLPPMatrix> MLPPMatrix::onematnm(int n, int m) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(1);

	return mat;
}
Ref<MLPPMatrix> MLPPMatrix::fullnm(int n, int m, int k) {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(k);

	return mat;
}

Ref<MLPPMatrix> MLPPMatrix::sinnm(const Ref<MLPPMatrix> &A) {
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
Ref<MLPPMatrix> MLPPMatrix::cosnm(const Ref<MLPPMatrix> &A) {
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

/*
std::vector<std::vector<real_t>> MLPPMatrix::rotate(std::vector<std::vector<real_t>> A, real_t theta, int axis) {
	std::vector<std::vector<real_t>> rotationMatrix = { { Math::cos(theta), -Math::sin(theta) }, { Math::sin(theta), Math::cos(theta) } };
	if (axis == 0) {
		rotationMatrix = { { 1, 0, 0 }, { 0, Math::cos(theta), -Math::sin(theta) }, { 0, Math::sin(theta), Math::cos(theta) } };
	} else if (axis == 1) {
		rotationMatrix = { { Math::cos(theta), 0, Math::sin(theta) }, { 0, 1, 0 }, { -Math::sin(theta), 0, Math::cos(theta) } };
	} else if (axis == 2) {
		rotationMatrix = { { Math::cos(theta), -Math::sin(theta), 0 }, { Math::sin(theta), Math::cos(theta), 0 }, { 1, 0, 0 } };
	}

	return matmult(A, rotationMatrix);
}
*/

Ref<MLPPMatrix> MLPPMatrix::maxnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

/*
real_t MLPPMatrix::max(std::vector<std::vector<real_t>> A) {
	return max(flatten(A));
}

real_t MLPPMatrix::min(std::vector<std::vector<real_t>> A) {
	return min(flatten(A));
}

std::vector<std::vector<real_t>> MLPPMatrix::round(std::vector<std::vector<real_t>> A) {
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
real_t MLPPMatrix::norm_2(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j] * A[i][j];
		}
	}
	return Math::sqrt(sum);
}
*/

Ref<MLPPMatrix> MLPPMatrix::identitym(int d) {
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

Ref<MLPPMatrix> MLPPMatrix::covnm(const Ref<MLPPMatrix> &A) {
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

MLPPMatrix::EigenResult MLPPMatrix::eigen(Ref<MLPPMatrix> A) {
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

		a_new = matmultnm(matmultnm(inversenm(P), A), P);

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

		eigenvectors = matmultnm(eigenvectors, P);
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

MLPPMatrix::SVDResult MLPPMatrix::svd(const Ref<MLPPMatrix> &A) {
	SVDResult res;

	ERR_FAIL_COND_V(!A.is_valid(), res);

	Size2i a_size = A->size();

	EigenResult left_eigen = eigen(matmultnm(A, transposenm(A)));
	EigenResult right_eigen = eigen(matmultnm(transposenm(A), A));

	Ref<MLPPMatrix> singularvals = sqrtnm(left_eigen.eigen_values);
	Ref<MLPPMatrix> sigma = zeromatnm(a_size.y, a_size.x);

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

/*
std::vector<real_t> MLPPMatrix::vectorProjection(std::vector<real_t> a, std::vector<real_t> b) {
	real_t product = dot(a, b) / dot(a, a);
	return scalarMultiply(product, a); // Projection of vector a onto b. Denotated as proj_a(b).
}
*/

/*
std::vector<std::vector<real_t>> MLPPMatrix::gramSchmidtProcess(std::vector<std::vector<real_t>> A) {
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
*/

/*
MLPPMatrix::QRDResult MLPPMatrix::qrd(std::vector<std::vector<real_t>> A) {
	QRDResult res;

	res.Q = gramSchmidtProcess(A);
	res.R = matmult(transpose(res.Q), A);

	return res;
}
*/

/*
MLPPMatrix::CholeskyResult MLPPMatrix::cholesky(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> L = zeromat(A.size(), A[0].size());
	for (uint32_t j = 0; j < L.size(); j++) { // Matrices entered must be square. No problem here.
		for (uint32_t i = j; i < L.size(); i++) {
			if (i == j) {
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[i][k];
				}
				L[i][j] = Math::sqrt(A[i][j] - sum);
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
*/

/*
real_t MLPPMatrix::sum_elements(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j];
		}
	}
	return sum;
}
*/

Ref<MLPPVector> MLPPMatrix::flattenvvnv(const Ref<MLPPMatrix> &A) {
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

/*
std::vector<real_t> MLPPMatrix::solve(std::vector<std::vector<real_t>> A, std::vector<real_t> b) {
	return mat_vec_mult(inverse(A), b);
}

bool MLPPMatrix::positiveDefiniteChecker(std::vector<std::vector<real_t>> A) {
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

bool MLPPMatrix::negativeDefiniteChecker(std::vector<std::vector<real_t>> A) {
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

bool MLPPMatrix::zeroEigenvalue(std::vector<std::vector<real_t>> A) {
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
*/

String MLPPMatrix::to_string() {
	String str;

	str += "[MLPPMatrix: \n";

	for (int y = 0; y < _size.y; ++y) {
		str += "  [ ";

		for (int x = 0; x < _size.x; ++x) {
			str += String::num(_data[_size.x * y + x]);
			str += " ";
		}

		str += "]\n";
	}

	str += "]";

	return str;
}

std::vector<real_t> MLPPMatrix::to_flat_std_vector() const {
	std::vector<real_t> ret;
	ret.resize(data_size());
	real_t *w = &ret[0];
	memcpy(w, _data, sizeof(real_t) * data_size());
	return ret;
}

void MLPPMatrix::set_from_std_vectors(const std::vector<std::vector<real_t>> &p_from) {
	if (p_from.size() == 0) {
		reset();
		return;
	}

	resize(Size2i(p_from[0].size(), p_from.size()));

	if (data_size() == 0) {
		reset();
		return;
	}

	for (uint32_t i = 0; i < p_from.size(); ++i) {
		const std::vector<real_t> &r = p_from[i];

		ERR_CONTINUE(r.size() != static_cast<uint32_t>(_size.x));

		int start_index = i * _size.x;

		const real_t *from_ptr = &r[0];
		for (int j = 0; j < _size.x; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
}

std::vector<std::vector<real_t>> MLPPMatrix::to_std_vector() {
	std::vector<std::vector<real_t>> ret;

	ret.resize(_size.y);

	for (int i = 0; i < _size.y; ++i) {
		std::vector<real_t> row;

		for (int j = 0; j < _size.x; ++j) {
			row.push_back(_data[calculate_index(i, j)]);
		}

		ret[i] = row;
	}

	return ret;
}

void MLPPMatrix::set_row_std_vector(int p_index_y, const std::vector<real_t> &p_row) {
	ERR_FAIL_COND(p_row.size() != static_cast<uint32_t>(_size.x));
	ERR_FAIL_INDEX(p_index_y, _size.y);

	int ind_start = p_index_y * _size.x;

	const real_t *row_ptr = &p_row[0];

	for (int i = 0; i < _size.x; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

MLPPMatrix::MLPPMatrix(const std::vector<std::vector<real_t>> &p_from) {
	_data = NULL;

	set_from_std_vectors(p_from);
}

void MLPPMatrix::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_row", "row"), &MLPPMatrix::add_row_pool_vector);
	ClassDB::bind_method(D_METHOD("add_row_mlpp_vector", "row"), &MLPPMatrix::add_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("add_rows_mlpp_matrix", "other"), &MLPPMatrix::add_rows_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("remove_row", "index"), &MLPPMatrix::remove_row);
	ClassDB::bind_method(D_METHOD("remove_row_unordered", "index"), &MLPPMatrix::remove_row_unordered);
	ClassDB::bind_method(D_METHOD("swap_row", "index_1", "index_2"), &MLPPMatrix::swap_row);

	ClassDB::bind_method(D_METHOD("clear"), &MLPPMatrix::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPMatrix::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPMatrix::empty);

	ClassDB::bind_method(D_METHOD("data_size"), &MLPPMatrix::data_size);
	ClassDB::bind_method(D_METHOD("size"), &MLPPMatrix::size);

	ClassDB::bind_method(D_METHOD("resize", "size"), &MLPPMatrix::resize);

	ClassDB::bind_method(D_METHOD("get_element_index", "index"), &MLPPMatrix::get_element_index);
	ClassDB::bind_method(D_METHOD("set_element_index", "index", "val"), &MLPPMatrix::set_element_index);

	ClassDB::bind_method(D_METHOD("get_element", "index_y", "index_x"), &MLPPMatrix::get_element);
	ClassDB::bind_method(D_METHOD("set_element", "index_y", "index_x", "val"), &MLPPMatrix::set_element);

	ClassDB::bind_method(D_METHOD("get_row_pool_vector", "index_y"), &MLPPMatrix::get_row_pool_vector);
	ClassDB::bind_method(D_METHOD("get_row_mlpp_vector", "index_y"), &MLPPMatrix::get_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("get_row_into_mlpp_vector", "index_y", "target"), &MLPPMatrix::get_row_into_mlpp_vector);

	ClassDB::bind_method(D_METHOD("set_row_pool_vector", "index_y", "row"), &MLPPMatrix::set_row_pool_vector);
	ClassDB::bind_method(D_METHOD("set_row_mlpp_vector", "index_y", "row"), &MLPPMatrix::set_row_mlpp_vector);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPMatrix::fill);

	ClassDB::bind_method(D_METHOD("to_flat_pool_vector"), &MLPPMatrix::to_flat_pool_vector);
	ClassDB::bind_method(D_METHOD("to_flat_byte_array"), &MLPPMatrix::to_flat_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPMatrix::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vectors_array", "from"), &MLPPMatrix::set_from_mlpp_vectors_array);
	ClassDB::bind_method(D_METHOD("set_from_arrays", "from"), &MLPPMatrix::set_from_arrays);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_matrix", "from"), &MLPPMatrix::set_from_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("is_equal_approx", "with", "tolerance"), &MLPPMatrix::is_equal_approx, CMP_EPSILON);
}
