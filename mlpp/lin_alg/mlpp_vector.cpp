
#include "mlpp_vector.h"

#include "mlpp_matrix.h"

void MLPPVector::flatten_vectors(const Vector<Ref<MLPPVector>> &A) {
	int vsize = 0;
	for (int i = 0; i < A.size(); ++i) {
		vsize += A[i]->size();
	}

	resize(vsize);

	int a_index = 0;
	real_t *a_ptr = ptrw();

	for (int i = 0; i < A.size(); ++i) {
		const Ref<MLPPVector> &r = A[i];

		int r_size = r->size();
		const real_t *r_ptr = r->ptr();

		for (int j = 0; j < r_size; ++j) {
			a_ptr[a_index] = r_ptr[j];
			++a_index;
		}
	}
}

Ref<MLPPVector> MLPPVector::flatten_vectorsn(const Vector<Ref<MLPPVector>> &A) {
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

void MLPPVector::hadamard_product(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!b.is_valid());

	ERR_FAIL_COND(_size != b->size());

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] * b_ptr[i];
	}
}
Ref<MLPPVector> MLPPVector::hadamard_productn(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	ERR_FAIL_COND_V(_size != b->size(), Ref<MLPPVector>());

	out->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] * b_ptr[i];
	}

	return out;
}
void MLPPVector::hadamard_productb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid());

	int s = a->size();

	ERR_FAIL_COND(s != b->size());

	if (unlikely(size() != s)) {
		resize(s);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = a_ptr[i] * b_ptr[i];
	}
}

void MLPPVector::element_wise_division(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!b.is_valid());

	Ref<MLPPVector> out;
	out.instance();

	ERR_FAIL_COND(_size != b->size());

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] / b_ptr[i];
	}
}

Ref<MLPPVector> MLPPVector::element_wise_divisionn(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();

	ERR_FAIL_COND_V(_size != b->size(), Ref<MLPPVector>());

	out->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] / b_ptr[i];
	}

	return out;
}

void MLPPVector::element_wise_divisionb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid());

	int s = a->size();

	ERR_FAIL_COND(s != b->size());

	resize(s);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = a_ptr[i] / b_ptr[i];
	}
}

void MLPPVector::scalar_multiply(real_t scalar) {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = out_ptr[i] * scalar;
	}
}
Ref<MLPPVector> MLPPVector::scalar_multiplyn(real_t scalar) {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] * scalar;
	}

	return out;
}
void MLPPVector::scalar_multiplyb(real_t scalar, const Ref<MLPPVector> &a) {
	int s = a->size();

	if (unlikely(size() != s)) {
		resize(s);
	}

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = a_ptr[i] * scalar;
	}
}

void MLPPVector::scalar_add(real_t scalar) {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = out_ptr[i] + scalar;
	}
}
Ref<MLPPVector> MLPPVector::scalar_addn(real_t scalar) {
	Ref<MLPPVector> out;
	out.instance();

	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] + scalar;
	}

	return out;
}
void MLPPVector::scalar_addb(real_t scalar, const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();

	if (unlikely(size() != s)) {
		resize(s);
	}

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = a_ptr[i] + scalar;
	}
}

void MLPPVector::add(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!b.is_valid());
	ERR_FAIL_COND(_size != b->size());

	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] += b_ptr[i];
	}
}
Ref<MLPPVector> MLPPVector::addn(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), Ref<MLPPVector>());
	ERR_FAIL_COND_V(_size != b->size(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] + b_ptr[i];
	}

	return out;
}
void MLPPVector::addb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid());

	int s = a->size();

	ERR_FAIL_COND(s != b->size());

	if (unlikely(size() != s)) {
		resize(s);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = a_ptr[i] + b_ptr[i];
	}
}

void MLPPVector::sub(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!b.is_valid());
	ERR_FAIL_COND(_size != b->size());

	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] -= b_ptr[i];
	}
}
Ref<MLPPVector> MLPPVector::subn(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), Ref<MLPPVector>());
	ERR_FAIL_COND_V(_size != b->size(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = a_ptr[i] - b_ptr[i];
	}

	return out;
}
void MLPPVector::subb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid());

	int s = a->size();

	ERR_FAIL_COND(s != b->size());

	if (unlikely(size() != s)) {
		resize(s);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = a_ptr[i] - b_ptr[i];
	}
}

void MLPPVector::log() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::log(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::logn() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::log(a_ptr[i]);
	}

	return out;
}
void MLPPVector::logb(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::log(a_ptr[i]);
	}
}

void MLPPVector::log10() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::log10(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::log10n() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::log10(a_ptr[i]);
	}

	return out;
}
void MLPPVector::log10b(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::log10(a_ptr[i]);
	}
}

void MLPPVector::exp() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::exp(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::expn() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::exp(a_ptr[i]);
	}

	return out;
}
void MLPPVector::expb(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::exp(a_ptr[i]);
	}
}

void MLPPVector::erf() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::erf(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::erfn() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::erf(a_ptr[i]);
	}

	return out;
}
void MLPPVector::erfb(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::erf(a_ptr[i]);
	}
}

void MLPPVector::exponentiate(real_t p) {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::pow(out_ptr[i], p);
	}
}
Ref<MLPPVector> MLPPVector::exponentiaten(real_t p) {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}

	return out;
}
void MLPPVector::exponentiateb(const Ref<MLPPVector> &a, real_t p) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}
}

void MLPPVector::sqrt() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::sqrt(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::sqrtn() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}

	return out;
}
void MLPPVector::sqrtb(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}
}

void MLPPVector::cbrt() {
	return exponentiate(static_cast<real_t>(1) / static_cast<real_t>(3));
}
Ref<MLPPVector> MLPPVector::cbrtn() {
	return exponentiaten(static_cast<real_t>(1) / static_cast<real_t>(3));
}
void MLPPVector::cbrtb(const Ref<MLPPVector> &a) {
	return exponentiateb(a, static_cast<real_t>(1) / static_cast<real_t>(3));
}

real_t MLPPVector::dot(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), 0);

	ERR_FAIL_COND_V(_size != b->size(), 0);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();

	real_t c = 0;
	for (int i = 0; i < _size; ++i) {
		c += a_ptr[i] * b_ptr[i];
	}
	return c;
}

/*
std::vector<real_t> MLPPVector::cross(std::vector<real_t> a, std::vector<real_t> b) {
	// Cross products exist in R^7 also. Though, I will limit it to R^3 as Wolfram does this.
	std::vector<std::vector<real_t>> mat = { onevec(3), a, b };

	real_t det1 = det({ { a[1], a[2] }, { b[1], b[2] } }, 2);
	real_t det2 = -det({ { a[0], a[2] }, { b[0], b[2] } }, 2);
	real_t det3 = det({ { a[0], a[1] }, { b[0], b[1] } }, 2);

	return { det1, det2, det3 };
}
*/

void MLPPVector::abs() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = ABS(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::absn() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}

	return out;
}
void MLPPVector::absb(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}
}

Ref<MLPPVector> MLPPVector::zero_vec(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(0);

	return vec;
}
Ref<MLPPVector> MLPPVector::one_vec(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(1);

	return vec;
}
Ref<MLPPVector> MLPPVector::full_vec(int n, int k) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(k);

	return vec;
}

void MLPPVector::sin() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::sin(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::sinn() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::sin(a_ptr[i]);
	}

	return out;
}
void MLPPVector::sinb(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::sin(a_ptr[i]);
	}
}

void MLPPVector::cos() {
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::sqrt(out_ptr[i]);
	}
}
Ref<MLPPVector> MLPPVector::cosn() {
	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		out_ptr[i] = Math::cos(a_ptr[i]);
	}

	return out;
}
void MLPPVector::cosb(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int s = a->size();
	resize(s);

	const real_t *a_ptr = a->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		out_ptr[i] = Math::cos(a_ptr[i]);
	}
}

void MLPPVector::maxv(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!b.is_valid());
	ERR_FAIL_COND(_size != b->size());

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		real_t aa_i = a_ptr[i];
		real_t bb_i = b_ptr[i];

		if (aa_i > bb_i) {
			out_ptr[i] = aa_i;
		} else {
			out_ptr[i] = bb_i;
		}
	}
}
Ref<MLPPVector> MLPPVector::maxvn(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), Ref<MLPPVector>());
	ERR_FAIL_COND_V(_size != b->size(), Ref<MLPPVector>());

	Ref<MLPPVector> out;
	out.instance();
	out->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < _size; ++i) {
		real_t aa_i = a_ptr[i];
		real_t bb_i = b_ptr[i];

		if (aa_i > bb_i) {
			out_ptr[i] = aa_i;
		} else {
			out_ptr[i] = bb_i;
		}
	}

	return out;
}
void MLPPVector::maxvb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid());

	int s = a->size();

	ERR_FAIL_COND(s != b->size());

	if (unlikely(size() != s)) {
		resize(s);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < s; ++i) {
		real_t aa_i = a_ptr[i];
		real_t bb_i = b_ptr[i];

		if (aa_i > bb_i) {
			out_ptr[i] = aa_i;
		} else {
			out_ptr[i] = bb_i;
		}
	}
}

real_t MLPPVector::max_element() {
	const real_t *aa = ptr();

	real_t max_element = -Math_INF;

	for (int i = 0; i < _size; i++) {
		real_t current_element = aa[i];

		if (current_element > max_element) {
			max_element = current_element;
		}
	}

	return max_element;
}
real_t MLPPVector::min_element() {
	const real_t *aa = ptr();

	real_t min_element = Math_INF;

	for (int i = 0; i < _size; i++) {
		real_t current_element = aa[i];

		if (current_element > min_element) {
			min_element = current_element;
		}
	}

	return min_element;
}

/*

std::vector<std::vector<real_t>> MLPPVector::round(std::vector<std::vector<real_t>> A) {
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

real_t MLPPVector::euclidean_distance(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), 0);

	ERR_FAIL_COND_V(_size != b->size(), 0);

	const real_t *aa = ptr();
	const real_t *ba = b->ptr();

	real_t dist = 0;

	for (int i = 0; i < _size; i++) {
		dist += (aa[i] - ba[i]) * (aa[i] - ba[i]);
	}

	return Math::sqrt(dist);
}
real_t MLPPVector::euclidean_distance_squared(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND_V(!b.is_valid(), 0);

	ERR_FAIL_COND_V(_size != b->size(), 0);

	const real_t *aa = ptr();
	const real_t *ba = b->ptr();

	real_t dist = 0;

	for (int i = 0; i < _size; i++) {
		dist += (aa[i] - ba[i]) * (aa[i] - ba[i]);
	}

	return dist;
}

/*
real_t MLPPVector::norm_2(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j] * A[i][j];
		}
	}
	return Math::sqrt(sum);
}
*/

real_t MLPPVector::norm_sq() {
	const real_t *a_ptr = ptr();

	real_t n_sq = 0;
	for (int i = 0; i < _size; ++i) {
		n_sq += a_ptr[i] * a_ptr[i];
	}
	return n_sq;
}

real_t MLPPVector::sum_elements() {
	const real_t *a_ptr = ptr();

	real_t sum = 0;
	for (int i = 0; i < _size; ++i) {
		sum += a_ptr[i];
	}
	return sum;
}

/*
real_t MLPPVector::cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b) {
	return dot(a, b) / (norm_2(a) * norm_2(b));
}
*/

void MLPPVector::subtract_matrix_rows(const Ref<MLPPMatrix> &B) {
	Size2i b_size = B->size();

	ERR_FAIL_COND(b_size.x != size());

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < b_size.y; ++i) {
		for (int j = 0; j < b_size.x; ++j) {
			c_ptr[j] -= b_ptr[B->calculate_index(i, j)];
		}
	}
}
Ref<MLPPVector> MLPPVector::subtract_matrix_rowsn(const Ref<MLPPMatrix> &B) {
	Ref<MLPPVector> c = duplicate();

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
void MLPPVector::subtract_matrix_rowsb(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B) {
	Size2i b_size = B->size();

	ERR_FAIL_COND(b_size.x != a->size());

	set_from_mlpp_vector(a);

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < b_size.y; ++i) {
		for (int j = 0; j < b_size.x; ++j) {
			c_ptr[j] -= b_ptr[B->calculate_index(i, j)];
		}
	}
}

Ref<MLPPMatrix> MLPPVector::outer_product(const Ref<MLPPVector> &b) {
	Ref<MLPPMatrix> C;
	C.instance();
	Size2i sm = Size2i(b->size(), size());
	C->resize(sm);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();

	for (int i = 0; i < sm.y; ++i) {
		real_t curr_a = a_ptr[i];

		for (int j = 0; j < sm.x; ++j) {
			C->set_element(i, j, curr_a * b_ptr[j]);
		}
	}

	return C;
}

Ref<MLPPMatrix> MLPPVector::diagnm() {
	Ref<MLPPMatrix> B;
	B.instance();

	B->resize(Size2i(_size, _size));
	B->fill(0);

	const real_t *a_ptr = ptr();
	real_t *b_ptr = B->ptrw();

	for (int i = 0; i < _size; ++i) {
		b_ptr[B->calculate_index(i, i)] = a_ptr[i];
	}

	return B;
}

String MLPPVector::to_string() {
	String str;

	str += "[MLPPVector: ";

	for (int x = 0; x < _size; ++x) {
		str += String::num(_data[x]);
		str += " ";
	}

	str += "]";

	return str;
}

std::vector<real_t> MLPPVector::to_std_vector() const {
	std::vector<real_t> ret;
	ret.resize(size());
	real_t *w = &ret[0];
	memcpy(w, _data, sizeof(real_t) * _size);
	return ret;
}

void MLPPVector::set_from_std_vector(const std::vector<real_t> &p_from) {
	resize(p_from.size());
	for (int i = 0; i < _size; i++) {
		_data[i] = p_from[i];
	}
}

MLPPVector::MLPPVector(const std::vector<real_t> &p_from) {
	_size = 0;
	_data = NULL;

	resize(p_from.size());
	for (int i = 0; i < _size; i++) {
		_data[i] = p_from[i];
	}
}

void MLPPVector::_bind_methods() {
	ClassDB::bind_method(D_METHOD("push_back", "elem"), &MLPPVector::push_back);
	ClassDB::bind_method(D_METHOD("add_mlpp_vector", "other"), &MLPPVector::push_back);
	ClassDB::bind_method(D_METHOD("remove", "index"), &MLPPVector::remove);
	ClassDB::bind_method(D_METHOD("remove_unordered", "index"), &MLPPVector::remove_unordered);
	ClassDB::bind_method(D_METHOD("erase", "val"), &MLPPVector::erase);
	ClassDB::bind_method(D_METHOD("erase_multiple_unordered", "val"), &MLPPVector::erase_multiple_unordered);
	ClassDB::bind_method(D_METHOD("invert"), &MLPPVector::invert);
	ClassDB::bind_method(D_METHOD("clear"), &MLPPVector::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPVector::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPVector::empty);

	ClassDB::bind_method(D_METHOD("size"), &MLPPVector::size);
	ClassDB::bind_method(D_METHOD("resize", "size"), &MLPPVector::resize);

	ClassDB::bind_method(D_METHOD("get_element", "index"), &MLPPVector::get_element);
	ClassDB::bind_method(D_METHOD("set_element", "index", "val"), &MLPPVector::set_element);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPVector::fill);
	ClassDB::bind_method(D_METHOD("insert", "pos", "val"), &MLPPVector::insert);
	ClassDB::bind_method(D_METHOD("find", "val", "from"), &MLPPVector::find, 0);
	ClassDB::bind_method(D_METHOD("sort"), &MLPPVector::sort);
	ClassDB::bind_method(D_METHOD("ordered_insert", "val"), &MLPPVector::ordered_insert);

	ClassDB::bind_method(D_METHOD("to_pool_vector"), &MLPPVector::to_pool_vector);
	ClassDB::bind_method(D_METHOD("to_byte_array"), &MLPPVector::to_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPVector::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vector", "from"), &MLPPVector::set_from_mlpp_vector);
	ClassDB::bind_method(D_METHOD("set_from_pool_vector", "from"), &MLPPVector::set_from_pool_vector);

	ClassDB::bind_method(D_METHOD("is_equal_approx", "with", "tolerance"), &MLPPVector::is_equal_approx, CMP_EPSILON);
}
