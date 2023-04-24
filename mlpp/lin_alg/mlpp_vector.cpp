
#include "mlpp_vector.h"

#include "mlpp_matrix.h"

Ref<MLPPVector> MLPPVector::flattenmnv(const Vector<Ref<MLPPVector>> &A) {
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

Ref<MLPPVector> MLPPVector::hadamard_productnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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
void MLPPVector::hadamard_productv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out) {
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

Ref<MLPPVector> MLPPVector::element_wise_divisionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

Ref<MLPPVector> MLPPVector::scalar_multiplynv(real_t scalar, const Ref<MLPPVector> &a) {
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
void MLPPVector::scalar_multiplyv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out) {
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

Ref<MLPPVector> MLPPVector::scalar_addnv(real_t scalar, const Ref<MLPPVector> &a) {
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
void MLPPVector::scalar_addv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out) {
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

Ref<MLPPVector> MLPPVector::additionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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
void MLPPVector::additionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out) {
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

Ref<MLPPVector> MLPPVector::subtractionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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
void MLPPVector::subtractionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out) {
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

Ref<MLPPVector> MLPPVector::lognv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPVector::log10nv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPVector::expnv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPVector::erfnv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPVector::exponentiatenv(const Ref<MLPPVector> &a, real_t p) {
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
Ref<MLPPVector> MLPPVector::sqrtnv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPVector::cbrtnv(const Ref<MLPPVector> &a) {
	return exponentiatenv(a, static_cast<real_t>(1) / static_cast<real_t>(3));
}

real_t MLPPVector::dotnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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
std::vector<real_t> MLPPVector::cross(std::vector<real_t> a, std::vector<real_t> b) {
	// Cross products exist in R^7 also. Though, I will limit it to R^3 as Wolfram does this.
	std::vector<std::vector<real_t>> mat = { onevec(3), a, b };

	real_t det1 = det({ { a[1], a[2] }, { b[1], b[2] } }, 2);
	real_t det2 = -det({ { a[0], a[2] }, { b[0], b[2] } }, 2);
	real_t det3 = det({ { a[0], a[1] }, { b[0], b[1] } }, 2);

	return { det1, det2, det3 };
}
*/

Ref<MLPPVector> MLPPVector::absv(const Ref<MLPPVector> &a) {
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

Ref<MLPPVector> MLPPVector::zerovecnv(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(0);

	return vec;
}
Ref<MLPPVector> MLPPVector::onevecnv(int n) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(1);

	return vec;
}
Ref<MLPPVector> MLPPVector::fullnv(int n, int k) {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(k);

	return vec;
}

Ref<MLPPVector> MLPPVector::sinnv(const Ref<MLPPVector> &a) {
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
Ref<MLPPVector> MLPPVector::cosnv(const Ref<MLPPVector> &a) {
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

Ref<MLPPVector> MLPPVector::maxnvv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

real_t MLPPVector::maxvr(const Ref<MLPPVector> &a) {
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
real_t MLPPVector::minvr(const Ref<MLPPVector> &a) {
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

real_t MLPPVector::euclidean_distance(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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
real_t MLPPVector::euclidean_distance_squared(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

real_t MLPPVector::norm_sqv(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND_V(!a.is_valid(), 0);

	int size = a->size();
	const real_t *a_ptr = a->ptr();

	real_t n_sq = 0;
	for (int i = 0; i < size; ++i) {
		n_sq += a_ptr[i] * a_ptr[i];
	}
	return n_sq;
}

real_t MLPPVector::sum_elementsv(const Ref<MLPPVector> &a) {
	int a_size = a->size();

	const real_t *a_ptr = a->ptr();

	real_t sum = 0;
	for (int i = 0; i < a_size; ++i) {
		sum += a_ptr[i];
	}
	return sum;
}

/*
real_t MLPPVector::cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b) {
	return dot(a, b) / (norm_2(a) * norm_2(b));
}
*/

Ref<MLPPVector> MLPPVector::subtract_matrix_rowsnv(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B) {
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

Ref<MLPPMatrix> MLPPVector::outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

Ref<MLPPMatrix> MLPPVector::diagnm(const Ref<MLPPVector> &a) {
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
