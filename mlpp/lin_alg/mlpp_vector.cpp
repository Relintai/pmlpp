
#include "mlpp_vector.h"

#include "mlpp_matrix.h"

PoolRealArray MLPPVector::get_data() {
	PoolRealArray pl;
	if (size()) {
		pl.resize(size());
		PoolRealArray::Write w = pl.write();
		real_t *dest = w.ptr();

		for (int i = 0; i < size(); ++i) {
			dest[i] = _data[i];
		}
	}
	return pl;
}
void MLPPVector::set_data(const PoolRealArray &p_from) {
	if (_size != p_from.size()) {
		resize(p_from.size());
	}

	PoolRealArray::Read r = p_from.read();
	for (int i = 0; i < _size; i++) {
		_data[i] = r[i];
	}
}

void MLPPVector::push_back(real_t p_elem) {
	++_size;

	_data = (real_t *)memrealloc(_data, _size * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");

	_data[_size - 1] = p_elem;
}

void MLPPVector::append_mlpp_vector(const Ref<MLPPVector> &p_other) {
	ERR_FAIL_COND(!p_other.is_valid());

	int other_size = p_other->size();

	if (other_size == 0) {
		return;
	}

	int start_offset = _size;

	_size += other_size;

	_data = (real_t *)memrealloc(_data, _size * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");

	const real_t *other_ptr = p_other->ptr();

	for (int i = 0; i < other_size; ++i) {
		_data[start_offset + i] = other_ptr[i];
	}
}

void MLPPVector::remove(real_t p_index) {
	ERR_FAIL_INDEX(p_index, _size);

	--_size;

	if (_size == 0) {
		memfree(_data);
		_data = NULL;
		return;
	}

	for (int i = p_index; i < _size; i++) {
		_data[i] = _data[i + 1];
	}

	_data = (real_t *)memrealloc(_data, _size * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

// Removes the item copying the last value into the position of the one to
// remove. It's generally faster than `remove`.
void MLPPVector::remove_unordered(int p_index) {
	ERR_FAIL_INDEX(p_index, _size);
	_size--;

	if (_size == 0) {
		memfree(_data);
		_data = NULL;
		return;
	}

	if (_size > p_index) {
		_data[p_index] = _data[_size];
	}

	_data = (real_t *)memrealloc(_data, _size * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

void MLPPVector::erase(const real_t &p_val) {
	int idx = find(p_val);
	if (idx >= 0) {
		remove(idx);
	}
}

int MLPPVector::erase_multiple_unordered(const real_t &p_val) {
	int from = 0;
	int count = 0;
	while (true) {
		int64_t idx = find(p_val, from);

		if (idx == -1) {
			break;
		}
		remove_unordered(idx);
		from = idx;
		count++;
	}
	return count;
}

void MLPPVector::invert() {
	for (int i = 0; i < _size / 2; i++) {
		SWAP(_data[i], _data[_size - i - 1]);
	}
}

void MLPPVector::resize(int p_size) {
	_size = p_size;

	if (_size == 0) {
		memfree(_data);
		_data = NULL;
		return;
	}

	_data = (real_t *)memrealloc(_data, _size * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

void MLPPVector::fill(real_t p_val) {
	for (int i = 0; i < _size; i++) {
		_data[i] = p_val;
	}
}

void MLPPVector::insert(int p_pos, real_t p_val) {
	ERR_FAIL_INDEX(p_pos, _size + 1);
	if (p_pos == _size) {
		push_back(p_val);
	} else {
		resize(_size + 1);
		for (int i = _size - 1; i > p_pos; i--) {
			_data[i] = _data[i - 1];
		}
		_data[p_pos] = p_val;
	}
}

int MLPPVector::find(const real_t &p_val, int p_from) const {
	for (int i = p_from; i < _size; i++) {
		if (_data[i] == p_val) {
			return i;
		}
	}
	return -1;
}

void MLPPVector::ordered_insert(real_t p_val) {
	int i;
	for (i = 0; i < _size; i++) {
		if (p_val < _data[i]) {
			break;
		}
	}
	insert(i, p_val);
}

Vector<real_t> MLPPVector::to_vector() const {
	Vector<real_t> ret;
	ret.resize(size());
	real_t *w = ret.ptrw();
	memcpy(w, _data, sizeof(real_t) * _size);
	return ret;
}

PoolRealArray MLPPVector::to_pool_vector() const {
	PoolRealArray pl;
	if (size()) {
		pl.resize(size());
		PoolRealArray::Write w = pl.write();
		real_t *dest = w.ptr();

		for (int i = 0; i < size(); ++i) {
			dest[i] = static_cast<real_t>(_data[i]);
		}
	}
	return pl;
}

Vector<uint8_t> MLPPVector::to_byte_array() const {
	Vector<uint8_t> ret;
	ret.resize(_size * sizeof(real_t));
	uint8_t *w = ret.ptrw();
	memcpy(w, _data, sizeof(real_t) * _size);
	return ret;
}

Ref<MLPPVector> MLPPVector::duplicate_fast() const {
	Ref<MLPPVector> ret;
	ret.instance();

	ret->set_from_mlpp_vectorr(*this);

	return ret;
}

void MLPPVector::set_from_mlpp_vectorr(const MLPPVector &p_from) {
	if (_size != p_from.size()) {
		resize(p_from.size());
	}

	for (int i = 0; i < p_from._size; i++) {
		_data[i] = p_from._data[i];
	}
}

void MLPPVector::set_from_mlpp_vector(const Ref<MLPPVector> &p_from) {
	ERR_FAIL_COND(!p_from.is_valid());

	if (_size != p_from->size()) {
		resize(p_from->size());
	}

	for (int i = 0; i < p_from->_size; i++) {
		_data[i] = p_from->_data[i];
	}
}

void MLPPVector::set_from_vector(const Vector<real_t> &p_from) {
	if (_size != p_from.size()) {
		resize(p_from.size());
	}

	resize(p_from.size());
	for (int i = 0; i < _size; i++) {
		_data[i] = p_from[i];
	}
}

void MLPPVector::set_from_pool_vector(const PoolRealArray &p_from) {
	if (_size != p_from.size()) {
		resize(p_from.size());
	}

	PoolRealArray::Read r = p_from.read();
	for (int i = 0; i < _size; i++) {
		_data[i] = r[i];
	}
}

bool MLPPVector::is_equal_approx(const Ref<MLPPVector> &p_with, real_t tolerance) const {
	ERR_FAIL_COND_V(!p_with.is_valid(), false);

	if (unlikely(this == p_with.ptr())) {
		return true;
	}

	if (_size != p_with->size()) {
		return false;
	}

	for (int i = 0; i < _size; ++i) {
		if (!Math::is_equal_approx(_data[i], p_with->_data[i], tolerance)) {
			return false;
		}
	}

	return true;
}

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

Ref<MLPPVector> MLPPVector::flatten_vectorsn(const Vector<Ref<MLPPVector>> &A) const {
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
Ref<MLPPVector> MLPPVector::hadamard_productn(const Ref<MLPPVector> &b) const {
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

void MLPPVector::division_element_wise(const Ref<MLPPVector> &b) {
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

Ref<MLPPVector> MLPPVector::division_element_wisen(const Ref<MLPPVector> &b) const {
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

void MLPPVector::division_element_wiseb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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
Ref<MLPPVector> MLPPVector::scalar_multiplyn(real_t scalar) const {
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
Ref<MLPPVector> MLPPVector::scalar_addn(real_t scalar) const {
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
Ref<MLPPVector> MLPPVector::addn(const Ref<MLPPVector> &b) const {
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
Ref<MLPPVector> MLPPVector::subn(const Ref<MLPPVector> &b) const {
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
Ref<MLPPVector> MLPPVector::logn() const {
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
Ref<MLPPVector> MLPPVector::log10n() const {
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
Ref<MLPPVector> MLPPVector::expn() const {
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
Ref<MLPPVector> MLPPVector::erfn() const {
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
Ref<MLPPVector> MLPPVector::exponentiaten(real_t p) const {
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
Ref<MLPPVector> MLPPVector::sqrtn() const {
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
Ref<MLPPVector> MLPPVector::cbrtn() const {
	return exponentiaten(static_cast<real_t>(1) / static_cast<real_t>(3));
}
void MLPPVector::cbrtb(const Ref<MLPPVector> &a) {
	return exponentiateb(a, static_cast<real_t>(1) / static_cast<real_t>(3));
}

real_t MLPPVector::dot(const Ref<MLPPVector> &b) const {
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
Ref<MLPPVector> MLPPVector::absn() const {
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

Ref<MLPPVector> MLPPVector::vecn_zero(int n) const {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(0);

	return vec;
}
Ref<MLPPVector> MLPPVector::vecn_one(int n) const {
	Ref<MLPPVector> vec;
	vec.instance();

	vec->resize(n);
	vec->fill(1);

	return vec;
}
Ref<MLPPVector> MLPPVector::vecn_full(int n, int k) const {
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
Ref<MLPPVector> MLPPVector::sinn() const {
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
Ref<MLPPVector> MLPPVector::cosn() const {
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

void MLPPVector::max(const Ref<MLPPVector> &b) {
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
Ref<MLPPVector> MLPPVector::maxn(const Ref<MLPPVector> &b) const {
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
void MLPPVector::maxb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

void MLPPVector::min(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!b.is_valid());
	ERR_FAIL_COND(_size != b->size());

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < _size; ++i) {
		real_t aa_i = a_ptr[i];
		real_t bb_i = b_ptr[i];

		if (aa_i < bb_i) {
			out_ptr[i] = aa_i;
		} else {
			out_ptr[i] = bb_i;
		}
	}
}
Ref<MLPPVector> MLPPVector::minn(const Ref<MLPPVector> &b) const {
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

		if (aa_i < bb_i) {
			out_ptr[i] = aa_i;
		} else {
			out_ptr[i] = bb_i;
		}
	}

	return out;
}
void MLPPVector::minb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
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

		if (aa_i < bb_i) {
			out_ptr[i] = aa_i;
		} else {
			out_ptr[i] = bb_i;
		}
	}
}

real_t MLPPVector::max_element() const {
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
int MLPPVector::max_element_index() const {
	const real_t *aa = ptr();

	real_t max_element = -Math_INF;
	int index = -1;

	for (int i = 0; i < _size; i++) {
		real_t current_element = aa[i];

		if (current_element > max_element) {
			max_element = current_element;
			index = i;
		}
	}

	return index;
}

real_t MLPPVector::min_element() const {
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
int MLPPVector::min_element_index() const {
	const real_t *aa = ptr();

	real_t min_element = Math_INF;
	int index = -1;

	for (int i = 0; i < _size; i++) {
		real_t current_element = aa[i];

		if (current_element > min_element) {
			min_element = current_element;
			index = i;
		}
	}

	return index;
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

real_t MLPPVector::euclidean_distance(const Ref<MLPPVector> &b) const {
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
real_t MLPPVector::euclidean_distance_squared(const Ref<MLPPVector> &b) const {
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

real_t MLPPVector::norm_sq() const {
	const real_t *a_ptr = ptr();

	real_t n_sq = 0;
	for (int i = 0; i < _size; ++i) {
		n_sq += a_ptr[i] * a_ptr[i];
	}
	return n_sq;
}

real_t MLPPVector::sum_elements() const {
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
Ref<MLPPVector> MLPPVector::subtract_matrix_rowsn(const Ref<MLPPMatrix> &B) const {
	Ref<MLPPVector> c = duplicate_fast();

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

Ref<MLPPMatrix> MLPPVector::outer_product(const Ref<MLPPVector> &b) const {
	Ref<MLPPMatrix> C;
	C.instance();
	Size2i sm = Size2i(b->size(), size());
	C->resize(sm);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();

	for (int i = 0; i < sm.y; ++i) {
		real_t curr_a = a_ptr[i];

		for (int j = 0; j < sm.x; ++j) {
			C->element_set(i, j, curr_a * b_ptr[j]);
		}
	}

	return C;
}

Ref<MLPPMatrix> MLPPVector::diagnm() const {
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

MLPPVector::MLPPVector() {
	_size = 0;
	_data = NULL;
}
MLPPVector::MLPPVector(const MLPPVector &p_from) {
	_size = 0;
	_data = NULL;

	resize(p_from.size());
	for (int i = 0; i < p_from._size; i++) {
		_data[i] = p_from._data[i];
	}
}

MLPPVector::MLPPVector(const Vector<real_t> &p_from) {
	_size = 0;
	_data = NULL;

	resize(p_from.size());
	for (int i = 0; i < _size; i++) {
		_data[i] = p_from[i];
	}
}

MLPPVector::MLPPVector(const PoolRealArray &p_from) {
	_size = 0;
	_data = NULL;

	resize(p_from.size());
	PoolRealArray::Read r = p_from.read();
	for (int i = 0; i < _size; i++) {
		_data[i] = r[i];
	}
}

MLPPVector::~MLPPVector() {
	if (_data) {
		reset();
	}
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
	ClassDB::bind_method(D_METHOD("get_data"), &MLPPVector::get_data);
	ClassDB::bind_method(D_METHOD("set_data", "data"), &MLPPVector::set_data);
	ADD_PROPERTY(PropertyInfo(Variant::POOL_REAL_ARRAY, "data"), "set_data", "get_data");

	ClassDB::bind_method(D_METHOD("push_back", "elem"), &MLPPVector::push_back);
	ClassDB::bind_method(D_METHOD("append_mlpp_vector", "other"), &MLPPVector::append_mlpp_vector);
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

	ClassDB::bind_method(D_METHOD("element_get", "index"), &MLPPVector::element_get);
	ClassDB::bind_method(D_METHOD("element_set", "index", "val"), &MLPPVector::element_set);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPVector::fill);
	ClassDB::bind_method(D_METHOD("insert", "pos", "val"), &MLPPVector::insert);
	ClassDB::bind_method(D_METHOD("find", "val", "from"), &MLPPVector::find, 0);
	ClassDB::bind_method(D_METHOD("sort"), &MLPPVector::sort);
	ClassDB::bind_method(D_METHOD("ordered_insert", "val"), &MLPPVector::ordered_insert);

	ClassDB::bind_method(D_METHOD("to_pool_vector"), &MLPPVector::to_pool_vector);
	ClassDB::bind_method(D_METHOD("to_byte_array"), &MLPPVector::to_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate_fast"), &MLPPVector::duplicate_fast);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vector", "from"), &MLPPVector::set_from_mlpp_vector);
	ClassDB::bind_method(D_METHOD("set_from_pool_vector", "from"), &MLPPVector::set_from_pool_vector);

	ClassDB::bind_method(D_METHOD("is_equal_approx", "with", "tolerance"), &MLPPVector::is_equal_approx, CMP_EPSILON);

	ClassDB::bind_method(D_METHOD("hadamard_product", "b"), &MLPPVector::hadamard_product);
	ClassDB::bind_method(D_METHOD("hadamard_productn", "b"), &MLPPVector::hadamard_productn);
	ClassDB::bind_method(D_METHOD("hadamard_productb", "a", "b"), &MLPPVector::hadamard_productb);

	ClassDB::bind_method(D_METHOD("division_element_wise", "b"), &MLPPVector::division_element_wise);
	ClassDB::bind_method(D_METHOD("division_element_wisen", "b"), &MLPPVector::division_element_wisen);
	ClassDB::bind_method(D_METHOD("division_element_wiseb", "a", "b"), &MLPPVector::division_element_wiseb);

	ClassDB::bind_method(D_METHOD("scalar_multiply", "scalar"), &MLPPVector::scalar_multiply);
	ClassDB::bind_method(D_METHOD("scalar_multiplyn", "scalar"), &MLPPVector::scalar_multiplyn);
	ClassDB::bind_method(D_METHOD("scalar_multiplyb", "scalar", "a"), &MLPPVector::scalar_multiplyb);

	ClassDB::bind_method(D_METHOD("scalar_add", "scalar"), &MLPPVector::scalar_add);
	ClassDB::bind_method(D_METHOD("scalar_addn", "scalar"), &MLPPVector::scalar_addn);
	ClassDB::bind_method(D_METHOD("scalar_addb", "scalar", "a"), &MLPPVector::scalar_addb);

	ClassDB::bind_method(D_METHOD("add", "b"), &MLPPVector::add);
	ClassDB::bind_method(D_METHOD("addn", "b"), &MLPPVector::addn);
	ClassDB::bind_method(D_METHOD("addb", "a", "b"), &MLPPVector::addb);

	ClassDB::bind_method(D_METHOD("sub", "b"), &MLPPVector::sub);
	ClassDB::bind_method(D_METHOD("subn", "b"), &MLPPVector::subn);
	ClassDB::bind_method(D_METHOD("subb", "a", "b"), &MLPPVector::subb);

	ClassDB::bind_method(D_METHOD("log"), &MLPPVector::log);
	ClassDB::bind_method(D_METHOD("logn"), &MLPPVector::logn);
	ClassDB::bind_method(D_METHOD("logb", "a"), &MLPPVector::logb);

	ClassDB::bind_method(D_METHOD("log10"), &MLPPVector::log10);
	ClassDB::bind_method(D_METHOD("log10n"), &MLPPVector::log10n);
	ClassDB::bind_method(D_METHOD("log10b", "a"), &MLPPVector::log10b);

	ClassDB::bind_method(D_METHOD("exp"), &MLPPVector::exp);
	ClassDB::bind_method(D_METHOD("expn"), &MLPPVector::expn);
	ClassDB::bind_method(D_METHOD("expb", "a"), &MLPPVector::expb);

	ClassDB::bind_method(D_METHOD("erf"), &MLPPVector::erf);
	ClassDB::bind_method(D_METHOD("erfn"), &MLPPVector::erfn);
	ClassDB::bind_method(D_METHOD("erfb", "a"), &MLPPVector::erfb);

	ClassDB::bind_method(D_METHOD("exponentiate", "p"), &MLPPVector::exponentiate);
	ClassDB::bind_method(D_METHOD("exponentiaten", "p"), &MLPPVector::exponentiaten);
	ClassDB::bind_method(D_METHOD("exponentiateb", "a", "p"), &MLPPVector::exponentiateb);

	ClassDB::bind_method(D_METHOD("sqrt"), &MLPPVector::sqrt);
	ClassDB::bind_method(D_METHOD("sqrtn"), &MLPPVector::sqrtn);
	ClassDB::bind_method(D_METHOD("sqrtb", "a"), &MLPPVector::sqrtb);

	ClassDB::bind_method(D_METHOD("cbrt"), &MLPPVector::cbrt);
	ClassDB::bind_method(D_METHOD("cbrtn"), &MLPPVector::cbrtn);
	ClassDB::bind_method(D_METHOD("cbrtb", "a"), &MLPPVector::cbrtb);

	ClassDB::bind_method(D_METHOD("dot", "b"), &MLPPVector::dot);

	ClassDB::bind_method(D_METHOD("abs"), &MLPPVector::abs);
	ClassDB::bind_method(D_METHOD("absn"), &MLPPVector::absn);
	ClassDB::bind_method(D_METHOD("absb", "a"), &MLPPVector::absb);

	ClassDB::bind_method(D_METHOD("vecn_zero", "n"), &MLPPVector::vecn_zero);
	ClassDB::bind_method(D_METHOD("vecn_one", "n"), &MLPPVector::vecn_one);
	ClassDB::bind_method(D_METHOD("vecn_full", "n", "k"), &MLPPVector::vecn_full);

	ClassDB::bind_method(D_METHOD("sin"), &MLPPVector::sin);
	ClassDB::bind_method(D_METHOD("sinn"), &MLPPVector::sinn);
	ClassDB::bind_method(D_METHOD("sinb", "a"), &MLPPVector::sinb);

	ClassDB::bind_method(D_METHOD("cos"), &MLPPVector::cos);
	ClassDB::bind_method(D_METHOD("cosn"), &MLPPVector::cosn);
	ClassDB::bind_method(D_METHOD("cosb", "a"), &MLPPVector::cosb);

	ClassDB::bind_method(D_METHOD("max", "b"), &MLPPVector::max);
	ClassDB::bind_method(D_METHOD("maxn", "b"), &MLPPVector::maxn);
	ClassDB::bind_method(D_METHOD("maxb", "a", "b"), &MLPPVector::maxb);

	ClassDB::bind_method(D_METHOD("min", "b"), &MLPPVector::min);
	ClassDB::bind_method(D_METHOD("minn", "b"), &MLPPVector::minn);
	ClassDB::bind_method(D_METHOD("minb", "a", "b"), &MLPPVector::minb);

	ClassDB::bind_method(D_METHOD("max_element"), &MLPPVector::max_element);
	ClassDB::bind_method(D_METHOD("max_element_index"), &MLPPVector::max_element_index);

	ClassDB::bind_method(D_METHOD("min_element"), &MLPPVector::min_element);
	ClassDB::bind_method(D_METHOD("min_element_index"), &MLPPVector::min_element_index);

	ClassDB::bind_method(D_METHOD("euclidean_distance", "b"), &MLPPVector::euclidean_distance);
	ClassDB::bind_method(D_METHOD("euclidean_distance_squared", "b"), &MLPPVector::euclidean_distance_squared);

	ClassDB::bind_method(D_METHOD("norm_sq"), &MLPPVector::norm_sq);
	ClassDB::bind_method(D_METHOD("sum_elements"), &MLPPVector::sum_elements);

	ClassDB::bind_method(D_METHOD("subtract_matrix_rows", "B"), &MLPPVector::subtract_matrix_rows);
	ClassDB::bind_method(D_METHOD("subtract_matrix_rowsn", "B"), &MLPPVector::subtract_matrix_rowsn);
	ClassDB::bind_method(D_METHOD("subtract_matrix_rowsb", "a", "B"), &MLPPVector::subtract_matrix_rowsb);

	ClassDB::bind_method(D_METHOD("outer_product", "b"), &MLPPVector::outer_product);

	ClassDB::bind_method(D_METHOD("diagnm"), &MLPPVector::diagnm);
}
