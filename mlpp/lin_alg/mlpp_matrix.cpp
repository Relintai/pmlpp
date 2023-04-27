
#include "mlpp_matrix.h"

#include "core/io/image.h"

#include "../stat/stat.h"
#include <random>

void MLPPMatrix::add_row(const Vector<real_t> &p_row) {
	if (p_row.size() == 0) {
		return;
	}

	if (_size.x == 0) {
		_size.x = p_row.size();
	}

	ERR_FAIL_COND(_size.x != p_row.size());

	int ci = data_size();

	++_size.y;

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");

	const real_t *row_arr = p_row.ptr();

	for (int i = 0; i < p_row.size(); ++i) {
		_data[ci + i] = row_arr[i];
	}
}

void MLPPMatrix::add_row_pool_vector(const PoolRealArray &p_row) {
	if (p_row.size() == 0) {
		return;
	}

	if (_size.x == 0) {
		_size.x = p_row.size();
	}

	ERR_FAIL_COND(_size.x != p_row.size());

	int ci = data_size();

	++_size.y;

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");

	PoolRealArray::Read rread = p_row.read();
	const real_t *row_arr = rread.ptr();

	for (int i = 0; i < p_row.size(); ++i) {
		_data[ci + i] = row_arr[i];
	}
}

void MLPPMatrix::add_row_mlpp_vector(const Ref<MLPPVector> &p_row) {
	ERR_FAIL_COND(!p_row.is_valid());

	int p_row_size = p_row->size();

	if (p_row_size == 0) {
		return;
	}

	if (_size.x == 0) {
		_size.x = p_row_size;
	}

	ERR_FAIL_COND(_size.x != p_row_size);

	int ci = data_size();

	++_size.y;

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");

	const real_t *row_ptr = p_row->ptr();

	for (int i = 0; i < p_row_size; ++i) {
		_data[ci + i] = row_ptr[i];
	}
}

void MLPPMatrix::add_rows_mlpp_matrix(const Ref<MLPPMatrix> &p_other) {
	ERR_FAIL_COND(!p_other.is_valid());

	int other_data_size = p_other->data_size();

	if (other_data_size == 0) {
		return;
	}

	Size2i other_size = p_other->size();

	if (_size.x == 0) {
		_size.x = other_size.x;
	}

	ERR_FAIL_COND(other_size.x != _size.x);

	int start_offset = data_size();

	_size.y += other_size.y;

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");

	const real_t *other_ptr = p_other->ptr();

	for (int i = 0; i < other_data_size; ++i) {
		_data[start_offset + i] = other_ptr[i];
	}
}

void MLPPMatrix::remove_row(int p_index) {
	ERR_FAIL_INDEX(p_index, _size.y);

	--_size.y;

	int ds = data_size();

	if (ds == 0) {
		memfree(_data);
		_data = NULL;
		return;
	}

	for (int i = p_index * _size.x; i < ds; ++i) {
		_data[i] = _data[i + _size.x];
	}

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

// Removes the item copying the last value into the position of the one to
// remove. It's generally faster than `remove`.
void MLPPMatrix::remove_row_unordered(int p_index) {
	ERR_FAIL_INDEX(p_index, _size.y);

	--_size.y;

	int ds = data_size();

	if (ds == 0) {
		memfree(_data);
		_data = NULL;
		return;
	}

	int start_ind = p_index * _size.x;
	int last_row_start_ind = _size.y * _size.x;

	if (start_ind != last_row_start_ind) {
		for (int i = 0; i < _size.x; ++i) {
			_data[start_ind + i] = _data[last_row_start_ind + i];
		}
	}

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

void MLPPMatrix::swap_row(int p_index_1, int p_index_2) {
	ERR_FAIL_INDEX(p_index_1, _size.y);
	ERR_FAIL_INDEX(p_index_2, _size.y);

	int ind1_start = p_index_1 * _size.x;
	int ind2_start = p_index_2 * _size.x;

	for (int i = 0; i < _size.x; ++i) {
		SWAP(_data[ind1_start + i], _data[ind2_start + i]);
	}
}

void MLPPMatrix::resize(const Size2i &p_size) {
	_size = p_size;

	int ds = data_size();

	if (ds == 0) {
		if (_data) {
			memfree(_data);
			_data = NULL;
		}

		return;
	}

	_data = (real_t *)memrealloc(_data, ds * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

Vector<real_t> MLPPMatrix::get_row_vector(int p_index_y) const {
	ERR_FAIL_INDEX_V(p_index_y, _size.y, Vector<real_t>());

	Vector<real_t> ret;

	if (unlikely(_size.x == 0)) {
		return ret;
	}

	ret.resize(_size.x);

	int ind_start = p_index_y * _size.x;

	real_t *row_ptr = ret.ptrw();

	for (int i = 0; i < _size.x; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}

	return ret;
}

PoolRealArray MLPPMatrix::get_row_pool_vector(int p_index_y) const {
	ERR_FAIL_INDEX_V(p_index_y, _size.y, PoolRealArray());

	PoolRealArray ret;

	if (unlikely(_size.x == 0)) {
		return ret;
	}

	ret.resize(_size.x);

	int ind_start = p_index_y * _size.x;

	PoolRealArray::Write w = ret.write();
	real_t *row_ptr = w.ptr();

	for (int i = 0; i < _size.x; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}

	return ret;
}

Ref<MLPPVector> MLPPMatrix::get_row_mlpp_vector(int p_index_y) const {
	ERR_FAIL_INDEX_V(p_index_y, _size.y, Ref<MLPPVector>());

	Ref<MLPPVector> ret;
	ret.instance();

	if (unlikely(_size.x == 0)) {
		return ret;
	}

	ret->resize(_size.x);

	int ind_start = p_index_y * _size.x;

	real_t *row_ptr = ret->ptrw();

	for (int i = 0; i < _size.x; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}

	return ret;
}

void MLPPMatrix::get_row_into_mlpp_vector(int p_index_y, Ref<MLPPVector> target) const {
	ERR_FAIL_COND(!target.is_valid());
	ERR_FAIL_INDEX(p_index_y, _size.y);

	if (unlikely(target->size() != _size.x)) {
		target->resize(_size.x);
	}

	int ind_start = p_index_y * _size.x;

	real_t *row_ptr = target->ptrw();

	for (int i = 0; i < _size.x; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}
}

void MLPPMatrix::set_row_vector(int p_index_y, const Vector<real_t> &p_row) {
	ERR_FAIL_COND(p_row.size() != _size.x);
	ERR_FAIL_INDEX(p_index_y, _size.y);

	int ind_start = p_index_y * _size.x;

	const real_t *row_ptr = p_row.ptr();

	for (int i = 0; i < _size.x; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPMatrix::set_row_pool_vector(int p_index_y, const PoolRealArray &p_row) {
	ERR_FAIL_COND(p_row.size() != _size.x);
	ERR_FAIL_INDEX(p_index_y, _size.y);

	int ind_start = p_index_y * _size.x;

	PoolRealArray::Read r = p_row.read();
	const real_t *row_ptr = r.ptr();

	for (int i = 0; i < _size.x; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPMatrix::set_row_mlpp_vector(int p_index_y, const Ref<MLPPVector> &p_row) {
	ERR_FAIL_COND(!p_row.is_valid());
	ERR_FAIL_COND(p_row->size() != _size.x);
	ERR_FAIL_INDEX(p_index_y, _size.y);

	int ind_start = p_index_y * _size.x;

	const real_t *row_ptr = p_row->ptr();

	for (int i = 0; i < _size.x; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPMatrix::fill(real_t p_val) {
	if (!_data) {
		return;
	}

	int ds = data_size();
	for (int i = 0; i < ds; ++i) {
		_data[i] = p_val;
	}
}

Vector<real_t> MLPPMatrix::to_flat_vector() const {
	Vector<real_t> ret;
	ret.resize(data_size());
	real_t *w = ret.ptrw();
	memcpy(w, _data, sizeof(real_t) * data_size());
	return ret;
}

PoolRealArray MLPPMatrix::to_flat_pool_vector() const {
	PoolRealArray pl;
	if (data_size()) {
		pl.resize(data_size());
		typename PoolRealArray::Write w = pl.write();
		real_t *dest = w.ptr();

		for (int i = 0; i < data_size(); ++i) {
			dest[i] = static_cast<real_t>(_data[i]);
		}
	}
	return pl;
}

Vector<uint8_t> MLPPMatrix::to_flat_byte_array() const {
	Vector<uint8_t> ret;
	ret.resize(data_size() * sizeof(real_t));
	uint8_t *w = ret.ptrw();
	memcpy(w, _data, sizeof(real_t) * data_size());
	return ret;
}

Ref<MLPPMatrix> MLPPMatrix::duplicate() const {
	Ref<MLPPMatrix> ret;
	ret.instance();

	ret->set_from_mlpp_matrixr(*this);

	return ret;
}

void MLPPMatrix::set_from_mlpp_matrix(const Ref<MLPPMatrix> &p_from) {
	ERR_FAIL_COND(!p_from.is_valid());

	resize(p_from->size());
	for (int i = 0; i < p_from->data_size(); ++i) {
		_data[i] = p_from->_data[i];
	}
}

void MLPPMatrix::set_from_mlpp_matrixr(const MLPPMatrix &p_from) {
	resize(p_from.size());
	for (int i = 0; i < p_from.data_size(); ++i) {
		_data[i] = p_from._data[i];
	}
}

void MLPPMatrix::set_from_mlpp_vectors(const Vector<Ref<MLPPVector>> &p_from) {
	if (p_from.size() == 0) {
		reset();
		return;
	}

	if (!p_from[0].is_valid()) {
		reset();
		return;
	}

	resize(Size2i(p_from[0]->size(), p_from.size()));

	if (data_size() == 0) {
		reset();
		return;
	}

	for (int i = 0; i < p_from.size(); ++i) {
		const Ref<MLPPVector> &r = p_from[i];

		ERR_CONTINUE(!r.is_valid());
		ERR_CONTINUE(r->size() != _size.x);

		int start_index = i * _size.x;

		const real_t *from_ptr = r->ptr();
		for (int j = 0; j < _size.x; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
}

void MLPPMatrix::set_from_mlpp_vectors_array(const Array &p_from) {
	if (p_from.size() == 0) {
		reset();
		return;
	}

	Ref<MLPPVector> v0 = p_from[0];

	if (!v0.is_valid()) {
		reset();
		return;
	}

	resize(Size2i(v0->size(), p_from.size()));

	if (data_size() == 0) {
		reset();
		return;
	}

	for (int i = 0; i < p_from.size(); ++i) {
		Ref<MLPPVector> r = p_from[i];

		ERR_CONTINUE(!r.is_valid());
		ERR_CONTINUE(r->size() != _size.x);

		int start_index = i * _size.x;

		const real_t *from_ptr = r->ptr();
		for (int j = 0; j < _size.x; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
}

void MLPPMatrix::set_from_vectors(const Vector<Vector<real_t>> &p_from) {
	if (p_from.size() == 0) {
		reset();
		return;
	}

	resize(Size2i(p_from[0].size(), p_from.size()));

	if (data_size() == 0) {
		reset();
		return;
	}

	for (int i = 0; i < p_from.size(); ++i) {
		const Vector<real_t> &r = p_from[i];

		ERR_CONTINUE(r.size() != _size.x);

		int start_index = i * _size.x;

		const real_t *from_ptr = r.ptr();
		for (int j = 0; j < _size.x; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
}

void MLPPMatrix::set_from_arrays(const Array &p_from) {
	if (p_from.size() == 0) {
		reset();
		return;
	}

	PoolRealArray p0arr = p_from[0];

	resize(Size2i(p0arr.size(), p_from.size()));

	if (data_size() == 0) {
		reset();
		return;
	}

	for (int i = 0; i < p_from.size(); ++i) {
		PoolRealArray r = p_from[i];

		ERR_CONTINUE(r.size() != _size.x);

		int start_index = i * _size.x;

		PoolRealArray::Read read = r.read();
		const real_t *from_ptr = read.ptr();
		for (int j = 0; j < _size.x; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
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

Ref<MLPPMatrix> MLPPMatrix::gaussian_noise(int n, int m) const {
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

void MLPPMatrix::gaussian_noise_fill() {
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<real_t> distribution(0, 1); // Standard normal distribution. Mean of 0, std of 1.

	int a_data_size = data_size();
	real_t *a_ptr = ptrw();

	for (int i = 0; i < a_data_size; ++i) {
		a_ptr[i] = distribution(generator);
	}
}

void MLPPMatrix::add(const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] += b_ptr[i];
	}
}
Ref<MLPPMatrix> MLPPMatrix::addn(const Ref<MLPPMatrix> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPMatrix>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = a_ptr[i] + b_ptr[i];
	}

	return C;
}
void MLPPMatrix::addb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size2i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (_size != a_size) {
		resize(a_size);
	}

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = a_ptr[i] + b_ptr[i];
	}
}

void MLPPMatrix::sub(const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] -= b_ptr[i];
	}
}
Ref<MLPPMatrix> MLPPMatrix::subn(const Ref<MLPPMatrix> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPMatrix>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = a_ptr[i] - b_ptr[i];
	}

	return C;
}
void MLPPMatrix::subb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size2i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (_size != a_size) {
		resize(a_size);
	}

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = a_ptr[i] - b_ptr[i];
	}
}

void MLPPMatrix::mult(const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!B.is_valid());

	Size2i b_size = B->size();

	ERR_FAIL_COND(_size.x != b_size.y || _size.y != b_size.x);

	Ref<MLPPMatrix> A = duplicate();
	Size2i a_size = A->size();

	Size2i rs = Size2i(b_size.x, a_size.y);

	if (_size != rs) {
		resize(rs);
	}

	fill(0);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int ay = 0; ay < a_size.y; ay++) {
		for (int by = 0; by < b_size.y; by++) {
			int ind_ay_by = A->calculate_index(ay, by);

			for (int bx = 0; bx < b_size.x; bx++) {
				int ind_ay_bx = calculate_index(ay, bx);
				int ind_by_bx = B->calculate_index(by, bx);

				c_ptr[ind_ay_bx] += a_ptr[ind_ay_by] * b_ptr[ind_by_bx];
			}
		}
	}
}
Ref<MLPPMatrix> MLPPMatrix::multn(const Ref<MLPPMatrix> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPMatrix>());

	Size2i b_size = B->size();

	ERR_FAIL_COND_V_MSG(_size.y != b_size.x || _size.x != b_size.y, Ref<MLPPMatrix>(), "_size.y != b_size.x || _size.x != b_size.y _size: " + _size.operator String() + " b_size: " + b_size.operator String());

	Size2i rs = Size2i(b_size.x, _size.y);

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(rs);
	C->fill(0);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < _size.y; i++) {
		for (int k = 0; k < b_size.y; k++) {
			int ind_i_k = calculate_index(i, k);

			for (int j = 0; j < b_size.x; j++) {
				int ind_i_j = C->calculate_index(i, j);
				int ind_k_j = B->calculate_index(k, j);

				c_ptr[ind_i_j] += a_ptr[ind_i_k] * b_ptr[ind_k_j];
			}
		}
	}

	return C;
}
void MLPPMatrix::multb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());

	Size2i a_size = A->size();
	Size2i b_size = B->size();

	ERR_FAIL_COND_MSG(a_size.y != b_size.x || a_size.x != b_size.y, "a_size.y != b_size.x || a_size.x != b_size.y: a_size: " + a_size.operator String() + " b_size: " + b_size.operator String());

	Size2i rs = Size2i(b_size.x, a_size.y);

	if (unlikely(_size != rs)) {
		resize(rs);
	}

	fill(0);

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < a_size.y; i++) {
		for (int k = 0; k < b_size.y; k++) {
			int ind_i_k = A->calculate_index(i, k);

			for (int j = 0; j < b_size.x; j++) {
				int ind_i_j = calculate_index(i, j);
				int ind_k_j = B->calculate_index(k, j);

				c_ptr[ind_i_j] += a_ptr[ind_i_k] * b_ptr[ind_k_j];
			}
		}
	}
}

void MLPPMatrix::hadamard_product(const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	int ds = data_size();

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = c_ptr[i] * b_ptr[i];
	}
}
Ref<MLPPMatrix> MLPPMatrix::hadamard_productn(const Ref<MLPPMatrix> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPMatrix>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPMatrix>());

	int ds = data_size();

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] * b_ptr[i];
	}

	return C;
}
void MLPPMatrix::hadamard_productb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size2i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (a_size != _size) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] * b_ptr[i];
	}
}

void MLPPMatrix::kronecker_product(const Ref<MLPPMatrix> &B) {
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

	ERR_FAIL_COND(!B.is_valid());
	Size2i a_size = size();
	Size2i b_size = B->size();

	Ref<MLPPMatrix> A = duplicate();

	resize(Size2i(b_size.x * a_size.x, b_size.y * a_size.y));

	const real_t *a_ptr = A->ptr();

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(b_size.x);

	for (int i = 0; i < _size.y; ++i) {
		for (int j = 0; j < b_size.y; ++j) {
			B->get_row_into_mlpp_vector(j, row_tmp);

			Vector<Ref<MLPPVector>> row;
			for (int k = 0; k < _size.x; ++k) {
				row.push_back(row_tmp->scalar_multiplyn(a_ptr[A->calculate_index(i, k)]));
			}

			Ref<MLPPVector> flattened_row = row_tmp->flatten_vectorsn(row);

			set_row_mlpp_vector(i * b_size.y + j, flattened_row);
		}
	}
}
Ref<MLPPMatrix> MLPPMatrix::kronecker_productn(const Ref<MLPPMatrix> &B) const {
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

	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPMatrix>());
	Size2i a_size = size();
	Size2i b_size = B->size();

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(Size2i(b_size.x * a_size.x, b_size.y * a_size.y));

	const real_t *a_ptr = ptr();

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(b_size.x);

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < b_size.y; ++j) {
			B->get_row_into_mlpp_vector(j, row_tmp);

			Vector<Ref<MLPPVector>> row;
			for (int k = 0; k < a_size.x; ++k) {
				row.push_back(row_tmp->scalar_multiplyn(a_ptr[calculate_index(i, k)]));
			}

			Ref<MLPPVector> flattened_row = row_tmp->flatten_vectorsn(row);

			C->set_row_mlpp_vector(i * b_size.y + j, flattened_row);
		}
	}

	return C;
}
void MLPPMatrix::kronecker_productb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
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

	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size2i a_size = A->size();
	Size2i b_size = B->size();

	resize(Size2i(b_size.x * a_size.x, b_size.y * a_size.y));

	const real_t *a_ptr = A->ptr();

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(b_size.x);

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < b_size.y; ++j) {
			B->get_row_into_mlpp_vector(j, row_tmp);

			Vector<Ref<MLPPVector>> row;
			for (int k = 0; k < a_size.x; ++k) {
				row.push_back(row_tmp->scalar_multiplyn(a_ptr[A->calculate_index(i, k)]));
			}

			Ref<MLPPVector> flattened_row = row_tmp->flatten_vectorsn(row);

			set_row_mlpp_vector(i * b_size.y + j, flattened_row);
		}
	}
}

void MLPPMatrix::element_wise_division(const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	int ds = data_size();

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] /= b_ptr[i];
	}
}
Ref<MLPPMatrix> MLPPMatrix::element_wise_divisionn(const Ref<MLPPMatrix> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPMatrix>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPMatrix>());

	int ds = data_size();

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] / b_ptr[i];
	}

	return C;
}
void MLPPMatrix::element_wise_divisionb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size2i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (a_size != _size) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] / b_ptr[i];
	}
}

void MLPPMatrix::transpose() {
	Ref<MLPPMatrix> A = duplicate();
	Size2i a_size = A->size();

	resize(Size2i(a_size.y, a_size.x));

	const real_t *a_ptr = A->ptr();
	real_t *at_ptr = ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			at_ptr[calculate_index(j, i)] = a_ptr[A->calculate_index(i, j)];
		}
	}
}
Ref<MLPPMatrix> MLPPMatrix::transposen() const {
	Ref<MLPPMatrix> AT;
	AT.instance();
	AT->resize(Size2i(_size.y, _size.x));

	const real_t *a_ptr = ptr();
	real_t *at_ptr = AT->ptrw();

	for (int i = 0; i < _size.y; ++i) {
		for (int j = 0; j < _size.x; ++j) {
			at_ptr[AT->calculate_index(j, i)] = a_ptr[calculate_index(i, j)];
		}
	}

	return AT;
}
void MLPPMatrix::transposeb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	Size2i s = Size2i(a_size.y, a_size.x);

	if (_size != s) {
		resize(s);
	}

	const real_t *a_ptr = A->ptr();
	real_t *at_ptr = ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			at_ptr[calculate_index(j, i)] = a_ptr[A->calculate_index(i, j)];
		}
	}
}

void MLPPMatrix::scalar_multiply(const real_t scalar) {
	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		_data[i] *= scalar;
	}
}
Ref<MLPPMatrix> MLPPMatrix::scalar_multiplyn(const real_t scalar) const {
	Ref<MLPPMatrix> AN = duplicate();
	int ds = AN->data_size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < ds; ++i) {
		an_ptr[i] *= scalar;
	}

	return AN;
}
void MLPPMatrix::scalar_multiplyb(const real_t scalar, const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	if (A->size() != _size) {
		resize(A->size());
	}

	int ds = data_size();
	real_t *an_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		_data[i] = an_ptr[i] * scalar;
	}
}

void MLPPMatrix::scalar_add(const real_t scalar) {
	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		_data[i] += scalar;
	}
}
Ref<MLPPMatrix> MLPPMatrix::scalar_addn(const real_t scalar) const {
	Ref<MLPPMatrix> AN = duplicate();
	int ds = AN->data_size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < ds; ++i) {
		an_ptr[i] += scalar;
	}

	return AN;
}
void MLPPMatrix::scalar_addb(const real_t scalar, const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	if (A->size() != _size) {
		resize(A->size());
	}

	int ds = data_size();
	real_t *an_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		_data[i] = an_ptr[i] + scalar;
	}
}

void MLPPMatrix::log() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::log(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::logn() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::log(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::logb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::log(a_ptr[i]);
	}
}

void MLPPMatrix::log10() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::log10(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::log10n() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::log10(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::log10b(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::log10(a_ptr[i]);
	}
}

void MLPPMatrix::exp() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::exp(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::expn() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::exp(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::expb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::exp(a_ptr[i]);
	}
}

void MLPPMatrix::erf() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::erf(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::erfn() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::erf(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::erfb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::erf(a_ptr[i]);
	}
}

void MLPPMatrix::exponentiate(real_t p) {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::pow(out_ptr[i], p);
	}
}
Ref<MLPPMatrix> MLPPMatrix::exponentiaten(real_t p) const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}

	return out;
}
void MLPPMatrix::exponentiateb(const Ref<MLPPMatrix> &A, real_t p) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}
}

void MLPPMatrix::sqrt() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sqrt(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::sqrtn() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::sqrtb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}
}

void MLPPMatrix::cbrt() {
	exponentiate(real_t(1) / real_t(3));
}
Ref<MLPPMatrix> MLPPMatrix::cbrtn() const {
	return exponentiaten(real_t(1) / real_t(3));
}
void MLPPMatrix::cbrtb(const Ref<MLPPMatrix> &A) {
	exponentiateb(A, real_t(1) / real_t(3));
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

void MLPPMatrix::abs() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = ABS(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::absn() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::absb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size2i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}
}

real_t MLPPMatrix::det(int d) const {
	if (d == -1) {
		return detb(Ref<MLPPMatrix>(this), _size.y);
	} else {
		return detb(Ref<MLPPMatrix>(this), d);
	}
}

real_t MLPPMatrix::detb(const Ref<MLPPMatrix> &A, int d) const {
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

			deter += Math::pow(static_cast<real_t>(-1), static_cast<real_t>(i)) * A->get_element(0, i) * B->det(d - 1);
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

Ref<MLPPMatrix> MLPPMatrix::cofactor(int n, int i, int j) const {
	Ref<MLPPMatrix> cof;
	cof.instance();
	cof->resize(_size);

	int sub_i = 0;
	int sub_j = 0;

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			if (row != i && col != j) {
				cof->set_element(sub_i, sub_j++, get_element(row, col));

				if (sub_j == n - 1) {
					sub_j = 0;
					sub_i++;
				}
			}
		}
	}

	return cof;
}
void MLPPMatrix::cofactoro(int n, int i, int j, Ref<MLPPMatrix> out) const {
	ERR_FAIL_COND(!out.is_valid());

	if (unlikely(out->size() != _size)) {
		out->resize(_size);
	}

	int sub_i = 0;
	int sub_j = 0;

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			if (row != i && col != j) {
				out->set_element(sub_i, sub_j++, get_element(row, col));

				if (sub_j == n - 1) {
					sub_j = 0;
					sub_i++;
				}
			}
		}
	}
}

Ref<MLPPMatrix> MLPPMatrix::adjoint() const {
	Ref<MLPPMatrix> adj;

	ERR_FAIL_COND_V(_size.x != _size.y, adj);

	//Resizing the initial adjoint matrix

	adj.instance();
	adj->resize(_size);

	// Checking for the case where the given N x N matrix is a scalar
	if (_size.y == 1) {
		adj->set_element(0, 0, 1);
		return adj;
	}

	if (_size.y == 2) {
		adj->set_element(0, 0, get_element(1, 1));
		adj->set_element(1, 1, get_element(0, 0));

		adj->set_element(0, 1, -get_element(0, 1));
		adj->set_element(1, 0, -get_element(1, 0));

		return adj;
	}

	for (int i = 0; i < _size.y; i++) {
		for (int j = 0; j < _size.x; j++) {
			Ref<MLPPMatrix> cof = cofactor(_size.y, i, j);
			// 1 if even, -1 if odd
			int sign = (i + j) % 2 == 0 ? 1 : -1;
			adj->set_element(j, i, sign * cof->det(int(_size.y) - 1));
		}
	}
	return adj;
}
void MLPPMatrix::adjointo(Ref<MLPPMatrix> out) const {
	ERR_FAIL_COND(!out.is_valid());

	ERR_FAIL_COND(_size.x != _size.y);

	//Resizing the initial adjoint matrix

	if (unlikely(out->size() != _size)) {
		out->resize(_size);
	}

	// Checking for the case where the given N x N matrix is a scalar
	if (_size.y == 1) {
		out->set_element(0, 0, 1);
		return;
	}

	if (_size.y == 2) {
		out->set_element(0, 0, get_element(1, 1));
		out->set_element(1, 1, get_element(0, 0));

		out->set_element(0, 1, -get_element(0, 1));
		out->set_element(1, 0, -get_element(1, 0));

		return;
	}

	for (int i = 0; i < _size.y; i++) {
		for (int j = 0; j < _size.x; j++) {
			Ref<MLPPMatrix> cof = cofactor(_size.y, i, j);
			// 1 if even, -1 if odd
			int sign = (i + j) % 2 == 0 ? 1 : -1;
			out->set_element(j, i, sign * cof->det(int(_size.y) - 1));
		}
	}
}

Ref<MLPPMatrix> MLPPMatrix::inverse() const {
	return adjoint()->scalar_multiplyn(1 / det());
}
void MLPPMatrix::inverseo(Ref<MLPPMatrix> out) const {
	ERR_FAIL_COND(!out.is_valid());

	out->set_from_mlpp_matrix(adjoint()->scalar_multiplyn(1 / det()));
}

Ref<MLPPMatrix> MLPPMatrix::pinverse() const {
	return multn(Ref<MLPPMatrix>(this))->transposen()->inverse()->multn(transposen());
}
void MLPPMatrix::pinverseo(Ref<MLPPMatrix> out) const {
	ERR_FAIL_COND(!out.is_valid());

	out->set_from_mlpp_matrix(multn(Ref<MLPPMatrix>(this))->transposen()->inverse()->multn(transposen()));
}

Ref<MLPPMatrix> MLPPMatrix::zero_mat(int n, int m) const {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(0);

	return mat;
}
Ref<MLPPMatrix> MLPPMatrix::one_mat(int n, int m) const {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(1);

	return mat;
}
Ref<MLPPMatrix> MLPPMatrix::full_mat(int n, int m, int k) const {
	Ref<MLPPMatrix> mat;
	mat.instance();

	mat->resize(Size2i(m, n));
	mat->fill(k);

	return mat;
}

void MLPPMatrix::sin() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sin(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::sinn() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sin(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::sinb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	if (A->size() != _size) {
		resize(A->size());
	}

	int ds = A->data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sin(a_ptr[i]);
	}
}

void MLPPMatrix::cos() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::cos(out_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::cosn() const {
	Ref<MLPPMatrix> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::cos(a_ptr[i]);
	}

	return out;
}
void MLPPMatrix::cosb(const Ref<MLPPMatrix> &A) {
	ERR_FAIL_COND(!A.is_valid());

	if (A->size() != _size) {
		resize(A->size());
	}

	int ds = A->data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::cos(a_ptr[i]);
	}
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

void MLPPMatrix::max(const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = MAX(c_ptr[i], b_ptr[i]);
	}
}
Ref<MLPPMatrix> MLPPMatrix::maxn(const Ref<MLPPMatrix> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPMatrix>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = MAX(a_ptr[i], b_ptr[i]);
	}

	return C;
}
void MLPPMatrix::maxb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size2i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (_size != a_size) {
		resize(a_size);
	}

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = MAX(a_ptr[i], b_ptr[i]);
	}
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

void MLPPMatrix::identity() {
	fill(0);

	real_t *im_ptr = ptrw();

	int d = MIN(_size.x, _size.y);

	for (int i = 0; i < d; i++) {
		im_ptr[calculate_index(i, i)] = 1;
	}
}
Ref<MLPPMatrix> MLPPMatrix::identityn() const {
	Ref<MLPPMatrix> identity_mat;
	identity_mat.instance();
	identity_mat->resize(_size);
	identity_mat->identity();

	return identity_mat;
}
Ref<MLPPMatrix> MLPPMatrix::identity_mat(int d) const {
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

Ref<MLPPMatrix> MLPPMatrix::cov() const {
	MLPPStat stat;

	Ref<MLPPMatrix> cov_mat;
	cov_mat.instance();

	cov_mat->resize(_size);

	Ref<MLPPVector> a_i_row_tmp;
	a_i_row_tmp.instance();
	a_i_row_tmp->resize(_size.x);

	Ref<MLPPVector> a_j_row_tmp;
	a_j_row_tmp.instance();
	a_j_row_tmp->resize(_size.x);

	for (int i = 0; i < _size.y; ++i) {
		get_row_into_mlpp_vector(i, a_i_row_tmp);

		for (int j = 0; j < _size.x; ++j) {
			get_row_into_mlpp_vector(j, a_j_row_tmp);

			cov_mat->set_element(i, j, stat.covariancev(a_i_row_tmp, a_j_row_tmp));
		}
	}

	return cov_mat;
}
void MLPPMatrix::covo(Ref<MLPPMatrix> out) const {
	ERR_FAIL_COND(!out.is_valid());

	MLPPStat stat;

	if (unlikely(out->size() != _size)) {
		out->resize(_size);
	}

	Ref<MLPPVector> a_i_row_tmp;
	a_i_row_tmp.instance();
	a_i_row_tmp->resize(_size.x);

	Ref<MLPPVector> a_j_row_tmp;
	a_j_row_tmp.instance();
	a_j_row_tmp->resize(_size.x);

	for (int i = 0; i < _size.y; ++i) {
		get_row_into_mlpp_vector(i, a_i_row_tmp);

		for (int j = 0; j < _size.x; ++j) {
			get_row_into_mlpp_vector(j, a_j_row_tmp);

			out->set_element(i, j, stat.covariancev(a_i_row_tmp, a_j_row_tmp));
		}
	}
}

MLPPMatrix::EigenResult MLPPMatrix::eigen() const {
	EigenResult res;

	/*
	A (the entered parameter) in most use cases will be X'X, XX', etc. and must be symmetric.
	That simply means that 1) X' = X and 2) X is a square matrix. This function that computes the
	eigenvalues of a matrix is utilizing Jacobi's method.
	*/

	real_t diagonal = true; // Perform the iterative Jacobi algorithm unless and until we reach a diagonal matrix which yields us the eigenvals.

	HashMap<int, int> val_to_vec;
	Ref<MLPPMatrix> a_new;
	Ref<MLPPMatrix> a_mat = Ref<MLPPMatrix>(this);
	Ref<MLPPMatrix> eigenvectors = identity_mat(a_mat->size().y);
	Size2i a_size = a_mat->size();

	do {
		real_t a_ij = a_mat->get_element(0, 1);
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (int i = 0; i < a_size.y; ++i) {
			for (int j = 0; j < a_size.x; ++j) {
				real_t ca_ij = a_mat->get_element(i, j);
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

		real_t a_ii = a_mat->get_element(sub_i, sub_i);
		real_t a_jj = a_mat->get_element(sub_j, sub_j);
		//real_t a_ji = a_mat->get_element(sub_j, sub_i);
		real_t theta;

		if (a_ii == a_jj) {
			theta = Math_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		Ref<MLPPMatrix> P = identity_mat(a_mat->size().y);
		P->set_element(sub_i, sub_j, -Math::sin(theta));
		P->set_element(sub_i, sub_i, Math::cos(theta));
		P->set_element(sub_j, sub_j, Math::cos(theta));
		P->set_element(sub_j, sub_i, Math::sin(theta));

		a_new = P->inverse()->multn(a_mat)->multn(P);

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

		if (a_new->is_equal_approx(a_mat)) {
			diagonal = true;
			for (int i = 0; i < a_new_size.y; ++i) {
				for (int j = 0; j < a_new_size.x; ++j) {
					if (i != j) {
						a_new->set_element(i, j, 0);
					}
				}
			}
		}

		eigenvectors = eigenvectors->multn(P);
		a_mat = a_new;

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
MLPPMatrix::EigenResult MLPPMatrix::eigenb(const Ref<MLPPMatrix> &A) const {
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
	Ref<MLPPMatrix> a_mat = A;
	Ref<MLPPMatrix> eigenvectors = identity_mat(a_mat->size().y);
	Size2i a_size = a_mat->size();

	do {
		real_t a_ij = a_mat->get_element(0, 1);
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (int i = 0; i < a_size.y; ++i) {
			for (int j = 0; j < a_size.x; ++j) {
				real_t ca_ij = a_mat->get_element(i, j);
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

		real_t a_ii = a_mat->get_element(sub_i, sub_i);
		real_t a_jj = a_mat->get_element(sub_j, sub_j);
		//real_t a_ji = a_mat->get_element(sub_j, sub_i);
		real_t theta;

		if (a_ii == a_jj) {
			theta = Math_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		Ref<MLPPMatrix> P = identity_mat(a_mat->size().y);
		P->set_element(sub_i, sub_j, -Math::sin(theta));
		P->set_element(sub_i, sub_i, Math::cos(theta));
		P->set_element(sub_j, sub_j, Math::cos(theta));
		P->set_element(sub_j, sub_i, Math::sin(theta));

		a_new = P->inverse()->multn(a_mat)->multn(P);

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

		if (a_new->is_equal_approx(a_mat)) {
			diagonal = true;
			for (int i = 0; i < a_new_size.y; ++i) {
				for (int j = 0; j < a_new_size.x; ++j) {
					if (i != j) {
						a_new->set_element(i, j, 0);
					}
				}
			}
		}

		eigenvectors = eigenvectors->multn(P);
		a_mat = a_new;

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
Array MLPPMatrix::eigen_bind() {
	Array arr;

	EigenResult r = eigen();

	arr.push_back(r.eigen_values);
	arr.push_back(r.eigen_vectors);

	return arr;
}
Array MLPPMatrix::eigenb_bind(const Ref<MLPPMatrix> &A) {
	Array arr;

	ERR_FAIL_COND_V(!A.is_valid(), arr);

	EigenResult r = eigenb(A);

	arr.push_back(r.eigen_values);
	arr.push_back(r.eigen_vectors);

	return arr;
}

MLPPMatrix::SVDResult MLPPMatrix::svd() const {
	SVDResult res;

	EigenResult left_eigen = multn(transposen())->eigen();
	EigenResult right_eigen = transposen()->multn(Ref<MLPPMatrix>(this))->eigen();

	Ref<MLPPMatrix> singularvals = left_eigen.eigen_values->sqrtn();
	Ref<MLPPMatrix> sigma = zero_mat(_size.y, _size.x);

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

MLPPMatrix::SVDResult MLPPMatrix::svdb(const Ref<MLPPMatrix> &A) const {
	SVDResult res;

	ERR_FAIL_COND_V(!A.is_valid(), res);

	Size2i a_size = A->size();

	EigenResult left_eigen = A->multn(A->transposen())->eigen();
	EigenResult right_eigen = A->transposen()->multn(A)->eigen();

	Ref<MLPPMatrix> singularvals = left_eigen.eigen_values->sqrtn();
	Ref<MLPPMatrix> sigma = zero_mat(a_size.y, a_size.x);

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

Array MLPPMatrix::svd_bind() {
	Array arr;

	SVDResult r = svd();

	arr.push_back(r.U);
	arr.push_back(r.S);
	arr.push_back(r.Vt);

	return arr;
}
Array MLPPMatrix::svdb_bind(const Ref<MLPPMatrix> &A) {
	Array arr;

	ERR_FAIL_COND_V(!A.is_valid(), arr);

	SVDResult r = svdb(A);

	arr.push_back(r.U);
	arr.push_back(r.S);
	arr.push_back(r.Vt);

	return arr;
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

Ref<MLPPVector> MLPPMatrix::flatten() const {
	int ds = data_size();

	Ref<MLPPVector> res;
	res.instance();
	res->resize(ds);

	real_t *res_ptr = res->ptrw();
	const real_t *a_ptr = ptr();

	for (int i = 0; i < ds; ++i) {
		res_ptr[i] = a_ptr[i];
	}

	return res;
}
void MLPPMatrix::flatteno(Ref<MLPPVector> out) const {
	ERR_FAIL_COND(!out.is_valid());

	int ds = data_size();

	if (unlikely(out->size() != ds)) {
		out->resize(ds);
	}

	real_t *res_ptr = out->ptrw();
	const real_t *a_ptr = ptr();

	for (int i = 0; i < ds; ++i) {
		res_ptr[i] = a_ptr[i];
	}
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

Ref<MLPPVector> MLPPMatrix::mult_vec(const Ref<MLPPVector> &b) const {
	ERR_FAIL_COND_V(!b.is_valid(), Ref<MLPPMatrix>());

	int b_size = b->size();

	ERR_FAIL_COND_V(_size.x < b->size(), Ref<MLPPMatrix>());

	Ref<MLPPVector> c;
	c.instance();
	c->resize(_size.y);
	c->fill(0);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *c_ptr = c->ptrw();

	for (int i = 0; i < _size.y; ++i) {
		for (int k = 0; k < b_size; ++k) {
			int mat_index = calculate_index(i, k);

			c_ptr[i] += a_ptr[mat_index] * b_ptr[k];
		}
	}

	return c;
}
void MLPPMatrix::mult_veco(const Ref<MLPPVector> &b, Ref<MLPPVector> out) {
	ERR_FAIL_COND(!out.is_valid() || !b.is_valid());

	int b_size = b->size();

	ERR_FAIL_COND(_size.x < b->size());

	if (unlikely(out->size() != _size.y)) {
		out->resize(_size.y);
	}

	out->fill(0);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *c_ptr = out->ptrw();

	for (int i = 0; i < _size.y; ++i) {
		for (int k = 0; k < b_size; ++k) {
			int mat_index = calculate_index(i, k);

			c_ptr[i] += a_ptr[mat_index] * b_ptr[k];
		}
	}
}

void MLPPMatrix::add_vec(const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!b.is_valid());
	ERR_FAIL_COND(_size.x != b->size());

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *ret_ptr = ptrw();

	for (int i = 0; i < _size.y; ++i) {
		for (int j = 0; j < _size.x; ++j) {
			int mat_index = calculate_index(i, j);

			ret_ptr[mat_index] = a_ptr[mat_index] + b_ptr[j];
		}
	}
}
Ref<MLPPMatrix> MLPPMatrix::add_vecn(const Ref<MLPPVector> &b) const {
	ERR_FAIL_COND_V(!b.is_valid(), Ref<MLPPMatrix>());
	ERR_FAIL_COND_V(_size.x != b->size(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> ret;
	ret.instance();
	ret->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = b->ptr();
	real_t *ret_ptr = ret->ptrw();

	for (int i = 0; i < _size.y; ++i) {
		for (int j = 0; j < _size.x; ++j) {
			int mat_index = calculate_index(i, j);

			ret_ptr[mat_index] = a_ptr[mat_index] + b_ptr[j];
		}
	}

	return ret;
}
void MLPPMatrix::add_vecb(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!A.is_valid() || !b.is_valid());
	Size2i a_size = A->size();
	ERR_FAIL_COND(a_size.x != b->size());

	if (unlikely(_size != a_size)) {
		resize(a_size);
	}

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = b->ptr();
	real_t *ret_ptr = ptrw();

	for (int i = 0; i < a_size.y; ++i) {
		for (int j = 0; j < a_size.x; ++j) {
			int mat_index = A->calculate_index(i, j);

			ret_ptr[mat_index] = a_ptr[mat_index] + b_ptr[j];
		}
	}
}

void MLPPMatrix::outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) {
	ERR_FAIL_COND(!a.is_valid() || !b.is_valid());

	Size2i s = Size2i(b->size(), a->size());

	if (unlikely(_size != s)) {
		resize(s);
	}

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();

	for (int i = 0; i < s.y; ++i) {
		real_t curr_a = a_ptr[i];

		for (int j = 0; j < s.x; ++j) {
			set_element(i, j, curr_a * b_ptr[j]);
		}
	}
}
Ref<MLPPMatrix> MLPPMatrix::outer_productn(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) const {
	ERR_FAIL_COND_V(!a.is_valid() || !b.is_valid(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> C;
	C.instance();

	Size2i s = Size2i(b->size(), a->size());
	C->resize(s);

	const real_t *a_ptr = a->ptr();
	const real_t *b_ptr = b->ptr();

	for (int i = 0; i < s.y; ++i) {
		real_t curr_a = a_ptr[i];

		for (int j = 0; j < s.x; ++j) {
			C->set_element(i, j, curr_a * b_ptr[j]);
		}
	}

	return C;
}

void MLPPMatrix::set_diagonal(const Ref<MLPPVector> &a) {
	ERR_FAIL_COND(!a.is_valid());

	int a_size = a->size();
	int ms = MIN(_size.x, _size.y);
	ms = MIN(ms, a_size);

	if (ms <= 0) {
		return;
	}

	const real_t *a_ptr = a->ptr();
	real_t *b_ptr = ptrw();

	for (int i = 0; i < ms; ++i) {
		b_ptr[calculate_index(i, i)] = a_ptr[i];
	}
}
Ref<MLPPMatrix> MLPPMatrix::set_diagonaln(const Ref<MLPPVector> &a) const {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> B = duplicate();

	int a_size = a->size();
	int ms = MIN(_size.x, _size.y);
	ms = MIN(ms, a_size);

	if (ms <= 0) {
		return B;
	}

	const real_t *a_ptr = a->ptr();
	real_t *b_ptr = B->ptrw();

	for (int i = 0; i < ms; ++i) {
		b_ptr[B->calculate_index(i, i)] = a_ptr[i];
	}

	return B;
}

void MLPPMatrix::diagonal_zeroed(const Ref<MLPPVector> &a) {
	fill(0);

	ERR_FAIL_COND(!a.is_valid());

	int a_size = a->size();
	int ms = MIN(_size.x, _size.y);
	ms = MIN(ms, a_size);

	if (ms <= 0) {
		return;
	}

	const real_t *a_ptr = a->ptr();
	real_t *b_ptr = ptrw();

	for (int i = 0; i < ms; ++i) {
		b_ptr[calculate_index(i, i)] = a_ptr[i];
	}
}
Ref<MLPPMatrix> MLPPMatrix::diagonal_zeroedn(const Ref<MLPPVector> &a) const {
	ERR_FAIL_COND_V(!a.is_valid(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> B;
	B.instance();
	B->resize(_size);
	B->fill(0);

	int a_size = a->size();
	int ms = MIN(_size.x, _size.y);
	ms = MIN(ms, a_size);

	if (ms <= 0) {
		return B;
	}

	const real_t *a_ptr = a->ptr();
	real_t *b_ptr = B->ptrw();

	for (int i = 0; i < ms; ++i) {
		b_ptr[B->calculate_index(i, i)] = a_ptr[i];
	}

	return B;
}

bool MLPPMatrix::is_equal_approx(const Ref<MLPPMatrix> &p_with, real_t tolerance) const {
	ERR_FAIL_COND_V(!p_with.is_valid(), false);

	if (unlikely(this == p_with.ptr())) {
		return true;
	}

	if (_size != p_with->size()) {
		return false;
	}

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		if (!Math::is_equal_approx(_data[i], p_with->_data[i], tolerance)) {
			return false;
		}
	}

	return true;
}

Ref<Image> MLPPMatrix::get_as_image() const {
	Ref<Image> image;
	image.instance();

	get_into_image(image);

	return image;
}

void MLPPMatrix::get_into_image(Ref<Image> out) const {
	ERR_FAIL_COND(!out.is_valid());

	if (data_size() == 0) {
		out->clear();
		return;
	}

	PoolByteArray arr;

	int ds = data_size();

	arr.resize(ds);

	PoolByteArray::Write w = arr.write();
	uint8_t *wptr = w.ptr();

	for (int i = 0; i < ds; ++i) {
		wptr[i] = static_cast<uint8_t>(_data[i] * 255.0);
	}

	out->create(_size.x, _size.y, false, Image::FORMAT_L8, arr);
}
void MLPPMatrix::set_from_image(const Ref<Image> &p_img, const int p_image_channel) {
	ERR_FAIL_COND(!p_img.is_valid());
	ERR_FAIL_INDEX(p_image_channel, 4);

	Size2i img_size = Size2i(p_img->get_width(), p_img->get_height());

	if (img_size != _size) {
		resize(img_size);
	}

	Ref<Image> img = p_img;

	img->lock();

	for (int y = 0; y < _size.y; ++y) {
		for (int x = 0; x < _size.x; ++x) {
			Color c = img->get_pixel(x, y);

			set_element(y, x, c[p_image_channel]);
		}
	}

	img->unlock();
}

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

MLPPMatrix::MLPPMatrix() {
	_data = NULL;
}

MLPPMatrix::MLPPMatrix(const MLPPMatrix &p_from) {
	_data = NULL;

	resize(p_from.size());
	for (int i = 0; i < p_from.data_size(); ++i) {
		_data[i] = p_from._data[i];
	}
}

MLPPMatrix::MLPPMatrix(const Vector<Vector<real_t>> &p_from) {
	_data = NULL;

	set_from_vectors(p_from);
}

MLPPMatrix::MLPPMatrix(const Array &p_from) {
	_data = NULL;

	set_from_arrays(p_from);
}

MLPPMatrix::~MLPPMatrix() {
	if (_data) {
		reset();
	}
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

	ClassDB::bind_method(D_METHOD("get_as_image"), &MLPPMatrix::get_as_image);
	ClassDB::bind_method(D_METHOD("get_into_image", "out"), &MLPPMatrix::get_into_image);
	ClassDB::bind_method(D_METHOD("set_from_image", "img", "image_channel"), &MLPPMatrix::set_from_image);

	ClassDB::bind_method(D_METHOD("gaussian_noise", "n", "m"), &MLPPMatrix::gaussian_noise);
	ClassDB::bind_method(D_METHOD("gaussian_noise_fill"), &MLPPMatrix::gaussian_noise_fill);

	ClassDB::bind_method(D_METHOD("add", "B"), &MLPPMatrix::add);
	ClassDB::bind_method(D_METHOD("addn", "B"), &MLPPMatrix::addn);
	ClassDB::bind_method(D_METHOD("addb", "A", "B"), &MLPPMatrix::addb);

	ClassDB::bind_method(D_METHOD("sub", "B"), &MLPPMatrix::sub);
	ClassDB::bind_method(D_METHOD("subn", "B"), &MLPPMatrix::subn);
	ClassDB::bind_method(D_METHOD("subb", "A", "B"), &MLPPMatrix::subb);

	ClassDB::bind_method(D_METHOD("mult", "B"), &MLPPMatrix::mult);
	ClassDB::bind_method(D_METHOD("multn", "B"), &MLPPMatrix::multn);
	ClassDB::bind_method(D_METHOD("multb", "A", "B"), &MLPPMatrix::multb);

	ClassDB::bind_method(D_METHOD("hadamard_product", "B"), &MLPPMatrix::hadamard_product);
	ClassDB::bind_method(D_METHOD("hadamard_productn", "B"), &MLPPMatrix::hadamard_productn);
	ClassDB::bind_method(D_METHOD("hadamard_productb", "A", "B"), &MLPPMatrix::hadamard_productb);

	ClassDB::bind_method(D_METHOD("kronecker_product", "B"), &MLPPMatrix::kronecker_product);
	ClassDB::bind_method(D_METHOD("kronecker_productn", "B"), &MLPPMatrix::kronecker_productn);
	ClassDB::bind_method(D_METHOD("kronecker_productb", "A", "B"), &MLPPMatrix::kronecker_productb);

	ClassDB::bind_method(D_METHOD("element_wise_division", "B"), &MLPPMatrix::element_wise_division);
	ClassDB::bind_method(D_METHOD("element_wise_divisionn", "B"), &MLPPMatrix::element_wise_divisionn);
	ClassDB::bind_method(D_METHOD("element_wise_divisionb", "A", "B"), &MLPPMatrix::element_wise_divisionb);

	ClassDB::bind_method(D_METHOD("transpose"), &MLPPMatrix::transpose);
	ClassDB::bind_method(D_METHOD("transposen"), &MLPPMatrix::transposen);
	ClassDB::bind_method(D_METHOD("transposeb", "A"), &MLPPMatrix::transposeb);

	ClassDB::bind_method(D_METHOD("scalar_multiply", "scalar"), &MLPPMatrix::scalar_multiply);
	ClassDB::bind_method(D_METHOD("scalar_multiplyn", "scalar"), &MLPPMatrix::scalar_multiplyn);
	ClassDB::bind_method(D_METHOD("scalar_multiplyb", "scalar", "A"), &MLPPMatrix::scalar_multiplyb);

	ClassDB::bind_method(D_METHOD("scalar_add", "scalar"), &MLPPMatrix::scalar_add);
	ClassDB::bind_method(D_METHOD("scalar_addn", "scalar"), &MLPPMatrix::scalar_addn);
	ClassDB::bind_method(D_METHOD("scalar_addb", "scalar", "A"), &MLPPMatrix::scalar_addb);

	ClassDB::bind_method(D_METHOD("log"), &MLPPMatrix::log);
	ClassDB::bind_method(D_METHOD("logn"), &MLPPMatrix::logn);
	ClassDB::bind_method(D_METHOD("logb", "A"), &MLPPMatrix::logb);

	ClassDB::bind_method(D_METHOD("log10"), &MLPPMatrix::log10);
	ClassDB::bind_method(D_METHOD("log10n"), &MLPPMatrix::log10n);
	ClassDB::bind_method(D_METHOD("log10b", "A"), &MLPPMatrix::log10b);

	ClassDB::bind_method(D_METHOD("exp"), &MLPPMatrix::exp);
	ClassDB::bind_method(D_METHOD("expn"), &MLPPMatrix::expn);
	ClassDB::bind_method(D_METHOD("expb", "A"), &MLPPMatrix::expb);

	ClassDB::bind_method(D_METHOD("erf"), &MLPPMatrix::erf);
	ClassDB::bind_method(D_METHOD("erfn"), &MLPPMatrix::erfn);
	ClassDB::bind_method(D_METHOD("erfb", "A"), &MLPPMatrix::erfb);

	ClassDB::bind_method(D_METHOD("exponentiate", "p"), &MLPPMatrix::exponentiate);
	ClassDB::bind_method(D_METHOD("exponentiaten", "p"), &MLPPMatrix::exponentiaten);
	ClassDB::bind_method(D_METHOD("exponentiateb", "A", "p"), &MLPPMatrix::exponentiateb);

	ClassDB::bind_method(D_METHOD("sqrt"), &MLPPMatrix::sqrt);
	ClassDB::bind_method(D_METHOD("sqrtn"), &MLPPMatrix::sqrtn);
	ClassDB::bind_method(D_METHOD("sqrtb", "A"), &MLPPMatrix::sqrtb);

	ClassDB::bind_method(D_METHOD("cbrt"), &MLPPMatrix::cbrt);
	ClassDB::bind_method(D_METHOD("cbrtn"), &MLPPMatrix::cbrtn);
	ClassDB::bind_method(D_METHOD("cbrtb", "A"), &MLPPMatrix::cbrtb);

	ClassDB::bind_method(D_METHOD("abs"), &MLPPMatrix::abs);
	ClassDB::bind_method(D_METHOD("absn"), &MLPPMatrix::absn);
	ClassDB::bind_method(D_METHOD("absb", "A"), &MLPPMatrix::absb);

	ClassDB::bind_method(D_METHOD("det", "d"), &MLPPMatrix::det, -1);
	ClassDB::bind_method(D_METHOD("detb", "A", "d"), &MLPPMatrix::detb);

	ClassDB::bind_method(D_METHOD("cofactor", "n", "i", "j"), &MLPPMatrix::cofactor);
	ClassDB::bind_method(D_METHOD("cofactoro", "n", "i", "j", "out"), &MLPPMatrix::cofactoro);

	ClassDB::bind_method(D_METHOD("adjoint"), &MLPPMatrix::adjoint);
	ClassDB::bind_method(D_METHOD("adjointo", "out"), &MLPPMatrix::adjointo);

	ClassDB::bind_method(D_METHOD("inverse"), &MLPPMatrix::inverse);
	ClassDB::bind_method(D_METHOD("inverseo", "out"), &MLPPMatrix::inverseo);

	ClassDB::bind_method(D_METHOD("pinverse"), &MLPPMatrix::pinverse);
	ClassDB::bind_method(D_METHOD("pinverseo", "out"), &MLPPMatrix::pinverseo);

	ClassDB::bind_method(D_METHOD("zero_mat", "n", "m"), &MLPPMatrix::zero_mat);
	ClassDB::bind_method(D_METHOD("one_mat", "n", "m"), &MLPPMatrix::one_mat);
	ClassDB::bind_method(D_METHOD("full_mat", "n", "m", "k"), &MLPPMatrix::full_mat);

	ClassDB::bind_method(D_METHOD("sin"), &MLPPMatrix::sin);
	ClassDB::bind_method(D_METHOD("sinn"), &MLPPMatrix::sinn);
	ClassDB::bind_method(D_METHOD("sinb", "A"), &MLPPMatrix::sinb);

	ClassDB::bind_method(D_METHOD("cos"), &MLPPMatrix::cos);
	ClassDB::bind_method(D_METHOD("cosn"), &MLPPMatrix::cosn);
	ClassDB::bind_method(D_METHOD("cosb", "A"), &MLPPMatrix::cosb);

	ClassDB::bind_method(D_METHOD("max", "B"), &MLPPMatrix::max);
	ClassDB::bind_method(D_METHOD("maxn", "B"), &MLPPMatrix::maxn);
	ClassDB::bind_method(D_METHOD("maxb", "A", "B"), &MLPPMatrix::maxb);

	ClassDB::bind_method(D_METHOD("identity"), &MLPPMatrix::identity);
	ClassDB::bind_method(D_METHOD("identityn"), &MLPPMatrix::identityn);
	ClassDB::bind_method(D_METHOD("identity_mat", "d"), &MLPPMatrix::identity_mat);

	ClassDB::bind_method(D_METHOD("cov"), &MLPPMatrix::cov);
	ClassDB::bind_method(D_METHOD("covo", "out"), &MLPPMatrix::covo);

	ClassDB::bind_method(D_METHOD("eigen"), &MLPPMatrix::eigen_bind);
	ClassDB::bind_method(D_METHOD("eigenb", "A"), &MLPPMatrix::eigenb_bind);

	ClassDB::bind_method(D_METHOD("svd"), &MLPPMatrix::svd_bind);
	ClassDB::bind_method(D_METHOD("svdb", "A"), &MLPPMatrix::svdb_bind);

	ClassDB::bind_method(D_METHOD("flatten"), &MLPPMatrix::flatten);
	ClassDB::bind_method(D_METHOD("flatteno", "out"), &MLPPMatrix::flatteno);

	ClassDB::bind_method(D_METHOD("mult_vec", "b"), &MLPPMatrix::mult_vec);
	ClassDB::bind_method(D_METHOD("mult_veco", "b", "out"), &MLPPMatrix::mult_veco);

	ClassDB::bind_method(D_METHOD("add_vec", "b"), &MLPPMatrix::add_vec);
	ClassDB::bind_method(D_METHOD("add_vecn", "b"), &MLPPMatrix::add_vecn);
	ClassDB::bind_method(D_METHOD("add_vecb", "A", "b"), &MLPPMatrix::add_vecb);

	ClassDB::bind_method(D_METHOD("outer_product", "a", "b"), &MLPPMatrix::outer_product);
	ClassDB::bind_method(D_METHOD("outer_productn", "a", "b"), &MLPPMatrix::outer_productn);

	ClassDB::bind_method(D_METHOD("set_diagonal", "a"), &MLPPMatrix::set_diagonal);
	ClassDB::bind_method(D_METHOD("set_diagonaln", "a"), &MLPPMatrix::set_diagonaln);

	ClassDB::bind_method(D_METHOD("diagonal_zeroed", "a"), &MLPPMatrix::diagonal_zeroed);
	ClassDB::bind_method(D_METHOD("diagonal_zeroedn", "a"), &MLPPMatrix::diagonal_zeroedn);
}
