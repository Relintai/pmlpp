#ifndef MLPP_MATRIX_H
#define MLPP_MATRIX_H

#include "core/math/math_defs.h"

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/math/vector2i.h"
#include "core/os/memory.h"

#include "core/object/reference.h"

#include "mlpp_vector.h"

class MLPPMatrix : public Reference {
	GDCLASS(MLPPMatrix, Reference);

public:
	real_t *ptrw() {
		return _data;
	}

	const real_t *ptr() const {
		return _data;
	}

	_FORCE_INLINE_ void add_row(const Vector<real_t> &p_row) {
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

	_FORCE_INLINE_ void add_row_pool_vector(const PoolRealArray &p_row) {
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

	_FORCE_INLINE_ void add_row_mlpp_vector(const Ref<MLPPVector> &p_row) {
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

	_FORCE_INLINE_ void add_rows_mlpp_matrix(const Ref<MLPPMatrix> &p_other) {
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

	void remove_row(int p_index) {
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
	void remove_row_unordered(int p_index) {
		ERR_FAIL_INDEX(p_index, _size.y);

		--_size.y;

		int ds = data_size();

		if (ds == 0) {
			memfree(_data);
			_data = NULL;
			return;
		}

		int start_ind = p_index * _size.x;
		int end_ind = (p_index + 1) * _size.x;

		for (int i = start_ind; i < end_ind; ++i) {
			_data[i] = _data[ds + i];
		}

		_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	void swap_row(int p_index_1, int p_index_2) {
		ERR_FAIL_INDEX(p_index_1, _size.y);
		ERR_FAIL_INDEX(p_index_2, _size.y);

		int ind1_start = p_index_1 * _size.x;
		int ind2_start = p_index_2 * _size.x;

		for (int i = 0; i < _size.x; ++i) {
			SWAP(_data[ind1_start + i], _data[ind2_start + i]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(Size2i()); }
	_FORCE_INLINE_ void reset() {
		if (_data) {
			memfree(_data);
			_data = NULL;
			_size = Vector2i();
		}
	}

	_FORCE_INLINE_ bool empty() const { return data_size() == 0; }
	_FORCE_INLINE_ int data_size() const { return _size.x * _size.y; }
	_FORCE_INLINE_ Size2i size() const { return _size; }

	void resize(const Size2i &p_size) {
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

	_FORCE_INLINE_ int calculate_index(int p_index_y, int p_index_x) const {
		return p_index_y * _size.x + p_index_x;
	}

	_FORCE_INLINE_ const real_t &operator[](int p_index) const {
		CRASH_BAD_INDEX(p_index, data_size());
		return _data[p_index];
	}
	_FORCE_INLINE_ real_t &operator[](int p_index) {
		CRASH_BAD_INDEX(p_index, data_size());
		return _data[p_index];
	}

	_FORCE_INLINE_ real_t get_element_index(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, data_size(), 0);

		return _data[p_index];
	}

	_FORCE_INLINE_ void set_element_index(int p_index, real_t p_val) {
		ERR_FAIL_INDEX(p_index, data_size());

		_data[p_index] = p_val;
	}

	_FORCE_INLINE_ real_t get_element(int p_index_y, int p_index_x) const {
		ERR_FAIL_INDEX_V(p_index_x, _size.x, 0);
		ERR_FAIL_INDEX_V(p_index_y, _size.y, 0);

		return _data[p_index_y * _size.x + p_index_x];
	}

	_FORCE_INLINE_ void set_element(int p_index_y, int p_index_x, real_t p_val) {
		ERR_FAIL_INDEX(p_index_x, _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);

		_data[p_index_y * _size.x + p_index_x] = p_val;
	}

	_FORCE_INLINE_ Vector<real_t> get_row_vector(int p_index_y) {
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

	_FORCE_INLINE_ PoolRealArray get_row_pool_vector(int p_index_y) {
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

	_FORCE_INLINE_ Ref<MLPPVector> get_row_mlpp_vector(int p_index_y) {
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

	_FORCE_INLINE_ void get_row_into_mlpp_vector(int p_index_y, Ref<MLPPVector> target) const {
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

	_FORCE_INLINE_ void set_row_vector(int p_index_y, const Vector<real_t> &p_row) {
		ERR_FAIL_COND(p_row.size() != _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);

		int ind_start = p_index_y * _size.x;

		const real_t *row_ptr = p_row.ptr();

		for (int i = 0; i < _size.x; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_row_pool_vector(int p_index_y, const PoolRealArray &p_row) {
		ERR_FAIL_COND(p_row.size() != _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);

		int ind_start = p_index_y * _size.x;

		PoolRealArray::Read r = p_row.read();
		const real_t *row_ptr = r.ptr();

		for (int i = 0; i < _size.x; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_row_mlpp_vector(int p_index_y, const Ref<MLPPVector> &p_row) {
		ERR_FAIL_COND(!p_row.is_valid());
		ERR_FAIL_COND(p_row->size() != _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);

		int ind_start = p_index_y * _size.x;

		const real_t *row_ptr = p_row->ptr();

		for (int i = 0; i < _size.x; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	void fill(real_t p_val) {
		if (!_data) {
			return;
		}

		int ds = data_size();
		for (int i = 0; i < ds; ++i) {
			_data[i] = p_val;
		}
	}

	Vector<real_t> to_flat_vector() const {
		Vector<real_t> ret;
		ret.resize(data_size());
		real_t *w = ret.ptrw();
		memcpy(w, _data, sizeof(real_t) * data_size());
		return ret;
	}

	PoolRealArray to_flat_pool_vector() const {
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

	Vector<uint8_t> to_flat_byte_array() const {
		Vector<uint8_t> ret;
		ret.resize(data_size() * sizeof(real_t));
		uint8_t *w = ret.ptrw();
		memcpy(w, _data, sizeof(real_t) * data_size());
		return ret;
	}

	Ref<MLPPMatrix> duplicate() const {
		Ref<MLPPMatrix> ret;
		ret.instance();

		ret->set_from_mlpp_matrixr(*this);

		return ret;
	}

	_FORCE_INLINE_ void set_from_mlpp_matrix(const Ref<MLPPMatrix> &p_from) {
		ERR_FAIL_COND(!p_from.is_valid());

		resize(p_from->size());
		for (int i = 0; i < p_from->data_size(); ++i) {
			_data[i] = p_from->_data[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_matrixr(const MLPPMatrix &p_from) {
		resize(p_from.size());
		for (int i = 0; i < p_from.data_size(); ++i) {
			_data[i] = p_from._data[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_vectors(const Vector<Ref<MLPPVector>> &p_from) {
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

	_FORCE_INLINE_ void set_from_mlpp_vectors_array(const Array &p_from) {
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

	_FORCE_INLINE_ void set_from_vectors(const Vector<Vector<real_t>> &p_from) {
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

	_FORCE_INLINE_ void set_from_arrays(const Array &p_from) {
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

	_FORCE_INLINE_ bool is_equal_approx(const Ref<MLPPMatrix> &p_with, real_t tolerance = static_cast<real_t>(CMP_EPSILON)) const {
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

	String to_string();

	_FORCE_INLINE_ MLPPMatrix() {
		_data = NULL;
	}
	_FORCE_INLINE_ MLPPMatrix(const MLPPMatrix &p_from) {
		_data = NULL;

		resize(p_from.size());
		for (int i = 0; i < p_from.data_size(); ++i) {
			_data[i] = p_from._data[i];
		}
	}

	MLPPMatrix(const Vector<Vector<real_t>> &p_from) {
		_data = NULL;

		set_from_vectors(p_from);
	}

	MLPPMatrix(const Array &p_from) {
		_data = NULL;

		set_from_arrays(p_from);
	}

	_FORCE_INLINE_ ~MLPPMatrix() {
		if (_data) {
			reset();
		}
	}

	// TODO: These are temporary
	std::vector<real_t> to_flat_std_vector() const;
	void set_from_std_vectors(const std::vector<std::vector<real_t>> &p_from);
	std::vector<std::vector<real_t>> to_std_vector();
	void set_row_std_vector(int p_index_y, const std::vector<real_t> &p_row);
	MLPPMatrix(const std::vector<std::vector<real_t>> &p_from);

protected:
	static void _bind_methods();

protected:
	Size2i _size;
	real_t *_data;
};

#endif
