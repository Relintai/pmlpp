#ifndef MLPP_MATRIX_H
#define MLPP_MATRIX_H

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/math/vector2i.h"
#include "core/os/memory.h"

#include "core/object/reference.h"

#include "mlpp_vector.h"

// Matrices are stored as rows first
// [x][y]

class MLPPMatrix : public Reference {
	GDCLASS(MLPPMatrix, Reference);

public:
	double *ptr() {
		return _data;
	}

	const double *ptr() const {
		return _data;
	}

	_FORCE_INLINE_ void add_row(const Vector<double> &p_row) {
		if (_size.x == 0) {
			_size.x = p_row.size();
		}

		ERR_FAIL_COND(_size.x != p_row.size());

		int ci = data_size();

		++_size.y;

		_data = (double *)memrealloc(_data, data_size() * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");

		const double *row_arr = p_row.ptr();

		for (int i = 0; i < p_row.size(); ++i) {
			_data[ci + i] = row_arr[i];
		}
	}

	_FORCE_INLINE_ void add_row_pool_vector(const PoolRealArray &p_row) {
		if (_size.x == 0) {
			_size.x = p_row.size();
		}

		ERR_FAIL_COND(_size.x != p_row.size());

		int ci = data_size();

		++_size.y;

		_data = (double *)memrealloc(_data, data_size() * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");

		PoolRealArray::Read rread = p_row.read();
		const real_t *row_arr = rread.ptr();

		for (int i = 0; i < p_row.size(); ++i) {
			_data[ci + i] = row_arr[i];
		}
	}

	void remove_row(double p_index) {
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

		_data = (double *)memrealloc(_data, data_size() * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_unordered(int p_index) {
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

		_data = (double *)memrealloc(_data, data_size() * sizeof(double));
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
		clear();
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
			memfree(_data);
			_data = NULL;
			return;
		}

		_data = (double *)memrealloc(_data, ds * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	_FORCE_INLINE_ const double &operator[](int p_index) const {
		CRASH_BAD_INDEX(p_index, data_size());
		return _data[p_index];
	}
	_FORCE_INLINE_ double &operator[](int p_index) {
		CRASH_BAD_INDEX(p_index, data_size());
		return _data[p_index];
	}

	_FORCE_INLINE_ double get_element(int p_index_x, int p_index_y) const {
		ERR_FAIL_INDEX_V(p_index_x, _size.x, 0);
		ERR_FAIL_INDEX_V(p_index_y, _size.y, 0);

		return _data[p_index_x * p_index_y];
	}
	_FORCE_INLINE_ double get_element(int p_index_x, int p_index_y) {
		ERR_FAIL_INDEX_V(p_index_x, _size.x, 0);
		ERR_FAIL_INDEX_V(p_index_y, _size.y, 0);

		return _data[p_index_x * p_index_y];
	}

	_FORCE_INLINE_ real_t get_element_bind(int p_index_x, int p_index_y) const {
		ERR_FAIL_INDEX_V(p_index_x, _size.x, 0);
		ERR_FAIL_INDEX_V(p_index_y, _size.y, 0);

		return static_cast<real_t>(_data[p_index_x * p_index_y]);
	}

	_FORCE_INLINE_ void set_element(int p_index_x, int p_index_y, double p_val) {
		ERR_FAIL_INDEX(p_index_x, _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);

		_data[p_index_x * p_index_y] = p_val;
	}

	_FORCE_INLINE_ void set_element_bind(int p_index_x, int p_index_y, real_t p_val) {
		ERR_FAIL_INDEX(p_index_x, _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);

		_data[p_index_x * p_index_y] = p_val;
	}

	_FORCE_INLINE_ void set_row_vector(int p_index_y, const Vector<double> &p_row) {
		ERR_FAIL_COND(p_row.size() != _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);

		int ind_start = p_index_y * _size.x;

		const double *row_ptr = p_row.ptr();

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

	void fill(double p_val) {
		int ds = data_size();
		for (int i = 0; i < ds; i++) {
			_data[i] = p_val;
		}
	}

	Vector<double> to_flat_vector() const {
		Vector<double> ret;
		ret.resize(data_size());
		double *w = ret.ptrw();
		memcpy(w, _data, sizeof(double) * data_size());
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
		ret.resize(data_size() * sizeof(double));
		uint8_t *w = ret.ptrw();
		memcpy(w, _data, sizeof(double) * data_size());
		return ret;
	}

	Ref<MLPPMatrix> duplicate() const {
		Ref<MLPPMatrix> ret;
		ret.instance();

		ret->set_from_mlpp_matrixr(*this);

		return ret;
	}

	_FORCE_INLINE_ void set_from_mlpp_matrixr(const MLPPMatrix &p_from) {
		resize(p_from.size());
		for (int i = 0; i < p_from.data_size(); i++) {
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

			const double *from_ptr = r->ptr();
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

			const double *from_ptr = r->ptr();
			for (int j = 0; j < _size.x; j++) {
				_data[start_index + j] = from_ptr[j];
			}
		}
	}

	_FORCE_INLINE_ void set_from_vectors(const Vector<Vector<double>> &p_from) {
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
			const Vector<double> &r = p_from[i];

			ERR_CONTINUE(r.size() != _size.x);

			int start_index = i * _size.x;

			const double *from_ptr = r.ptr();
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

	MLPPMatrix(const Vector<Vector<double>> &p_from) {
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
	std::vector<double> to_flat_std_vector() const {
		std::vector<double> ret;
		ret.resize(data_size());
		double *w = &ret[0];
		memcpy(w, _data, sizeof(double) * data_size());
		return ret;
	}

	_FORCE_INLINE_ void set_from_std_vectors(const std::vector<std::vector<double>> &p_from) {
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
			const std::vector<double> &r = p_from[i];

			ERR_CONTINUE(r.size() != static_cast<uint32_t>(_size.x));

			int start_index = i * _size.x;

			const double *from_ptr = &r[0];
			for (int j = 0; j < _size.x; j++) {
				_data[start_index + j] = from_ptr[j];
			}
		}
	}

	_FORCE_INLINE_ void set_row_std_vector(int p_index_y, const std::vector<double> &p_row) {
		ERR_FAIL_COND(p_row.size() != static_cast<uint32_t>(_size.x));
		ERR_FAIL_INDEX(p_index_y, _size.y);

		int ind_start = p_index_y * _size.x;

		const double *row_ptr = &p_row[0];

		for (int i = 0; i < _size.x; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	MLPPMatrix(const std::vector<std::vector<double>> &p_from) {
		_data = NULL;

		set_from_std_vectors(p_from);
	}

protected:
	static void _bind_methods();

protected:
	Size2i _size;
	double *_data;
};

#endif
