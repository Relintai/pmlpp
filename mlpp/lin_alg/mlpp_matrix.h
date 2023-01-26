#ifndef MLPP_MATRIX_H
#define MLPP_MATRIX_H

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/math/vector2i.h"

#include "core/object/reference.h"

#include "mlpp_vector.h"

class MLPPMatrix : public Reference {
	GDCLASS(MLPPMatrix, Reference);

public:
	double *ptr() {
		return _data;
	}

	const double *ptr() const {
		return _data;
	}

	_FORCE_INLINE_ void push_back(double p_elem) {
		++_data_size;

		_data = (double *)memrealloc(_data, _data_size * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");

		_data[_data_size - 1] = p_elem;
	}

	void remove(double p_index) {
		ERR_FAIL_INDEX(p_index, _data_size);

		--_data_size;

		for (int i = p_index; i < _data_size; i++) {
			_data[i] = _data[i + 1];
		}

		_data = (double *)memrealloc(_data, _data_size * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_unordered(int p_index) {
		ERR_FAIL_INDEX(p_index, _data_size);
		_data_size--;

		if (_data_size > p_index) {
			_data[p_index] = _data[_data_size];
		}

		_data = (double *)memrealloc(_data, _data_size * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	void erase(const double &p_val) {
		int idx = find(p_val);
		if (idx >= 0) {
			remove(idx);
		}
	}

	int erase_multiple_unordered(const double &p_val) {
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

	void invert() {
		for (int i = 0; i < _data_size / 2; i++) {
			SWAP(_data[i], _data[_data_size - i - 1]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ void reset() {
		clear();
		if (_data) {
			memfree(_data);
			_data = NULL;
			_data_size = 0;
		}
	}

	_FORCE_INLINE_ bool empty() const { return _data_size == 0; }
	_FORCE_INLINE_ int data_size() const { return _data_size; }

	void resize(int p_size) {
		_data_size = p_size;

		_data = (double *)memrealloc(_data, _data_size * sizeof(double));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	_FORCE_INLINE_ const double &operator[](int p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _data_size);
		return _data[p_index];
	}
	_FORCE_INLINE_ double &operator[](int p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _data_size);
		return _data[p_index];
	}

	_FORCE_INLINE_ double get_element(int p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _data_size);
		return _data[p_index];
	}
	_FORCE_INLINE_ double get_element(int p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _data_size);
		return _data[p_index];
	}

	_FORCE_INLINE_ real_t get_element_bind(int p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _data_size);
		return static_cast<real_t>(_data[p_index]);
	}

	_FORCE_INLINE_ void set_element(int p_index, double p_val) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _data_size);
		_data[p_index] = p_val;
	}

	_FORCE_INLINE_ void set_element_bind(int p_index, real_t p_val) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _data_size);
		_data[p_index] = p_val;
	}

	void fill(double p_val) {
		for (int i = 0; i < _data_size; i++) {
			_data[i] = p_val;
		}
	}

	void insert(int p_pos, double p_val) {
		ERR_FAIL_INDEX(p_pos, _data_size + 1);
		if (p_pos == _data_size) {
			push_back(p_val);
		} else {
			resize(_data_size + 1);
			for (int i = _data_size - 1; i > p_pos; i--) {
				_data[i] = _data[i - 1];
			}
			_data[p_pos] = p_val;
		}
	}

	int find(const double &p_val, int p_from = 0) const {
		for (int i = p_from; i < _data_size; i++) {
			if (_data[i] == p_val) {
				return i;
			}
		}
		return -1;
	}

	template <class C>
	void sort_custom() {
		int len = _data_size;
		if (len == 0) {
			return;
		}

		SortArray<double, C> sorter;
		sorter.sort(_data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<double>>();
	}

	void ordered_insert(double p_val) {
		int i;
		for (i = 0; i < _data_size; i++) {
			if (p_val < _data[i]) {
				break;
			}
		}
		insert(i, p_val);
	}

	Vector<double> to_vector() const {
		Vector<double> ret;
		ret.resize(data_size());
		double *w = ret.ptrw();
		memcpy(w, _data, sizeof(double) * _data_size);
		return ret;
	}

	PoolRealArray to_pool_vector() const {
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

	Vector<uint8_t> to_byte_array() const {
		Vector<uint8_t> ret;
		ret.resize(_data_size * sizeof(double));
		uint8_t *w = ret.ptrw();
		memcpy(w, _data, sizeof(double) * _data_size);
		return ret;
	}

	Ref<MLPPMatrix> duplicate() const {
		Ref<MLPPMatrix> ret;
		ret.instance();

		ret->set_from_mlpp_matrixr(*this);

		return ret;
	}

	_FORCE_INLINE_ void set_from_mlpp_matrixr(const MLPPMatrix &p_from) {
		resize(p_from.data_size());
		for (int i = 0; i < p_from._data_size; i++) {
			_data[i] = p_from._data[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_vectorr(const MLPPVector &p_from) {
		resize(p_from.size());
		const double *from_ptr = p_from.ptr();
		for (int i = 0; i < p_from.size(); i++) {
			_data[i] = from_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_vector(const Ref<MLPPVector> &p_from) {
		ERR_FAIL_COND(!p_from.is_valid());
		resize(p_from->size());
		const double *from_ptr = p_from->ptr();
		for (int i = 0; i < p_from->size(); i++) {
			_data[i] = from_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_from_vector(const Vector<double> &p_from) {
		resize(p_from.size());
		for (int i = 0; i < _data_size; i++) {
			_data[i] = p_from[i];
		}
	}

	_FORCE_INLINE_ void set_from_pool_vector(const PoolRealArray &p_from) {
		resize(p_from.size());
		typename PoolRealArray::Read r = p_from.read();
		for (int i = 0; i < _data_size; i++) {
			_data[i] = r[i];
		}
	}

	_FORCE_INLINE_ MLPPMatrix() {
		_data_size = 0;
		_data = NULL;
	}
	_FORCE_INLINE_ MLPPMatrix(const MLPPMatrix &p_from) {
		_data_size = 0;
		_data = NULL;

		resize(p_from.data_size());
		for (int i = 0; i < p_from._data_size; i++) {
			_data[i] = p_from._data[i];
		}
	}

	MLPPMatrix(const Vector<double> &p_from) {
		_data_size = 0;
		_data = NULL;

		resize(p_from.size());
		for (int i = 0; i < _data_size; i++) {
			_data[i] = p_from[i];
		}
	}

	MLPPMatrix(const PoolRealArray &p_from) {
		_data_size = 0;
		_data = NULL;

		resize(p_from.size());
		typename PoolRealArray::Read r = p_from.read();
		for (int i = 0; i < _data_size; i++) {
			_data[i] = r[i];
		}
	}

	_FORCE_INLINE_ ~MLPPMatrix() {
		if (_data) {
			reset();
		}
	}

	// TODO: These are temporary
	std::vector<double> to_std_vector() const {
		std::vector<double> ret;
		ret.resize(data_size());
		double *w = &ret[0];
		memcpy(w, _data, sizeof(double) * _data_size);
		return ret;
	}

	_FORCE_INLINE_ void set_from_std_vector(const std::vector<double> &p_from) {
		resize(p_from.size());
		for (int i = 0; i < _data_size; i++) {
			_data[i] = p_from[i];
		}
	}

	MLPPMatrix(const std::vector<double> &p_from) {
		_data_size = 0;
		_data = NULL;

		resize(p_from.size());
		for (int i = 0; i < _data_size; i++) {
			_data[i] = p_from[i];
		}
	}

protected:
	static void _bind_methods();

protected:
	Size2i _size;
	int _data_size;
	double *_data;
};

#endif
