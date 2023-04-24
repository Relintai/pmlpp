#ifndef MLPP_VECTOR_H
#define MLPP_VECTOR_H

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/os/memory.h"

#include "core/object/reference.h"

//REMOVE
#include <vector>

class MLPPMatrix;

class MLPPVector : public Reference {
	GDCLASS(MLPPVector, Reference);

public:
	real_t *ptrw() {
		return _data;
	}

	const real_t *ptr() const {
		return _data;
	}

	_FORCE_INLINE_ void push_back(real_t p_elem) {
		++_size;

		_data = (real_t *)memrealloc(_data, _size * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");

		_data[_size - 1] = p_elem;
	}

	_FORCE_INLINE_ void add_mlpp_vector(const Ref<MLPPVector> &p_other) {
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

	void remove(real_t p_index) {
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
	void remove_unordered(int p_index) {
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

	void erase(const real_t &p_val) {
		int idx = find(p_val);
		if (idx >= 0) {
			remove(idx);
		}
	}

	int erase_multiple_unordered(const real_t &p_val) {
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
		for (int i = 0; i < _size / 2; i++) {
			SWAP(_data[i], _data[_size - i - 1]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ void reset() {
		if (_data) {
			memfree(_data);
			_data = NULL;
			_size = 0;
		}
	}

	_FORCE_INLINE_ bool empty() const { return _size == 0; }
	_FORCE_INLINE_ int size() const { return _size; }

	void resize(int p_size) {
		_size = p_size;

		if (_size == 0) {
			memfree(_data);
			_data = NULL;
			return;
		}

		_data = (real_t *)memrealloc(_data, _size * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	_FORCE_INLINE_ const real_t &operator[](int p_index) const {
		CRASH_BAD_INDEX(p_index, _size);
		return _data[p_index];
	}
	_FORCE_INLINE_ real_t &operator[](int p_index) {
		CRASH_BAD_INDEX(p_index, _size);
		return _data[p_index];
	}

	_FORCE_INLINE_ real_t get_element(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, _size, 0);
		return _data[p_index];
	}

	_FORCE_INLINE_ void set_element(int p_index, real_t p_val) {
		ERR_FAIL_INDEX(p_index, _size);
		_data[p_index] = p_val;
	}

	void fill(real_t p_val) {
		for (int i = 0; i < _size; i++) {
			_data[i] = p_val;
		}
	}

	void insert(int p_pos, real_t p_val) {
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

	int find(const real_t &p_val, int p_from = 0) const {
		for (int i = p_from; i < _size; i++) {
			if (_data[i] == p_val) {
				return i;
			}
		}
		return -1;
	}

	template <class C>
	void sort_custom() {
		int len = _size;
		if (len == 0) {
			return;
		}

		SortArray<real_t, C> sorter;
		sorter.sort(_data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<real_t>>();
	}

	void ordered_insert(real_t p_val) {
		int i;
		for (i = 0; i < _size; i++) {
			if (p_val < _data[i]) {
				break;
			}
		}
		insert(i, p_val);
	}

	Vector<real_t> to_vector() const {
		Vector<real_t> ret;
		ret.resize(size());
		real_t *w = ret.ptrw();
		memcpy(w, _data, sizeof(real_t) * _size);
		return ret;
	}

	PoolRealArray to_pool_vector() const {
		PoolRealArray pl;
		if (size()) {
			pl.resize(size());
			typename PoolRealArray::Write w = pl.write();
			real_t *dest = w.ptr();

			for (int i = 0; i < size(); ++i) {
				dest[i] = static_cast<real_t>(_data[i]);
			}
		}
		return pl;
	}

	Vector<uint8_t> to_byte_array() const {
		Vector<uint8_t> ret;
		ret.resize(_size * sizeof(real_t));
		uint8_t *w = ret.ptrw();
		memcpy(w, _data, sizeof(real_t) * _size);
		return ret;
	}

	Ref<MLPPVector> duplicate() const {
		Ref<MLPPVector> ret;
		ret.instance();

		ret->set_from_mlpp_vectorr(*this);

		return ret;
	}

	_FORCE_INLINE_ void set_from_mlpp_vectorr(const MLPPVector &p_from) {
		if (_size != p_from.size()) {
			resize(p_from.size());
		}

		for (int i = 0; i < p_from._size; i++) {
			_data[i] = p_from._data[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_vector(const Ref<MLPPVector> &p_from) {
		ERR_FAIL_COND(!p_from.is_valid());

		if (_size != p_from->size()) {
			resize(p_from->size());
		}

		for (int i = 0; i < p_from->_size; i++) {
			_data[i] = p_from->_data[i];
		}
	}

	_FORCE_INLINE_ void set_from_vector(const Vector<real_t> &p_from) {
		if (_size != p_from.size()) {
			resize(p_from.size());
		}

		resize(p_from.size());
		for (int i = 0; i < _size; i++) {
			_data[i] = p_from[i];
		}
	}

	_FORCE_INLINE_ void set_from_pool_vector(const PoolRealArray &p_from) {
		if (_size != p_from.size()) {
			resize(p_from.size());
		}

		PoolRealArray::Read r = p_from.read();
		for (int i = 0; i < _size; i++) {
			_data[i] = r[i];
		}
	}

	_FORCE_INLINE_ bool is_equal_approx(const Ref<MLPPVector> &p_with, real_t tolerance = static_cast<real_t>(CMP_EPSILON)) const {
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

	// New apis should look like this:
	//void substract(const Ref<MLPPVector> &b); <- this should be the simplest / most obvious method
	//Ref<MLPPVector> substractn(const Ref<MLPPVector> &b);
	//void substractb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b); -> result is in this (subtractionv like)

	// Or:
	//void hadamard_product(const Ref<MLPPVector> &b); <- this should be the simplest / most obvious method
	//Ref<MLPPVector> hadamard_productn(const Ref<MLPPVector> &b); <- n -> new
	//void hadamard_productb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b); <- b -> between, result is stored in *this

	void flatten_vectors(const Vector<Ref<MLPPVector>> &A);
	Ref<MLPPVector> flatten_vectorsn(const Vector<Ref<MLPPVector>> &A);

	void hadamard_product(const Ref<MLPPVector> &b);
	Ref<MLPPVector> hadamard_productn(const Ref<MLPPVector> &b);
	void hadamard_productb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void element_wise_division(const Ref<MLPPVector> &b);
	Ref<MLPPVector> element_wise_divisionn(const Ref<MLPPVector> &b);
	void element_wise_divisionb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void scalar_multiply(real_t scalar);
	Ref<MLPPVector> scalar_multiplyn(real_t scalar);
	void scalar_multiplyb(real_t scalar, const Ref<MLPPVector> &a);

	void scalar_add(real_t scalar);
	Ref<MLPPVector> scalar_addn(real_t scalar);
	void scalar_addb(real_t scalar, const Ref<MLPPVector> &a);

	void add(const Ref<MLPPVector> &b);
	Ref<MLPPVector> addn(const Ref<MLPPVector> &b);
	void addb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void sub(const Ref<MLPPVector> &b);
	Ref<MLPPVector> subn(const Ref<MLPPVector> &b);
	void subb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void log();
	Ref<MLPPVector> logn();
	void logb(const Ref<MLPPVector> &a);

	void log10();
	Ref<MLPPVector> log10n();
	void log10b(const Ref<MLPPVector> &a);

	void exp();
	Ref<MLPPVector> expn();
	void expb(const Ref<MLPPVector> &a);

	void erf();
	Ref<MLPPVector> erfn();
	void erfb(const Ref<MLPPVector> &a);

	void exponentiate(real_t p);
	Ref<MLPPVector> exponentiaten(real_t p);
	void exponentiateb(const Ref<MLPPVector> &a, real_t p);

	void sqrt();
	Ref<MLPPVector> sqrtn();
	void sqrtb(const Ref<MLPPVector> &a);

	void cbrt();
	Ref<MLPPVector> cbrtn();
	void cbrtb(const Ref<MLPPVector> &a);

	real_t dotnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	//std::vector<real_t> cross(std::vector<real_t> a, std::vector<real_t> b);

	Ref<MLPPVector> absv(const Ref<MLPPVector> &a);

	Ref<MLPPVector> zerovecnv(int n);
	Ref<MLPPVector> onevecnv(int n);
	Ref<MLPPVector> fullnv(int n, int k);

	Ref<MLPPVector> sinnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> cosnv(const Ref<MLPPVector> &a);

	Ref<MLPPVector> maxnvv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	real_t maxvr(const Ref<MLPPVector> &a);
	real_t minvr(const Ref<MLPPVector> &a);

	//std::vector<real_t> round(std::vector<real_t> a);

	real_t euclidean_distance(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	real_t euclidean_distance_squared(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	/*
	real_t norm_2(std::vector<real_t> a);
	*/

	real_t norm_sqv(const Ref<MLPPVector> &a);

	real_t sum_elementsv(const Ref<MLPPVector> &a);

	//real_t cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b);

	Ref<MLPPVector> subtract_matrix_rowsnv(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b); // This multiplies a, bT

	// as_diagonal_matrix / to_diagonal_matrix
	Ref<MLPPMatrix> diagnm(const Ref<MLPPVector> &a);

	String to_string();

	_FORCE_INLINE_ MLPPVector() {
		_size = 0;
		_data = NULL;
	}
	_FORCE_INLINE_ MLPPVector(const MLPPVector &p_from) {
		_size = 0;
		_data = NULL;

		resize(p_from.size());
		for (int i = 0; i < p_from._size; i++) {
			_data[i] = p_from._data[i];
		}
	}

	MLPPVector(const Vector<real_t> &p_from) {
		_size = 0;
		_data = NULL;

		resize(p_from.size());
		for (int i = 0; i < _size; i++) {
			_data[i] = p_from[i];
		}
	}

	MLPPVector(const PoolRealArray &p_from) {
		_size = 0;
		_data = NULL;

		resize(p_from.size());
		typename PoolRealArray::Read r = p_from.read();
		for (int i = 0; i < _size; i++) {
			_data[i] = r[i];
		}
	}

	_FORCE_INLINE_ ~MLPPVector() {
		if (_data) {
			reset();
		}
	}

	// TODO: These are temporary
	std::vector<real_t> to_std_vector() const;
	void set_from_std_vector(const std::vector<real_t> &p_from);
	MLPPVector(const std::vector<real_t> &p_from);

protected:
	static void _bind_methods();

protected:
	int _size;
	real_t *_data;
};

#endif
