#ifndef MLPP_VECTOR_H
#define MLPP_VECTOR_H

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/os/memory.h"

#include "core/object/reference.h"

//REMOVE
#include <vector>

class MLPPVector : public Reference {
	GDCLASS(MLPPVector, Reference);

protected:
	int count;
	int capacity;
	double *data;

public:
	double *ptr() {
		return data;
	}

	const double *ptr() const {
		return data;
	}

	_FORCE_INLINE_ void push_back(double p_elem) {
		if (unlikely(count == capacity)) {
			if (capacity == 0) {
				capacity = 1;
			} else {
				capacity <<= 1;
			}

			data = (double *)memrealloc(data, capacity * sizeof(double));
			CRASH_COND_MSG(!data, "Out of memory");
		}

		data[count++] = p_elem;
	}

	void remove(double p_index) {
		ERR_FAIL_INDEX(p_index, count);
		count--;
		for (int i = p_index; i < count; i++) {
			data[i] = data[i + 1];
		}
	}

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_unordered(int p_index) {
		ERR_FAIL_INDEX(p_index, count);
		count--;

		if (count > p_index) {
			data[p_index] = data[count];
		}
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
		for (int i = 0; i < count / 2; i++) {
			SWAP(data[i], data[count - i - 1]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(0); }
	_FORCE_INLINE_ void reset() {
		clear();
		if (data) {
			memfree(data);
			data = NULL;
			capacity = 0;
		}
	}
	_FORCE_INLINE_ bool empty() const { return count == 0; }
	_FORCE_INLINE_ int get_capacity() const { return capacity; }
	_FORCE_INLINE_ void reserve(int p_size) {
		p_size = nearest_power_of_2_templated(p_size);
		if (p_size > capacity) {
			capacity = p_size;
			data = (double *)memrealloc(data, capacity * sizeof(double));
			CRASH_COND_MSG(!data, "Out of memory");
		}
	}

	_FORCE_INLINE_ int size() const { return count; }
	void resize(int p_size) {
		if (p_size < count) {
			count = p_size;
		} else if (p_size > count) {
			if (unlikely(p_size > capacity)) {
				if (capacity == 0) {
					capacity = 1;
				}
				while (capacity < p_size) {
					capacity <<= 1;
				}
				data = (double *)memrealloc(data, capacity * sizeof(double));
				CRASH_COND_MSG(!data, "Out of memory");
			}

			count = p_size;
		}
	}
	_FORCE_INLINE_ const double &operator[](int p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}
	_FORCE_INLINE_ double &operator[](int p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}

	_FORCE_INLINE_ const double &get_element(int p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}
	_FORCE_INLINE_ double &get_element(int p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return data[p_index];
	}

	_FORCE_INLINE_ real_t get_element_bind(int p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);
		return static_cast<real_t>(data[p_index]);
	}

	void fill(double p_val) {
		for (int i = 0; i < count; i++) {
			data[i] = p_val;
		}
	}

	void insert(int p_pos, double p_val) {
		ERR_FAIL_UNSIGNED_INDEX(p_pos, count + 1);
		if (p_pos == count) {
			push_back(p_val);
		} else {
			resize(count + 1);
			for (int i = count - 1; i > p_pos; i--) {
				data[i] = data[i - 1];
			}
			data[p_pos] = p_val;
		}
	}

	int64_t find(const double &p_val, int p_from = 0) const {
		for (int i = p_from; i < count; i++) {
			if (data[i] == p_val) {
				return int64_t(i);
			}
		}
		return -1;
	}

	template <class C>
	void sort_custom() {
		int len = count;
		if (len == 0) {
			return;
		}

		SortArray<double, C> sorter;
		sorter.sort(data, len);
	}

	void sort() {
		sort_custom<_DefaultComparator<double>>();
	}

	void ordered_insert(double p_val) {
		int i;
		for (i = 0; i < count; i++) {
			if (p_val < data[i]) {
				break;
			}
		}
		insert(i, p_val);
	}

	Vector<double> to_vector() const {
		Vector<double> ret;
		ret.resize(size());
		double *w = ret.ptrw();
		memcpy(w, data, sizeof(double) * count);
		return ret;
	}

	PoolRealArray to_pool_vector() const {
		PoolRealArray pl;
		if (size()) {
			pl.resize(size());
			typename PoolRealArray::Write w = pl.write();
			real_t *dest = w.ptr();

			for (int i = 0; i < size(); ++i) {
				dest[i] = static_cast<real_t>(data[i]);
			}
		}
		return pl;
	}

	Vector<uint8_t> to_byte_array() const {
		Vector<uint8_t> ret;
		ret.resize(count * sizeof(double));
		uint8_t *w = ret.ptrw();
		memcpy(w, data, sizeof(double) * count);
		return ret;
	}

	Ref<MLPPVector> duplicate() const {
		Ref<MLPPVector> ret;
		ret.instance();

		ret->set_from_mlpp_vectorr(*this);

		return ret;
	}

	_FORCE_INLINE_ void set_from_mlpp_vectorr(const MLPPVector &p_from) {
		resize(p_from.size());
		for (int i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_vector(const Ref<MLPPVector> &p_from) {
		ERR_FAIL_COND(!p_from.is_valid());
		resize(p_from->size());
		for (int i = 0; i < p_from->count; i++) {
			data[i] = p_from->data[i];
		}
	}

	_FORCE_INLINE_ void set_from_vector(const Vector<double> &p_from) {
		resize(p_from.size());
		for (int i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}

	_FORCE_INLINE_ void set_from_pool_vector(const PoolRealArray &p_from) {
		resize(p_from.size());
		typename PoolRealArray::Read r = p_from.read();
		for (int i = 0; i < count; i++) {
			data[i] = r[i];
		}
	}

	_FORCE_INLINE_ MLPPVector() {
		count = 0;
		capacity = 0;
		data = NULL;
	}
	_FORCE_INLINE_ MLPPVector(const MLPPVector &p_from) {
		count = 0;
		capacity = 0;
		data = NULL;

		resize(p_from.size());
		for (int i = 0; i < p_from.count; i++) {
			data[i] = p_from.data[i];
		}
	}

	MLPPVector(const Vector<double> &p_from) {
		count = 0;
		capacity = 0;
		data = NULL;

		resize(p_from.size());
		for (int i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}

	MLPPVector(const PoolRealArray &p_from) {
		count = 0;
		capacity = 0;
		data = NULL;

		resize(p_from.size());
		typename PoolRealArray::Read r = p_from.read();
		for (int i = 0; i < count; i++) {
			data[i] = r[i];
		}
	}

	_FORCE_INLINE_ ~MLPPVector() {
		if (data) {
			reset();
		}
	}

	// TODO: These are temporary
	std::vector<double> to_std_vector() const {
		std::vector<double> ret;
		ret.resize(size());
		double *w = &ret[0];
		memcpy(w, data, sizeof(double) * count);
		return ret;
	}

	_FORCE_INLINE_ void set_from_std_vector(const std::vector<double> &p_from) {
		resize(p_from.size());
		for (int i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}

	MLPPVector(const std::vector<double> &p_from) {
		count = 0;
		capacity = 0;
		data = NULL;

		resize(p_from.size());
		for (int i = 0; i < count; i++) {
			data[i] = p_from[i];
		}
	}

protected:
	static void _bind_methods();
};

#endif
