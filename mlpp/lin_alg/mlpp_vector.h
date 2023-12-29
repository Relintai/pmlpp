#ifndef MLPP_VECTOR_H
#define MLPP_VECTOR_H

#ifndef GDNATIVE

#include "core/math/math_defs.h"
#include "core/math/math_funcs.h"

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/os/memory.h"

#include "core/object/resource.h"

#else

#include "core/containers/vector.h"
#include "core/defs.h"
#include "core/math_funcs.h"
#include "core/os/memory.h"
#include "core/pool_arrays.h"

#include "gen/resource.h"

#endif

//REMOVE
#include <vector>

class MLPPMatrix;

class MLPPVector : public Resource {
	GDCLASS(MLPPVector, Resource);

public:
	PoolRealArray get_data();
	void set_data(const PoolRealArray &p_from);

	_FORCE_INLINE_ real_t *ptrw() {
		return _data;
	}

	_FORCE_INLINE_ const real_t *ptr() const {
		return _data;
	}

	void push_back(real_t p_elem);
	void append_mlpp_vector(const Ref<MLPPVector> &p_other);

	void remove(int p_index);

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_unordered(int p_index);

	void erase(const real_t &p_val);
	int erase_multiple_unordered(const real_t &p_val);

	void invert();

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

	void resize(int p_size);

	_FORCE_INLINE_ const real_t &operator[](int p_index) const {
		CRASH_BAD_INDEX(p_index, _size);
		return _data[p_index];
	}
	_FORCE_INLINE_ real_t &operator[](int p_index) {
		CRASH_BAD_INDEX(p_index, _size);
		return _data[p_index];
	}

	_FORCE_INLINE_ real_t element_get(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, _size, 0);
		return _data[p_index];
	}

	_FORCE_INLINE_ void element_set(int p_index, real_t p_val) {
		ERR_FAIL_INDEX(p_index, _size);
		_data[p_index] = p_val;
	}

	_FORCE_INLINE_ const real_t &element_get_ref(int p_index) const {
		CRASH_BAD_INDEX(p_index, _size);
		return _data[p_index];
	}

	_FORCE_INLINE_ real_t &element_get_ref(int p_index) {
		CRASH_BAD_INDEX(p_index, _size);
		return _data[p_index];
	}

	void fill(real_t p_val);
	void insert(int p_pos, real_t p_val);

	int find(const real_t &p_val, int p_from = 0) const;

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

	void ordered_insert(real_t p_val);

	Vector<real_t> to_vector() const;
	PoolRealArray to_pool_vector() const;
	Vector<uint8_t> to_byte_array() const;

	Ref<MLPPVector> duplicate_fast() const;

	void set_from_mlpp_vectorr(const MLPPVector &p_from);
	void set_from_mlpp_vector(const Ref<MLPPVector> &p_from);
	void set_from_vector(const Vector<real_t> &p_from);
	void set_from_pool_vector(const PoolRealArray &p_from);

	bool is_equal_approx(const Ref<MLPPVector> &p_with, real_t tolerance = static_cast<real_t>(CMP_EPSILON)) const;

	void flatten_vectors(const Vector<Ref<MLPPVector>> &A);
	Ref<MLPPVector> flatten_vectorsn(const Vector<Ref<MLPPVector>> &A) const;

	void hadamard_product(const Ref<MLPPVector> &b);
	Ref<MLPPVector> hadamard_productn(const Ref<MLPPVector> &b) const;
	void hadamard_productb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void division_element_wise(const Ref<MLPPVector> &b);
	Ref<MLPPVector> division_element_wisen(const Ref<MLPPVector> &b) const;
	void division_element_wiseb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void scalar_multiply(real_t scalar);
	Ref<MLPPVector> scalar_multiplyn(real_t scalar) const;
	void scalar_multiplyb(real_t scalar, const Ref<MLPPVector> &a);

	void scalar_add(real_t scalar);
	Ref<MLPPVector> scalar_addn(real_t scalar) const;
	void scalar_addb(real_t scalar, const Ref<MLPPVector> &a);

	void add(const Ref<MLPPVector> &b);
	Ref<MLPPVector> addn(const Ref<MLPPVector> &b) const;
	void addb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void sub(const Ref<MLPPVector> &b);
	Ref<MLPPVector> subn(const Ref<MLPPVector> &b) const;
	void subb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void log();
	Ref<MLPPVector> logn() const;
	void logb(const Ref<MLPPVector> &a);

	void log10();
	Ref<MLPPVector> log10n() const;
	void log10b(const Ref<MLPPVector> &a);

	void exp();
	Ref<MLPPVector> expn() const;
	void expb(const Ref<MLPPVector> &a);

	void erf();
	Ref<MLPPVector> erfn() const;
	void erfb(const Ref<MLPPVector> &a);

	void exponentiate(real_t p);
	Ref<MLPPVector> exponentiaten(real_t p) const;
	void exponentiateb(const Ref<MLPPVector> &a, real_t p);

	void sqrt();
	Ref<MLPPVector> sqrtn() const;
	void sqrtb(const Ref<MLPPVector> &a);

	void cbrt();
	Ref<MLPPVector> cbrtn() const;
	void cbrtb(const Ref<MLPPVector> &a);

	real_t dot(const Ref<MLPPVector> &b) const;

	Ref<MLPPVector> cross(const Ref<MLPPVector> &b);

	void abs();
	Ref<MLPPVector> absn() const;
	void absb(const Ref<MLPPVector> &a);

	Ref<MLPPVector> vecn_zero(int n) const;
	Ref<MLPPVector> vecn_one(int n) const;
	Ref<MLPPVector> vecn_full(int n, int k) const;

	static Ref<MLPPVector> create_vec_zero(int n);
	static Ref<MLPPVector> create_vec_one(int n);
	static Ref<MLPPVector> create_vec_full(int n, int k);

	void sin();
	Ref<MLPPVector> sinn() const;
	void sinb(const Ref<MLPPVector> &a);

	void cos();
	Ref<MLPPVector> cosn() const;
	void cosb(const Ref<MLPPVector> &a);

	void max(const Ref<MLPPVector> &b);
	Ref<MLPPVector> maxn(const Ref<MLPPVector> &b) const;
	void maxb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	void min(const Ref<MLPPVector> &b);
	Ref<MLPPVector> minn(const Ref<MLPPVector> &b) const;
	void minb(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	real_t max_element() const;
	int max_element_index() const;

	real_t min_element() const;
	int min_element_index() const;

	//std::vector<real_t> round(std::vector<real_t> a);

	real_t euclidean_distance(const Ref<MLPPVector> &b) const;
	real_t euclidean_distance_squared(const Ref<MLPPVector> &b) const;

	real_t norm_2() const;
	real_t norm_sq() const;

	real_t sum_elements() const;

	//real_t cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b);

	void subtract_matrix_rows(const Ref<MLPPMatrix> &B);
	Ref<MLPPVector> subtract_matrix_rowsn(const Ref<MLPPMatrix> &B) const;
	void subtract_matrix_rowsb(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B);

	// This multiplies a, bT
	Ref<MLPPMatrix> outer_product(const Ref<MLPPVector> &b) const;

	// as_diagonal_matrix / to_diagonal_matrix
	Ref<MLPPMatrix> diagnm() const;

	String to_string();

	MLPPVector();
	MLPPVector(const MLPPVector &p_from);
	MLPPVector(const Vector<real_t> &p_from);
	MLPPVector(const PoolRealArray &p_from);
	MLPPVector(const real_t *p_from, const int p_size);

	~MLPPVector();

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
