#ifndef MLPP_MATRIX_H
#define MLPP_MATRIX_H

#include "core/math/math_defs.h"

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/math/vector2i.h"
#include "core/os/memory.h"

#include "core/object/resource.h"

#include "mlpp_vector.h"

class Image;

class MLPPMatrix : public Resource {
	GDCLASS(MLPPMatrix, Resource);

public:
	Array get_data();
	void set_data(const Array &p_from);

	_FORCE_INLINE_ real_t *ptrw() {
		return _data;
	}

	_FORCE_INLINE_ const real_t *ptr() const {
		return _data;
	}

	void add_row(const Vector<real_t> &p_row);
	void add_row_pool_vector(const PoolRealArray &p_row);
	void add_row_mlpp_vector(const Ref<MLPPVector> &p_row);
	void add_rows_mlpp_matrix(const Ref<MLPPMatrix> &p_other);

	void remove_row(int p_index);

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_row_unordered(int p_index);

	void swap_row(int p_index_1, int p_index_2);

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

	void resize(const Size2i &p_size);

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

	Vector<real_t> get_row_vector(int p_index_y) const;
	PoolRealArray get_row_pool_vector(int p_index_y) const;
	Ref<MLPPVector> get_row_mlpp_vector(int p_index_y) const;
	void get_row_into_mlpp_vector(int p_index_y, Ref<MLPPVector> target) const;

	void set_row_vector(int p_index_y, const Vector<real_t> &p_row);
	void set_row_pool_vector(int p_index_y, const PoolRealArray &p_row);
	void set_row_mlpp_vector(int p_index_y, const Ref<MLPPVector> &p_row);

	void fill(real_t p_val);

	Vector<real_t> to_flat_vector() const;
	PoolRealArray to_flat_pool_vector() const;
	Vector<uint8_t> to_flat_byte_array() const;

	Ref<MLPPMatrix> duplicate_fast() const;

	void set_from_mlpp_matrix(const Ref<MLPPMatrix> &p_from);
	void set_from_mlpp_matrixr(const MLPPMatrix &p_from);
	void set_from_mlpp_vectors(const Vector<Ref<MLPPVector>> &p_from);
	void set_from_mlpp_vectors_array(const Array &p_from);
	void set_from_vectors(const Vector<Vector<real_t>> &p_from);
	void set_from_arrays(const Array &p_from);

	//std::vector<std::vector<real_t>> gramMatrix(std::vector<std::vector<real_t>> A);
	//bool linearIndependenceChecker(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> gaussian_noise(int n, int m) const;
	void gaussian_noise_fill();

	void add(const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> addn(const Ref<MLPPMatrix> &B) const;
	void addb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	void sub(const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> subn(const Ref<MLPPMatrix> &B) const;
	void subb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	void mult(const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> multn(const Ref<MLPPMatrix> &B) const;
	void multb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	void hadamard_product(const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> hadamard_productn(const Ref<MLPPMatrix> &B) const;
	void hadamard_productb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	void kronecker_product(const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> kronecker_productn(const Ref<MLPPMatrix> &B) const;
	void kronecker_productb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	void element_wise_division(const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> element_wise_divisionn(const Ref<MLPPMatrix> &B) const;
	void element_wise_divisionb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	void transpose();
	Ref<MLPPMatrix> transposen() const;
	void transposeb(const Ref<MLPPMatrix> &A);

	void scalar_multiply(const real_t scalar);
	Ref<MLPPMatrix> scalar_multiplyn(const real_t scalar) const;
	void scalar_multiplyb(const real_t scalar, const Ref<MLPPMatrix> &A);

	void scalar_add(const real_t scalar);
	Ref<MLPPMatrix> scalar_addn(const real_t scalar) const;
	void scalar_addb(const real_t scalar, const Ref<MLPPMatrix> &A);

	void log();
	Ref<MLPPMatrix> logn() const;
	void logb(const Ref<MLPPMatrix> &A);

	void log10();
	Ref<MLPPMatrix> log10n() const;
	void log10b(const Ref<MLPPMatrix> &A);

	void exp();
	Ref<MLPPMatrix> expn() const;
	void expb(const Ref<MLPPMatrix> &A);

	void erf();
	Ref<MLPPMatrix> erfn() const;
	void erfb(const Ref<MLPPMatrix> &A);

	void exponentiate(real_t p);
	Ref<MLPPMatrix> exponentiaten(real_t p) const;
	void exponentiateb(const Ref<MLPPMatrix> &A, real_t p);

	void sqrt();
	Ref<MLPPMatrix> sqrtn() const;
	void sqrtb(const Ref<MLPPMatrix> &A);

	void cbrt();
	Ref<MLPPMatrix> cbrtn() const;
	void cbrtb(const Ref<MLPPMatrix> &A);

	//std::vector<std::vector<real_t>> matrixPower(std::vector<std::vector<real_t>> A, int n);

	void abs();
	Ref<MLPPMatrix> absn() const;
	void absb(const Ref<MLPPMatrix> &A);

	real_t det(int d = -1) const;
	real_t detb(const Ref<MLPPMatrix> &A, int d) const;

	//real_t trace(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> cofactor(int n, int i, int j) const;
	void cofactoro(int n, int i, int j, Ref<MLPPMatrix> out) const;

	Ref<MLPPMatrix> adjoint() const;
	void adjointo(Ref<MLPPMatrix> out) const;

	Ref<MLPPMatrix> inverse() const;
	void inverseo(Ref<MLPPMatrix> out) const;

	Ref<MLPPMatrix> pinverse() const;
	void pinverseo(Ref<MLPPMatrix> out) const;

	Ref<MLPPMatrix> zero_mat(int n, int m) const;
	Ref<MLPPMatrix> one_mat(int n, int m) const;
	Ref<MLPPMatrix> full_mat(int n, int m, int k) const;

	void sin();
	Ref<MLPPMatrix> sinn() const;
	void sinb(const Ref<MLPPMatrix> &A);

	void cos();
	Ref<MLPPMatrix> cosn() const;
	void cosb(const Ref<MLPPMatrix> &A);

	//std::vector<std::vector<real_t>> rotate(std::vector<std::vector<real_t>> A, real_t theta, int axis = -1);

	void max(const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> maxn(const Ref<MLPPMatrix> &B) const;
	void maxb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	//real_t max(std::vector<std::vector<real_t>> A);
	//real_t min(std::vector<std::vector<real_t>> A);

	//std::vector<std::vector<real_t>> round(std::vector<std::vector<real_t>> A);

	//real_t norm_2(std::vector<std::vector<real_t>> A);

	void identity();
	Ref<MLPPMatrix> identityn() const;
	Ref<MLPPMatrix> identity_mat(int d) const;

	Ref<MLPPMatrix> cov() const;
	void covo(Ref<MLPPMatrix> out) const;

	struct EigenResult {
		Ref<MLPPMatrix> eigen_vectors;
		Ref<MLPPMatrix> eigen_values;
	};

	EigenResult eigen() const;
	EigenResult eigenb(const Ref<MLPPMatrix> &A) const;
	Array eigen_bind();
	Array eigenb_bind(const Ref<MLPPMatrix> &A);

	struct SVDResult {
		Ref<MLPPMatrix> U;
		Ref<MLPPMatrix> S;
		Ref<MLPPMatrix> Vt;
	};

	SVDResult svd() const;
	SVDResult svdb(const Ref<MLPPMatrix> &A) const;
	Array svd_bind();
	Array svdb_bind(const Ref<MLPPMatrix> &A);

	//std::vector<real_t> vectorProjection(std::vector<real_t> a, std::vector<real_t> b);

	//std::vector<std::vector<real_t>> gramSchmidtProcess(std::vector<std::vector<real_t>> A);

	/*
	struct QRDResult {
		std::vector<std::vector<real_t>> Q;
		std::vector<std::vector<real_t>> R;
	};
	*/

	//QRDResult qrd(std::vector<std::vector<real_t>> A);

	/*
	struct CholeskyResult {
		std::vector<std::vector<real_t>> L;
		std::vector<std::vector<real_t>> Lt;
	};

	CholeskyResult cholesky(std::vector<std::vector<real_t>> A);
	*/

	//real_t sum_elements(std::vector<std::vector<real_t>> A);

	Ref<MLPPVector> flatten() const;
	void flatteno(Ref<MLPPVector> out) const;

	/*
	std::vector<real_t> solve(std::vector<std::vector<real_t>> A, std::vector<real_t> b);

	bool positiveDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool negativeDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool zeroEigenvalue(std::vector<std::vector<real_t>> A);
	*/

	Ref<MLPPVector> mult_vec(const Ref<MLPPVector> &b) const;
	void mult_veco(const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	void add_vec(const Ref<MLPPVector> &b);
	Ref<MLPPMatrix> add_vecn(const Ref<MLPPVector> &b) const;
	void add_vecb(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);

	// This multiplies a, bT
	void outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	Ref<MLPPMatrix> outer_productn(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b) const;

	// Just sets the diagonal
	void set_diagonal(const Ref<MLPPVector> &a);
	Ref<MLPPMatrix> set_diagonaln(const Ref<MLPPVector> &a) const;

	// Sets the diagonals, everythign else will get zeroed
	void diagonal_zeroed(const Ref<MLPPVector> &a);
	Ref<MLPPMatrix> diagonal_zeroedn(const Ref<MLPPVector> &a) const;

	bool is_equal_approx(const Ref<MLPPMatrix> &p_with, real_t tolerance = static_cast<real_t>(CMP_EPSILON)) const;

	Ref<Image> get_as_image() const;
	void get_into_image(Ref<Image> out) const;
	void set_from_image(const Ref<Image> &p_img, const int p_image_channel);

	String to_string();

	MLPPMatrix();
	MLPPMatrix(const MLPPMatrix &p_from);
	MLPPMatrix(const Vector<Vector<real_t>> &p_from);
	MLPPMatrix(const Array &p_from);

	~MLPPMatrix();

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
