#ifndef MLPP_MATRIX_H
#define MLPP_MATRIX_H

/*************************************************************************/
/*  mlpp_matrix.h                                                        */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifdef USING_SFW
#include "sfw.h"
#include "image.h"
#else
#include "core/math/math_defs.h"

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/math/vector2i.h"
#include "core/os/memory.h"

#include "core/object/resource.h"
#endif

#include "mlpp_vector.h"

class Image;

class MLPPMatrix : public Resource {
  GDCLASS(MLPPMatrix, Resource);

public:
  Array get_data();
  void set_data(const Array &p_from);

  _FORCE_INLINE_ real_t *ptrw() { return _data; }

  _FORCE_INLINE_ const real_t *ptr() const { return _data; }

  void row_add(const Vector<real_t> &p_row);
  void row_add_pool_vector(const PoolRealArray &p_row);
  void row_add_mlpp_vector(const Ref<MLPPVector> &p_row);
  void rows_add_mlpp_matrix(const Ref<MLPPMatrix> &p_other);

  void row_remove(int p_index);

  // Removes the item copying the last value into the position of the one to
  // remove. It's generally faster than `remove`.
  void row_remove_unordered(int p_index);

  void row_swap(int p_index_1, int p_index_2);

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

  _FORCE_INLINE_ real_t element_get_index(int p_index) const {
    ERR_FAIL_INDEX_V(p_index, data_size(), 0);

    return _data[p_index];
  }

  _FORCE_INLINE_ void element_set_index(int p_index, real_t p_val) {
    ERR_FAIL_INDEX(p_index, data_size());

    _data[p_index] = p_val;
  }

  _FORCE_INLINE_ real_t element_get(int p_index_y, int p_index_x) const {
    ERR_FAIL_INDEX_V(p_index_x, _size.x, 0);
    ERR_FAIL_INDEX_V(p_index_y, _size.y, 0);

    return _data[p_index_y * _size.x + p_index_x];
  }

  _FORCE_INLINE_ void element_set(int p_index_y, int p_index_x, real_t p_val) {
    ERR_FAIL_INDEX(p_index_x, _size.x);
    ERR_FAIL_INDEX(p_index_y, _size.y);

    _data[p_index_y * _size.x + p_index_x] = p_val;
  }

  Vector<real_t> row_get_vector(int p_index_y) const;
  PoolRealArray row_get_pool_vector(int p_index_y) const;
  Ref<MLPPVector> row_get_mlpp_vector(int p_index_y) const;
  void row_get_into_mlpp_vector(int p_index_y, Ref<MLPPVector> target) const;

  void row_set_vector(int p_index_y, const Vector<real_t> &p_row);
  void row_set_pool_vector(int p_index_y, const PoolRealArray &p_row);
  void row_set_mlpp_vector(int p_index_y, const Ref<MLPPVector> &p_row);

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
  void set_from_ptr(const real_t *p_from, const int p_size_y,
                    const int p_size_x);

  // std::vector<std::vector<real_t>>
  // gramMatrix(std::vector<std::vector<real_t>> A); bool
  // linearIndependenceChecker(std::vector<std::vector<real_t>> A);

  Ref<MLPPMatrix> gaussian_noise(int n, int m) const;
  void gaussian_noise_fill();

  static Ref<MLPPMatrix> create_gaussian_noise(int n, int m);

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

  void division_element_wise(const Ref<MLPPMatrix> &B);
  Ref<MLPPMatrix> division_element_wisen(const Ref<MLPPMatrix> &B) const;
  void division_element_wiseb(const Ref<MLPPMatrix> &A,
                              const Ref<MLPPMatrix> &B);

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

  Ref<MLPPMatrix> matrix_powern(const int n) const;

  void abs();
  Ref<MLPPMatrix> absn() const;
  void absb(const Ref<MLPPMatrix> &A);

  real_t det(int d = -1) const;
  real_t detb(const Ref<MLPPMatrix> &A, int d) const;

  real_t trace() const;

  Ref<MLPPMatrix> cofactor(int n, int i, int j) const;
  void cofactoro(int n, int i, int j, Ref<MLPPMatrix> out) const;

  Ref<MLPPMatrix> adjoint() const;
  void adjointo(Ref<MLPPMatrix> out) const;

  Ref<MLPPMatrix> inverse() const;
  void inverseo(Ref<MLPPMatrix> out) const;

  Ref<MLPPMatrix> pinverse() const;
  void pinverseo(Ref<MLPPMatrix> out) const;

  Ref<MLPPMatrix> matn_zero(int n, int m) const;
  Ref<MLPPMatrix> matn_one(int n, int m) const;
  Ref<MLPPMatrix> matn_full(int n, int m, int k) const;

  void sin();
  Ref<MLPPMatrix> sinn() const;
  void sinb(const Ref<MLPPMatrix> &A);

  void cos();
  Ref<MLPPMatrix> cosn() const;
  void cosb(const Ref<MLPPMatrix> &A);

  Ref<MLPPMatrix> create_rotation_matrix(real_t theta, int axis = -1);

  void rotate(real_t theta, int axis = -1);
  Ref<MLPPMatrix> rotaten(real_t theta, int axis = -1);
  void rotateb(const Ref<MLPPMatrix> &A, real_t theta, int axis = -1);

  void max(const Ref<MLPPMatrix> &B);
  Ref<MLPPMatrix> maxn(const Ref<MLPPMatrix> &B) const;
  void maxb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

  void min(const Ref<MLPPMatrix> &B);
  Ref<MLPPMatrix> minn(const Ref<MLPPMatrix> &B) const;
  void minb(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

  // real_t max(std::vector<std::vector<real_t>> A);
  // real_t min(std::vector<std::vector<real_t>> A);

  // std::vector<std::vector<real_t>> round(std::vector<std::vector<real_t>> A);

  // real_t norm_2(std::vector<std::vector<real_t>> A);

  void identity();
  Ref<MLPPMatrix> identityn() const;
  Ref<MLPPMatrix> identity_mat(int d) const;

  static Ref<MLPPMatrix> create_identity_mat(int d);

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

  // std::vector<real_t> vectorProjection(std::vector<real_t> a,
  // std::vector<real_t> b);

  // std::vector<std::vector<real_t>>
  // gramSchmidtProcess(std::vector<std::vector<real_t>> A);

  /*
  struct QRDResult {
          std::vector<std::vector<real_t>> Q;
          std::vector<std::vector<real_t>> R;
  };
  */

  // QRDResult qrd(std::vector<std::vector<real_t>> A);

  /*
  struct CholeskyResult {
          std::vector<std::vector<real_t>> L;
          std::vector<std::vector<real_t>> Lt;
  };

  CholeskyResult cholesky(std::vector<std::vector<real_t>> A);
  */

  // real_t sum_elements(std::vector<std::vector<real_t>> A);

  Ref<MLPPVector> flatten() const;
  void flatteno(Ref<MLPPVector> out) const;

  Ref<MLPPVector> solve(const Ref<MLPPVector> &b) const;

  /*
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
  Ref<MLPPMatrix> outer_productn(const Ref<MLPPVector> &a,
                                 const Ref<MLPPVector> &b) const;

  // Just sets the diagonal
  void diagonal_set(const Ref<MLPPVector> &a);
  Ref<MLPPMatrix> diagonal_setn(const Ref<MLPPVector> &a) const;

  // Sets the diagonals, everythign else will get zeroed
  void diagonal_zeroed(const Ref<MLPPVector> &a);
  Ref<MLPPMatrix> diagonal_zeroedn(const Ref<MLPPVector> &a) const;

  bool
  is_equal_approx(const Ref<MLPPMatrix> &p_with,
                  real_t tolerance = static_cast<real_t>(CMP_EPSILON)) const;

  Ref<Image> get_as_image() const;
  void get_into_image(Ref<Image> out) const;
  void set_from_image(const Ref<Image> &p_img, const int p_image_channel);

  String to_string();

  MLPPMatrix();
  MLPPMatrix(const MLPPMatrix &p_from);
  MLPPMatrix(const Vector<Vector<real_t>> &p_from);
  MLPPMatrix(const Array &p_from);
  MLPPMatrix(const real_t *p_from, const int p_size_y, const int p_size_x);

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
