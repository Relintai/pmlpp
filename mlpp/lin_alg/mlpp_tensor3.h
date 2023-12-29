#ifndef MLPP_TENSOR3_H
#define MLPP_TENSOR3_H

#ifndef GDNATIVE

#include "core/math/math_defs.h"

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/math/vector2i.h"
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

#include "mlpp_matrix.h"
#include "mlpp_vector.h"

class Image;

class MLPPTensor3 : public Resource {
	GDCLASS(MLPPTensor3, Resource);

public:
	Array get_data();
	void set_data(const Array &p_from);

	_FORCE_INLINE_ real_t *ptrw() {
		return _data;
	}

	_FORCE_INLINE_ const real_t *ptr() const {
		return _data;
	}

	void z_slice_add(const Vector<real_t> &p_row);
	void z_slice_add_pool_vector(const PoolRealArray &p_row);
	void z_slice_add_mlpp_vector(const Ref<MLPPVector> &p_row);
	void z_slice_add_mlpp_matrix(const Ref<MLPPMatrix> &p_matrix);
	void z_slice_remove(int p_index);

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void z_slice_remove_unordered(int p_index);

	void z_slice_swap(int p_index_1, int p_index_2);

	_FORCE_INLINE_ void clear() { resize(Size3i()); }
	_FORCE_INLINE_ void reset() {
		if (_data) {
			memfree(_data);
			_data = NULL;
			_size = Size3i();
		}
	}

	_FORCE_INLINE_ bool empty() const { return _size == Size3i(); }
	_FORCE_INLINE_ int z_slice_data_size() const { return _size.x * _size.y; }
	_FORCE_INLINE_ Size2i z_slice_size() const { return Size2i(_size.x, _size.y); }
	_FORCE_INLINE_ int data_size() const { return _size.x * _size.y * _size.z; }
	_FORCE_INLINE_ Size3i size() const { return _size; }

	void resize(const Size3i &p_size);
	void shape_set(const Size3i &p_size);

	_FORCE_INLINE_ int calculate_index(int p_index_z, int p_index_y, int p_index_x) const {
		return p_index_y * _size.x + p_index_x + _size.x * _size.y * p_index_z;
	}

	_FORCE_INLINE_ int calculate_z_slice_index(int p_index_z) const {
		return _size.x * _size.y * p_index_z;
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

	_FORCE_INLINE_ real_t element_get(int p_index_z, int p_index_y, int p_index_x) const {
		ERR_FAIL_INDEX_V(p_index_x, _size.x, 0);
		ERR_FAIL_INDEX_V(p_index_y, _size.y, 0);
		ERR_FAIL_INDEX_V(p_index_z, _size.z, 0);

		return _data[p_index_y * _size.x + p_index_x + _size.x * _size.y * p_index_z];
	}

	_FORCE_INLINE_ void element_set(int p_index_z, int p_index_y, int p_index_x, real_t p_val) {
		ERR_FAIL_INDEX(p_index_x, _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);
		ERR_FAIL_INDEX(p_index_z, _size.z);

		_data[p_index_y * _size.x + p_index_x + _size.x * _size.y * p_index_z] = p_val;
	}

	Vector<real_t> row_get_vector(int p_index_z, int p_index_y) const;
	PoolRealArray row_get_pool_vector(int p_index_z, int p_index_y) const;
	Ref<MLPPVector> row_get_mlpp_vector(int p_index_z, int p_index_y) const;
	void row_get_into_mlpp_vector(int p_index_z, int p_index_y, Ref<MLPPVector> target) const;

	void row_set_vector(int p_index_z, int p_index_y, const Vector<real_t> &p_row);
	void row_set_pool_vector(int p_index_z, int p_index_y, const PoolRealArray &p_row);
	void row_set_mlpp_vector(int p_index_z, int p_index_y, const Ref<MLPPVector> &p_row);

	Vector<real_t> z_slice_get_vector(int p_index_z) const;
	PoolRealArray z_slice_get_pool_vector(int p_index_z) const;
	Ref<MLPPVector> z_slice_get_mlpp_vector(int p_index_z) const;
	void z_slice_get_into_mlpp_vector(int p_index_z, Ref<MLPPVector> target) const;
	Ref<MLPPMatrix> z_slice_get_mlpp_matrix(int p_index_z) const;
	void z_slice_get_into_mlpp_matrix(int p_index_z, Ref<MLPPMatrix> target) const;

	void z_slice_set_vector(int p_index_z, const Vector<real_t> &p_row);
	void z_slice_set_pool_vector(int p_index_z, const PoolRealArray &p_row);
	void z_slice_set_mlpp_vector(int p_index_z, const Ref<MLPPVector> &p_row);
	void z_slice_set_mlpp_matrix(int p_index_z, const Ref<MLPPMatrix> &p_mat);

	//TODO resize() need to be reworked for add and remove to work, in any other direction than z
	//void x_slice_add(const Ref<MLPPMatrix> &p_matrix);
	//void x_slice_remove(int p_index);
	void x_slice_get_into(int p_index_x, Ref<MLPPMatrix> target) const;
	Ref<MLPPMatrix> x_slice_get(int p_index_x) const;
	void x_slice_set(int p_index_x, const Ref<MLPPMatrix> &p_mat);

	//void y_slice_add(const Ref<MLPPMatrix> &p_matrix);
	//void y_slice_remove(int p_index);
	void y_slice_get_into(int p_index_y, Ref<MLPPMatrix> target) const;
	Ref<MLPPMatrix> y_slice_get(int p_index_y) const;
	void y_slice_set(int p_index_y, const Ref<MLPPMatrix> &p_mat);

public:
	//Image api

	enum ImageChannelFlags {
		IMAGE_CHANNEL_FLAG_R = 1 << 0,
		IMAGE_CHANNEL_FLAG_G = 1 << 1,
		IMAGE_CHANNEL_FLAG_B = 1 << 2,
		IMAGE_CHANNEL_FLAG_A = 1 << 3,

		IMAGE_CHANNEL_FLAG_NONE = 0,
		IMAGE_CHANNEL_FLAG_RG = IMAGE_CHANNEL_FLAG_R | IMAGE_CHANNEL_FLAG_G,
		IMAGE_CHANNEL_FLAG_RGB = IMAGE_CHANNEL_FLAG_R | IMAGE_CHANNEL_FLAG_G | IMAGE_CHANNEL_FLAG_B,
		IMAGE_CHANNEL_FLAG_GB = IMAGE_CHANNEL_FLAG_G | IMAGE_CHANNEL_FLAG_B,
		IMAGE_CHANNEL_FLAG_GBA = IMAGE_CHANNEL_FLAG_G | IMAGE_CHANNEL_FLAG_B | IMAGE_CHANNEL_FLAG_A,
		IMAGE_CHANNEL_FLAG_BA = IMAGE_CHANNEL_FLAG_B | IMAGE_CHANNEL_FLAG_A,
		IMAGE_CHANNEL_FLAG_RGBA = IMAGE_CHANNEL_FLAG_R | IMAGE_CHANNEL_FLAG_G | IMAGE_CHANNEL_FLAG_B | IMAGE_CHANNEL_FLAG_A,
	};

	void z_slices_add_image(const Ref<Image> &p_img, const int p_channels = IMAGE_CHANNEL_FLAG_RGBA);

	Ref<Image> z_slice_get_image(const int p_index_z) const;
	Ref<Image> z_slices_get_image(const int p_index_r = -1, const int p_index_g = -1, const int p_index_b = -1, const int p_index_a = -1) const;

	void z_slice_get_into_image(Ref<Image> p_target, const int p_index_z, const int p_target_channels = IMAGE_CHANNEL_FLAG_RGB) const;
	void z_slices_get_into_image(Ref<Image> p_target, const int p_index_r = -1, const int p_index_g = -1, const int p_index_b = -1, const int p_index_a = -1) const;

	void z_slice_set_image(const Ref<Image> &p_img, const int p_index_z, const int p_image_channel_flag = IMAGE_CHANNEL_FLAG_R);
	void z_slices_set_image(const Ref<Image> &p_img, const int p_index_r = -1, const int p_index_g = -1, const int p_index_b = -1, const int p_index_a = -1);

	void set_from_image(const Ref<Image> &p_img, const int p_channels = IMAGE_CHANNEL_FLAG_RGBA);

	//void x_slices_add_image(const Ref<Image> &p_img, const int p_channels = IMAGE_CHANNEL_FLAG_RGBA);
	Ref<Image> x_slice_get_image(const int p_index_x) const;
	void x_slice_get_into_image(Ref<Image> p_target, const int p_index_x, const int p_target_channels = IMAGE_CHANNEL_FLAG_RGB) const;
	void x_slice_set_image(const Ref<Image> &p_img, const int p_index_x, const int p_image_channel_flag = IMAGE_CHANNEL_FLAG_R);

	//void y_slices_add_image(const Ref<Image> &p_img, const int p_channels = IMAGE_CHANNEL_FLAG_RGBA);
	Ref<Image> y_slice_get_image(const int p_index_y) const;
	void y_slice_get_into_image(Ref<Image> p_target, const int p_index_y, const int p_target_channels = IMAGE_CHANNEL_FLAG_RGB) const;
	void y_slice_set_image(const Ref<Image> &p_img, const int p_index_y, const int p_image_channel_flag = IMAGE_CHANNEL_FLAG_R);

public:
	//math api

	void add(const Ref<MLPPTensor3> &B);
	Ref<MLPPTensor3> addn(const Ref<MLPPTensor3> &B) const;
	void addb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B);

	void sub(const Ref<MLPPTensor3> &B);
	Ref<MLPPTensor3> subn(const Ref<MLPPTensor3> &B) const;
	void subb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B);

	void division_element_wise(const Ref<MLPPTensor3> &B);
	Ref<MLPPTensor3> division_element_wisen(const Ref<MLPPTensor3> &B) const;
	void division_element_wiseb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B);

	void sqrt();
	Ref<MLPPTensor3> sqrtn() const;
	void sqrtb(const Ref<MLPPTensor3> &A);

	void exponentiate(real_t p);
	Ref<MLPPTensor3> exponentiaten(real_t p) const;
	void exponentiateb(const Ref<MLPPTensor3> &A, real_t p);

	void scalar_multiply(const real_t scalar);
	Ref<MLPPTensor3> scalar_multiplyn(const real_t scalar) const;
	void scalar_multiplyb(const real_t scalar, const Ref<MLPPTensor3> &A);

	void scalar_add(const real_t scalar);
	Ref<MLPPTensor3> scalar_addn(const real_t scalar) const;
	void scalar_addb(const real_t scalar, const Ref<MLPPTensor3> &A);

	void hadamard_product(const Ref<MLPPTensor3> &B);
	Ref<MLPPTensor3> hadamard_productn(const Ref<MLPPTensor3> &B) const;
	void hadamard_productb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B);

	void max(const Ref<MLPPTensor3> &B);
	Ref<MLPPTensor3> maxn(const Ref<MLPPTensor3> &B) const;
	void maxb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B);

	void abs();
	Ref<MLPPTensor3> absn() const;
	void absb(const Ref<MLPPTensor3> &A);

	Ref<MLPPVector> flatten() const;
	void flatteno(Ref<MLPPVector> out) const;

	//real_t norm_2(std::vector<std::vector<std::vector<real_t>>> A);

	Ref<MLPPMatrix> tensor_vec_mult(const Ref<MLPPVector> &b);
	//std::vector<std::vector<std::vector<real_t>>> vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B);

public:
	void fill(real_t p_val);

	Vector<real_t> to_flat_vector() const;
	PoolRealArray to_flat_pool_vector() const;
	Vector<uint8_t> to_flat_byte_array() const;

	Ref<MLPPTensor3> duplicate_fast() const;

	void set_from_mlpp_tensor3(const Ref<MLPPTensor3> &p_from);
	void set_from_mlpp_tensor3r(const MLPPTensor3 &p_from);

	void set_from_mlpp_matrix(const Ref<MLPPMatrix> &p_from);
	void set_from_mlpp_matrixr(const MLPPMatrix &p_from);
	void set_from_mlpp_vectors(const Vector<Ref<MLPPVector>> &p_from);
	void set_from_mlpp_matricess(const Vector<Ref<MLPPMatrix>> &p_from);

	void set_from_mlpp_vectors_array(const Array &p_from);
	void set_from_mlpp_matrices_array(const Array &p_from);

	bool is_equal_approx(const Ref<MLPPTensor3> &p_with, real_t tolerance = static_cast<real_t>(CMP_EPSILON)) const;

	String to_string();

	MLPPTensor3();
	MLPPTensor3(const MLPPMatrix &p_from);
	MLPPTensor3(const Array &p_from);
	~MLPPTensor3();

	// TODO: These are temporary
	std::vector<real_t> to_flat_std_vector() const;
	void set_from_std_vectors(const std::vector<std::vector<std::vector<real_t>>> &p_from);
	std::vector<std::vector<std::vector<real_t>>> to_std_vector();
	MLPPTensor3(const std::vector<std::vector<std::vector<real_t>>> &p_from);

protected:
	static void _bind_methods();

protected:
	Size3i _size;
	real_t *_data;
};

VARIANT_ENUM_CAST(MLPPTensor3::ImageChannelFlags);

#endif
