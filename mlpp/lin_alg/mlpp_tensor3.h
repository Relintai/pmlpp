#ifndef MLPP_TENSOR3_H
#define MLPP_TENSOR3_H

#include "core/math/math_defs.h"

#include "core/containers/pool_vector.h"
#include "core/containers/sort_array.h"
#include "core/containers/vector.h"
#include "core/error/error_macros.h"
#include "core/math/vector2i.h"
#include "core/os/memory.h"

#include "core/object/reference.h"

#include "mlpp_matrix.h"
#include "mlpp_vector.h"

class Image;

class MLPPTensor3 : public Reference {
	GDCLASS(MLPPTensor3, Reference);

public:
	real_t *ptrw() {
		return _data;
	}

	const real_t *ptr() const {
		return _data;
	}

	_FORCE_INLINE_ void add_feature_map(const Vector<real_t> &p_row) {
		if (p_row.size() == 0) {
			return;
		}

		int fms = feature_map_data_size();

		ERR_FAIL_COND(fms != p_row.size());

		int ci = data_size();

		++_size.z;

		_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");

		const real_t *row_arr = p_row.ptr();

		for (int i = 0; i < p_row.size(); ++i) {
			_data[ci + i] = row_arr[i];
		}
	}

	_FORCE_INLINE_ void add_feature_map_pool_vector(const PoolRealArray &p_row) {
		if (p_row.size() == 0) {
			return;
		}

		int fms = feature_map_data_size();

		ERR_FAIL_COND(fms != p_row.size());

		int ci = data_size();

		++_size.z;

		_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");

		PoolRealArray::Read rread = p_row.read();
		const real_t *row_arr = rread.ptr();

		for (int i = 0; i < p_row.size(); ++i) {
			_data[ci + i] = row_arr[i];
		}
	}

	_FORCE_INLINE_ void add_feature_map_mlpp_vector(const Ref<MLPPVector> &p_row) {
		ERR_FAIL_COND(!p_row.is_valid());

		int p_row_size = p_row->size();

		if (p_row_size == 0) {
			return;
		}

		int fms = feature_map_data_size();

		ERR_FAIL_COND(fms != p_row_size);

		int ci = data_size();

		++_size.z;

		_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");

		const real_t *row_ptr = p_row->ptr();

		for (int i = 0; i < p_row_size; ++i) {
			_data[ci + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void add_feature_map_mlpp_matrix(const Ref<MLPPMatrix> &p_matrix) {
		ERR_FAIL_COND(!p_matrix.is_valid());

		int other_data_size = p_matrix->data_size();

		if (other_data_size == 0) {
			return;
		}

		Size2i matrix_size = p_matrix->size();
		Size2i fms = feature_map_size();

		ERR_FAIL_COND(fms != matrix_size);

		int start_offset = data_size();

		++_size.z;

		_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");

		const real_t *other_ptr = p_matrix->ptr();

		for (int i = 0; i < other_data_size; ++i) {
			_data[start_offset + i] = other_ptr[i];
		}
	}

	void remove_feature_map(int p_index) {
		ERR_FAIL_INDEX(p_index, _size.z);

		--_size.z;

		int ds = data_size();

		if (ds == 0) {
			memfree(_data);
			_data = NULL;
			return;
		}

		int fmds = feature_map_data_size();

		for (int i = calculate_feature_map_index(p_index); i < ds; ++i) {
			_data[i] = _data[i + fmds];
		}

		_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	// Removes the item copying the last value into the position of the one to
	// remove. It's generally faster than `remove`.
	void remove_feature_map_unordered(int p_index) {
		ERR_FAIL_INDEX(p_index, _size.z);

		--_size.z;

		int ds = data_size();

		if (ds == 0) {
			memfree(_data);
			_data = NULL;
			return;
		}

		int start_ind = calculate_feature_map_index(p_index);
		int end_ind = calculate_feature_map_index(p_index + 1);

		for (int i = start_ind; i < end_ind; ++i) {
			_data[i] = _data[ds + i];
		}

		_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
		CRASH_COND_MSG(!_data, "Out of memory");
	}

	void swap_feature_map(int p_index_1, int p_index_2) {
		ERR_FAIL_INDEX(p_index_1, _size.z);
		ERR_FAIL_INDEX(p_index_2, _size.z);

		int ind1_start = calculate_feature_map_index(p_index_1);
		int ind2_start = calculate_feature_map_index(p_index_2);

		int fmds = feature_map_data_size();

		for (int i = 0; i < fmds; ++i) {
			SWAP(_data[ind1_start + i], _data[ind2_start + i]);
		}
	}

	_FORCE_INLINE_ void clear() { resize(Size3i()); }
	_FORCE_INLINE_ void reset() {
		if (_data) {
			memfree(_data);
			_data = NULL;
			_size = Size3i();
		}
	}

	_FORCE_INLINE_ bool empty() const { return _size == Size3i(); }
	_FORCE_INLINE_ int feature_map_data_size() const { return _size.x * _size.y; }
	_FORCE_INLINE_ Size2i feature_map_size() const { return Size2i(_size.x, _size.y); }
	_FORCE_INLINE_ int data_size() const { return _size.x * _size.y * _size.z; }
	_FORCE_INLINE_ Size3i size() const { return _size; }

	void resize(const Size3i &p_size) {
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

	void set_shape(const Size3i &p_size) {
		int ds = data_size();
		int new_data_size = p_size.x * p_size.y * p_size.z;

		ERR_FAIL_COND_MSG(ds != new_data_size, "The new size has a different volume than the old. If this is intended use resize()!");

		_size = p_size;
	}

	_FORCE_INLINE_ int calculate_index(int p_index_y, int p_index_x, int p_index_z) const {
		return p_index_y * _size.x + p_index_x + _size.x * _size.y * p_index_z;
	}

	_FORCE_INLINE_ int calculate_feature_map_index(int p_index_z) const {
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

	_FORCE_INLINE_ real_t get_element_index(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, data_size(), 0);

		return _data[p_index];
	}

	_FORCE_INLINE_ void set_element_index(int p_index, real_t p_val) {
		ERR_FAIL_INDEX(p_index, data_size());

		_data[p_index] = p_val;
	}

	_FORCE_INLINE_ real_t get_element(int p_index_y, int p_index_x, int p_index_z) const {
		ERR_FAIL_INDEX_V(p_index_x, _size.x, 0);
		ERR_FAIL_INDEX_V(p_index_y, _size.y, 0);
		ERR_FAIL_INDEX_V(p_index_z, _size.z, 0);

		return _data[p_index_y * _size.x + p_index_x + _size.x * _size.y * p_index_z];
	}

	_FORCE_INLINE_ void set_element(int p_index_y, int p_index_x, int p_index_z, real_t p_val) {
		ERR_FAIL_INDEX(p_index_x, _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);
		ERR_FAIL_INDEX(p_index_z, _size.z);

		_data[p_index_y * _size.x + p_index_x + _size.x * _size.y * p_index_z] = p_val;
	}

	_FORCE_INLINE_ Vector<real_t> get_row_vector(int p_index_y, int p_index_z) {
		ERR_FAIL_INDEX_V(p_index_y, _size.y, Vector<real_t>());
		ERR_FAIL_INDEX_V(p_index_z, _size.z, Vector<real_t>());

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

	_FORCE_INLINE_ PoolRealArray get_row_pool_vector(int p_index_y, int p_index_z) {
		ERR_FAIL_INDEX_V(p_index_y, _size.y, PoolRealArray());
		ERR_FAIL_INDEX_V(p_index_z, _size.z, PoolRealArray());

		PoolRealArray ret;

		if (unlikely(_size.x == 0)) {
			return ret;
		}

		ret.resize(_size.x);

		int ind_start = p_index_y * _size.x + _size.x * _size.y * p_index_z;

		PoolRealArray::Write w = ret.write();
		real_t *row_ptr = w.ptr();

		for (int i = 0; i < _size.x; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}

		return ret;
	}

	_FORCE_INLINE_ Ref<MLPPVector> get_row_mlpp_vector(int p_index_y, int p_index_z) {
		ERR_FAIL_INDEX_V(p_index_y, _size.y, Ref<MLPPVector>());
		ERR_FAIL_INDEX_V(p_index_z, _size.z, Ref<MLPPVector>());

		Ref<MLPPVector> ret;
		ret.instance();

		if (unlikely(_size.x == 0)) {
			return ret;
		}

		ret->resize(_size.x);

		int ind_start = p_index_y * _size.x + _size.x * _size.y * p_index_z;

		real_t *row_ptr = ret->ptrw();

		for (int i = 0; i < _size.x; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}

		return ret;
	}

	_FORCE_INLINE_ void get_row_into_mlpp_vector(int p_index_y, int p_index_z, Ref<MLPPVector> target) const {
		ERR_FAIL_COND(!target.is_valid());
		ERR_FAIL_INDEX(p_index_y, _size.y);
		ERR_FAIL_INDEX(p_index_z, _size.z);

		if (unlikely(target->size() != _size.x)) {
			target->resize(_size.x);
		}

		int ind_start = p_index_y * _size.x + _size.x * _size.y * p_index_z;

		real_t *row_ptr = target->ptrw();

		for (int i = 0; i < _size.x; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}
	}

	_FORCE_INLINE_ void set_row_vector(int p_index_y, int p_index_z, const Vector<real_t> &p_row) {
		ERR_FAIL_COND(p_row.size() != _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);
		ERR_FAIL_INDEX(p_index_z, _size.z);

		int ind_start = p_index_y * _size.x + _size.x * _size.y * p_index_z;

		const real_t *row_ptr = p_row.ptr();

		for (int i = 0; i < _size.x; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_row_pool_vector(int p_index_y, int p_index_z, const PoolRealArray &p_row) {
		ERR_FAIL_COND(p_row.size() != _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);
		ERR_FAIL_INDEX(p_index_z, _size.z);

		int ind_start = p_index_y * _size.x + _size.x * _size.y * p_index_z;

		PoolRealArray::Read r = p_row.read();
		const real_t *row_ptr = r.ptr();

		for (int i = 0; i < _size.x; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_row_mlpp_vector(int p_index_y, int p_index_z, const Ref<MLPPVector> &p_row) {
		ERR_FAIL_COND(!p_row.is_valid());
		ERR_FAIL_COND(p_row->size() != _size.x);
		ERR_FAIL_INDEX(p_index_y, _size.y);
		ERR_FAIL_INDEX(p_index_z, _size.z);

		int ind_start = p_index_y * _size.x + _size.x * _size.y * p_index_z;

		const real_t *row_ptr = p_row->ptr();

		for (int i = 0; i < _size.x; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ Vector<real_t> get_feature_map_vector(int p_index_z) {
		ERR_FAIL_INDEX_V(p_index_z, _size.z, Vector<real_t>());

		Vector<real_t> ret;

		int fmds = feature_map_data_size();

		if (unlikely(fmds == 0)) {
			return ret;
		}

		ret.resize(fmds);

		int ind_start = calculate_feature_map_index(p_index_z);

		real_t *row_ptr = ret.ptrw();

		for (int i = 0; i < fmds; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}

		return ret;
	}

	_FORCE_INLINE_ PoolRealArray get_feature_map_pool_vector(int p_index_z) {
		ERR_FAIL_INDEX_V(p_index_z, _size.z, PoolRealArray());

		PoolRealArray ret;

		int fmds = feature_map_data_size();

		if (unlikely(fmds == 0)) {
			return ret;
		}

		ret.resize(fmds);

		int ind_start = calculate_feature_map_index(p_index_z);

		PoolRealArray::Write w = ret.write();
		real_t *row_ptr = w.ptr();

		for (int i = 0; i < fmds; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}

		return ret;
	}

	_FORCE_INLINE_ Ref<MLPPVector> get_feature_map_mlpp_vector(int p_index_z) {
		ERR_FAIL_INDEX_V(p_index_z, _size.z, Ref<MLPPVector>());

		Ref<MLPPVector> ret;
		ret.instance();

		int fmds = feature_map_data_size();

		if (unlikely(fmds == 0)) {
			return ret;
		}

		ret->resize(fmds);

		int ind_start = calculate_feature_map_index(p_index_z);

		real_t *row_ptr = ret->ptrw();

		for (int i = 0; i < fmds; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}

		return ret;
	}

	_FORCE_INLINE_ void get_feature_map_into_mlpp_vector(int p_index_z, Ref<MLPPVector> target) const {
		ERR_FAIL_INDEX(p_index_z, _size.z);

		int fmds = feature_map_data_size();

		if (unlikely(target->size() != fmds)) {
			target->resize(fmds);
		}

		int ind_start = calculate_feature_map_index(p_index_z);

		real_t *row_ptr = target->ptrw();

		for (int i = 0; i < fmds; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}
	}

	_FORCE_INLINE_ Ref<MLPPMatrix> get_feature_map_mlpp_matrix(int p_index_z) {
		ERR_FAIL_INDEX_V(p_index_z, _size.z, Ref<MLPPMatrix>());

		Ref<MLPPMatrix> ret;
		ret.instance();

		int fmds = feature_map_data_size();

		if (unlikely(fmds == 0)) {
			return ret;
		}

		ret->resize(feature_map_size());

		int ind_start = calculate_feature_map_index(p_index_z);

		real_t *row_ptr = ret->ptrw();

		for (int i = 0; i < fmds; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}

		return ret;
	}

	_FORCE_INLINE_ void get_feature_map_into_mlpp_vector(int p_index_z, Ref<MLPPMatrix> target) const {
		ERR_FAIL_INDEX(p_index_z, _size.z);

		int fmds = feature_map_data_size();
		Size2i fms = feature_map_size();

		if (unlikely(target->size() != fms)) {
			target->resize(fms);
		}

		int ind_start = calculate_feature_map_index(p_index_z);

		real_t *row_ptr = target->ptrw();

		for (int i = 0; i < fmds; ++i) {
			row_ptr[i] = _data[ind_start + i];
		}
	}

	_FORCE_INLINE_ void set_feature_map_vector(int p_index_z, const Vector<real_t> &p_row) {
		ERR_FAIL_INDEX(p_index_z, _size.z);

		int fmds = feature_map_data_size();

		ERR_FAIL_COND(p_row.size() != fmds);

		int ind_start = calculate_feature_map_index(p_index_z);

		const real_t *row_ptr = p_row.ptr();

		for (int i = 0; i < fmds; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_feature_map_pool_vector(int p_index_z, const PoolRealArray &p_row) {
		ERR_FAIL_INDEX(p_index_z, _size.z);

		int fmds = feature_map_data_size();

		ERR_FAIL_COND(p_row.size() != fmds);

		int ind_start = calculate_feature_map_index(p_index_z);

		PoolRealArray::Read r = p_row.read();
		const real_t *row_ptr = r.ptr();

		for (int i = 0; i < fmds; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_feature_map_mlpp_vector(int p_index_z, const Ref<MLPPVector> &p_row) {
		ERR_FAIL_INDEX(p_index_z, _size.z);
		ERR_FAIL_COND(!p_row.is_valid());

		int fmds = feature_map_data_size();

		ERR_FAIL_COND(p_row->size() != fmds);

		int ind_start = calculate_feature_map_index(p_index_z);

		const real_t *row_ptr = p_row->ptr();

		for (int i = 0; i < fmds; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

	_FORCE_INLINE_ void set_feature_map_mlpp_matrix(int p_index_z, const Ref<MLPPMatrix> &p_mat) {
		ERR_FAIL_INDEX(p_index_z, _size.z);
		ERR_FAIL_COND(!p_mat.is_valid());

		int fmds = feature_map_data_size();

		ERR_FAIL_COND(p_mat->size() != feature_map_size());

		int ind_start = calculate_feature_map_index(p_index_z);

		const real_t *row_ptr = p_mat->ptr();

		for (int i = 0; i < fmds; ++i) {
			_data[ind_start + i] = row_ptr[i];
		}
	}

public:
	//Image api

	enum ImageChannels {
		IMAGE_CHANNEL_R = 1 << 0,
		IMAGE_CHANNEL_G = 1 << 1,
		IMAGE_CHANNEL_B = 1 << 2,
		IMAGE_CHANNEL_A = 1 << 3,

		IMAGE_CHANNEL_NONE = 0,
		IMAGE_CHANNEL_RG = IMAGE_CHANNEL_R | IMAGE_CHANNEL_G,
		IMAGE_CHANNEL_RGB = IMAGE_CHANNEL_R | IMAGE_CHANNEL_G | IMAGE_CHANNEL_B,
		IMAGE_CHANNEL_GB = IMAGE_CHANNEL_G | IMAGE_CHANNEL_B,
		IMAGE_CHANNEL_GBA = IMAGE_CHANNEL_G | IMAGE_CHANNEL_B | IMAGE_CHANNEL_A,
		IMAGE_CHANNEL_BA = IMAGE_CHANNEL_B | IMAGE_CHANNEL_A,
		IMAGE_CHANNEL_RGBA = IMAGE_CHANNEL_R | IMAGE_CHANNEL_G | IMAGE_CHANNEL_B | IMAGE_CHANNEL_A,
	};

	void add_feature_maps_image(const Ref<Image> &p_img, const int p_channels = IMAGE_CHANNEL_RGBA);

	Ref<Image> get_feature_map_image(const int p_index_z);
	Ref<Image> get_feature_maps_image(const int p_index_r = -1, const int p_index_g = -1, const int p_index_b = -1, const int p_index_a = -1);

	void get_feature_map_into_image(Ref<Image> p_target, const int p_index_z, const int p_target_channels = IMAGE_CHANNEL_RGBA) const;
	void get_feature_map_into_image(Ref<Image> p_target, const int p_index_r = -1, const int p_index_g = -1, const int p_index_b = -1, const int p_index_a = -1) const;

	void set_feature_map_image(const Ref<Image> &p_img, const int p_index_z, const int p_image_channel = IMAGE_CHANNEL_R);
	void set_feature_maps_image(const Ref<Image> &p_img, const int p_index_r = -1, const int p_index_g = -1, const int p_index_b = -1, const int p_index_a = -1);

	void set_from_image(const Ref<Image> &p_img, const int p_channels = IMAGE_CHANNEL_RGBA);

public:
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

	Ref<MLPPTensor3> duplicate() const {
		Ref<MLPPTensor3> ret;
		ret.instance();

		ret->set_from_mlpp_tensor3r(*this);

		return ret;
	}

	_FORCE_INLINE_ void set_from_mlpp_tensor3(const Ref<MLPPTensor3> &p_from) {
		ERR_FAIL_COND(!p_from.is_valid());

		resize(p_from->size());

		int ds = p_from->data_size();
		const real_t *ptr = p_from->ptr();

		for (int i = 0; i < ds; ++i) {
			_data[i] = ptr[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_tensor3r(const MLPPTensor3 &p_from) {
		resize(p_from.size());

		int ds = p_from.data_size();
		const real_t *ptr = p_from.ptr();

		for (int i = 0; i < ds; ++i) {
			_data[i] = ptr[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_matrix(const Ref<MLPPMatrix> &p_from) {
		ERR_FAIL_COND(!p_from.is_valid());

		Size2i mat_size = p_from->size();
		resize(Size3i(mat_size.x, mat_size.y, 1));

		int ds = p_from->data_size();
		const real_t *ptr = p_from->ptr();

		for (int i = 0; i < ds; ++i) {
			_data[i] = ptr[i];
		}
	}

	_FORCE_INLINE_ void set_from_mlpp_matrixr(const MLPPMatrix &p_from) {
		Size2i mat_size = p_from.size();
		resize(Size3i(mat_size.x, mat_size.y, 1));

		int ds = p_from.data_size();
		const real_t *ptr = p_from.ptr();

		for (int i = 0; i < ds; ++i) {
			_data[i] = ptr[i];
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

		resize(Size3i(p_from[0]->size(), p_from.size(), 1));

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

	_FORCE_INLINE_ void set_from_mlpp_matricess(const Vector<Ref<MLPPMatrix>> &p_from) {
		if (p_from.size() == 0) {
			reset();
			return;
		}

		if (!p_from[0].is_valid()) {
			reset();
			return;
		}

		resize(Size3i(p_from[0]->size().x, p_from[0]->size().y, p_from.size()));

		if (data_size() == 0) {
			reset();
			return;
		}

		Size2i fms = feature_map_size();
		int fmds = feature_map_data_size();

		for (int i = 0; i < p_from.size(); ++i) {
			const Ref<MLPPMatrix> &r = p_from[i];

			ERR_CONTINUE(!r.is_valid());
			ERR_CONTINUE(r->size() != fms);

			int start_index = calculate_feature_map_index(i);

			const real_t *from_ptr = r->ptr();
			for (int j = 0; j < fmds; j++) {
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

		resize(Size3i(v0->size(), p_from.size(), 1));

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

	_FORCE_INLINE_ void set_from_mlpp_matrices_array(const Array &p_from) {
		if (p_from.size() == 0) {
			reset();
			return;
		}

		Ref<MLPPMatrix> v0 = p_from[0];

		if (!v0.is_valid()) {
			reset();
			return;
		}

		resize(Size3i(v0->size().x, v0->size().y, p_from.size()));

		if (data_size() == 0) {
			reset();
			return;
		}

		Size2i fms = feature_map_size();
		int fmds = feature_map_data_size();

		for (int i = 0; i < p_from.size(); ++i) {
			Ref<MLPPMatrix> r = p_from[i];

			ERR_CONTINUE(!r.is_valid());
			ERR_CONTINUE(r->size() != fms);

			int start_index = calculate_feature_map_index(i);

			const real_t *from_ptr = r->ptr();
			for (int j = 0; j < fmds; j++) {
				_data[start_index + j] = from_ptr[j];
			}
		}
	}

	_FORCE_INLINE_ bool is_equal_approx(const Ref<MLPPTensor3> &p_with, real_t tolerance = static_cast<real_t>(CMP_EPSILON)) const {
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

	_FORCE_INLINE_ MLPPTensor3() {
		_data = NULL;
	}

	_FORCE_INLINE_ MLPPTensor3(const MLPPMatrix &p_from) {
		_data = NULL;

		Size2i mat_size = p_from.size();
		resize(Size3i(mat_size.x, mat_size.y, 1));

		int ds = p_from.data_size();
		const real_t *ptr = p_from.ptr();

		for (int i = 0; i < ds; ++i) {
			_data[i] = ptr[i];
		}
	}

	MLPPTensor3(const Array &p_from) {
		_data = NULL;

		set_from_mlpp_matrices_array(p_from);
	}

	_FORCE_INLINE_ ~MLPPTensor3() {
		if (_data) {
			reset();
		}
	}

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

VARIANT_ENUM_CAST(MLPPTensor3::ImageChannels);

#endif
