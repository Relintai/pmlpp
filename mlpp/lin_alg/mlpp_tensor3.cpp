
#include "mlpp_tensor3.h"

#include "core/io/image.h"

Array MLPPTensor3::get_data() {
	PoolRealArray pl;

	int ds = data_size();

	if (ds) {
		pl.resize(ds);
		PoolRealArray::Write w = pl.write();
		real_t *dest = w.ptr();

		for (int i = 0; i < ds; ++i) {
			dest[i] = _data[i];
		}
	}

	Array arr;
	arr.push_back(size());
	arr.push_back(pl);

	return arr;
}
void MLPPTensor3::set_data(const Array &p_from) {
	if (p_from.size() != 2) {
		return;
	}

	Size3i s = p_from[0];
	PoolRealArray pl = p_from[1];

	int ds = s.x * s.y * s.z;

	if (ds != pl.size()) {
		return;
	}

	if (_size != s) {
		resize(s);
	}

	PoolRealArray::Read r = pl.read();
	for (int i = 0; i < ds; ++i) {
		_data[i] = r[i];
	}
}

void MLPPTensor3::add_z_slice(const Vector<real_t> &p_row) {
	if (p_row.size() == 0) {
		return;
	}

	int fms = z_slice_data_size();

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

void MLPPTensor3::add_z_slice_pool_vector(const PoolRealArray &p_row) {
	if (p_row.size() == 0) {
		return;
	}

	int fms = z_slice_data_size();

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

void MLPPTensor3::add_z_slice_mlpp_vector(const Ref<MLPPVector> &p_row) {
	ERR_FAIL_COND(!p_row.is_valid());

	int p_row_size = p_row->size();

	if (p_row_size == 0) {
		return;
	}

	int fms = z_slice_data_size();

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

void MLPPTensor3::add_z_slice_mlpp_matrix(const Ref<MLPPMatrix> &p_matrix) {
	ERR_FAIL_COND(!p_matrix.is_valid());

	int other_data_size = p_matrix->data_size();

	if (other_data_size == 0) {
		return;
	}

	Size2i matrix_size = p_matrix->size();
	Size2i fms = z_slice_size();

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

void MLPPTensor3::remove_z_slice(int p_index) {
	ERR_FAIL_INDEX(p_index, _size.z);

	--_size.z;

	int ds = data_size();

	if (ds == 0) {
		memfree(_data);
		_data = NULL;
		return;
	}

	int fmds = z_slice_data_size();

	for (int i = calculate_z_slice_index(p_index); i < ds; ++i) {
		_data[i] = _data[i + fmds];
	}

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

// Removes the item copying the last value into the position of the one to
// remove. It's generally faster than `remove`.
void MLPPTensor3::remove_z_slice_unordered(int p_index) {
	ERR_FAIL_INDEX(p_index, _size.z);

	--_size.z;

	int ds = data_size();

	if (ds == 0) {
		memfree(_data);
		_data = NULL;
		return;
	}

	int start_ind = calculate_z_slice_index(p_index);
	int end_ind = calculate_z_slice_index(p_index + 1);

	for (int i = start_ind; i < end_ind; ++i) {
		_data[i] = _data[ds + i];
	}

	_data = (real_t *)memrealloc(_data, data_size() * sizeof(real_t));
	CRASH_COND_MSG(!_data, "Out of memory");
}

void MLPPTensor3::swap_z_slice(int p_index_1, int p_index_2) {
	ERR_FAIL_INDEX(p_index_1, _size.z);
	ERR_FAIL_INDEX(p_index_2, _size.z);

	int ind1_start = calculate_z_slice_index(p_index_1);
	int ind2_start = calculate_z_slice_index(p_index_2);

	int fmds = z_slice_data_size();

	for (int i = 0; i < fmds; ++i) {
		SWAP(_data[ind1_start + i], _data[ind2_start + i]);
	}
}

void MLPPTensor3::resize(const Size3i &p_size) {
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

void MLPPTensor3::set_shape(const Size3i &p_size) {
	int ds = data_size();
	int new_data_size = p_size.x * p_size.y * p_size.z;

	ERR_FAIL_COND_MSG(ds != new_data_size, "The new size has a different volume than the old. If this is intended use resize()!");

	_size = p_size;
}

Vector<real_t> MLPPTensor3::get_row_vector(int p_index_y, int p_index_z) const {
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

PoolRealArray MLPPTensor3::get_row_pool_vector(int p_index_y, int p_index_z) const {
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

Ref<MLPPVector> MLPPTensor3::get_row_mlpp_vector(int p_index_y, int p_index_z) const {
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

void MLPPTensor3::get_row_into_mlpp_vector(int p_index_y, int p_index_z, Ref<MLPPVector> target) const {
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

void MLPPTensor3::set_row_vector(int p_index_y, int p_index_z, const Vector<real_t> &p_row) {
	ERR_FAIL_COND(p_row.size() != _size.x);
	ERR_FAIL_INDEX(p_index_y, _size.y);
	ERR_FAIL_INDEX(p_index_z, _size.z);

	int ind_start = p_index_y * _size.x + _size.x * _size.y * p_index_z;

	const real_t *row_ptr = p_row.ptr();

	for (int i = 0; i < _size.x; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPTensor3::set_row_pool_vector(int p_index_y, int p_index_z, const PoolRealArray &p_row) {
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

void MLPPTensor3::set_row_mlpp_vector(int p_index_y, int p_index_z, const Ref<MLPPVector> &p_row) {
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

Vector<real_t> MLPPTensor3::get_z_slice_vector(int p_index_z) const {
	ERR_FAIL_INDEX_V(p_index_z, _size.z, Vector<real_t>());

	Vector<real_t> ret;

	int fmds = z_slice_data_size();

	if (unlikely(fmds == 0)) {
		return ret;
	}

	ret.resize(fmds);

	int ind_start = calculate_z_slice_index(p_index_z);

	real_t *row_ptr = ret.ptrw();

	for (int i = 0; i < fmds; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}

	return ret;
}

PoolRealArray MLPPTensor3::get_z_slice_pool_vector(int p_index_z) const {
	ERR_FAIL_INDEX_V(p_index_z, _size.z, PoolRealArray());

	PoolRealArray ret;

	int fmds = z_slice_data_size();

	if (unlikely(fmds == 0)) {
		return ret;
	}

	ret.resize(fmds);

	int ind_start = calculate_z_slice_index(p_index_z);

	PoolRealArray::Write w = ret.write();
	real_t *row_ptr = w.ptr();

	for (int i = 0; i < fmds; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}

	return ret;
}

Ref<MLPPVector> MLPPTensor3::get_z_slice_mlpp_vector(int p_index_z) const {
	ERR_FAIL_INDEX_V(p_index_z, _size.z, Ref<MLPPVector>());

	Ref<MLPPVector> ret;
	ret.instance();

	int fmds = z_slice_data_size();

	if (unlikely(fmds == 0)) {
		return ret;
	}

	ret->resize(fmds);

	int ind_start = calculate_z_slice_index(p_index_z);

	real_t *row_ptr = ret->ptrw();

	for (int i = 0; i < fmds; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}

	return ret;
}

void MLPPTensor3::get_z_slice_into_mlpp_vector(int p_index_z, Ref<MLPPVector> target) const {
	ERR_FAIL_INDEX(p_index_z, _size.z);

	int fmds = z_slice_data_size();

	if (unlikely(target->size() != fmds)) {
		target->resize(fmds);
	}

	int ind_start = calculate_z_slice_index(p_index_z);

	real_t *row_ptr = target->ptrw();

	for (int i = 0; i < fmds; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}
}

Ref<MLPPMatrix> MLPPTensor3::get_z_slice_mlpp_matrix(int p_index_z) const {
	ERR_FAIL_INDEX_V(p_index_z, _size.z, Ref<MLPPMatrix>());

	Ref<MLPPMatrix> ret;
	ret.instance();

	int fmds = z_slice_data_size();

	if (unlikely(fmds == 0)) {
		return ret;
	}

	ret->resize(z_slice_size());

	int ind_start = calculate_z_slice_index(p_index_z);

	real_t *row_ptr = ret->ptrw();

	for (int i = 0; i < fmds; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}

	return ret;
}

void MLPPTensor3::get_z_slice_into_mlpp_matrix(int p_index_z, Ref<MLPPMatrix> target) const {
	ERR_FAIL_INDEX(p_index_z, _size.z);

	int fmds = z_slice_data_size();
	Size2i fms = z_slice_size();

	if (unlikely(target->size() != fms)) {
		target->resize(fms);
	}

	int ind_start = calculate_z_slice_index(p_index_z);

	real_t *row_ptr = target->ptrw();

	for (int i = 0; i < fmds; ++i) {
		row_ptr[i] = _data[ind_start + i];
	}
}

void MLPPTensor3::set_z_slice_vector(int p_index_z, const Vector<real_t> &p_row) {
	ERR_FAIL_INDEX(p_index_z, _size.z);

	int fmds = z_slice_data_size();

	ERR_FAIL_COND(p_row.size() != fmds);

	int ind_start = calculate_z_slice_index(p_index_z);

	const real_t *row_ptr = p_row.ptr();

	for (int i = 0; i < fmds; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPTensor3::set_z_slice_pool_vector(int p_index_z, const PoolRealArray &p_row) {
	ERR_FAIL_INDEX(p_index_z, _size.z);

	int fmds = z_slice_data_size();

	ERR_FAIL_COND(p_row.size() != fmds);

	int ind_start = calculate_z_slice_index(p_index_z);

	PoolRealArray::Read r = p_row.read();
	const real_t *row_ptr = r.ptr();

	for (int i = 0; i < fmds; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPTensor3::set_z_slice_mlpp_vector(int p_index_z, const Ref<MLPPVector> &p_row) {
	ERR_FAIL_INDEX(p_index_z, _size.z);
	ERR_FAIL_COND(!p_row.is_valid());

	int fmds = z_slice_data_size();

	ERR_FAIL_COND(p_row->size() != fmds);

	int ind_start = calculate_z_slice_index(p_index_z);

	const real_t *row_ptr = p_row->ptr();

	for (int i = 0; i < fmds; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPTensor3::set_z_slice_mlpp_matrix(int p_index_z, const Ref<MLPPMatrix> &p_mat) {
	ERR_FAIL_INDEX(p_index_z, _size.z);
	ERR_FAIL_COND(!p_mat.is_valid());

	int fmds = z_slice_data_size();

	ERR_FAIL_COND(p_mat->size() != z_slice_size());

	int ind_start = calculate_z_slice_index(p_index_z);

	const real_t *row_ptr = p_mat->ptr();

	for (int i = 0; i < fmds; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

void MLPPTensor3::get_x_slice_into(int p_index_x, Ref<MLPPMatrix> target) const {
	ERR_FAIL_INDEX(p_index_x, _size.x);
	ERR_FAIL_COND(!target.is_valid());

	if (unlikely(target->size() != Size2i(_size.y, _size.z))) {
		target->resize(Size2i(_size.y, _size.z));
	}

	for (int z = 0; z < _size.z; ++z) {
		for (int y = 0; y < _size.y; ++y) {
			target->set_element(z, y, get_element(p_index_x, y, z));
		}
	}
}
Ref<MLPPMatrix> MLPPTensor3::get_x_slice(int p_index_x) const {
	ERR_FAIL_INDEX_V(p_index_x, _size.x, Ref<MLPPMatrix>());

	Ref<MLPPMatrix> m;
	m.instance();

	get_x_slice_into(p_index_x, m);

	return m;
}
void MLPPTensor3::set_x_slice(int p_index_x, const Ref<MLPPMatrix> &p_mat) {
	ERR_FAIL_INDEX(p_index_x, _size.x);
	ERR_FAIL_COND(!p_mat.is_valid());
	ERR_FAIL_COND(p_mat->size() != Size2i(_size.y, _size.z));

	for (int z = 0; z < _size.z; ++z) {
		for (int y = 0; y < _size.y; ++y) {
			set_element(p_index_x, y, z, p_mat->get_element(z, y));
		}
	}
}

void MLPPTensor3::get_y_slice_into(int p_index_y, Ref<MLPPMatrix> target) const {
	ERR_FAIL_INDEX(p_index_y, _size.y);
	ERR_FAIL_COND(!target.is_valid());

	if (unlikely(target->size() != Size2i(_size.y, _size.z))) {
		target->resize(Size2i(_size.x, _size.z));
	}

	for (int z = 0; z < _size.z; ++z) {
		for (int x = 0; x < _size.x; ++x) {
			target->set_element(z, x, get_element(x, p_index_y, z));
		}
	}
}
Ref<MLPPMatrix> MLPPTensor3::get_y_slice(int p_index_y) const {
	ERR_FAIL_INDEX_V(p_index_y, _size.y, Ref<MLPPMatrix>());

	Ref<MLPPMatrix> m;
	m.instance();

	get_y_slice_into(p_index_y, m);

	return m;
}
void MLPPTensor3::set_y_slice(int p_index_y, const Ref<MLPPMatrix> &p_mat) {
	ERR_FAIL_INDEX(p_index_y, _size.y);
	ERR_FAIL_COND(!p_mat.is_valid());
	ERR_FAIL_COND(p_mat->size() != Size2i(_size.y, _size.z));

	for (int z = 0; z < _size.z; ++z) {
		for (int x = 0; x < _size.x; ++x) {
			set_element(x, p_index_y, z, p_mat->get_element(z, x));
		}
	}
}

void MLPPTensor3::add_z_slices_image(const Ref<Image> &p_img, const int p_channels) {
	ERR_FAIL_COND(!p_img.is_valid());

	Size2i img_size = Size2i(p_img->get_width(), p_img->get_height());

	int channel_count = 0;
	int channels[4];

	if (p_channels & IMAGE_CHANNEL_FLAG_R) {
		channels[channel_count] = 0;
		++channel_count;
	}

	if (p_channels & IMAGE_CHANNEL_FLAG_G) {
		channels[channel_count] = 1;
		++channel_count;
	}

	if (p_channels & IMAGE_CHANNEL_FLAG_B) {
		channels[channel_count] = 2;
		++channel_count;
	}

	if (p_channels & IMAGE_CHANNEL_FLAG_A) {
		channels[channel_count] = 3;
		++channel_count;
	}

	ERR_FAIL_COND(channel_count == 0);

	if (unlikely(_size == Size3i())) {
		resize(Size3i(img_size.x, img_size.y, channel_count));
	}

	Size2i fms = z_slice_size();

	ERR_FAIL_COND(img_size != fms);

	int start_channel = _size.y;

	_size.y += channel_count;

	resize(_size);

	Ref<Image> img = p_img;

	img->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int x = 0; x < fms.x; ++x) {
			Color c = img->get_pixel(x, y);

			for (int i = 0; i < channel_count; ++i) {
				set_element(y, x, start_channel + i, c[channels[i]]);
			}
		}
	}

	img->unlock();
}

Ref<Image> MLPPTensor3::get_z_slice_image(const int p_index_z) const {
	ERR_FAIL_INDEX_V(p_index_z, _size.z, Ref<Image>());

	Ref<Image> image;
	image.instance();

	if (data_size() == 0) {
		return image;
	}

	PoolByteArray arr;

	int fmsi = calculate_z_slice_index(p_index_z);
	int fms = z_slice_data_size();

	arr.resize(fms);

	PoolByteArray::Write w = arr.write();
	uint8_t *wptr = w.ptr();

	for (int i = 0; i < fms; ++i) {
		wptr[i] = static_cast<uint8_t>(_data[fmsi + i] * 255.0);
	}

	image->create(_size.x, _size.y, false, Image::FORMAT_L8, arr);

	return image;
}
Ref<Image> MLPPTensor3::get_z_slices_image(const int p_index_r, const int p_index_g, const int p_index_b, const int p_index_a) const {
	if (p_index_r != -1) {
		ERR_FAIL_INDEX_V(p_index_r, _size.z, Ref<Image>());
	}

	if (p_index_g != -1) {
		ERR_FAIL_INDEX_V(p_index_g, _size.z, Ref<Image>());
	}

	if (p_index_b != -1) {
		ERR_FAIL_INDEX_V(p_index_b, _size.z, Ref<Image>());
	}

	if (p_index_a != -1) {
		ERR_FAIL_INDEX_V(p_index_a, _size.z, Ref<Image>());
	}

	Ref<Image> image;
	image.instance();

	if (data_size() == 0) {
		return image;
	}

	Size2i fms = z_slice_size();

	image->create(_size.x, _size.y, false, Image::FORMAT_RGBA8);

	image->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int x = 0; x < fms.x; ++x) {
			Color c;

			if (p_index_r != -1) {
				c.r = get_element(y, x, p_index_r);
			}

			if (p_index_g != -1) {
				c.g = get_element(y, x, p_index_g);
			}

			if (p_index_b != -1) {
				c.b = get_element(y, x, p_index_b);
			}

			if (p_index_a != -1) {
				c.a = get_element(y, x, p_index_a);
			}

			image->set_pixel(x, y, c);
		}
	}

	image->unlock();

	return image;
}

void MLPPTensor3::get_z_slice_into_image(Ref<Image> p_target, const int p_index_z, const int p_target_channels) const {
	ERR_FAIL_INDEX(p_index_z, _size.z);
	ERR_FAIL_COND(!p_target.is_valid());

	int channel_count = 0;
	int channels[4];

	if (p_target_channels & IMAGE_CHANNEL_FLAG_R) {
		channels[channel_count] = 0;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_G) {
		channels[channel_count] = 1;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_B) {
		channels[channel_count] = 2;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_A) {
		channels[channel_count] = 3;
		++channel_count;
	}

	ERR_FAIL_COND(channel_count == 0);

	if (data_size() == 0) {
		p_target->clear();
		return;
	}

	Size2i img_size = Size2i(p_target->get_width(), p_target->get_height());
	Size2i fms = z_slice_size();
	if (img_size != fms) {
		bool mip_maps = p_target->has_mipmaps();
		p_target->resize(fms.x, fms.y, Image::INTERPOLATE_NEAREST);

		if (p_target->has_mipmaps() != mip_maps) {
			if (mip_maps) {
				p_target->generate_mipmaps();
			} else {
				p_target->clear_mipmaps();
			}
		}
	}

	p_target->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int x = 0; x < fms.x; ++x) {
			Color c;

			float e = get_element(y, x, p_index_z);

			for (int i = 0; i < channel_count; ++i) {
				c[channels[i]] = e;
			}

			p_target->set_pixel(x, y, c);
		}
	}

	p_target->unlock();
}
void MLPPTensor3::get_z_slices_into_image(Ref<Image> p_target, const int p_index_r, const int p_index_g, const int p_index_b, const int p_index_a) const {
	ERR_FAIL_COND(!p_target.is_valid());

	if (p_index_r != -1) {
		ERR_FAIL_INDEX(p_index_r, _size.z);
	}

	if (p_index_g != -1) {
		ERR_FAIL_INDEX(p_index_g, _size.z);
	}

	if (p_index_b != -1) {
		ERR_FAIL_INDEX(p_index_b, _size.z);
	}

	if (p_index_a != -1) {
		ERR_FAIL_INDEX(p_index_a, _size.z);
	}

	if (data_size() == 0) {
		p_target->clear();
		return;
	}

	Size2i img_size = Size2i(p_target->get_width(), p_target->get_height());
	Size2i fms = z_slice_size();
	if (img_size != fms) {
		bool mip_maps = p_target->has_mipmaps();
		p_target->resize(fms.x, fms.y, Image::INTERPOLATE_NEAREST);

		if (p_target->has_mipmaps() != mip_maps) {
			if (mip_maps) {
				p_target->generate_mipmaps();
			} else {
				p_target->clear_mipmaps();
			}
		}
	}

	p_target->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int x = 0; x < fms.x; ++x) {
			Color c;

			if (p_index_r != -1) {
				c.r = get_element(y, x, p_index_r);
			}

			if (p_index_g != -1) {
				c.g = get_element(y, x, p_index_g);
			}

			if (p_index_b != -1) {
				c.b = get_element(y, x, p_index_b);
			}

			if (p_index_a != -1) {
				c.a = get_element(y, x, p_index_a);
			}

			p_target->set_pixel(x, y, c);
		}
	}

	p_target->unlock();
}

void MLPPTensor3::set_z_slice_image(const Ref<Image> &p_img, const int p_index_z, const int p_image_channel_flag) {
	ERR_FAIL_COND(!p_img.is_valid());
	ERR_FAIL_INDEX(p_index_z, _size.z);

	int channel_index = -1;

	for (int i = 0; i < 4; ++i) {
		if (((p_image_channel_flag & (1 << i)) != 0)) {
			channel_index = i;
			break;
		}
	}

	ERR_FAIL_INDEX(channel_index, 4);

	Size2i img_size = Size2i(p_img->get_width(), p_img->get_height());
	Size2i fms = z_slice_size();

	ERR_FAIL_COND(img_size != fms);

	Ref<Image> img = p_img;

	img->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int x = 0; x < fms.x; ++x) {
			Color c = img->get_pixel(x, y);

			set_element(y, x, p_index_z, c[channel_index]);
		}
	}

	img->unlock();
}
void MLPPTensor3::set_z_slices_image(const Ref<Image> &p_img, const int p_index_r, const int p_index_g, const int p_index_b, const int p_index_a) {
	ERR_FAIL_COND(!p_img.is_valid());

	if (p_index_r != -1) {
		ERR_FAIL_INDEX(p_index_r, _size.z);
	}

	if (p_index_g != -1) {
		ERR_FAIL_INDEX(p_index_g, _size.z);
	}

	if (p_index_b != -1) {
		ERR_FAIL_INDEX(p_index_b, _size.z);
	}

	if (p_index_a != -1) {
		ERR_FAIL_INDEX(p_index_a, _size.z);
	}

	Size2i img_size = Size2i(p_img->get_width(), p_img->get_height());
	Size2i fms = z_slice_size();

	ERR_FAIL_COND(img_size != fms);

	Ref<Image> img = p_img;

	img->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int x = 0; x < fms.x; ++x) {
			Color c = img->get_pixel(x, y);

			if (p_index_r != -1) {
				set_element(y, x, p_index_r, c.r);
			}

			if (p_index_g != -1) {
				set_element(y, x, p_index_g, c.g);
			}

			if (p_index_b != -1) {
				set_element(y, x, p_index_b, c.b);
			}

			if (p_index_a != -1) {
				set_element(y, x, p_index_a, c.a);
			}
		}
	}

	img->unlock();
}

void MLPPTensor3::set_from_image(const Ref<Image> &p_img, const int p_channels) {
	ERR_FAIL_COND(!p_img.is_valid());

	int channel_count = 0;
	int channels[4];

	if (p_channels & IMAGE_CHANNEL_FLAG_R) {
		channels[channel_count] = 0;
		++channel_count;
	}

	if (p_channels & IMAGE_CHANNEL_FLAG_G) {
		channels[channel_count] = 1;
		++channel_count;
	}

	if (p_channels & IMAGE_CHANNEL_FLAG_B) {
		channels[channel_count] = 2;
		++channel_count;
	}

	if (p_channels & IMAGE_CHANNEL_FLAG_A) {
		channels[channel_count] = 3;
		++channel_count;
	}

	ERR_FAIL_COND(channel_count == 0);

	Size2i img_size = Size2i(p_img->get_width(), p_img->get_height());

	resize(Size3i(img_size.x, img_size.y, channel_count));

	Size2i fms = z_slice_size();

	Ref<Image> img = p_img;

	img->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int x = 0; x < fms.x; ++x) {
			Color c = img->get_pixel(x, y);

			for (int i = 0; i < channel_count; ++i) {
				set_element(y, x, i, c[channels[i]]);
			}
		}
	}

	img->unlock();
}

Ref<Image> MLPPTensor3::get_x_slice_image(const int p_index_x) const {
	ERR_FAIL_INDEX_V(p_index_x, _size.x, Ref<Image>());

	Ref<Image> image;
	image.instance();

	if (data_size() == 0) {
		return image;
	}

	PoolByteArray arr;
	arr.resize(_size.y * _size.z);

	PoolByteArray::Write w = arr.write();
	uint8_t *wptr = w.ptr();
	int i = 0;

	for (int z = 0; z < _size.z; ++z) {
		for (int y = 0; y < _size.y; ++y) {
			wptr[i] = static_cast<uint8_t>(get_element(p_index_x, y, z) * 255.0);

			++i;
		}
	}

	image->create(_size.y, _size.z, false, Image::FORMAT_L8, arr);

	return image;
}
void MLPPTensor3::get_x_slice_into_image(Ref<Image> p_target, const int p_index_x, const int p_target_channels) const {
	ERR_FAIL_INDEX(p_index_x, _size.x);
	ERR_FAIL_COND(!p_target.is_valid());

	int channel_count = 0;
	int channels[4];

	if (p_target_channels & IMAGE_CHANNEL_FLAG_R) {
		channels[channel_count] = 0;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_G) {
		channels[channel_count] = 1;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_B) {
		channels[channel_count] = 2;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_A) {
		channels[channel_count] = 3;
		++channel_count;
	}

	ERR_FAIL_COND(channel_count == 0);

	if (data_size() == 0) {
		p_target->clear();
		return;
	}

	Size2i img_size = Size2i(p_target->get_width(), p_target->get_height());
	Size2i fms = Size2i(_size.y, _size.z);
	if (img_size != fms) {
		bool mip_maps = p_target->has_mipmaps();
		p_target->resize(fms.x, fms.y, Image::INTERPOLATE_NEAREST);

		if (p_target->has_mipmaps() != mip_maps) {
			if (mip_maps) {
				p_target->generate_mipmaps();
			} else {
				p_target->clear_mipmaps();
			}
		}
	}

	p_target->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int z = 0; z < fms.x; ++z) {
			Color c;

			float e = get_element(y, p_index_x, z);

			for (int i = 0; i < channel_count; ++i) {
				c[channels[i]] = e;
			}

			p_target->set_pixel(z, y, c);
		}
	}

	p_target->unlock();
}
void MLPPTensor3::set_x_slice_image(const Ref<Image> &p_img, const int p_index_x, const int p_image_channel_flag) {
	ERR_FAIL_COND(!p_img.is_valid());
	ERR_FAIL_INDEX(p_index_x, _size.x);

	int channel_index = -1;

	for (int i = 0; i < 4; ++i) {
		if (((p_image_channel_flag & (1 << i)) != 0)) {
			channel_index = i;
			break;
		}
	}

	ERR_FAIL_INDEX(channel_index, 4);

	Size2i img_size = Size2i(p_img->get_width(), p_img->get_height());
	Size2i fms = Size2i(_size.y, _size.z);

	ERR_FAIL_COND(img_size != fms);

	Ref<Image> img = p_img;

	img->lock();

	for (int y = 0; y < fms.y; ++y) {
		for (int z = 0; z < fms.x; ++z) {
			Color c = img->get_pixel(z, y);

			set_element(y, p_index_x, z, c[channel_index]);
		}
	}

	img->unlock();
}

Ref<Image> MLPPTensor3::get_y_slice_image(const int p_index_y) const {
	ERR_FAIL_INDEX_V(p_index_y, _size.y, Ref<Image>());

	Ref<Image> image;
	image.instance();

	if (data_size() == 0) {
		return image;
	}

	PoolByteArray arr;
	arr.resize(_size.x * _size.z);

	PoolByteArray::Write w = arr.write();
	uint8_t *wptr = w.ptr();
	int i = 0;

	for (int z = 0; z < _size.z; ++z) {
		for (int x = 0; x < _size.x; ++x) {
			wptr[i] = static_cast<uint8_t>(get_element(x, p_index_y, z) * 255.0);

			++i;
		}
	}

	image->create(_size.x, _size.z, false, Image::FORMAT_L8, arr);

	return image;
}
void MLPPTensor3::get_y_slice_into_image(Ref<Image> p_target, const int p_index_y, const int p_target_channels) const {
	ERR_FAIL_INDEX(p_index_y, _size.y);
	ERR_FAIL_COND(!p_target.is_valid());

	int channel_count = 0;
	int channels[4];

	if (p_target_channels & IMAGE_CHANNEL_FLAG_R) {
		channels[channel_count] = 0;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_G) {
		channels[channel_count] = 1;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_B) {
		channels[channel_count] = 2;
		++channel_count;
	}

	if (p_target_channels & IMAGE_CHANNEL_FLAG_A) {
		channels[channel_count] = 3;
		++channel_count;
	}

	ERR_FAIL_COND(channel_count == 0);

	if (data_size() == 0) {
		p_target->clear();
		return;
	}

	Size2i img_size = Size2i(p_target->get_width(), p_target->get_height());
	Size2i fms = Size2i(_size.x, _size.z);
	if (img_size != fms) {
		bool mip_maps = p_target->has_mipmaps();
		p_target->resize(fms.x, fms.y, Image::INTERPOLATE_NEAREST);

		if (p_target->has_mipmaps() != mip_maps) {
			if (mip_maps) {
				p_target->generate_mipmaps();
			} else {
				p_target->clear_mipmaps();
			}
		}
	}

	p_target->lock();

	for (int x = 0; x < fms.y; ++x) {
		for (int z = 0; z < fms.x; ++z) {
			Color c;

			float e = get_element(p_index_y, x, z);

			for (int i = 0; i < channel_count; ++i) {
				c[channels[i]] = e;
			}

			p_target->set_pixel(z, x, c);
		}
	}

	p_target->unlock();
}
void MLPPTensor3::set_y_slice_image(const Ref<Image> &p_img, const int p_index_y, const int p_image_channel_flag) {
	ERR_FAIL_COND(!p_img.is_valid());
	ERR_FAIL_INDEX(p_index_y, _size.y);

	int channel_index = -1;

	for (int i = 0; i < 4; ++i) {
		if (((p_image_channel_flag & (1 << i)) != 0)) {
			channel_index = i;
			break;
		}
	}

	ERR_FAIL_INDEX(channel_index, 4);

	Size2i img_size = Size2i(p_img->get_width(), p_img->get_height());
	Size2i fms = Size2i(_size.x, _size.z);

	ERR_FAIL_COND(img_size != fms);

	Ref<Image> img = p_img;

	img->lock();

	for (int z = 0; z < fms.y; ++z) {
		for (int x = 0; x < fms.x; ++x) {
			Color c = img->get_pixel(x, z);

			set_element(p_index_y, x, z, c[channel_index]);
		}
	}

	img->unlock();
}

void MLPPTensor3::add(const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] += b_ptr[i];
	}
}
Ref<MLPPTensor3> MLPPTensor3::addn(const Ref<MLPPTensor3> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPTensor3>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPTensor3>());

	Ref<MLPPTensor3> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = a_ptr[i] + b_ptr[i];
	}

	return C;
}
void MLPPTensor3::addb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size3i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (_size != a_size) {
		resize(a_size);
	}

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = a_ptr[i] + b_ptr[i];
	}
}

void MLPPTensor3::sub(const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] -= b_ptr[i];
	}
}
Ref<MLPPTensor3> MLPPTensor3::subn(const Ref<MLPPTensor3> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPTensor3>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPTensor3>());

	Ref<MLPPTensor3> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = a_ptr[i] - b_ptr[i];
	}

	return C;
}
void MLPPTensor3::subb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size3i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (_size != a_size) {
		resize(a_size);
	}

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = a_ptr[i] - b_ptr[i];
	}
}

void MLPPTensor3::element_wise_division(const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	int ds = data_size();

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] /= b_ptr[i];
	}
}
Ref<MLPPTensor3> MLPPTensor3::element_wise_divisionn(const Ref<MLPPTensor3> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPTensor3>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPTensor3>());

	int ds = data_size();

	Ref<MLPPTensor3> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] / b_ptr[i];
	}

	return C;
}
void MLPPTensor3::element_wise_divisionb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size3i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (a_size != _size) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] / b_ptr[i];
	}
}

void MLPPTensor3::sqrt() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sqrt(out_ptr[i]);
	}
}
Ref<MLPPTensor3> MLPPTensor3::sqrtn() const {
	Ref<MLPPTensor3> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}

	return out;
}
void MLPPTensor3::sqrtb(const Ref<MLPPTensor3> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size3i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::sqrt(a_ptr[i]);
	}
}

void MLPPTensor3::exponentiate(real_t p) {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::pow(out_ptr[i], p);
	}
}
Ref<MLPPTensor3> MLPPTensor3::exponentiaten(real_t p) const {
	Ref<MLPPTensor3> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}

	return out;
}
void MLPPTensor3::exponentiateb(const Ref<MLPPTensor3> &A, real_t p) {
	ERR_FAIL_COND(!A.is_valid());

	Size3i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = Math::pow(a_ptr[i], p);
	}
}

void MLPPTensor3::scalar_multiply(const real_t scalar) {
	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		_data[i] *= scalar;
	}
}
Ref<MLPPTensor3> MLPPTensor3::scalar_multiplyn(const real_t scalar) const {
	Ref<MLPPTensor3> AN = duplicate_fast();
	int ds = AN->data_size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < ds; ++i) {
		an_ptr[i] *= scalar;
	}

	return AN;
}
void MLPPTensor3::scalar_multiplyb(const real_t scalar, const Ref<MLPPTensor3> &A) {
	ERR_FAIL_COND(!A.is_valid());

	if (A->size() != _size) {
		resize(A->size());
	}

	int ds = data_size();
	real_t *an_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		_data[i] = an_ptr[i] * scalar;
	}
}

void MLPPTensor3::scalar_add(const real_t scalar) {
	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		_data[i] += scalar;
	}
}
Ref<MLPPTensor3> MLPPTensor3::scalar_addn(const real_t scalar) const {
	Ref<MLPPTensor3> AN = duplicate_fast();
	int ds = AN->data_size();
	real_t *an_ptr = AN->ptrw();

	for (int i = 0; i < ds; ++i) {
		an_ptr[i] += scalar;
	}

	return AN;
}
void MLPPTensor3::scalar_addb(const real_t scalar, const Ref<MLPPTensor3> &A) {
	ERR_FAIL_COND(!A.is_valid());

	if (A->size() != _size) {
		resize(A->size());
	}

	int ds = data_size();
	real_t *an_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		_data[i] = an_ptr[i] + scalar;
	}
}

void MLPPTensor3::hadamard_product(const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	int ds = data_size();

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = c_ptr[i] * b_ptr[i];
	}
}
Ref<MLPPTensor3> MLPPTensor3::hadamard_productn(const Ref<MLPPTensor3> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPTensor3>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPTensor3>());

	int ds = data_size();

	Ref<MLPPTensor3> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] * b_ptr[i];
	}

	return C;
}
void MLPPTensor3::hadamard_productb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size3i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (a_size != _size) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	for (int i = 0; i < ds; i++) {
		c_ptr[i] = a_ptr[i] * b_ptr[i];
	}
}

void MLPPTensor3::max(const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!B.is_valid());
	ERR_FAIL_COND(_size != B->size());

	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = MAX(c_ptr[i], b_ptr[i]);
	}
}
Ref<MLPPTensor3> MLPPTensor3::maxn(const Ref<MLPPTensor3> &B) const {
	ERR_FAIL_COND_V(!B.is_valid(), Ref<MLPPTensor3>());
	ERR_FAIL_COND_V(_size != B->size(), Ref<MLPPTensor3>());

	Ref<MLPPTensor3> C;
	C.instance();
	C->resize(_size);

	const real_t *a_ptr = ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = C->ptrw();

	int ds = data_size();

	for (int i = 0; i < ds; ++i) {
		c_ptr[i] = MAX(a_ptr[i], b_ptr[i]);
	}

	return C;
}
void MLPPTensor3::maxb(const Ref<MLPPTensor3> &A, const Ref<MLPPTensor3> &B) {
	ERR_FAIL_COND(!A.is_valid() || !B.is_valid());
	Size3i a_size = A->size();
	ERR_FAIL_COND(a_size != B->size());

	if (_size != a_size) {
		resize(a_size);
	}

	const real_t *a_ptr = A->ptr();
	const real_t *b_ptr = B->ptr();
	real_t *c_ptr = ptrw();

	int data_size = A->data_size();

	for (int i = 0; i < data_size; ++i) {
		c_ptr[i] = MAX(a_ptr[i], b_ptr[i]);
	}
}

void MLPPTensor3::abs() {
	int ds = data_size();

	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = ABS(out_ptr[i]);
	}
}
Ref<MLPPTensor3> MLPPTensor3::absn() const {
	Ref<MLPPTensor3> out;
	out.instance();
	out->resize(size());

	int ds = data_size();

	const real_t *a_ptr = ptr();
	real_t *out_ptr = out->ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}

	return out;
}
void MLPPTensor3::absb(const Ref<MLPPTensor3> &A) {
	ERR_FAIL_COND(!A.is_valid());

	Size3i a_size = A->size();

	if (a_size != size()) {
		resize(a_size);
	}

	int ds = data_size();

	const real_t *a_ptr = A->ptr();
	real_t *out_ptr = ptrw();

	for (int i = 0; i < ds; ++i) {
		out_ptr[i] = ABS(a_ptr[i]);
	}
}

Ref<MLPPVector> MLPPTensor3::flatten() const {
	int ds = data_size();

	Ref<MLPPVector> res;
	res.instance();
	res->resize(ds);

	real_t *res_ptr = res->ptrw();
	const real_t *a_ptr = ptr();

	for (int i = 0; i < ds; ++i) {
		res_ptr[i] = a_ptr[i];
	}

	return res;
}
void MLPPTensor3::flatteno(Ref<MLPPVector> out) const {
	ERR_FAIL_COND(!out.is_valid());

	int ds = data_size();

	if (unlikely(out->size() != ds)) {
		out->resize(ds);
	}

	real_t *res_ptr = out->ptrw();
	const real_t *a_ptr = ptr();

	for (int i = 0; i < ds; ++i) {
		res_ptr[i] = a_ptr[i];
	}
}

/*
real_t MLPPTensor3::norm_2(std::vector<std::vector<std::vector<real_t>>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			for (uint32_t k = 0; k < A[i][j].size(); k++) {
				sum += A[i][j][k] * A[i][j][k];
			}
		}
	}
	return Math::sqrt(sum);
}
*/

/*
std::vector<std::vector<real_t>> MLPPTensor3::tensor_vec_mult(std::vector<std::vector<std::vector<real_t>>> A, std::vector<real_t> b) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < C.size(); i++) {
		for (uint32_t j = 0; j < C[i].size(); j++) {
			C[i][j] = dot(A[i][j], b);
		}
	}
	return C;
}
*/

/*
// Bad implementation. Change this later.
std::vector<std::vector<std::vector<real_t>>> MLPPTensor3::vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<std::vector<real_t>>> C;
	C = resize(C, A);
	for (uint32_t i = 0; i < A[0].size(); i++) {
		for (uint32_t j = 0; j < A[0][i].size(); j++) {
			std::vector<real_t> currentVector;
			currentVector.resize(A.size());

			for (uint32_t k = 0; k < C.size(); k++) {
				currentVector[k] = A[k][i][j];
			}

			currentVector = mat_vec_mult(B, currentVector);

			for (uint32_t k = 0; k < C.size(); k++) {
				C[k][i][j] = currentVector[k];
			}
		}
	}
	return C;
}
*/

void MLPPTensor3::fill(real_t p_val) {
	if (!_data) {
		return;
	}

	int ds = data_size();
	for (int i = 0; i < ds; ++i) {
		_data[i] = p_val;
	}
}

Vector<real_t> MLPPTensor3::to_flat_vector() const {
	Vector<real_t> ret;
	ret.resize(data_size());
	real_t *w = ret.ptrw();
	memcpy(w, _data, sizeof(real_t) * data_size());
	return ret;
}

PoolRealArray MLPPTensor3::to_flat_pool_vector() const {
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

Vector<uint8_t> MLPPTensor3::to_flat_byte_array() const {
	Vector<uint8_t> ret;
	ret.resize(data_size() * sizeof(real_t));
	uint8_t *w = ret.ptrw();
	memcpy(w, _data, sizeof(real_t) * data_size());
	return ret;
}

Ref<MLPPTensor3> MLPPTensor3::duplicate_fast() const {
	Ref<MLPPTensor3> ret;
	ret.instance();

	ret->set_from_mlpp_tensor3r(*this);

	return ret;
}

void MLPPTensor3::set_from_mlpp_tensor3(const Ref<MLPPTensor3> &p_from) {
	ERR_FAIL_COND(!p_from.is_valid());

	resize(p_from->size());

	int ds = p_from->data_size();
	const real_t *ptr = p_from->ptr();

	for (int i = 0; i < ds; ++i) {
		_data[i] = ptr[i];
	}
}

void MLPPTensor3::set_from_mlpp_tensor3r(const MLPPTensor3 &p_from) {
	resize(p_from.size());

	int ds = p_from.data_size();
	const real_t *ptr = p_from.ptr();

	for (int i = 0; i < ds; ++i) {
		_data[i] = ptr[i];
	}
}

void MLPPTensor3::set_from_mlpp_matrix(const Ref<MLPPMatrix> &p_from) {
	ERR_FAIL_COND(!p_from.is_valid());

	Size2i mat_size = p_from->size();
	resize(Size3i(mat_size.x, mat_size.y, 1));

	int ds = p_from->data_size();
	const real_t *ptr = p_from->ptr();

	for (int i = 0; i < ds; ++i) {
		_data[i] = ptr[i];
	}
}

void MLPPTensor3::set_from_mlpp_matrixr(const MLPPMatrix &p_from) {
	Size2i mat_size = p_from.size();
	resize(Size3i(mat_size.x, mat_size.y, 1));

	int ds = p_from.data_size();
	const real_t *ptr = p_from.ptr();

	for (int i = 0; i < ds; ++i) {
		_data[i] = ptr[i];
	}
}

void MLPPTensor3::set_from_mlpp_vectors(const Vector<Ref<MLPPVector>> &p_from) {
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

void MLPPTensor3::set_from_mlpp_matricess(const Vector<Ref<MLPPMatrix>> &p_from) {
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

	Size2i fms = z_slice_size();
	int fmds = z_slice_data_size();

	for (int i = 0; i < p_from.size(); ++i) {
		const Ref<MLPPMatrix> &r = p_from[i];

		ERR_CONTINUE(!r.is_valid());
		ERR_CONTINUE(r->size() != fms);

		int start_index = calculate_z_slice_index(i);

		const real_t *from_ptr = r->ptr();
		for (int j = 0; j < fmds; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
}

void MLPPTensor3::set_from_mlpp_vectors_array(const Array &p_from) {
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

void MLPPTensor3::set_from_mlpp_matrices_array(const Array &p_from) {
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

	Size2i fms = z_slice_size();
	int fmds = z_slice_data_size();

	for (int i = 0; i < p_from.size(); ++i) {
		Ref<MLPPMatrix> r = p_from[i];

		ERR_CONTINUE(!r.is_valid());
		ERR_CONTINUE(r->size() != fms);

		int start_index = calculate_z_slice_index(i);

		const real_t *from_ptr = r->ptr();
		for (int j = 0; j < fmds; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
}

bool MLPPTensor3::is_equal_approx(const Ref<MLPPTensor3> &p_with, real_t tolerance) const {
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

String MLPPTensor3::to_string() {
	String str;

	str += "[MLPPTensor3: \n";

	for (int z = 0; z < _size.z; ++z) {
		int z_ofs = _size.x * _size.y * z;

		str += "  [ ";

		for (int y = 0; y < _size.y; ++y) {
			str += "    [ ";

			for (int x = 0; x < _size.x; ++x) {
				str += String::num(_data[_size.x * y + x + z_ofs]);
				str += " ";
			}

			str += "  ]\n";
		}

		str += "],\n";
	}

	str += "]\n";

	return str;
}

MLPPTensor3::MLPPTensor3() {
	_data = NULL;
}

MLPPTensor3::MLPPTensor3(const MLPPMatrix &p_from) {
	_data = NULL;

	Size2i mat_size = p_from.size();
	resize(Size3i(mat_size.x, mat_size.y, 1));

	int ds = p_from.data_size();
	const real_t *ptr = p_from.ptr();

	for (int i = 0; i < ds; ++i) {
		_data[i] = ptr[i];
	}
}

MLPPTensor3::MLPPTensor3(const Array &p_from) {
	_data = NULL;

	set_from_mlpp_matrices_array(p_from);
}

MLPPTensor3::~MLPPTensor3() {
	if (_data) {
		reset();
	}
}

std::vector<real_t> MLPPTensor3::to_flat_std_vector() const {
	std::vector<real_t> ret;
	ret.resize(data_size());
	real_t *w = &ret[0];
	memcpy(w, _data, sizeof(real_t) * data_size());
	return ret;
}

void MLPPTensor3::set_from_std_vectors(const std::vector<std::vector<std::vector<real_t>>> &p_from) {
	if (p_from.size() == 0) {
		reset();
		return;
	}

	resize(Size3i(p_from[1].size(), p_from[0].size(), p_from.size()));

	if (data_size() == 0) {
		reset();
		return;
	}

	for (uint32_t k = 0; k < p_from.size(); ++k) {
		const std::vector<std::vector<real_t>> &fm = p_from[k];

		for (uint32_t i = 0; i < p_from.size(); ++i) {
			const std::vector<real_t> &r = fm[i];

			ERR_CONTINUE(r.size() != static_cast<uint32_t>(_size.x));

			int start_index = i * _size.x;

			const real_t *from_ptr = &r[0];
			for (int j = 0; j < _size.x; j++) {
				_data[start_index + j] = from_ptr[j];
			}
		}
	}
}

std::vector<std::vector<std::vector<real_t>>> MLPPTensor3::to_std_vector() {
	std::vector<std::vector<std::vector<real_t>>> ret;

	ret.resize(_size.z);

	for (int k = 0; k < _size.z; ++k) {
		ret[k].resize(_size.y);

		for (int i = 0; i < _size.y; ++i) {
			std::vector<real_t> row;

			for (int j = 0; j < _size.x; ++j) {
				row.push_back(_data[calculate_index(i, j, 1)]);
			}

			ret[k][i] = row;
		}
	}

	return ret;
}

MLPPTensor3::MLPPTensor3(const std::vector<std::vector<std::vector<real_t>>> &p_from) {
	_data = NULL;

	set_from_std_vectors(p_from);
}

void MLPPTensor3::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_data"), &MLPPTensor3::get_data);
	ClassDB::bind_method(D_METHOD("set_data", "data"), &MLPPTensor3::set_data);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "data"), "set_data", "get_data");

	ClassDB::bind_method(D_METHOD("add_z_slice_pool_vector", "row"), &MLPPTensor3::add_z_slice_pool_vector);
	ClassDB::bind_method(D_METHOD("add_z_slice_mlpp_vector", "row"), &MLPPTensor3::add_z_slice_mlpp_vector);
	ClassDB::bind_method(D_METHOD("add_z_slice_mlpp_matrix", "matrix"), &MLPPTensor3::add_z_slice_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("remove_z_slice", "index"), &MLPPTensor3::remove_z_slice);
	ClassDB::bind_method(D_METHOD("remove_z_slice_unordered", "index"), &MLPPTensor3::remove_z_slice_unordered);

	ClassDB::bind_method(D_METHOD("swap_z_slice", "index_1", "index_2"), &MLPPTensor3::swap_z_slice);

	ClassDB::bind_method(D_METHOD("clear"), &MLPPTensor3::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPTensor3::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPTensor3::empty);

	ClassDB::bind_method(D_METHOD("z_slice_data_size"), &MLPPTensor3::z_slice_data_size);
	ClassDB::bind_method(D_METHOD("z_slice_size"), &MLPPTensor3::z_slice_size);

	ClassDB::bind_method(D_METHOD("data_size"), &MLPPTensor3::data_size);
	ClassDB::bind_method(D_METHOD("size"), &MLPPTensor3::size);

	ClassDB::bind_method(D_METHOD("resize", "size"), &MLPPTensor3::resize);

	ClassDB::bind_method(D_METHOD("set_shape", "size"), &MLPPTensor3::set_shape);
	ClassDB::bind_method(D_METHOD("calculate_index", "index_y", "index_x", "index_z"), &MLPPTensor3::calculate_index);
	ClassDB::bind_method(D_METHOD("calculate_z_slice_index", "index_z"), &MLPPTensor3::calculate_z_slice_index);

	ClassDB::bind_method(D_METHOD("get_element_index", "index"), &MLPPTensor3::get_element_index);
	ClassDB::bind_method(D_METHOD("set_element_index", "index", "val"), &MLPPTensor3::set_element_index);

	ClassDB::bind_method(D_METHOD("get_element", "index_y", "index_x", "index_z"), &MLPPTensor3::get_element);
	ClassDB::bind_method(D_METHOD("set_element", "index_y", "index_x", "index_z", "val"), &MLPPTensor3::set_element);

	ClassDB::bind_method(D_METHOD("get_row_pool_vector", "index_y", "index_z"), &MLPPTensor3::get_row_pool_vector);
	ClassDB::bind_method(D_METHOD("get_row_mlpp_vector", "index_y", "index_z"), &MLPPTensor3::get_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("get_row_into_mlpp_vector", "index_y", "index_z", "target"), &MLPPTensor3::get_row_into_mlpp_vector);

	ClassDB::bind_method(D_METHOD("set_row_pool_vector", "index_y", "index_z", "row"), &MLPPTensor3::set_row_pool_vector);
	ClassDB::bind_method(D_METHOD("set_row_mlpp_vector", "index_y", "index_z", "row"), &MLPPTensor3::set_row_mlpp_vector);

	ClassDB::bind_method(D_METHOD("get_z_slice_pool_vector", "index_z"), &MLPPTensor3::get_z_slice_pool_vector);
	ClassDB::bind_method(D_METHOD("get_z_slice_mlpp_vector", "index_z"), &MLPPTensor3::get_z_slice_mlpp_vector);
	ClassDB::bind_method(D_METHOD("get_z_slice_into_mlpp_vector", "index_z", "target"), &MLPPTensor3::get_z_slice_into_mlpp_vector);

	ClassDB::bind_method(D_METHOD("get_z_slice_mlpp_matrix", "index_z"), &MLPPTensor3::get_z_slice_mlpp_matrix);
	ClassDB::bind_method(D_METHOD("get_z_slice_into_mlpp_matrix", "index_z", "target"), &MLPPTensor3::get_z_slice_into_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("set_z_slice_pool_vector", "index_z", "row"), &MLPPTensor3::set_z_slice_pool_vector);
	ClassDB::bind_method(D_METHOD("set_z_slice_mlpp_vector", "index_z", "row"), &MLPPTensor3::set_z_slice_mlpp_vector);
	ClassDB::bind_method(D_METHOD("set_z_slice_mlpp_matrix", "index_z", "mat"), &MLPPTensor3::set_z_slice_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("get_x_slice_into", "index_x", "target"), &MLPPTensor3::get_x_slice_into);
	ClassDB::bind_method(D_METHOD("get_x_slice", "index_x"), &MLPPTensor3::get_x_slice);
	ClassDB::bind_method(D_METHOD("set_x_slice", "index_x", "mat"), &MLPPTensor3::set_x_slice);

	ClassDB::bind_method(D_METHOD("get_y_slice_into", "index_y", "target"), &MLPPTensor3::get_y_slice_into);
	ClassDB::bind_method(D_METHOD("get_y_slice", "index_y"), &MLPPTensor3::get_y_slice);
	ClassDB::bind_method(D_METHOD("set_y_slice", "index_y", "mat"), &MLPPTensor3::set_y_slice);

	ClassDB::bind_method(D_METHOD("add_z_slices_image", "img", "channels"), &MLPPTensor3::add_z_slices_image, IMAGE_CHANNEL_FLAG_RGBA);

	ClassDB::bind_method(D_METHOD("get_z_slice_image", "index_z"), &MLPPTensor3::get_z_slice_image);
	ClassDB::bind_method(D_METHOD("get_z_slices_image", "index_r", "index_g", "index_b", "index_a"), &MLPPTensor3::get_z_slices_image, -1, -1, -1, -1);

	ClassDB::bind_method(D_METHOD("get_z_slice_into_image", "target", "index_z", "target_channels"), &MLPPTensor3::get_z_slice_into_image, IMAGE_CHANNEL_FLAG_RGB);
	ClassDB::bind_method(D_METHOD("get_z_slices_into_image", "target", "index_r", "index_g", "index_b", "index_a"), &MLPPTensor3::get_z_slices_into_image, -1, -1, -1, -1);

	ClassDB::bind_method(D_METHOD("set_z_slice_image", "img", "index_z", "image_channel_flag"), &MLPPTensor3::set_z_slice_image, IMAGE_CHANNEL_FLAG_R);
	ClassDB::bind_method(D_METHOD("set_z_slices_image", "img", "index_r", "index_g", "index_b", "index_a"), &MLPPTensor3::set_z_slices_image);

	ClassDB::bind_method(D_METHOD("set_from_image", "img", "channels"), &MLPPTensor3::set_from_image, IMAGE_CHANNEL_FLAG_RGBA);

	ClassDB::bind_method(D_METHOD("get_x_slice_image", "index_x"), &MLPPTensor3::get_x_slice_image);
	ClassDB::bind_method(D_METHOD("get_x_slice_into_image", "target", "index_x", "target_channels"), &MLPPTensor3::get_x_slice_into_image, IMAGE_CHANNEL_FLAG_RGB);
	ClassDB::bind_method(D_METHOD("set_x_slice_image", "img", "index_x", "image_channel_flag"), &MLPPTensor3::set_x_slice_image, IMAGE_CHANNEL_FLAG_R);

	ClassDB::bind_method(D_METHOD("get_y_slice_image", "index_x"), &MLPPTensor3::get_y_slice_image);
	ClassDB::bind_method(D_METHOD("get_y_slice_into_image", "target", "index_x", "target_channels"), &MLPPTensor3::get_y_slice_into_image, IMAGE_CHANNEL_FLAG_RGB);
	ClassDB::bind_method(D_METHOD("set_y_slice_image", "img", "index_x", "image_channel_flag"), &MLPPTensor3::set_y_slice_image, IMAGE_CHANNEL_FLAG_R);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPTensor3::fill);

	ClassDB::bind_method(D_METHOD("to_flat_pool_vector"), &MLPPTensor3::to_flat_pool_vector);
	ClassDB::bind_method(D_METHOD("to_flat_byte_array"), &MLPPTensor3::to_flat_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate_fast"), &MLPPTensor3::duplicate_fast);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_tensor3", "from"), &MLPPTensor3::set_from_mlpp_tensor3);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_matrix", "from"), &MLPPTensor3::set_from_mlpp_matrix);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_vectors_array", "from"), &MLPPTensor3::set_from_mlpp_vectors_array);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_matrices_array", "from"), &MLPPTensor3::set_from_mlpp_matrices_array);

	ClassDB::bind_method(D_METHOD("is_equal_approx", "with", "tolerance"), &MLPPTensor3::is_equal_approx, CMP_EPSILON);

	ClassDB::bind_method(D_METHOD("add", "B"), &MLPPTensor3::add);
	ClassDB::bind_method(D_METHOD("addn", "B"), &MLPPTensor3::addn);
	ClassDB::bind_method(D_METHOD("addb", "A", "B"), &MLPPTensor3::addb);

	ClassDB::bind_method(D_METHOD("sub", "B"), &MLPPTensor3::sub);
	ClassDB::bind_method(D_METHOD("subn", "B"), &MLPPTensor3::subn);
	ClassDB::bind_method(D_METHOD("subb", "A", "B"), &MLPPTensor3::subb);

	ClassDB::bind_method(D_METHOD("hadamard_product", "B"), &MLPPTensor3::hadamard_product);
	ClassDB::bind_method(D_METHOD("hadamard_productn", "B"), &MLPPTensor3::hadamard_productn);
	ClassDB::bind_method(D_METHOD("hadamard_productb", "A", "B"), &MLPPTensor3::hadamard_productb);

	ClassDB::bind_method(D_METHOD("element_wise_division", "B"), &MLPPTensor3::element_wise_division);
	ClassDB::bind_method(D_METHOD("element_wise_divisionn", "B"), &MLPPTensor3::element_wise_divisionn);
	ClassDB::bind_method(D_METHOD("element_wise_divisionb", "A", "B"), &MLPPTensor3::element_wise_divisionb);

	ClassDB::bind_method(D_METHOD("scalar_multiply", "scalar"), &MLPPTensor3::scalar_multiply);
	ClassDB::bind_method(D_METHOD("scalar_multiplyn", "scalar"), &MLPPTensor3::scalar_multiplyn);
	ClassDB::bind_method(D_METHOD("scalar_multiplyb", "scalar", "A"), &MLPPTensor3::scalar_multiplyb);

	ClassDB::bind_method(D_METHOD("scalar_add", "scalar"), &MLPPTensor3::scalar_add);
	ClassDB::bind_method(D_METHOD("scalar_addn", "scalar"), &MLPPTensor3::scalar_addn);
	ClassDB::bind_method(D_METHOD("scalar_addb", "scalar", "A"), &MLPPTensor3::scalar_addb);

	ClassDB::bind_method(D_METHOD("exponentiate", "p"), &MLPPTensor3::exponentiate);
	ClassDB::bind_method(D_METHOD("exponentiaten", "p"), &MLPPTensor3::exponentiaten);
	ClassDB::bind_method(D_METHOD("exponentiateb", "A", "p"), &MLPPTensor3::exponentiateb);

	ClassDB::bind_method(D_METHOD("sqrt"), &MLPPTensor3::sqrt);
	ClassDB::bind_method(D_METHOD("sqrtn"), &MLPPTensor3::sqrtn);
	ClassDB::bind_method(D_METHOD("sqrtb", "A"), &MLPPTensor3::sqrtb);

	ClassDB::bind_method(D_METHOD("abs"), &MLPPTensor3::abs);
	ClassDB::bind_method(D_METHOD("absn"), &MLPPTensor3::absn);
	ClassDB::bind_method(D_METHOD("absb", "A"), &MLPPTensor3::absb);

	ClassDB::bind_method(D_METHOD("max", "B"), &MLPPTensor3::max);
	ClassDB::bind_method(D_METHOD("maxn", "B"), &MLPPTensor3::maxn);
	ClassDB::bind_method(D_METHOD("maxb", "A", "B"), &MLPPTensor3::maxb);

	ClassDB::bind_method(D_METHOD("flatten"), &MLPPTensor3::flatten);
	ClassDB::bind_method(D_METHOD("flatteno", "out"), &MLPPTensor3::flatteno);

	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_R);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_G);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_B);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_A);

	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_NONE);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_RG);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_RGB);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_GB);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_GBA);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_BA);
	BIND_ENUM_CONSTANT(IMAGE_CHANNEL_FLAG_RGBA);
}
