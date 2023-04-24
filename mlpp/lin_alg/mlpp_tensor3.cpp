
#include "mlpp_tensor3.h"

#include "core/io/image.h"

void MLPPTensor3::add_feature_maps_image(const Ref<Image> &p_img, const int p_channels) {
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

	Size2i fms = feature_map_size();

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

Ref<Image> MLPPTensor3::get_feature_map_image(const int p_index_z) {
	ERR_FAIL_INDEX_V(p_index_z, _size.z, Ref<Image>());

	Ref<Image> image;
	image.instance();

	if (data_size() == 0) {
		return image;
	}

	PoolByteArray arr;

	int fmsi = calculate_feature_map_index(p_index_z);
	int fms = feature_map_data_size();

	arr.resize(fms);

	PoolByteArray::Write w = arr.write();
	uint8_t *wptr = w.ptr();

	for (int i = 0; i < fms; ++i) {
		wptr[i] = static_cast<uint8_t>(_data[fmsi + i] * 255.0);
	}

	image->create(_size.x, _size.y, false, Image::FORMAT_L8, arr);

	return image;
}
Ref<Image> MLPPTensor3::get_feature_maps_image(const int p_index_r, const int p_index_g, const int p_index_b, const int p_index_a) {
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

	Size2i fms = feature_map_size();

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

void MLPPTensor3::get_feature_map_into_image(Ref<Image> p_target, const int p_index_z, const int p_target_channels) const {
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
	Size2i fms = feature_map_size();
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
void MLPPTensor3::get_feature_maps_into_image(Ref<Image> p_target, const int p_index_r, const int p_index_g, const int p_index_b, const int p_index_a) const {
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
	Size2i fms = feature_map_size();
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

void MLPPTensor3::set_feature_map_image(const Ref<Image> &p_img, const int p_index_z, const int p_image_channel_flag) {
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
	Size2i fms = feature_map_size();

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
void MLPPTensor3::set_feature_maps_image(const Ref<Image> &p_img, const int p_index_r, const int p_index_g, const int p_index_b, const int p_index_a) {
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
	Size2i fms = feature_map_size();

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

	Size2i fms = feature_map_size();

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

/*
Vector<Ref<MLPPMatrix>> MLPPTensor3::additionnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < res.size(); i++) {
		res.write[i] = additionnm(A[i], B[i]);
	}

	return res;
}
*/
/*
Vector<Ref<MLPPMatrix>> MLPPTensor3::element_wise_divisionnvnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = element_wise_divisionnvnm(A[i], B[i]);
	}

	return res;
}
*/
/*
Vector<Ref<MLPPMatrix>> MLPPTensor3::sqrtnvt(const Vector<Ref<MLPPMatrix>> &A) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = sqrtnm(A[i]);
	}

	return res;
}
*/
/*
Vector<Ref<MLPPMatrix>> MLPPTensor3::exponentiatenvt(const Vector<Ref<MLPPMatrix>> &A, real_t p) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = exponentiatenm(A[i], p);
	}

	return res;
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
std::vector<real_t> MLPPTensor3::flatten(std::vector<std::vector<std::vector<real_t>>> A) {
	std::vector<real_t> c;
	for (uint32_t i = 0; i < A.size(); i++) {
		std::vector<real_t> flattenedVec = flatten(A[i]);
		c.insert(c.end(), flattenedVec.begin(), flattenedVec.end());
	}
	return c;
}
*/

/*
Vector<Ref<MLPPMatrix>> MLPPTensor3::scalar_multiplynvt(real_t scalar, Vector<Ref<MLPPMatrix>> A) {
	for (int i = 0; i < A.size(); i++) {
		A.write[i] = scalar_multiplynm(scalar, A[i]);
	}
	return A;
}
Vector<Ref<MLPPMatrix>> MLPPTensor3::scalar_addnvt(real_t scalar, Vector<Ref<MLPPMatrix>> A) {
	for (int i = 0; i < A.size(); i++) {
		A.write[i] = scalar_addnm(scalar, A[i]);
	}
	return A;
}

Vector<Ref<MLPPMatrix>> MLPPTensor3::resizenvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(B.size());

	for (int i = 0; i < res.size(); i++) {
		Ref<MLPPMatrix> m;
		m.instance();
		m->resize(B[i]->size());

		res.write[i] = m;
	}

	return res;
}
*/

//std::vector<std::vector<std::vector<real_t>>> hadamard_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

/*
Vector<Ref<MLPPMatrix>> MLPPTensor3::maxnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = maxnm(A[i], B[i]);
	}

	return res;
}

Vector<Ref<MLPPMatrix>> MLPPTensor3::absnvt(const Vector<Ref<MLPPMatrix>> &A) {
	Vector<Ref<MLPPMatrix>> res;
	res.resize(A.size());

	for (int i = 0; i < A.size(); i++) {
		res.write[i] = absnm(A[i]);
	}

	return A;
}
*/

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
	ClassDB::bind_method(D_METHOD("add_feature_map_pool_vector", "row"), &MLPPTensor3::add_feature_map_pool_vector);
	ClassDB::bind_method(D_METHOD("add_feature_map_mlpp_vector", "row"), &MLPPTensor3::add_feature_map_mlpp_vector);
	ClassDB::bind_method(D_METHOD("add_feature_map_mlpp_matrix", "matrix"), &MLPPTensor3::add_feature_map_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("remove_feature_map", "index"), &MLPPTensor3::remove_feature_map);
	ClassDB::bind_method(D_METHOD("remove_feature_map_unordered", "index"), &MLPPTensor3::remove_feature_map_unordered);

	ClassDB::bind_method(D_METHOD("swap_feature_map", "index_1", "index_2"), &MLPPTensor3::swap_feature_map);

	ClassDB::bind_method(D_METHOD("clear"), &MLPPTensor3::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPTensor3::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPTensor3::empty);

	ClassDB::bind_method(D_METHOD("feature_map_data_size"), &MLPPTensor3::feature_map_data_size);
	ClassDB::bind_method(D_METHOD("feature_map_size"), &MLPPTensor3::feature_map_size);

	ClassDB::bind_method(D_METHOD("data_size"), &MLPPTensor3::data_size);
	ClassDB::bind_method(D_METHOD("size"), &MLPPTensor3::size);

	ClassDB::bind_method(D_METHOD("resize", "size"), &MLPPTensor3::resize);

	ClassDB::bind_method(D_METHOD("set_shape", "size"), &MLPPTensor3::set_shape);
	ClassDB::bind_method(D_METHOD("calculate_index", "index_y", "index_x", "index_z"), &MLPPTensor3::calculate_index);
	ClassDB::bind_method(D_METHOD("calculate_feature_map_index", "index_z"), &MLPPTensor3::calculate_feature_map_index);

	ClassDB::bind_method(D_METHOD("get_element_index", "index"), &MLPPTensor3::get_element_index);
	ClassDB::bind_method(D_METHOD("set_element_index", "index", "val"), &MLPPTensor3::set_element_index);

	ClassDB::bind_method(D_METHOD("get_element", "index_y", "index_x", "index_z"), &MLPPTensor3::get_element);
	ClassDB::bind_method(D_METHOD("set_element", "index_y", "index_x", "index_z", "val"), &MLPPTensor3::set_element);

	ClassDB::bind_method(D_METHOD("get_row_pool_vector", "index_y", "index_z"), &MLPPTensor3::get_row_pool_vector);
	ClassDB::bind_method(D_METHOD("get_row_mlpp_vector", "index_y", "index_z"), &MLPPTensor3::get_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("get_row_into_mlpp_vector", "index_y", "index_z", "target"), &MLPPTensor3::get_row_into_mlpp_vector);

	ClassDB::bind_method(D_METHOD("set_row_pool_vector", "index_y", "index_z", "row"), &MLPPTensor3::set_row_pool_vector);
	ClassDB::bind_method(D_METHOD("set_row_mlpp_vector", "index_y", "index_z", "row"), &MLPPTensor3::set_row_mlpp_vector);

	ClassDB::bind_method(D_METHOD("get_feature_map_pool_vector", "index_z"), &MLPPTensor3::get_feature_map_pool_vector);
	ClassDB::bind_method(D_METHOD("get_feature_map_mlpp_vector", "index_z"), &MLPPTensor3::get_feature_map_mlpp_vector);
	ClassDB::bind_method(D_METHOD("get_feature_map_into_mlpp_vector", "index_z", "target"), &MLPPTensor3::get_feature_map_into_mlpp_vector);

	ClassDB::bind_method(D_METHOD("get_feature_map_mlpp_matrix", "index_z"), &MLPPTensor3::get_feature_map_mlpp_matrix);
	ClassDB::bind_method(D_METHOD("get_feature_map_into_mlpp_matrix", "index_z", "target"), &MLPPTensor3::get_feature_map_into_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("set_feature_map_pool_vector", "index_z", "row"), &MLPPTensor3::set_feature_map_pool_vector);
	ClassDB::bind_method(D_METHOD("set_feature_map_mlpp_vector", "index_z", "row"), &MLPPTensor3::set_feature_map_mlpp_vector);
	ClassDB::bind_method(D_METHOD("set_feature_map_mlpp_matrix", "index_z", "mat"), &MLPPTensor3::set_feature_map_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("add_feature_maps_image", "img", "channels"), &MLPPTensor3::add_feature_maps_image, IMAGE_CHANNEL_FLAG_RGBA);

	ClassDB::bind_method(D_METHOD("get_feature_map_image", "index_z"), &MLPPTensor3::get_feature_map_image);
	ClassDB::bind_method(D_METHOD("get_feature_maps_image", "index_r", "index_g", "index_b", "index_a"), &MLPPTensor3::get_feature_maps_image, -1, -1, -1, -1);

	ClassDB::bind_method(D_METHOD("get_feature_map_into_image", "target", "index_z", "target_channels"), &MLPPTensor3::get_feature_map_into_image, IMAGE_CHANNEL_FLAG_RGB);
	ClassDB::bind_method(D_METHOD("get_feature_maps_into_image", "target", "index_r", "index_g", "index_b", "index_a"), &MLPPTensor3::get_feature_maps_into_image, -1, -1, -1, -1);

	ClassDB::bind_method(D_METHOD("set_feature_map_image", "img", "index_z", "image_channel_flag"), &MLPPTensor3::set_feature_map_image, IMAGE_CHANNEL_FLAG_R);
	ClassDB::bind_method(D_METHOD("set_feature_maps_image", "img", "index_r", "index_g", "index_b", "index_a"), &MLPPTensor3::set_feature_maps_image);

	ClassDB::bind_method(D_METHOD("set_from_image", "img", "channels"), &MLPPTensor3::set_from_image, IMAGE_CHANNEL_FLAG_RGBA);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPTensor3::fill);

	ClassDB::bind_method(D_METHOD("to_flat_pool_vector"), &MLPPTensor3::to_flat_pool_vector);
	ClassDB::bind_method(D_METHOD("to_flat_byte_array"), &MLPPTensor3::to_flat_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPTensor3::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_tensor3", "from"), &MLPPTensor3::set_from_mlpp_tensor3);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_matrix", "from"), &MLPPTensor3::set_from_mlpp_matrix);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_vectors_array", "from"), &MLPPTensor3::set_from_mlpp_vectors_array);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_matrices_array", "from"), &MLPPTensor3::set_from_mlpp_matrices_array);

	ClassDB::bind_method(D_METHOD("is_equal_approx", "with", "tolerance"), &MLPPTensor3::is_equal_approx, CMP_EPSILON);

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
