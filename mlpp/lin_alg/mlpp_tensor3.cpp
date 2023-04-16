
#include "mlpp_tensor3.h"

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

	resize(Size3i(p_from[1].size(), p_from.size(), p_from[0].size()));

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

void MLPPTensor3::set_row_std_vector(int p_index_y, const std::vector<real_t> &p_row) {
	ERR_FAIL_COND(p_row.size() != static_cast<uint32_t>(_size.x));
	ERR_FAIL_INDEX(p_index_y, _size.y);

	int ind_start = p_index_y * _size.x;

	const real_t *row_ptr = &p_row[0];

	for (int i = 0; i < _size.x; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

MLPPTensor3::MLPPTensor3(const std::vector<std::vector<std::vector<real_t>>> &p_from) {
	_data = NULL;

	set_from_std_vectors(p_from);
}

void MLPPTensor3::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("add_row", "row"), &MLPPTensor3::add_row_pool_vector);
	ClassDB::bind_method(D_METHOD("add_row_mlpp_vector", "row"), &MLPPTensor3::add_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("add_rows_mlpp_matrix", "other"), &MLPPTensor3::add_rows_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("remove_row", "index"), &MLPPTensor3::remove_row);
	ClassDB::bind_method(D_METHOD("remove_row_unordered", "index"), &MLPPTensor3::remove_row_unordered);
	ClassDB::bind_method(D_METHOD("swap_row", "index_1", "index_2"), &MLPPTensor3::swap_row);

	ClassDB::bind_method(D_METHOD("clear"), &MLPPTensor3::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPTensor3::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPTensor3::empty);

	ClassDB::bind_method(D_METHOD("data_size"), &MLPPTensor3::data_size);
	ClassDB::bind_method(D_METHOD("size"), &MLPPTensor3::size);

	ClassDB::bind_method(D_METHOD("resize", "size"), &MLPPTensor3::resize);

	ClassDB::bind_method(D_METHOD("get_element", "index_x", "index_y"), &MLPPTensor3::get_element);
	ClassDB::bind_method(D_METHOD("set_element", "index_x", "index_y", "val"), &MLPPTensor3::set_element);

	ClassDB::bind_method(D_METHOD("get_row_pool_vector", "index_y"), &MLPPTensor3::get_row_pool_vector);
	ClassDB::bind_method(D_METHOD("get_row_mlpp_vector", "index_y"), &MLPPTensor3::get_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("get_row_into_mlpp_vector", "index_y", "target"), &MLPPTensor3::get_row_into_mlpp_vector);

	ClassDB::bind_method(D_METHOD("set_row_pool_vector", "index_y", "row"), &MLPPTensor3::set_row_pool_vector);
	ClassDB::bind_method(D_METHOD("set_row_mlpp_vector", "index_y", "row"), &MLPPTensor3::set_row_mlpp_vector);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPTensor3::fill);

	ClassDB::bind_method(D_METHOD("to_flat_pool_vector"), &MLPPTensor3::to_flat_pool_vector);
	ClassDB::bind_method(D_METHOD("to_flat_byte_array"), &MLPPTensor3::to_flat_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPTensor3::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vectors_array", "from"), &MLPPTensor3::set_from_mlpp_vectors_array);
	ClassDB::bind_method(D_METHOD("set_from_arrays", "from"), &MLPPTensor3::set_from_arrays);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_matrix", "from"), &MLPPTensor3::set_from_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("is_equal_approx", "with", "tolerance"), &MLPPTensor3::is_equal_approx, CMP_EPSILON);
	*/
}
