
#include "mlpp_matrix.h"

String MLPPMatrix::to_string() {
	String str;

	str += "[MLPPMatrix: \n";

	for (int y = 0; y < _size.y; ++y) {
		str += "  [ ";

		for (int x = 0; x < _size.x; ++x) {
			str += String::num(_data[_size.x * y + x]);
			str += " ";
		}

		str += "]\n";
	}

	str += "]";

	return str;
}

std::vector<real_t> MLPPMatrix::to_flat_std_vector() const {
	std::vector<real_t> ret;
	ret.resize(data_size());
	real_t *w = &ret[0];
	memcpy(w, _data, sizeof(real_t) * data_size());
	return ret;
}

void MLPPMatrix::set_from_std_vectors(const std::vector<std::vector<real_t>> &p_from) {
	if (p_from.size() == 0) {
		reset();
		return;
	}

	resize(Size2i(p_from[0].size(), p_from.size()));

	if (data_size() == 0) {
		reset();
		return;
	}

	for (uint32_t i = 0; i < p_from.size(); ++i) {
		const std::vector<real_t> &r = p_from[i];

		ERR_CONTINUE(r.size() != static_cast<uint32_t>(_size.x));

		int start_index = i * _size.x;

		const real_t *from_ptr = &r[0];
		for (int j = 0; j < _size.x; j++) {
			_data[start_index + j] = from_ptr[j];
		}
	}
}

std::vector<std::vector<real_t>> MLPPMatrix::to_std_vector() {
	std::vector<std::vector<real_t>> ret;

	ret.resize(_size.y);

	for (int i = 0; i < _size.y; ++i) {
		std::vector<real_t> row;

		for (int j = 0; j < _size.x; ++j) {
			row.push_back(_data[calculate_index(i, j)]);
		}

		ret[i] = row;
	}

	return ret;
}

void MLPPMatrix::set_row_std_vector(int p_index_y, const std::vector<real_t> &p_row) {
	ERR_FAIL_COND(p_row.size() != static_cast<uint32_t>(_size.x));
	ERR_FAIL_INDEX(p_index_y, _size.y);

	int ind_start = p_index_y * _size.x;

	const real_t *row_ptr = &p_row[0];

	for (int i = 0; i < _size.x; ++i) {
		_data[ind_start + i] = row_ptr[i];
	}
}

MLPPMatrix::MLPPMatrix(const std::vector<std::vector<real_t>> &p_from) {
	_data = NULL;

	set_from_std_vectors(p_from);
}

void MLPPMatrix::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_row", "row"), &MLPPMatrix::add_row_pool_vector);
	ClassDB::bind_method(D_METHOD("add_row_mlpp_vector", "row"), &MLPPMatrix::add_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("add_rows_mlpp_matrix", "other"), &MLPPMatrix::add_rows_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("remove_row", "index"), &MLPPMatrix::remove_row);
	ClassDB::bind_method(D_METHOD("remove_row_unordered", "index"), &MLPPMatrix::remove_row_unordered);
	ClassDB::bind_method(D_METHOD("swap_row", "index_1", "index_2"), &MLPPMatrix::swap_row);

	ClassDB::bind_method(D_METHOD("clear"), &MLPPMatrix::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPMatrix::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPMatrix::empty);

	ClassDB::bind_method(D_METHOD("data_size"), &MLPPMatrix::data_size);
	ClassDB::bind_method(D_METHOD("size"), &MLPPMatrix::size);

	ClassDB::bind_method(D_METHOD("resize", "size"), &MLPPMatrix::resize);

	ClassDB::bind_method(D_METHOD("get_element_index", "index"), &MLPPMatrix::get_element_index);
	ClassDB::bind_method(D_METHOD("set_element_index", "index", "val"), &MLPPMatrix::set_element_index);

	ClassDB::bind_method(D_METHOD("get_element", "index_x", "index_y"), &MLPPMatrix::get_element);
	ClassDB::bind_method(D_METHOD("set_element", "index_x", "index_y", "val"), &MLPPMatrix::set_element);

	ClassDB::bind_method(D_METHOD("get_row_pool_vector", "index_y"), &MLPPMatrix::get_row_pool_vector);
	ClassDB::bind_method(D_METHOD("get_row_mlpp_vector", "index_y"), &MLPPMatrix::get_row_mlpp_vector);
	ClassDB::bind_method(D_METHOD("get_row_into_mlpp_vector", "index_y", "target"), &MLPPMatrix::get_row_into_mlpp_vector);

	ClassDB::bind_method(D_METHOD("set_row_pool_vector", "index_y", "row"), &MLPPMatrix::set_row_pool_vector);
	ClassDB::bind_method(D_METHOD("set_row_mlpp_vector", "index_y", "row"), &MLPPMatrix::set_row_mlpp_vector);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPMatrix::fill);

	ClassDB::bind_method(D_METHOD("to_flat_pool_vector"), &MLPPMatrix::to_flat_pool_vector);
	ClassDB::bind_method(D_METHOD("to_flat_byte_array"), &MLPPMatrix::to_flat_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPMatrix::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vectors_array", "from"), &MLPPMatrix::set_from_mlpp_vectors_array);
	ClassDB::bind_method(D_METHOD("set_from_arrays", "from"), &MLPPMatrix::set_from_arrays);
	ClassDB::bind_method(D_METHOD("set_from_mlpp_matrix", "from"), &MLPPMatrix::set_from_mlpp_matrix);

	ClassDB::bind_method(D_METHOD("is_equal_approx", "with", "tolerance"), &MLPPMatrix::is_equal_approx, CMP_EPSILON);
}
