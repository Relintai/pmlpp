
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

void MLPPMatrix::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_row", "row"), &MLPPMatrix::add_row_pool_vector);
	ClassDB::bind_method(D_METHOD("remove_row", "index"), &MLPPMatrix::remove_row);
	ClassDB::bind_method(D_METHOD("remove_unordered", "index"), &MLPPMatrix::remove_unordered);
	ClassDB::bind_method(D_METHOD("swap_row", "index_1", "index_2"), &MLPPMatrix::swap_row);

	ClassDB::bind_method(D_METHOD("clear"), &MLPPMatrix::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPMatrix::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPMatrix::empty);

	ClassDB::bind_method(D_METHOD("data_size"), &MLPPMatrix::data_size);
	ClassDB::bind_method(D_METHOD("size"), &MLPPMatrix::size);

	ClassDB::bind_method(D_METHOD("resize", "size"), &MLPPMatrix::resize);

	ClassDB::bind_method(D_METHOD("get_element", "index_x", "index_y"), &MLPPMatrix::get_element_bind);
	ClassDB::bind_method(D_METHOD("set_element", "index_x", "index_y", "val"), &MLPPMatrix::set_element_bind);

	ClassDB::bind_method(D_METHOD("set_row_pool_vector", "index_y", "row"), &MLPPMatrix::set_row_pool_vector);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPMatrix::fill);

	ClassDB::bind_method(D_METHOD("to_flat_pool_vector"), &MLPPMatrix::to_flat_pool_vector);
	ClassDB::bind_method(D_METHOD("to_flat_byte_array"), &MLPPMatrix::to_flat_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPMatrix::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vectors_array", "from"), &MLPPMatrix::set_from_mlpp_vectors_array);
	ClassDB::bind_method(D_METHOD("set_from_arrays", "from"), &MLPPMatrix::set_from_arrays);
}
