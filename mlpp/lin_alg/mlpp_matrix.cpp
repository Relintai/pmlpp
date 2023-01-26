
#include "mlpp_matrix.h"

void MLPPMatrix::_bind_methods() {
	ClassDB::bind_method(D_METHOD("push_back", "elem"), &MLPPMatrix::push_back);
	ClassDB::bind_method(D_METHOD("remove", "index"), &MLPPMatrix::remove);
	ClassDB::bind_method(D_METHOD("remove_unordered", "index"), &MLPPMatrix::remove_unordered);
	ClassDB::bind_method(D_METHOD("erase", "val"), &MLPPMatrix::erase);
	ClassDB::bind_method(D_METHOD("erase_multiple_unordered", "val"), &MLPPMatrix::erase_multiple_unordered);
	ClassDB::bind_method(D_METHOD("invert"), &MLPPMatrix::invert);
	ClassDB::bind_method(D_METHOD("clear"), &MLPPMatrix::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPMatrix::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPMatrix::empty);

	ClassDB::bind_method(D_METHOD("data_size"), &MLPPMatrix::data_size);
	ClassDB::bind_method(D_METHOD("resize_data", "size"), &MLPPMatrix::resize_data);

	ClassDB::bind_method(D_METHOD("get_element", "index"), &MLPPMatrix::get_element_bind);
	ClassDB::bind_method(D_METHOD("set_element", "index", "val"), &MLPPMatrix::set_element_bind);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPMatrix::fill);
	ClassDB::bind_method(D_METHOD("insert", "pos", "val"), &MLPPMatrix::insert);
	ClassDB::bind_method(D_METHOD("find", "val", "from"), &MLPPMatrix::find, 0);
	ClassDB::bind_method(D_METHOD("sort"), &MLPPMatrix::sort);
	ClassDB::bind_method(D_METHOD("ordered_insert", "val"), &MLPPMatrix::ordered_insert);

	ClassDB::bind_method(D_METHOD("to_pool_vector"), &MLPPMatrix::to_pool_vector);
	ClassDB::bind_method(D_METHOD("to_byte_array"), &MLPPMatrix::to_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPMatrix::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vector", "from"), &MLPPMatrix::set_from_mlpp_vector);
	ClassDB::bind_method(D_METHOD("set_from_pool_vector", "from"), &MLPPMatrix::set_from_pool_vector);
}
