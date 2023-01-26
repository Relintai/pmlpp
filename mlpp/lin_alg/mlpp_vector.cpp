
#include "mlpp_vector.h"

void MLPPVector::_bind_methods() {
	ClassDB::bind_method(D_METHOD("push_back", "elem"), &MLPPVector::push_back);
	ClassDB::bind_method(D_METHOD("remove", "index"), &MLPPVector::remove);
	ClassDB::bind_method(D_METHOD("remove_unordered", "index"), &MLPPVector::remove_unordered);
	ClassDB::bind_method(D_METHOD("erase", "val"), &MLPPVector::erase);
	ClassDB::bind_method(D_METHOD("erase_multiple_unordered", "val"), &MLPPVector::erase_multiple_unordered);
	ClassDB::bind_method(D_METHOD("invert"), &MLPPVector::invert);
	ClassDB::bind_method(D_METHOD("clear"), &MLPPVector::clear);
	ClassDB::bind_method(D_METHOD("reset"), &MLPPVector::reset);
	ClassDB::bind_method(D_METHOD("empty"), &MLPPVector::empty);

	ClassDB::bind_method(D_METHOD("size"), &MLPPVector::size);

	ClassDB::bind_method(D_METHOD("get_element", "index"), &MLPPVector::get_element_bind);
	ClassDB::bind_method(D_METHOD("set_element", "index", "val"), &MLPPVector::set_element_bind);

	ClassDB::bind_method(D_METHOD("fill", "val"), &MLPPVector::fill);
	ClassDB::bind_method(D_METHOD("insert", "pos", "val"), &MLPPVector::insert);
	ClassDB::bind_method(D_METHOD("find", "val", "from"), &MLPPVector::find, 0);
	ClassDB::bind_method(D_METHOD("sort"), &MLPPVector::sort);
	ClassDB::bind_method(D_METHOD("ordered_insert", "val"), &MLPPVector::ordered_insert);

	ClassDB::bind_method(D_METHOD("to_pool_vector"), &MLPPVector::to_pool_vector);
	ClassDB::bind_method(D_METHOD("to_byte_array"), &MLPPVector::to_byte_array);

	ClassDB::bind_method(D_METHOD("duplicate"), &MLPPVector::duplicate);

	ClassDB::bind_method(D_METHOD("set_from_mlpp_vector", "from"), &MLPPVector::set_from_mlpp_vector);
	ClassDB::bind_method(D_METHOD("set_from_pool_vector", "from"), &MLPPVector::set_from_pool_vector);
}
