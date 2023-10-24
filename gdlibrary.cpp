#include "core/pandemonium_global.h"



extern "C" void GDN_EXPORT pandemonium_gdnative_init(pandemonium_gdnative_init_options *o) {
	Pandemonium::gdnative_init(o);
}

extern "C" void GDN_EXPORT pandemonium_gdnative_terminate(pandemonium_gdnative_terminate_options *o) {
	Pandemonium::gdnative_terminate(o);
	
}

extern "C" void GDN_EXPORT pandemonium_nativescript_init(void *handle) {
	Pandemonium::nativescript_init(handle);
}
