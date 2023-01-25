#ifndef MLPP_TEST_H
#define MLPP_TEST_H

// TODO port this class to use the test module once it's working
// Also don't forget to remove it's bindings

#include "core/object/reference.h"

class MLPPTest : public Reference {
	GDCLASS(MLPPTest, Reference);

public:
	MLPPTest();
	~MLPPTest();

protected:
	static void _bind_methods();
};

#endif
