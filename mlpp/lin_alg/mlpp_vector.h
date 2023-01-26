#ifndef MLPP_VECTOR_H
#define MLPP_VECTOR_H

#include "core/object/reference.h"

class MLPPVector : public Reference {
	GDCLASS(MLPPVector, Reference);

public:
	MLPPVector();
	~MLPPVector();

protected:
	static void _bind_methods();
};

#endif
