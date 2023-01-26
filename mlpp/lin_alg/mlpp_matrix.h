#ifndef MLPP_MATRIX_H
#define MLPP_MATRIX_H

#include "core/object/reference.h"

class MLPPMatrix : public Reference {
	GDCLASS(MLPPMatrix, Reference);

public:
	MLPPMatrix();
	~MLPPMatrix();

protected:
	static void _bind_methods();
};

#endif
