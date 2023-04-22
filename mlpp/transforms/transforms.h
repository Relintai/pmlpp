
#ifndef MLPP_TRANSFORMS_H
#define MLPP_TRANSFORMS_H

//
//  Transforms.hpp
//
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include <string>
#include <vector>

class MLPPTransforms : public Reference {
	GDCLASS(MLPPTransforms, Reference);

public:
	//std::vector<std::vector<real_t>> discreteCosineTransform(std::vector<std::vector<real_t>> A);

protected:
	static void _bind_methods();
};

#endif /* Transforms_hpp */
