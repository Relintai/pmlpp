
#ifndef MLPP_TRANSFORMS_OLD_H
#define MLPP_TRANSFORMS_OLD_H

//
//  Transforms.hpp
//
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include <string>
#include <vector>

class MLPPTransformsOld : public Reference {
	GDCLASS(MLPPTransformsOld, Reference);

public:
	std::vector<std::vector<real_t>> discreteCosineTransform(std::vector<std::vector<real_t>> A);

protected:
	static void _bind_methods();
};

#endif /* Transforms_hpp */
