
#ifndef MLPP_TRANSFORMS_H
#define MLPP_TRANSFORMS_H

//
//  Transforms.hpp
//
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>


class MLPPTransforms {
public:
	std::vector<std::vector<real_t>> discreteCosineTransform(std::vector<std::vector<real_t>> A);
};


#endif /* Transforms_hpp */
