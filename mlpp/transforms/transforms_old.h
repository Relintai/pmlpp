
#ifndef MLPP_TRANSFORMS_OLD_H
#define MLPP_TRANSFORMS_OLD_H

//
//  Transforms.hpp
//
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>

class MLPPTransformsOld {
public:
	std::vector<std::vector<real_t>> discreteCosineTransform(std::vector<std::vector<real_t>> A);
};

#endif /* Transforms_hpp */
