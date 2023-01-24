
#ifndef MLPP_TRANSFORMS_H
#define MLPP_TRANSFORMS_H

//
//  Transforms.hpp
//
//

#include <string>
#include <vector>

namespace MLPP {
class Transforms {
public:
	std::vector<std::vector<double>> discreteCosineTransform(std::vector<std::vector<double>> A);
};
} //namespace MLPP

#endif /* Transforms_hpp */
