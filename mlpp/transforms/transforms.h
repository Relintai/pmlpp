
#ifndef MLPP_TRANSFORMS_H
#define MLPP_TRANSFORMS_H

//
//  Transforms.hpp
//
//

#include <string>
#include <vector>


class MLPPTransforms {
public:
	std::vector<std::vector<double>> discreteCosineTransform(std::vector<std::vector<double>> A);
};


#endif /* Transforms_hpp */
