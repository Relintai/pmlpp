//
//  Transforms.hpp
//
//

#ifndef MLPP_TRANSFORMS_H
#define MLPP_TRANSFORMS_H

#include <vector>
#include <string>

namespace MLPP{
    class Transforms{
        public:
            std::vector<std::vector<double>> discreteCosineTransform(std::vector<std::vector<double>> A);
            
    };
}

#endif /* Transforms_hpp */
