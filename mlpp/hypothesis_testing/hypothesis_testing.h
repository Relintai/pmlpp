//
//  HypothesisTesting.hpp
//
//  Created by Marc Melikyan on 3/10/21.
//

#ifndef MLPP_HYPOTHESIS_TESTING_H
#define MLPP_HYPOTHESIS_TESTING_H

#include <vector>
#include <tuple>

namespace MLPP{
    class HypothesisTesting{
      
        public:
            std::tuple<bool, double> chiSquareTest(std::vector<double> observed, std::vector<double> expected);
        
        private:
            
    };
}

#endif /* HypothesisTesting_hpp */
