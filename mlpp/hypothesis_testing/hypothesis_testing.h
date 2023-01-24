
#ifndef MLPP_HYPOTHESIS_TESTING_H
#define MLPP_HYPOTHESIS_TESTING_H

//
//  HypothesisTesting.hpp
//
//  Created by Marc Melikyan on 3/10/21.
//

#include <tuple>
#include <vector>

namespace MLPP {
class HypothesisTesting {
public:
	std::tuple<bool, double> chiSquareTest(std::vector<double> observed, std::vector<double> expected);

private:
};
} //namespace MLPP

#endif /* HypothesisTesting_hpp */
