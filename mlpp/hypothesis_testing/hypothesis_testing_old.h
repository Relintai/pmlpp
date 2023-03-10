
#ifndef MLPP_HYPOTHESIS_TESTING_OLD_H
#define MLPP_HYPOTHESIS_TESTING_OLD_H

//
//  HypothesisTesting.hpp
//
//  Created by Marc Melikyan on 3/10/21.
//

#include "core/math/math_defs.h"
#include "core/int_types.h"

#include <tuple>
#include <vector>

class MLPPHypothesisTestingOld {
public:
	std::tuple<bool, real_t> chiSquareTest(std::vector<real_t> observed, std::vector<real_t> expected);

protected:
	static void _bind_methods();
};

#endif /* HypothesisTesting_hpp */
