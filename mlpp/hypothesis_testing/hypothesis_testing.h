
#ifndef MLPP_HYPOTHESIS_TESTING_H
#define MLPP_HYPOTHESIS_TESTING_H

//
//  HypothesisTesting.hpp
//
//  Created by Marc Melikyan on 3/10/21.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include <tuple>
#include <vector>

class MLPPHypothesisTesting : public Reference {
	GDCLASS(MLPPHypothesisTesting, Reference);

public:
	std::tuple<bool, real_t> chiSquareTest(std::vector<real_t> observed, std::vector<real_t> expected);

protected:
	static void _bind_methods();
};

#endif /* HypothesisTesting_hpp */
