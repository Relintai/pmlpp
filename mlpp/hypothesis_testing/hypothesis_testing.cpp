//
//  HypothesisTesting.cpp
//
//  Created by Marc Melikyan on 3/10/21.
//

#include "hypothesis_testing.h"



std::tuple<bool, real_t> MLPPHypothesisTesting::chiSquareTest(std::vector<real_t> observed, std::vector<real_t> expected) {
	real_t df = observed.size() - 1; // These are our degrees of freedom
	real_t sum = 0;
	for (int i = 0; i < observed.size(); i++) {
		sum += (observed[i] - expected[i]) * (observed[i] - expected[i]) / expected[i];
	}
}

