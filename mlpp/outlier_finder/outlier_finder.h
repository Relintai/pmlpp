
#ifndef MLPP_OUTLIER_FINDER_H
#define MLPP_OUTLIER_FINDER_H

//
//  OutlierFinder.hpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "core/math/math_defs.h"

#include <vector>


class MLPPOutlierFinder {
public:
	// Cnstr
	MLPPOutlierFinder(int threshold);

	std::vector<std::vector<real_t>> modelSetTest(std::vector<std::vector<real_t>> inputSet);
	std::vector<real_t> modelTest(std::vector<real_t> inputSet);

	// Variables required
	int threshold;
};


#endif /* OutlierFinder_hpp */
