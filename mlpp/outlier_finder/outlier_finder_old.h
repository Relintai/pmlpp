
#ifndef MLPP_OUTLIER_FINDER_OLD_H
#define MLPP_OUTLIER_FINDER_OLD_H

//
//  OutlierFinder.hpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "core/math/math_defs.h"
#include "core/int_types.h"

#include <vector>


class MLPPOutlierFinderOld {
public:
	// Cnstr
	MLPPOutlierFinderOld(int threshold);

	std::vector<std::vector<real_t>> modelSetTest(std::vector<std::vector<real_t>> inputSet);
	std::vector<real_t> modelTest(std::vector<real_t> inputSet);

	// Variables required
	int threshold;
};


#endif /* OutlierFinder_hpp */
