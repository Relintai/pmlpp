//
//  OutlierFinder.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "outlier_finder_old.h"

#include "../stat/stat.h"
#include <iostream>


MLPPOutlierFinderOld::MLPPOutlierFinderOld(int threshold) :
		threshold(threshold) {
}

std::vector<std::vector<real_t>> MLPPOutlierFinderOld::modelSetTest(std::vector<std::vector<real_t>> inputSet) {
	MLPPStat  stat;
	std::vector<std::vector<real_t>> outliers;
	outliers.resize(inputSet.size());
	for (uint32_t i = 0; i < inputSet.size(); i++) {
		for (uint32_t j = 0; j < inputSet[i].size(); j++) {
			real_t z = (inputSet[i][j] - stat.mean(inputSet[i])) / stat.standardDeviation(inputSet[i]);
			if (abs(z) > threshold) {
				outliers[i].push_back(inputSet[i][j]);
			}
		}
	}
	return outliers;
}

std::vector<real_t> MLPPOutlierFinderOld::modelTest(std::vector<real_t> inputSet) {
	MLPPStat  stat;
	std::vector<real_t> outliers;
	for (uint32_t i = 0; i < inputSet.size(); i++) {
		real_t z = (inputSet[i] - stat.mean(inputSet)) / stat.standardDeviation(inputSet);
		if (abs(z) > threshold) {
			outliers.push_back(inputSet[i]);
		}
	}
	return outliers;
}
