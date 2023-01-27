//
//  OutlierFinder.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "outlier_finder.h"
#include "../stat/stat.h"
#include <iostream>


MLPPOutlierFinder::MLPPOutlierFinder(int threshold) :
		threshold(threshold) {
}

std::vector<std::vector<real_t>> MLPPOutlierFinder::modelSetTest(std::vector<std::vector<real_t>> inputSet) {
	MLPPStat  stat;
	std::vector<std::vector<real_t>> outliers;
	outliers.resize(inputSet.size());
	for (int i = 0; i < inputSet.size(); i++) {
		for (int j = 0; j < inputSet[i].size(); j++) {
			real_t z = (inputSet[i][j] - stat.mean(inputSet[i])) / stat.standardDeviation(inputSet[i]);
			if (abs(z) > threshold) {
				outliers[i].push_back(inputSet[i][j]);
			}
		}
	}
	return outliers;
}

std::vector<real_t> MLPPOutlierFinder::modelTest(std::vector<real_t> inputSet) {
	MLPPStat  stat;
	std::vector<real_t> outliers;
	for (int i = 0; i < inputSet.size(); i++) {
		real_t z = (inputSet[i] - stat.mean(inputSet)) / stat.standardDeviation(inputSet);
		if (abs(z) > threshold) {
			outliers.push_back(inputSet[i]);
		}
	}
	return outliers;
}
