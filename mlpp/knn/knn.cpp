//
//  kNN.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "knn.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <algorithm>
#include <iostream>
#include <map>


MLPPKNN::MLPPKNN(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int k) :
		inputSet(inputSet), outputSet(outputSet), k(k) {
}

std::vector<real_t> MLPPKNN::modelSetTest(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	for (int i = 0; i < X.size(); i++) {
		y_hat.push_back(modelTest(X[i]));
	}
	return y_hat;
}

int MLPPKNN::modelTest(std::vector<real_t> x) {
	return determineClass(nearestNeighbors(x));
}

real_t MLPPKNN::score() {
	MLPPUtilities   util;
	return util.performance(modelSetTest(inputSet), outputSet);
}

int MLPPKNN::determineClass(std::vector<real_t> knn) {
	std::map<int, int> class_nums;
	for (int i = 0; i < outputSet.size(); i++) {
		class_nums[outputSet[i]] = 0;
	}
	for (int i = 0; i < knn.size(); i++) {
		for (int j = 0; j < outputSet.size(); j++) {
			if (knn[i] == outputSet[j]) {
				class_nums[outputSet[j]]++;
			}
		}
	}
	int max = class_nums[outputSet[0]];
	int final_class = outputSet[0];

	for (int i = 0; i < outputSet.size(); i++) {
		if (class_nums[outputSet[i]] > max) {
			max = class_nums[outputSet[i]];
		}
	}
	for (auto [c, v] : class_nums) {
		if (v == max) {
			final_class = c;
		}
	}
	return final_class;
}

std::vector<real_t> MLPPKNN::nearestNeighbors(std::vector<real_t> x) {
	MLPPLinAlg alg;
	// The nearest neighbors
	std::vector<real_t> knn;

	std::vector<std::vector<real_t>> inputUseSet = inputSet;
	//Perfom this loop unless and until all k nearest neighbors are found, appended, and returned
	for (int i = 0; i < k; i++) {
		int neighbor = 0;
		for (int j = 0; j < inputUseSet.size(); j++) {
			bool isNeighborNearer = alg.euclideanDistance(x, inputUseSet[j]) < alg.euclideanDistance(x, inputUseSet[neighbor]);
			if (isNeighborNearer) {
				neighbor = j;
			}
		}
		knn.push_back(neighbor);
		inputUseSet.erase(inputUseSet.begin() + neighbor); // This is why we maintain an extra input"Use"Set
	}
	return knn;
}
