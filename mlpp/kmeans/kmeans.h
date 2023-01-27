
#ifndef MLPP_K_MEANS_H
#define MLPP_K_MEANS_H

//
//  KMeans.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>


class MLPPKMeans {
public:
	MLPPKMeans(std::vector<std::vector<real_t>> inputSet, int k, std::string init_type = "Default");
	std::vector<std::vector<real_t>> modelSetTest(std::vector<std::vector<real_t>> X);
	std::vector<real_t> modelTest(std::vector<real_t> x);
	void train(int epoch_num, bool UI = 1);
	real_t score();
	std::vector<real_t> silhouette_scores();

private:
	void Evaluate();
	void computeMu();

	void centroidInitialization(int k);
	void kmeansppInitialization(int k);
	real_t Cost();

	std::vector<std::vector<real_t>> inputSet;
	std::vector<std::vector<real_t>> mu;
	std::vector<std::vector<real_t>> r;

	real_t euclideanDistance(std::vector<real_t> A, std::vector<real_t> B);

	real_t accuracy_threshold;
	int k;

	std::string init_type;
};


#endif /* KMeans_hpp */
