
#ifndef MLPP_KNN_H
#define MLPP_KNN_H

//
//  kNN.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include <vector>


class MLPPKNN {
public:
	MLPPKNN(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int k);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	int modelTest(std::vector<real_t> x);
	real_t score();

private:
	// Private Model Functions
	std::vector<real_t> nearestNeighbors(std::vector<real_t> x);
	int determineClass(std::vector<real_t> knn);

	// Model Inputs and Parameters
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	int k;
};


#endif /* kNN_hpp */
