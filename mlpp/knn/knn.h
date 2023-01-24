
#ifndef MLPP_KNN_H
#define MLPP_KNN_H

//
//  kNN.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include <vector>

namespace MLPP {
class kNN {
public:
	kNN(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int k);
	std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
	int modelTest(std::vector<double> x);
	double score();

private:
	// Private Model Functions
	std::vector<double> nearestNeighbors(std::vector<double> x);
	int determineClass(std::vector<double> knn);

	// Model Inputs and Parameters
	std::vector<std::vector<double>> inputSet;
	std::vector<double> outputSet;
	int k;
};
} //namespace MLPP

#endif /* kNN_hpp */
