//
//  GaussMarkovChecker.hpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#ifndef MLPP_GAUSS_MARKOV_CHECKER_H
#define MLPP_GAUSS_MARKOV_CHECKER_H

#include <string>
#include <vector>

namespace MLPP{
    class GaussMarkovChecker{
        public:
            void checkGMConditions(std::vector<double> eps);
        
            // Independent, 3 Gauss-Markov Conditions
            bool arithmeticMean(std::vector<double> eps); // 1) Arithmetic Mean of 0.
            bool homoscedasticity(std::vector<double> eps); // 2) Homoscedasticity
            bool exogeneity(std::vector<double> eps); // 3) Cov of any 2 non-equal eps values = 0.
        private:
        
    };
}

#endif /* GaussMarkovChecker_hpp */
