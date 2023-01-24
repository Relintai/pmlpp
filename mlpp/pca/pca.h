//
//  PCA.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#ifndef MLPP_PCA_H
#define MLPP_PCA_H

#include <vector>

namespace MLPP{
    class PCA{
        
        public:
            PCA(std::vector<std::vector<double>> inputSet, int k);
            std::vector<std::vector<double>> principalComponents();
            double score(); 
        private:
            std::vector<std::vector<double>> inputSet;
            std::vector<std::vector<double>> X_normalized;
            std::vector<std::vector<double>> U_reduce;
            std::vector<std::vector<double>> Z;  
            int k;
    };
}

#endif /* PCA_hpp */
