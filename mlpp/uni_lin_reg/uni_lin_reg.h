
#ifndef MLPP_UNI_LIN_REG_H
#define MLPP_UNI_LIN_REG_H

//
//  UniLinReg.hpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include <vector>

namespace MLPP{
    class UniLinReg{
        
        public:
            UniLinReg(std::vector <double> x, std::vector<double> y);
            std::vector<double> modelSetTest(std::vector<double> x);
            double modelTest(double x);
        
        private:
            std::vector <double> inputSet;
            std::vector <double> outputSet;
        
            double b0;
            double b1;
        
    };
}

#endif /* UniLinReg_hpp */
