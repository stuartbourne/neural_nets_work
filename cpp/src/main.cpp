#include <iostream>
#include <Neuron.hpp>
#include <Eigen/Dense>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(int argc, char** argv){
    //First will directly emulate the initial python implementation, integrate eigen, matplotlibcpp, etc.
    //Then will restructure
    Eigen::MatrixXd m(2, 2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    plt::plot({1,3,2,4});
    plt::show();
    
    return 0;
}
