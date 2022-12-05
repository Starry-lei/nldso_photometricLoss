#include "deltaCompute/deltaCompute.h"

int main()
{
    Eigen::Vector3d x;
    Eigen::Vector3d y;
    x << 1, 1, -1;
    y << 0, 0, 1;
    double c = DSONL::dot(x, y);
    printf("dot = %f\n", c);
    printf("mod = %f\n", DSONL::mod(10.002, 1.0));
    printf("clamp = %f\n", DSONL::clamp(10.002, 1.0, 3));
    printf("clamp = %f\n", DSONL::clamp(0.002, 1.111, 3));
    // std::cout << DSONL::pow(0.002, x) << std::endl;
    // std::cout << DSONL::pow(y, y) << std::endl;

    std::cout << DSONL::normalize(y) << std::endl
              << std::endl;
    std::cout << DSONL::mix(x, y, 0.1) << std::endl
              << std::endl;

    std::cout << DSONL::reflect(x, y) << std::endl
              << std::endl;
    return 0;
}