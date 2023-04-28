//
// Created by lei on 28.04.23.
//

#include "types.h"
#include <iostream>

std::ostream& operator<<(std::ostream& os, const ImageSize& s)
{
    os << "[" << s.rows << "," << s.cols << "]";
    return os;
}
