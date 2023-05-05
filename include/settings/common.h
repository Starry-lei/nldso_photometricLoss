//
// Created by lei on 15.01.23.
//

#ifndef NLDSO_PHOTOMETRICLOSS_COMMON_H
#define NLDSO_PHOTOMETRICLOSS_COMMON_H

#include <fstream>
#include <iomanip>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <string>

#include "sophus/se3.hpp"

using namespace std;

namespace DSONL{

    template <class T>
    struct hash3d
    {
        size_t operator()(const T &key) const
        {
            float mult = 10000.0;
            size_t hash = 137 * std::round(mult * (key.x+10.0)) + 149 * std::round(mult * (key.y+10.0)) + 163 * std::round(mult * (key.z+10.0));
            return hash;
        }
    };

    template <class T>
    struct hash2d
    {
        size_t operator()(const T &key) const
        {
            size_t hash = 137 *key.x + 149 *key.y;
            return hash;
        }
    };

    template <class T>
    struct equalTo2D
    {
        bool operator()(const T &key1, const T &key2) const
        {

            bool res= key1.x == key2.x && key1.y == key2.y;
            return res ;
        }
    };

    template <class T>
    struct equalTo
    {
        bool operator()(const T &key1, const T &key2) const
        {

            bool res= key1.x == key2.x && key1.y == key2.y && key1.z == key2.z;
//            cout<<"\n Hash res:"<< res<<endl;
            return res ;
        }
    };







}



#endif //NLDSO_PHOTOMETRICLOSS_COMMON_H
