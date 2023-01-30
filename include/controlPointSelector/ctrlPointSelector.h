//
// Created by lei on 15.01.23.
//

#ifndef NLDSO_PHOTOMETRICLOSS_CTRLPOINTSELECTOR_H
#define NLDSO_PHOTOMETRICLOSS_CTRLPOINTSELECTOR_H

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "sophus/se3.hpp"
#include <unordered_map>

#include "settings/common.h"






namespace DSONL{

    class ctrlPointSelector {

    public:
        ctrlPointSelector(Sophus::SE3d Camera1_extrin,string controlPointPose_path, Mat Image, Mat depthImage, Eigen::Matrix<float,3,3>& K
        , Mat pointOfInterest
        );
        ~ctrlPointSelector();
        int kNearest;
        vector<int> selectedIndex_vec;
        std::unordered_map<int, int> selectedIndex;

    };





}










#endif //NLDSO_PHOTOMETRICLOSS_CTRLPOINTSELECTOR_H
