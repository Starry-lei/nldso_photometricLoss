//
// Created by cheng on 13.09.22.
//
#pragma once

#include "cameraProjection/reprojection.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <ceres/rotation.h>
#include <thread>

#include "utils/ultils.h"

namespace DSONL {

using namespace cv;
using namespace std;

void PhotometricBA(Mat &image, Mat &image_right,
                   const PhotometricBAOptions &options,
                   const Eigen::Matrix3f &K, Sophus::SO3d &Rotation,
                   Eigen::Vector3d &Translation, Mat &depth_ref, Mat deltaMap,
                   const double &depth_upper_bound,
                   const double &depth_lower_bound, float *statusMap,
                   bool *statusMapB

) {
  ceres::Problem problem;
  double rows_ = image.rows, cols_ = image.cols;
  deltaMap.convertTo(deltaMap, CV_64FC1);

  cv::Mat flat =
      image_right.reshape(1, image_right.total() * image_right.channels());
  std::vector<double> grayImage_right_values =
      image_right.isContinuous() ? flat : flat.clone();

  //problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new
  // Sophus::test::LocalParameterizationSE3);
  problem.AddParameterBlock(Rotation.data(), Sophus::SO3d::num_parameters);
  problem.AddParameterBlock(Translation.data(), 3);

  Eigen::Matrix3f Kinv = K.inverse();
  double fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);

  std::unordered_map<int, int> inliers_filter;
  // new image
  //	inliers_filter.emplace(366, 326); ///(352, 562)        baseline_label:
  // control experiment baseline is the smallest
  // inliers_filter.emplace(436, 127); ///(439, 338)

  inliers_filter.emplace(148, 66); ///(33, 265)

  //	at (366, 326)  B value is: 52  G value is: 111  R value is: 255
  //	at (352, 562)  B value is: 53  G value is: 109  R value is: 255

  // correspondence2

  //	at (436, 127)  B value is: 97  G value is: 106  R value is: 95
  //	at (439, 338)  B value is: 98  G value is: 109  R value is: 111

  // extrem baseline
  //	at (148, 66)  B value is: 53  G value is: 112  R value is: 255
  //	at (33, 265)  B value is: 46  G value is: 105  R value is: 255

  int counter = 0;
  for (int u = 0; u < image.rows; u++) // colId, cols: 0 to 480
  {
    for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
    {
      if (statusMap != NULL && statusMap[u * image.cols + v] != 0) {
        inliers_filter.emplace(u, v);
        counter++;
      }
    }
  }

  cerr << "show counter for confirmation:" << counter << endl;
  double intensity_ref;
  double deltaMap_val;
  double *Rotation_ = Rotation.data();
  double *Translation_ = Translation.data();

  Eigen::Matrix<float, 3, 3> KRKi =
      K * Rotation.matrix().cast<float>() * K.inverse();
  Eigen::Matrix<float, 3, 1> Kt = K * Translation.cast<float>();

  int step = 50;
  int pixelSkip = 0;

  // use pixels,depth and delta to optimize pose and depth itself
  for (int u = 0; u < image.rows; u++) // colId, cols: 0 to 480
  {
    for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
    {

      // use DSO pixel selector
      //			if (statusMap!=NULL &&
      // statusMap[u*image.cols+v]==0 ){ continue;}

      // use the inlier filter
      if (inliers_filter.count(u) == 0) {
        continue;
      } // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
      if (inliers_filter[u] != v) {
        continue;
      } // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~

      // if(pixelSkip%step!=0){ pixelSkip++;continue;
      // }///----------------------current PhotoBA---------------------------
      // pixelSkip++;

      // remove way far points
      double gray_values[9]{};
      double delta_values[9]{};

      int k = 0;

      // residual size: 9
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          int rowId = u + i;
          int colId = v + j;
          if (colId > 0.0 && colId < image.cols && rowId > 0.0 &&
              rowId < image.rows) {
            gray_values[k] = image.at<double>(rowId, colId);

            //							cout<<"show
            // gray_values:"<<gray_values[k]<<endl;
            delta_values[k] = deltaMap.at<double>(rowId, colId);
          } else {
            gray_values[k] = image.at<double>(u, v);
            delta_values[k] = deltaMap.at<double>(u, v);
          }
          k++;
        }
      }

      if (depth_ref.at<double>(u, v) < 0) {
        continue;
      }

//      cout << "show the current depth:" << depth_ref.at<double>(u, v) << endl;

      intensity_ref = image.at<double>(u, v);
      deltaMap_val = deltaMap.at<double>(u, v);
      Eigen::Vector2d pixelCoord((double)v, (double)u);

      Eigen::Matrix<float, 2, 1> pt2d;

      double newIDepth;
//      cout << "show parameter before:<<\n"
//           << Rotation.matrix() << "," << Translation << endl;

      //if (!project((double) v, (double) u,
      // depth_ref.at<double>(u, v), cols_, rows_, pt2d, Rotation,
      // Translation)) { continue; }
      if (!project((float)v, (float)u, (float)depth_ref.at<double>(u, v),
                   (int)cols_, (int)rows_, KRKi, Kt, pt2d)) {
        continue;
      }

      if (options.use_huber) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9,
                                            Sophus::SO3d::num_parameters, 3, 1>(
                new PhotometricCostFunctor(pixelCoord, K, Kinv, rows_, cols_,
                                           grayImage_right_values, gray_values,
                                           delta_values)),

            new ceres::HuberLoss(options.huber_parameter), Rotation_,
            Translation_, &depth_ref.at<double>(u, v));
      } else {
        problem.AddResidualBlock(

            new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9,
                                            Sophus::SO3d::num_parameters, 3, 1>(
                new PhotometricCostFunctor(pixelCoord, K, Kinv, rows_, cols_,
                                           grayImage_right_values, gray_values,
                                           delta_values)),

            NULL, // new ceres::HuberLoss(options.huber_parameter),
            Rotation_, Translation_, &depth_ref.at<double>(u, v));
      }

      //				problem.SetParameterLowerBound(&depth_ref.at<double>(u,
      // v), 0, depth_lower_bound);
      //				problem.SetParameterUpperBound(&depth_ref.at<double>(u,
      // v), 0, depth_upper_bound);

      if (!options.optimize_pose) {
        problem.SetParameterBlockConstant(Rotation_);
        problem.SetParameterBlockConstant(Translation_);
      }
      if (!options.optimize_depth) {
        if (inliers_filter.count(u) != 0 && inliers_filter[u] == v) {
          //					std::cerr<<"optimized  depth: "<< u<<
          //","<< v<<endl;
          problem.SetParameterBlockVariable(&depth_ref.at<double>(u, v));
        } else {
          problem.SetParameterBlockConstant(&depth_ref.at<double>(u, v));
        }
      }
    }
  }
  // Solve
  std::cout << "\n Solving ceres directBA ... " << endl;
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 600;

  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres_options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;

  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
  // 0: silent
  case 1:
    std::cout << summary.BriefReport() << std::endl;
    break;
  case 2:
    std::cout << summary.FullReport() << std::endl;
    break;
  }
}

} // namespace DSONL