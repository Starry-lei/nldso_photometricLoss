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

	void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3f &K, Sophus::SO3d &Rotation, Eigen::Vector3d &Translation, Mat &depth_ref, Mat deltaMap,
	                   const double &depth_upper_bound, const double &depth_lower_bound, float *statusMap, bool *statusMapB

	) {
		ceres::Problem problem;
		double rows_ = image.rows, cols_ = image.cols;
		deltaMap.convertTo(deltaMap, CV_64FC1);

		cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
		std::vector<double> grayImage_right_values = image_right.isContinuous() ? flat : flat.clone();

		//problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new
		// Sophus::test::LocalParameterizationSE3);
		problem.AddParameterBlock(Rotation.data(), Sophus::SO3d::num_parameters);
		problem.AddParameterBlock(Translation.data(), 3);

		Eigen::Matrix3f Kinv = K.inverse();
		//		double fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);

		std::unordered_map<int, int> inliers_filter;

		//		at (240, 189)  B value is: 36  G value is: 25  R value is: 22
		//		at (233, 146)  B value is: 35  G value is: 27  R value is: 23

//        at (339, 266)  B value is: 0.225064  G value is: 0.237635  R value is: 0.180505
//        at (343, 247)  B value is: 0.247288  G value is: 0.253032  R value is: 0.196374

//        at (243, 518)  B value is: 0.0998543  G value is: 0.17071  R value is: 0.330281
//        at (245, 496)  B value is: 0.0882254  G value is: 0.151583  R value is: 0.278097

// 10 to 14

//        at (339, 266)  B value is: 0.225064  G value is: 0.237635  R value is: 0.180505
//        at (355, 229)  B value is: 0.178958  G value is: 0.182782  R value is: 0.134759


//        at (232, 122)  B value is: 0.101819  G value is: 0.0435376  R value is: 0.0233006
//        at (251, 80)  B value is: 0.0512972  G value is: 0.0257137  R value is: 0.0161647

		inliers_filter.emplace(339, 266);///(355, 229)

		int counter = 0;
		for (int u = 0; u < image.rows; u++)// colId, cols: 0 to 480
		{
			for (int v = 0; v < image.cols; v++)// rowId,  rows: 0 to 640
			{
				if (statusMap != NULL && statusMap[u * image.cols + v] != 0) {
					inliers_filter.emplace(u, v);
					counter++;
				}
			}
		}

		//  cerr << "show counter for confirmation:" << counter << endl;
		//		double intensity_ref;
		//		double deltaMap_val;
		double *Rotation_ = Rotation.data();
		double *Translation_ = Translation.data();

		Eigen::Matrix<float, 3, 3> KRKi = K * Rotation.matrix().cast<float>() * K.inverse();
		Eigen::Matrix<float, 3, 1> Kt = K * Translation.cast<float>();

		//		int step = 50;
		//		int pixelSkip = 0;

		// use pixels,depth and delta to optimize pose and depth itself
        int counter_outlier= 0;

		for (int u = 0; u < image.rows; u++)// colId, cols: 0 to 480
		{
			for (int v = 0; v < image.cols; v++)// rowId,  rows: 0 to 640
			{

				// use DSO pixel selector
				// if (statusMap!=NULL && statusMap[u*image.cols+v]==0 ){ continue;}

				// use the inlier filter
//				if (inliers_filter.count(u) == 0) { continue; }// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				if (inliers_filter[u] != v) { continue; }      // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~

				// if(pixelSkip%step!=0){ pixelSkip++;continue;}
				//----------------------current PhotoBA---------------------------
				// pixelSkip++;





				float iDepth = depth_ref.at<double>(u, v);
//				if (round(1.0 / iDepth) == 15.0) { continue; }
				if (depth_ref.at<double>(u, v) <0.0) { continue; }



				// remove way far points
				double gray_values[9]{};
				double delta_values[9]{};

				int k = 0;

				// residual size: 9
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						int rowId = u + i;
						int colId = v + j;
						if (colId > 0.0 && colId < image.cols && rowId > 0.0 && rowId < image.rows) {
							gray_values[k] = image.at<double>(rowId, colId);

							//cout<<"show gray_values:"<<gray_values[k]<<endl;
							delta_values[k] = deltaMap.at<double>(rowId, colId);
						} else {
							gray_values[k] = image.at<double>(u, v);
							delta_values[k] = deltaMap.at<double>(u, v);
						}
						k++;
					}
				}



				//cout << "show the current depth:" << depth_ref.at<double>(u, v) << endl;
				//				intensity_ref = image.at<double>(u, v);
				//				deltaMap_val = deltaMap.at<double>(u, v);
				Eigen::Vector2d pixelCoord((double) v, (double) u);
				Eigen::Matrix<float, 2, 1> pt2d;
				double newIDepth;
				//cout << "show parameter before:<<\n" << Rotation.matrix() << "," << Translation << endl;
				//if (!project((double) v, (double) u,depth_ref.at<double>(u, v), cols_, rows_, pt2d, Rotation,Translation)) { continue; }


				if (!project((float) v, (float) u, (float) depth_ref.at<double>(u, v), (int) cols_, (int) rows_, KRKi, Kt, pt2d)) {  counter_outlier+=1;
                    cout<<"show counter_outlier:"<< counter_outlier<<endl;

                    continue; }

				if (options.use_huber) {
					problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SO3d::num_parameters, 3, 1>(
					                                 new PhotometricCostFunctor(pixelCoord, K, Kinv, rows_, cols_, grayImage_right_values, gray_values, delta_values)),

					                         new ceres::HuberLoss(options.huber_parameter), Rotation_, Translation_, &depth_ref.at<double>(u, v));
				} else {
					problem.AddResidualBlock(

					        new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SO3d::num_parameters, 3, 1>(
					                new PhotometricCostFunctor(pixelCoord, K, Kinv, rows_, cols_, grayImage_right_values, gray_values, delta_values)),

					        NULL,// new ceres::HuberLoss(options.huber_parameter),
					        Rotation_, Translation_, &depth_ref.at<double>(u, v));
				}

				//				problem.SetParameterLowerBound(&depth_ref.at<double>(u,v), 0, depth_lower_bound);
				//				problem.SetParameterUpperBound(&depth_ref.at<double>(u, v), 0, depth_upper_bound);

				if (!options.optimize_pose) {
					problem.SetParameterBlockConstant(Rotation_);
					problem.SetParameterBlockConstant(Translation_);
				}
				if (options.optimize_depth) {
//					if (inliers_filter.count(u) != 0 && inliers_filter[u] == v) {
						//std::cerr<<"optimized  depth: "<< u<<","<< v<<endl;
						problem.SetParameterBlockVariable(&depth_ref.at<double>(u, v));
//					}
				} else {
					problem.SetParameterBlockConstant(&depth_ref.at<double>(u, v));
				}
			}
		}
		// Solve
		std::cout << "\n Solving ceres directBA ... " << endl;
		ceres::Solver::Options ceres_options;
		ceres_options.max_num_iterations = 300;

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

}// namespace DSONL