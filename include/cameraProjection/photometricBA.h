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
	                   const double &depth_upper_bound, const double &depth_lower_bound, float *statusMap, bool *statusMapB, Mat statusMap_NonLambCand

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


//		inliers_filter.emplace(339, 266);///(355, 229)


        Vec2i boundingBoxUpperLeft( 145,180);
        Vec2i boundingBoxBotRight(173,242);


        Vec2i boundingBoxUpperLeft_AOI2( 246,14);
        Vec2i boundingBoxBotRight_AOI2(460,163);

        Vec2i boundingBoxUpperLeft_AOI3( 217,378);
        Vec2i boundingBoxBotRight_AOI3(282,444);

        int counter = 0;

        Mat AOI( image.rows, image.cols, CV_8UC1, Scalar(0));

//		for (int u = 0; u < image.rows; u++)// colId, cols: 0 to 480
//		{
//			for (int v = 0; v < image.cols; v++)// rowId,  rows: 0 to 640
//			{
//
//                if ( (v<boundingBoxUpperLeft.val[1] || v>boundingBoxBotRight.val[1]) || (u< boundingBoxUpperLeft.val[0] ||  u> boundingBoxBotRight.val[0])){ continue;}
//                if (statusMap!=NULL && statusMap[u*image.cols+v]==0 ){ continue;}
//                inliers_filter.emplace(u, v);
//                counter++;
//                AOI.at<uchar>(u,v)=255;
//
////                if (statusMap != NULL && statusMap[u * image.cols + v] != 0) {
////					inliers_filter.emplace(u, v);
////					counter++;
////				}
//			}
//		}

//        for (int u = 0; u < image.rows; u++)// colId, cols: 0 to 480
//        {
//            for (int v = 0; v < image.cols; v++)// rowId,  rows: 0 to 640
//            {
//
//                if ( (v<boundingBoxUpperLeft_AOI2.val[1] || v>boundingBoxBotRight_AOI2.val[1]) || (u< boundingBoxUpperLeft_AOI2.val[0] ||  u> boundingBoxBotRight_AOI2.val[0])){ continue;}
//                if (statusMap!=NULL && statusMap[u*image.cols+v]==0 ){ continue;}
//                inliers_filter.emplace(u, v);
//                counter++;
//                AOI.at<uchar>(u,v)=255;
////                if (statusMap != NULL && statusMap[u * image.cols + v] != 0) {
////					inliers_filter.emplace(u, v);
////					counter++;
////				}
//            }
//        }
//
//        for (int u = 0; u < image.rows; u++)// colId, cols: 0 to 480
//        {
//            for (int v = 0; v < image.cols; v++)// rowId,  rows: 0 to 640
//            {
//
//                if ( (v<boundingBoxUpperLeft_AOI3.val[1] || v>boundingBoxBotRight_AOI3.val[1]) || (u< boundingBoxUpperLeft_AOI3.val[0] ||  u> boundingBoxBotRight_AOI3.val[0])){ continue;}
//                if (statusMap!=NULL && statusMap[u*image.cols+v]==0 ){ continue;}
//                inliers_filter.emplace(u, v);
//                counter++;
//                AOI.at<uchar>(u,v)=255;
//
////                if (statusMap != NULL && statusMap[u * image.cols + v] != 0) {
////					inliers_filter.emplace(u, v);
////					counter++;
////				}
//            }
//        }


//
//		  cerr << "show counter for used points in BA:" << counter << endl;


		//		double intensity_ref;
		//		double deltaMap_val;
		double *Rotation_ = Rotation.data();
		double *Translation_ = Translation.data();

		Eigen::Matrix<float, 3, 3> KRKi = K * Rotation.matrix().cast<float>() * K.inverse();
		Eigen::Matrix<float, 3, 1> Kt = K * Translation.cast<float>();


		// use pixels,depth and delta to optimize pose and depth itself
        int counter_outlier= 0;
        int num_points_used= 0;


        AOI= statusMap_NonLambCand.clone();

//        imshow("AOI", AOI);
//        waitKey(0);



		for (int u = 0; u < image.rows; u++)// colId, cols: 0 to 480
		{
			for (int v = 0; v < image.cols; v++)// rowId,  rows: 0 to 640
			{

                //  use interest of area bounding box here
//                 if ( (v<boundingBoxUpperLeft.val[1] || v>boundingBoxBotRight.val[1]) || (u< boundingBoxUpperLeft.val[0] ||  u> boundingBoxBotRight.val[0])){ continue;}

                // use DSO pixel selector
//                 if (statusMap!=NULL && statusMap[u*image.cols+v]==0 ){ continue;}

                // use non lambertian point selector
//                if (statusMap!=NULL && static_cast<int>(statusMap[u * image.cols + v])!= 255){ continue;}



				// use the inlier filter
//                 if (inliers_filter.count(u) == 0) { continue; }// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//                 if (inliers_filter[u] != v) { continue; }      // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~

                if (AOI.at<uchar>(u,v)!=255){ continue;}
                num_points_used+=1;


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

                    continue; }
//                cout<<"show counter_outlier:"<< counter_outlier<<endl;

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



        std::cout << "\n Showing number of point used in directBA: " <<num_points_used<< endl;
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