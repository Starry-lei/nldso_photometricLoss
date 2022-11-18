//
// Created by cheng on 13.09.22.
//
#pragma once

#include <reprojection.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>
#include <thread>

#include "ultils.h"


namespace DSONL{

	using namespace cv;
	using namespace std;

	void PhotometricBA_old(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3d &K,
	                   Sophus::SE3d& pose,
					   Mat&         depth_ref,
					   Mat deltaMap,
					   const double& depth_upper_bound,
					   const double& depth_lower_bound
					   )
					   {

		ceres::Problem problem;
		double rows_= image.rows, cols_= image.cols;

		deltaMap.convertTo(deltaMap, CV_64FC1);

		cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
		std::vector<double> grayImage_right_values = image_right.isContinuous() ? flat : flat.clone();
		ceres::Grid2D<double> grid2d_grayImage_right(&grayImage_right_values[0],0, rows_, 0, cols_);



		problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);

		for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
			{
				problem.AddParameterBlock(&depth_ref.at<double>(u,v), 1);
				if (!options.optimize_depth) {
					problem.SetParameterBlockConstant(&depth_ref.at<double>(u,v));
				}


			}
		}


		std::unordered_map<int, int> inliers_filter;
		//new image
		inliers_filter.emplace(173,333); //yes
		inliers_filter.emplace(378,268); //yes


		double gray_values[1]{};
		double *transformation = pose.data();

		double depth_var;


		// use pixels,depth and delta to optimize pose and depth itself
		for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
			{
				// use the inlier filter
//				if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~


				gray_values[0] =  image.at<double>(u, v);
				Eigen::Vector2d pixelCoord((double)v,(double)u);
				problem.AddResidualBlock(
						new ceres::AutoDiffCostFunction<GetPixelGrayValue, 1, Sophus::SE3d::num_parameters, 1>(
								new GetPixelGrayValue(
								                      pixelCoord,
								                      K,
								                      image.rows,
								                      image.cols,
								                      grid2d_grayImage_right,
                                                      image,
													  deltaMap
								)
						),
						new ceres::HuberLoss(options.huber_parameter),
						transformation,
						&depth_ref.at<double>(u,v)
				);
				problem.SetParameterLowerBound(&depth_ref.at<double>(u,v), 0, depth_lower_bound);
				problem.SetParameterUpperBound(&depth_ref.at<double>(u,v), 0, depth_upper_bound);



			}
		}
		// Solve
		std::cout << "\n Solving ceres directBA ... " << endl;
		ceres::Solver::Options ceres_options;
		ceres_options.max_num_iterations = 300;

		ceres_options.linear_solver_type =ceres::SPARSE_SCHUR;
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



void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options,
				   const Eigen::Matrix3f &K,
                   Sophus::SO3d& Rotation,
				   Eigen::Vector3d& Translation,
                   Mat&         depth_ref,
                   Mat deltaMap,
                   const double& depth_upper_bound,
                   const double& depth_lower_bound,
				   float* statusMap,
				   bool* statusMapB

) {
	ceres::Problem problem;
	double rows_= image.rows, cols_= image.cols;
	deltaMap.convertTo(deltaMap, CV_64FC1);


	cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
	std::vector<double> grayImage_right_values = image_right.isContinuous() ? flat : flat.clone();



//	problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);
	problem.AddParameterBlock(Rotation.data(), Sophus::SO3d::num_parameters);
	problem.AddParameterBlock(Translation.data(), 3);

	Eigen::Matrix3f Kinv= K.inverse();
	double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy =K(1, 2);





	std::unordered_map<int, int> inliers_filter;
	//new image
//	inliers_filter.emplace(366, 326); ///(352, 562)        baseline_label: control experiment baseline is the smallest
//	inliers_filter.emplace(436, 127); ///(439, 338)

	inliers_filter.emplace(148, 66); ///(33, 265)

//	at (366, 326)  B value is: 52  G value is: 111  R value is: 255
//	at (352, 562)  B value is: 53  G value is: 109  R value is: 255

// correspondence2

//	at (436, 127)  B value is: 97  G value is: 106  R value is: 95
//	at (439, 338)  B value is: 98  G value is: 109  R value is: 111

// extrem baseline
//	at (148, 66)  B value is: 53  G value is: 112  R value is: 255
//	at (33, 265)  B value is: 46  G value is: 105  R value is: 255

	int counter=0;
	for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
		{
			if (statusMap!=NULL && statusMap[u*image.cols+v]!=0 ){
				inliers_filter.emplace(u,v);
				counter++;

			}
		}
	}

	cerr<<"show counter for confirmation:"<<counter<<endl;
			double intensity_ref;
	double deltaMap_val;
    double * Rotation_=Rotation.data();
	double * Translation_= Translation.data();


	Eigen::Matrix<float,3,3> KRKi = K *Rotation.matrix().cast<float>() * K.inverse();
	Eigen::Matrix<float,3,1> Kt = K *Translation.cast<float>();


	int step= 50;
	int pixelSkip=0;

	// use pixels,depth and delta to optimize pose and depth itself
	for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 480
	{
		for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
		{

			// use DSO pixel selector
//			if (statusMap!=NULL && statusMap[u*image.cols+v]==0 ){ continue;}

			// use the inlier filter
			if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
			if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~

			//if(pixelSkip%step!=0){ pixelSkip++;continue; }///----------------------current PhotoBA---------------------------
			//pixelSkip++;

//			cout<<"148, 66 depth"<<depth_ref.at<float>(366, 326 )<<endl;

			// remove way far points
			double gray_values[9]{};
			double delta_values[9]{};

			int k=0;

			// residual size: 9
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					int rowId=u+i;
					int colId=v+j;
					if (colId >0.0 && colId<image.cols && rowId >0.0 && rowId <image.rows ){
						gray_values[k]= image.at<double>(rowId,colId);

//							cout<<"show gray_values:"<<gray_values[k]<<endl;
						delta_values[k]=deltaMap.at<double>(rowId,colId);
					}else{
						gray_values[k]=image.at<double>(u, v);
						delta_values[k]=deltaMap.at<double>(u, v);
					}
					k++;

				}
			}


			if (depth_ref.at<double>(u,v)< 0) { continue;}

			cout<<"show the current depth:"<<depth_ref.at<double>(u,v)<<endl;

			intensity_ref=  image.at<double>(u, v);
			deltaMap_val=  deltaMap.at<double>(u, v);
			Eigen::Vector2d pixelCoord((double)v,(double)u);


//			Eigen::Matrix<double,3,1> p_3d_no_d;
//			p_3d_no_d<< (v-cx)/fx, (u-cy)/fy,(double )1.0;
//			Eigen::Matrix<double, 3,1> p_c1 ;
//			p_c1 <<  p_3d_no_d.x() /depth_ref.at<double>(u,v),  p_3d_no_d.y() /depth_ref.at<double>(u,v) ,p_3d_no_d.z() /depth_ref.at<double>(u,v);
//			Eigen::Matrix<double, 3, 1> p1 = pose * p_c1 ;
//			Eigen::Matrix<double, 2, 1> pt = project(p1,fx, fy,cx, cy);
//			if(pt.y()< 0.0 && pt.y()>image.cols && pt.x() <0.0 && pt.x()> image.rows ){ continue;}
			Eigen::Matrix<double, 2, 1> pt2d;
			double newIDepth;
			cout<<"show parameter before:<<\n"<<Rotation.matrix()<<","<<Translation<<endl;

			if (!project( (double)v,(double)u, depth_ref.at<double>(u,v),cols_,rows_,pt2d,Rotation,Translation)){ continue;}
//			if (!project(float (v),float (u), float(depth_ref.at<double>(u,v)),cols_,rows_,KRKi,Kt,pt2d,newIDepth)){ continue;}

			if (options.use_huber){
				problem.AddResidualBlock(
						new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SO3d::num_parameters, 3, 1>(
								new PhotometricCostFunctor(
										pixelCoord,
										K,
										Kinv,
										rows_,
										cols_,
										grayImage_right_values,
										gray_values,
										delta_values
								)
						),

//						new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 16, Sophus::SE3d::num_parameters,1>(
//								new PhotometricCostFunctor(
//										pixelCoord,
//										K,
//										image.rows,
//										image.cols,
//										grayImage_right_values,
//										gray_values,
//										delta_values
//								)
//						),
						new ceres::HuberLoss(options.huber_parameter),
						Rotation_,
						Translation_,
						&depth_ref.at<double>(u,v)
				);
			} else{
				problem.AddResidualBlock(

						new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SO3d::num_parameters, 3, 1>(
								new PhotometricCostFunctor(
										pixelCoord,
										K,
										Kinv,
										rows_,
										cols_,
										grayImage_right_values,
										gray_values,
										delta_values
								)
						),

						NULL, //new ceres::HuberLoss(options.huber_parameter),
						Rotation_,
						Translation_,
						&depth_ref.at<double>(u,v)
				);
			}


			problem.SetParameterLowerBound(&depth_ref.at<double>(u,v), 0,   depth_lower_bound);
			problem.SetParameterUpperBound(&depth_ref.at<double>(u,v), 0,   depth_upper_bound);
			if (!options.optimize_pose){
				problem.SetParameterBlockConstant(Rotation_);
				problem.SetParameterBlockConstant(Translation_);
			}
			if (!options.optimize_depth) {
				if(   inliers_filter.count(u)!=0 &&inliers_filter[u]==v ){
//					std::cerr<<"optimized  depth: "<< u<< ","<< v<<endl;
					problem.SetParameterBlockVariable(&depth_ref.at<double>(u,v));
				}else{
					problem.SetParameterBlockConstant(&depth_ref.at<double>(u,v));
				}

			}
		}
	}
	// Solve
	std::cout << "\n Solving ceres directBA ... " << endl;
	ceres::Solver::Options ceres_options;
	ceres_options.max_num_iterations = 600;

	ceres_options.linear_solver_type =ceres::SPARSE_SCHUR;
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


	//  ----------------------------------------------overload PhotometricBA-------------------------------------------------
//	void PhotometricBA
//					(Mat &image, Mat &image_right,
//					 const PhotometricBAOptions &options,
//					 const Eigen::Matrix3f &K,
//	                   Sophus::SE3d& pose,
//	                   Mat&       depth_ref,
//	                   Mat deltaMap,
//	                   const double& depth_upper_bound,
//	                   const double& depth_lower_bound,
//					   const Mat& outlier_mask
//
//	) {
//		ceres::Problem problem;
//		double rows_= image.rows, cols_= image.cols;
//		deltaMap.convertTo(deltaMap, CV_64FC1);
//
//
//
//
////		cv::Mat flat_depth_map = img_ref_depth.reshape(1, img_ref_depth.total() * img_ref_depth.channels());
////		std::vector<double> img_ref_depth_values=img_ref_depth.isContinuous() ? flat_depth_map : flat_depth_map.clone();
////		ceres::Grid2D<double> grid2d_depth(&img_ref_depth_values[0],0, rows_, 0, cols_);
////		ceres::BiCubicInterpolator<ceres::Grid2D<double>> interpolator_depth(grid2d_depth);
//
//
//
//		cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
//		std::vector<double> grayImage_right_values = image_right.isContinuous() ? flat : flat.clone();
//
//
//		problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, new Sophus::test::LocalParameterizationSE3);
//
//
//
//		std::unordered_map<int, int> inliers_filter;
//		//new image
//		inliers_filter.emplace(321,296); //yes
//
//		inliers_filter.emplace(102,136);
//		inliers_filter.emplace(241,33);
//		inliers_filter.emplace(340,107);
//		inliers_filter.emplace(242,6);
//
//		inliers_filter.emplace(113,94);
//		inliers_filter.emplace(393,37);
//		inliers_filter.emplace(112,93);
//		inliers_filter.emplace(255,564);
//
//
//
//
//
//
//		double intensity_ref;
//		double deltaMap_val;
//		double *transformation = pose.data();
//
//
//		int step= 100;
//		int pixelSkip=0;
//
//		// use pixels,depth and delta to optimize pose and depth itself
//		for (int u = 0; u< image.rows; u++) // colId, cols: 0 to 480
//		{
//			for (int v = 0; v < image.cols; v++) // rowId,  rows: 0 to 640
//			{
//				// use the inlier filter
////				if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
////				if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
////				if(pixelSkip%step!=0){ pixelSkip++;continue; }
////				pixelSkip++;
//
//
//                // red outlier mask
//				if (outlier_mask.at<uchar>(u,v)==0){ continue;}
//				// remove way far points
//				if (depth_ref.at<double>(u,v)< 0) { continue;}
//
//
//
//
//				intensity_ref=  image.at<double>(u, v);
//				double gray_values[9]{};
//				double delta_values[9]{};
//
////				gray_values[0]=intensity_ref;
////				gray_values[1]=image.at<double>(u, v);
//
//				int k=0;
//				for (int i = -1; i <= 1; i++)
//				{
//					for (int j = -1; j <= 1; j++)
//					{
//						int rowId=u+i;
//						int colId=v+j;
//						if (colId >0.0 && colId<image.cols && rowId >0.0 && rowId <image.rows ){
//							gray_values[k]= image.at<double>(rowId,colId);
//
////							cout<<"show gray_values:"<<gray_values[k]<<endl;
//							delta_values[k]=deltaMap.at<double>(rowId,colId);
//						}else{
//							gray_values[k]=image.at<double>(u, v);
//							delta_values[k]=deltaMap.at<double>(u, v);
//						}
//						k++;
//
//					}
//				}
//
////int a=0;
//
//
//
//
//
//				deltaMap_val=  deltaMap.at<double>(u, v);
//				Eigen::Vector2d pixelCoord((double)v,(double)u);
//
//				double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy =K(1, 2);
//				Eigen::Matrix<double,3,1> p_3d_no_d;
//				p_3d_no_d<< (v-cx)/fx, (u-cy)/fy,(double )1.0;
//				Eigen::Matrix<double, 3,1> p_c1 ;
//				double cur_detph=depth_ref.at<double>(u,v);
//				p_c1 <<  p_3d_no_d.x() /depth_ref.at<double>(u,v),  p_3d_no_d.y() /depth_ref.at<double>(u,v) ,p_3d_no_d.z() /depth_ref.at<double>(u,v);
//				Eigen::Matrix<double, 3, 1> p1 = pose * p_c1 ;
//				Eigen::Matrix<double, 2, 1> pt = project(p1,fx, fy,cx, cy);
//				if(pt.y()< 0.0 && pt.y()>image.cols && pt.x() <0.0 && pt.x()> image.rows ){
//					cout<<"show outside points:"<<pt.x()<<","<<pt.y()<<endl;
//				 int a=0;
//					continue;
//				}
//
//
//				if (options.use_huber){
//					problem.AddResidualBlock(
//							new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SE3d::num_parameters,1>(
//									new PhotometricCostFunctor(
//											pixelCoord,
//											K,
//											image.rows,
//											image.cols,
//											grayImage_right_values,
//											gray_values,
//											delta_values
//									)
//							),
//							new ceres::HuberLoss(options.huber_parameter),
//							transformation,
//							&depth_ref.at<double>(u,v)
//					);
//				} else{
//					problem.AddResidualBlock(
//							new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SE3d::num_parameters,1>(
//									new PhotometricCostFunctor(
//											pixelCoord,
//											K,
//											image.rows,
//											image.cols,
//											grayImage_right_values,
//											gray_values,
//											delta_values
//									)
//							),
//							NULL, //new ceres::HuberLoss(options.huber_parameter),
//							transformation,
//							&depth_ref.at<double>(u,v)
//					);
//				}
//
//
//				problem.SetParameterLowerBound(&depth_ref.at<double>(u,v), 0,   depth_lower_bound);
//				problem.SetParameterUpperBound(&depth_ref.at<double>(u,v), 0,   depth_upper_bound);
//				if (!options.optimize_pose){
//					problem.SetParameterBlockConstant(transformation);
//				}
//				if (!options.optimize_depth) {
//					problem.SetParameterBlockConstant(&depth_ref.at<double>(u,v));
//				}
//			}
//		}
//		// Solve
//		std::cout << "\n Solving ceres directBA ... " << endl;
//		ceres::Solver::Options ceres_options;
//		ceres_options.max_num_iterations = 600;
//
//		ceres_options.linear_solver_type =ceres::SPARSE_SCHUR;
//		ceres_options.num_threads = std::thread::hardware_concurrency();
//		ceres_options.minimizer_progress_to_stdout = true;
//		ceres::Solver::Summary summary;
//
//		Solve(ceres_options, &problem, &summary);
//		switch (options.verbosity_level) {
//			// 0: silent
//			case 1:
//				std::cout << summary.BriefReport() << std::endl;
//				break;
//			case 2:
//				std::cout << summary.FullReport() << std::endl;
//				break;
//		}
//
//
//	}





}