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
#include "pba_cost_functor.h"

#include "utils/ultils.h"

namespace DSONL {

	using namespace cv;
	using namespace std;


    vector<double> vectorizeImage(Mat &inputMat){
        std::vector<double> image_1_vectorized;

        if (inputMat.type()==CV_32F){
            for (int r = 0; r < inputMat.rows; r++){
                for (int c = 0; c < inputMat.cols; c++){
                    image_1_vectorized.push_back(static_cast<double>(inputMat.at<float>(r, c)));
                }
            }
        }else if (inputMat.type()==CV_8U){
            for (int r = 0; r < inputMat.rows; r++){
                for (int c = 0; c < inputMat.cols; c++){
                    image_1_vectorized.push_back(static_cast<double>(inputMat.at<uchar>(r, c)));
                }
            }
        }
        return image_1_vectorized;

    }



    void pbaRelativePose(float huberPara, Mat &image_left,float* statusMapPoints_ref,Mat &idepth_1_float, Mat &image_right, float* statusMapPoints_tar, const Eigen::Matrix3d &K, double* camera_poses, std::vector<cv::Point3f>& points3D){
        // construct image patches
        ceres::Problem problem;

        // intrinsics  // 574.540648625183
        float fx = K(0,0);
        float fy = K(1,1);
        float cx =K(0,2);
        float cy =K(1,2);

        ceres::LossFunction* loss_function = new ceres::HuberLoss(huberPara);
        std::vector<double> image_1_vectorized;
        std::vector<double> image_2_vectorized;
        // vectorize images
        image_1_vectorized=vectorizeImage(image_left);
        image_2_vectorized=vectorizeImage(image_right);

        Mat image_1 = image_left.clone();
        Mat image_2 = image_right.clone();

        size_t num_points = image_1.rows * image_1.cols;
        double depth_array[num_points];

        std::unique_ptr<ceres::Grid2D<double, 1>> image_grid;
        std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_interpolation;

        image_grid.reset(new ceres::Grid2D<double, 1>(&image_1_vectorized[0], 0, image_1.rows, 0, image_1.cols));
        compute_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> >(*image_grid));

//    std::vector<cv::Point3f> points3D;

        std::unordered_map<int, int> inliers_filter;
        inliers_filter.emplace(250,250);


        for (int r = 0; r < image_1.rows; r++){
            for (int c = 0; c < image_1.cols; c++){

//                 if (inliers_filter.count(r) == 0) { continue; }// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//                 if (inliers_filter[r] != c) { continue; }      // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~


                // use DSO pixel selector
                 if (statusMapPoints_ref!=NULL && statusMapPoints_ref[r*image_1.cols+c]==0 ){ continue;}

                    std::vector<double> patch(PATTERN_SIZE, 0.0);
                    std::vector<double> patch_weigts(PATTERN_SIZE, 1.0);

                for (size_t i = 0; i < PATTERN_SIZE; i++){
                    int du = PATTERN_OFFSETS[i][0];
                    int dv = PATTERN_OFFSETS[i][1];
                    float u_new = c + du;
                    float v_new = r + dv;
                    compute_interpolation->Evaluate(v_new, u_new, &patch[i]);


                }
                depth_array[r*idepth_1_float.cols + c] = idepth_1_float.at<float>(r, c);

                float x_norm = (c - cx) / fx;
                float y_norm = (r - cy) / fy;
                ceres::CostFunction* cost_fun = PhotometricBundleAdjustment_vanilla::Create(image_2_vectorized,
                                                                                    image_2.cols, image_2.rows,
                                                                                    patch, x_norm, y_norm,
                                                                                    fx, fy, cx, cy);
                problem.AddResidualBlock(cost_fun, loss_function, &(camera_poses[7]), &(depth_array[r*idepth_1_float.cols + c]));
                // optimize depth
                problem.SetParameterBlockConstant(&(depth_array[r*idepth_1_float.cols + c]));



//            ceres::CostFunction* cost_fun_orig = PhotometricBundleAdjustment::Create(image_1_vectorized,
//                                                                                image_1.cols, image_1.rows,
//                                                                                patch, x_norm, y_norm,
//                                                                                fx, fy, cx, cy);
//            problem.AddResidualBlock(cost_fun_orig, loss_function, &(camera_poses[0]), &(depth_array[r*idepth_1_float.cols + c]));


                //problem.SetParameterBlockConstant(&(depth_array[r*idepth_1_float.cols + c]));
                points3D.push_back(cv::Point3f(x_norm / idepth_1_float.at<float>(r, c), y_norm / idepth_1_float.at<float>(r, c), 1.0 / idepth_1_float.at<float>(r, c)));
            }
        }

        ceres::LocalParameterization* camera_parameterization = new ceres::ProductParameterization(new ceres::QuaternionParameterization(),
                                                                                                   new ceres::IdentityParameterization(3));
//    for (size_t i = 0; i < num_cameras; i++) {
//        problem.SetParameterization(&(camera_poses[7*i]), camera_parameterization);
//    }

        problem.SetParameterization(&(camera_poses[7]), camera_parameterization);
        problem.SetParameterBlockVariable(&(camera_poses[7]));


//    problem.SetParameterBlockConstant(&(camera_poses[0]));


// output the residuals distribution

        vector<double> residuals;
        double cost;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);
        std::cout<<"\n Initial cost:"<<cost<<endl;
        std::cout << "\n Residuals size: " << residuals.size() << std::endl;

        if (image_left.cols==640){
            drawResidualDistribution(residuals,"residualsDistri_withoutCorrection", image_1.rows,image_1.cols);
        }










        ceres::Solver::Options options;

        options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.num_threads = std::thread::hardware_concurrency();

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;




    }

    void pbaRelativePose(float huberPara, Mat& W_specularity,  Mat &image_left,float* statusMapPoints_ref,Mat &idepth_1_float, Mat &image_right, float* statusMapPoints_tar, const Eigen::Matrix3d &K, double* camera_poses, std::vector<cv::Point3f>& points3D){
        // construct image patches
        ceres::Problem problem;

        // intrinsics  // 574.540648625183
        float fx = K(0,0);
        float fy = K(1,1);
        float cx =K(0,2);
        float cy =K(1,2);

        ceres::LossFunction* loss_function = new ceres::HuberLoss(huberPara);
        std::vector<double> image_1_vectorized;
        std::vector<double> image_1_weight_vectorized;
        std::vector<double> image_2_vectorized;
        // vectorize images
        image_1_vectorized=vectorizeImage(image_left);
        image_1_weight_vectorized=vectorizeImage(W_specularity);

        image_2_vectorized=vectorizeImage(image_right);

        Mat image_1 = image_left.clone();
        Mat image_2 = image_right.clone();

        size_t num_points = image_1.rows * image_1.cols;
        double depth_array[num_points];

        std::unique_ptr<ceres::Grid2D<double, 1>> image_grid;
        std::unique_ptr<ceres::Grid2D<double, 1>> image_weight_grid;

        std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_interpolation;
        std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_weight_interpolation;

        image_grid.reset(new ceres::Grid2D<double, 1>(&image_1_vectorized[0], 0, image_1.rows, 0, image_1.cols));
        image_weight_grid.reset(new ceres::Grid2D<double, 1>(&image_1_weight_vectorized[0], 0, image_1.rows, 0, image_1.cols));
        compute_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> >(*image_grid));
        compute_weight_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double,1> >(*image_weight_grid));

//    std::vector<cv::Point3f> points3D;

        std::unordered_map<int, int> inliers_filter;
        inliers_filter.emplace(250,250);


        for (int r = 0; r < image_1.rows; r++){
            for (int c = 0; c < image_1.cols; c++){

//                 if (inliers_filter.count(r) == 0) { continue; }// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//                 if (inliers_filter[r] != c) { continue; }      // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~

                // use DSO pixel selector
//                if (statusMapPoints_ref!=NULL && statusMapPoints_ref[r*image_1.cols+c]==0 ){ continue;}
                if (W_specularity.at<float>(r,c)==0.0f ){continue;}// for enlarge the performance gap



                std::vector<double> patch(PATTERN_SIZE, 0.0);
                std::vector<double> patch_weight(PATTERN_SIZE, 1.0);
                for (size_t i = 0; i < PATTERN_SIZE; i++){
                    int du = PATTERN_OFFSETS[i][0];
                    int dv = PATTERN_OFFSETS[i][1];
                    float u_new = c + du;
                    float v_new = r + dv;
                    compute_interpolation->Evaluate(v_new, u_new, &patch[i]);
                    // mark here
//                    compute_weight_interpolation->Evaluate(v_new, u_new, &patch_weight[i]);

                    if(u_new > 1.1 && u_new < image_1.cols - 2.1 && v_new > 1.1 && v_new < image_1.rows - 2.1){
                        patch_weight[i]=W_specularity.at<float>(r, c);
//                        cout<<"show patch_weight[i]:"<<patch_weight[i]<<endl;
                    }
                }
                depth_array[r*idepth_1_float.cols + c] = idepth_1_float.at<float>(r, c);

                float x_norm = (c - cx) / fx;
                float y_norm = (r - cy) / fy;
                ceres::CostFunction* cost_fun = PhotometricBundleAdjustment::Create(image_2_vectorized,
                                                                                    image_2.cols, image_2.rows,
                                                                                    patch, patch_weight, x_norm, y_norm,
                                                                                    fx, fy, cx, cy);
                problem.AddResidualBlock(cost_fun, loss_function, &(camera_poses[7]), &(depth_array[r*idepth_1_float.cols + c]));
                // optimize depth
                problem.SetParameterBlockConstant(&(depth_array[r*idepth_1_float.cols + c]));



//            ceres::CostFunction* cost_fun_orig = PhotometricBundleAdjustment::Create(image_1_vectorized,
//                                                                                image_1.cols, image_1.rows,
//                                                                                patch, x_norm, y_norm,
//                                                                                fx, fy, cx, cy);
//            problem.AddResidualBlock(cost_fun_orig, loss_function, &(camera_poses[0]), &(depth_array[r*idepth_1_float.cols + c]));


                //problem.SetParameterBlockConstant(&(depth_array[r*idepth_1_float.cols + c]));
                points3D.push_back(cv::Point3f(x_norm / idepth_1_float.at<float>(r, c), y_norm / idepth_1_float.at<float>(r, c), 1.0 / idepth_1_float.at<float>(r, c)));
            }
        }

        ceres::LocalParameterization* camera_parameterization = new ceres::ProductParameterization(new ceres::QuaternionParameterization(),
                                                                                                   new ceres::IdentityParameterization(3));
//    for (size_t i = 0; i < num_cameras; i++) {
//        problem.SetParameterization(&(camera_poses[7*i]), camera_parameterization);
//    }

        problem.SetParameterization(&(camera_poses[7]), camera_parameterization);
        problem.SetParameterBlockVariable(&(camera_poses[7]));


        // problem.SetParameterBlockConstant(&(camera_poses[0]));
        // output the residuals

        vector<double> residuals;
        double cost;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);
        std::cout<<"\n Initial cost:"<<cost<<endl;
        std::cout << "\n Residuals size: " << residuals.size() << std::endl;


        if (image_left.cols==640){
            drawResidualDistribution(residuals,"residualsDistri_withCorrection", image_1.rows,image_1.cols);
        }







        //    drawResidualPerPixels(residuals, 5.0,"residuals", image_1.cols, image_1.rows);
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        //    options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.num_threads = std::thread::hardware_concurrency();

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;




    }

    void pbaRelativePose(float huberPara, Mat &image_left,Mat &idepth_1_float, Mat &image_right, const Eigen::Matrix3d &K, double* camera_poses, std::vector<cv::Point3f>& points3D){
        // construct image patches
        ceres::Problem problem;

        // intrinsics  // 574.540648625183
        float fx = K(0,0);
        float fy = K(1,1);
        float cx =K(0,2);
        float cy =K(1,2);



        ceres::LossFunction* loss_function = new ceres::HuberLoss(huberPara);
        std::vector<double> image_1_vectorized;
        std::vector<double> image_2_vectorized;
        // vectorize images
        image_1_vectorized=vectorizeImage(image_left);
        image_2_vectorized=vectorizeImage(image_right);

        Mat image_1 = image_left.clone();
        Mat image_2 = image_right.clone();

        size_t num_points = image_1.rows * image_1.cols;
        double depth_array[num_points];

        std::unique_ptr<ceres::Grid2D<double, 1>> image_grid;
        std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_interpolation;

        image_grid.reset(new ceres::Grid2D<double, 1>(&image_1_vectorized[0], 0, image_1.rows, 0, image_1.cols));
        compute_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> >(*image_grid));

//    std::vector<cv::Point3f> points3D;

        std::unordered_map<int, int> inliers_filter;
        inliers_filter.emplace(250,250);


        for (int r = 0; r < image_1.rows; r++){
            for (int c = 0; c < image_1.cols; c++){

//                 if (inliers_filter.count(r) == 0) { continue; }// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//                 if (inliers_filter[r] != c) { continue; }      // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
                std::vector<double> patch(PATTERN_SIZE, 0.0);
                for (size_t i = 0; i < PATTERN_SIZE; i++){
                    int du = PATTERN_OFFSETS[i][0];
                    int dv = PATTERN_OFFSETS[i][1];
                    float u_new = c + du;
                    float v_new = r + dv;
                    compute_interpolation->Evaluate(v_new, u_new, &patch[i]);
                }
                depth_array[r*idepth_1_float.cols + c] = idepth_1_float.at<float>(r, c);

                float x_norm = (c - cx) / fx;
                float y_norm = (r - cy) / fy;
                ceres::CostFunction* cost_fun = PhotometricBundleAdjustment_vanilla::Create(image_2_vectorized,
                                                                                    image_2.cols, image_2.rows,
                                                                                    patch, x_norm, y_norm,
                                                                                    fx, fy, cx, cy);
                problem.AddResidualBlock(cost_fun, loss_function, &(camera_poses[7]), &(depth_array[r*idepth_1_float.cols + c]));
                // optimize depth
                problem.SetParameterBlockConstant(&(depth_array[r*idepth_1_float.cols + c]));



//            ceres::CostFunction* cost_fun_orig = PhotometricBundleAdjustment::Create(image_1_vectorized,
//                                                                                image_1.cols, image_1.rows,
//                                                                                patch, x_norm, y_norm,
//                                                                                fx, fy, cx, cy);
//            problem.AddResidualBlock(cost_fun_orig, loss_function, &(camera_poses[0]), &(depth_array[r*idepth_1_float.cols + c]));


                //problem.SetParameterBlockConstant(&(depth_array[r*idepth_1_float.cols + c]));
                points3D.push_back(cv::Point3f(x_norm / idepth_1_float.at<float>(r, c), y_norm / idepth_1_float.at<float>(r, c), 1.0 / idepth_1_float.at<float>(r, c)));
            }
        }

        ceres::LocalParameterization* camera_parameterization = new ceres::ProductParameterization(new ceres::QuaternionParameterization(),
                                                                                                   new ceres::IdentityParameterization(3));
//    for (size_t i = 0; i < num_cameras; i++) {
//        problem.SetParameterization(&(camera_poses[7*i]), camera_parameterization);
//    }

        problem.SetParameterization(&(camera_poses[7]), camera_parameterization);
        problem.SetParameterBlockVariable(&(camera_poses[7]));


//    problem.SetParameterBlockConstant(&(camera_poses[0]));


// output the residuals

        vector<double> residuals;
        double cost;
        problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);
        std::cout<<"\n Initial cost:"<<cost<<endl;
        std::cout << "\n Residuals size: " << residuals.size() << std::endl;

//    drawResidualPerPixels(residuals, 5.0,"residuals", image_1.cols, image_1.rows);



        ceres::Solver::Options options;

        options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.num_threads = std::thread::hardware_concurrency();

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;




    }






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