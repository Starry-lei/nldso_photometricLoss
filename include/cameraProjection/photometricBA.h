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

#include <tbb/parallel_for.h>
#include <vector>
#include <mutex>

namespace DSONL {

	using namespace cv;
	using namespace std;
    typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

    // Camera intrinsics
    double _fx = 577.8705, _fy = 577.8705, _cx = 320, _cy = 240;

//    577.8705, 0, 320,
//    0, 577.8705, 240,
//    0, 0, 1;
    // useful typedefs
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;
    typedef Eigen::Matrix<double, 2, 6> Matrix26d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
/// class for accumulator jacobians in parallel
    class JacobianAccumulator {
    public:
        JacobianAccumulator(
                const cv::Mat &img1_,
                const cv::Mat &img2_,
                const VecVector2d &px_ref_,
                const vector<double> depth_ref_,
                Sophus::SE3d &T21_) :
                img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
            projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
        }

        /// accumulate jacobians in a range
        void accumulate_jacobian(const cv::Range &range);

        /// get hessian matrix
        Matrix6d hessian() const { return H; }

        /// get bias
        Vector6d bias() const { return b; }

        /// get total cost
        double cost_func() const { return cost; }

        /// get projected points
        VecVector2d projected_points() const { return projection; }

        /// reset h, b, cost to zero
        void reset() {
            H = Matrix6d::Zero();
            b = Vector6d::Zero();
            cost = 0;
        }

    private:
        const cv::Mat &img1;
        const cv::Mat &img2;
        const VecVector2d &px_ref;
        const vector<double> depth_ref;
        Sophus::SE3d &T21;
        VecVector2d projection; // projected points

        std::mutex hessian_mutex;
        Matrix6d H = Matrix6d::Zero();
        Vector6d b = Vector6d::Zero();
        double cost = 0;
    };


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


    // bilinear interpolation
    inline float GetPixelValue(const cv::Mat &img, float x, float y) {
        // boundary check
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= img.cols) x = img.cols - 1;
        if (y >= img.rows) y = img.rows - 1;
        uchar *data = &img.data[int(y) * img.step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
                (1 - xx) * (1 - yy) * data[0] +
                xx * (1 - yy) * data[1] +
                (1 - xx) * yy * data[img.step] +
                xx * yy * data[img.step + 1]
        );
    }

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21) {

    const int iterations = 60;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);;
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method time used for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;

    img2_show=img2;
//    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {

    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {

        // compute the projection in the second image
        Eigen::Vector3d point_ref =
                depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - _cx) / _fx, (px_ref[i][1] - _cy) / _fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0)   // depth invalid
            continue;

        float u = _fx * point_cur[0] / point_cur[2] + _cx, v = _fy * point_cur[1] / point_cur[2] + _cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
                Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = _fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -_fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -_fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = _fx + _fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -_fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = _fy * Z_inv;
                J_pixel_xi(1, 2) = -_fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -_fy - _fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = _fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = _fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                        0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                        0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good) {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = _fx, fyG = _fy, cxG = _cx, cyG = _cy;  // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        _fx = fxG * scales[level];
        _fy = fyG * scales[level];
        _cx = cxG * scales[level];
        _cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}

    void PhotometricBA(Mat &image, Mat &image_right, const PhotometricBAOptions &options, const Eigen::Matrix3f &K, Mat &depth_ref_map, Mat deltaMap, Mat statusMap_NonLambCand, Sophus::SE3d &T21
    ) {

        // variable
        ceres::Problem problem;
        double rows_ = image.rows, cols_ = image.cols;
        deltaMap.convertTo(deltaMap, CV_64FC1);

        cv::Mat flat = image_right.reshape(1, image_right.total() * image_right.channels());
        std::vector<double> grayImage_right_values = image_right.isContinuous() ? flat : flat.clone();

        // depth vector

        vector<double> depth_ref;
        VecVector2d pixels_ref;
        Mat AOI( image.rows, image.cols, CV_8UC1, Scalar(0));
        AOI= statusMap_NonLambCand.clone();
        // use pixels,depth and delta to optimize pose and depth itself
        int counter_outlier= 0;
        int num_points_used= 0;



        // generate pixels in ref and load depth data
//        for (int i = 0; i < nPoints; i++) {
//            int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
//            int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
//            int disparity = disparity_img.at<uchar>(y, x);
//            double depth = fx * baseline / disparity; // you know this is disparity to depth
//            depth_ref.push_back(depth);
//            pixels_ref.push_back(Eigen::Vector2d(x, y));
//        }

        int boarder = 0;

        Eigen::Matrix<float, 2, 1> pt2d;
        Eigen::Matrix<float, 3, 3> KRKi = K * T21.rotationMatrix().matrix().cast<float>() * K.inverse();
        Eigen::Matrix<float, 3, 1> Kt = K * T21.translation().cast<float>();
//        Eigen::Vector2d pixelCoord((double) v, (double) u);



        for (int u = 0+boarder; u < image.rows-boarder; u++)// colId, cols: 20 to 460
		{
			for (int v = 0+boarder; v < image.cols-boarder; v++)// rowId,  rows: 20 to 620
			{
                if (AOI.at<uchar>(u,v)!=255){ continue;}
                double depth= depth_ref_map.at<double>(u, v);
                float iDepth = 1.0 / static_cast<float>(depth);
                if (!project((float) v, (float) u, iDepth, (int) cols_, (int) rows_, KRKi, Kt, pt2d)) {  counter_outlier+=1;continue; }
                num_points_used+=1;
                cout<<"show depth val:"<<depth<<endl;
                depth_ref.push_back(depth);
                pixels_ref.push_back(Eigen::Vector2d(v, u));

            }
        }


        // estimates 01~05.png's pose using this information
        Sophus::SE3d T_cur_ref;

        DirectPoseEstimationSingleLayer(image, image_right, pixels_ref, depth_ref, T_cur_ref);

//        DirectPoseEstimationMultiLayer(image, image_right, pixels_ref, depth_ref, T_cur_ref);

        T21=T_cur_ref;


//        Show initial rotation:
//        1  1.15556e-32  1.66533e-16
//        1.15556e-32            1 -1.38778e-16
//                               -1.66533e-16  1.38778e-16            1
//        Show initial translation:
//        -0.16933
//        0.097636
//        -0.0423614

//
//        Eigen::Matrix3f Kinv = K.inverse();
//
//
//        std::unordered_map<int, int> inliers_filter;
//

//
//        imshow("AOI", AOI);
//        waitKey(0);



//        for (int u = 0; u < image.rows; u++)// colId, cols: 0 to 480
//        {
//            for (int v = 0; v < image.cols; v++)// rowId,  rows: 0 to 640
//            {
//
//                //  use interest of area bounding box here
////                 if ( (v<boundingBoxUpperLeft.val[1] || v>boundingBoxBotRight.val[1]) || (u< boundingBoxUpperLeft.val[0] ||  u> boundingBoxBotRight.val[0])){ continue;}
//
//                // use DSO pixel selector
////                 if (statusMap!=NULL && statusMap[u*image.cols+v]==0 ){ continue;}
//
//                // use non lambertian point selector
////                if (statusMap!=NULL && static_cast<int>(statusMap[u * image.cols + v])!= 255){ continue;}
//
//
//
//                // use the inlier filter
////                 if (inliers_filter.count(u) == 0) { continue; }// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
////                 if (inliers_filter[u] != v) { continue; }      // ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//
//                if (AOI.at<uchar>(u,v)!=255){ continue;}
//                num_points_used+=1;
//
//
//                // if(pixelSkip%step!=0){ pixelSkip++;continue;}
//                //----------------------current PhotoBA---------------------------
//                // pixelSkip++;
//
//
//                float iDepth = depth_ref.at<double>(u, v);
////				if (round(1.0 / iDepth) == 15.0) { continue; }
//                if (depth_ref.at<double>(u, v) <0.0) { continue; }
//
//
//
//                // remove way far points
//                double gray_values[9]{};
//                double delta_values[9]{};
//
//                int k = 0;
//
//                // residual size: 9
//                for (int i = -1; i <= 1; i++) {
//                    for (int j = -1; j <= 1; j++) {
//                        int rowId = u + i;
//                        int colId = v + j;
//                        if (colId > 0.0 && colId < image.cols && rowId > 0.0 && rowId < image.rows) {
//                            gray_values[k] = image.at<double>(rowId, colId);
//
//                            //cout<<"show gray_values:"<<gray_values[k]<<endl;
//                            delta_values[k] = deltaMap.at<double>(rowId, colId);
//                        } else {
//                            gray_values[k] = image.at<double>(u, v);
//                            delta_values[k] = deltaMap.at<double>(u, v);
//                        }
//                        k++;
//                    }
//                }
//
//
//
//                //cout << "show the current depth:" << depth_ref.at<double>(u, v) << endl;
//                //				intensity_ref = image.at<double>(u, v);
//                //				deltaMap_val = deltaMap.at<double>(u, v);
//                Eigen::Vector2d pixelCoord((double) v, (double) u);
//                Eigen::Matrix<float, 2, 1> pt2d;
//                double newIDepth;
//                //cout << "show parameter before:<<\n" << Rotation.matrix() << "," << Translation << endl;
//                //if (!project((double) v, (double) u,depth_ref.at<double>(u, v), cols_, rows_, pt2d, Rotation,Translation)) { continue; }
//
//
//                if (!project((float) v, (float) u, (float) depth_ref.at<double>(u, v), (int) cols_, (int) rows_, KRKi, Kt, pt2d)) {  counter_outlier+=1;
//
//                    continue; }
////                cout<<"show counter_outlier:"<< counter_outlier<<endl;
//
//                if (options.use_huber) {
//                    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SO3d::num_parameters, 3, 1>(
//                                                     new PhotometricCostFunctor(pixelCoord, K, Kinv, rows_, cols_, grayImage_right_values, gray_values, delta_values)),
//
//                                             new ceres::HuberLoss(options.huber_parameter), Rotation_, Translation_, &depth_ref.at<double>(u, v));
//                } else {
//                    problem.AddResidualBlock(
//
//                            new ceres::AutoDiffCostFunction<PhotometricCostFunctor, 9, Sophus::SO3d::num_parameters, 3, 1>(
//                                    new PhotometricCostFunctor(pixelCoord, K, Kinv, rows_, cols_, grayImage_right_values, gray_values, delta_values)),
//
//                            NULL,// new ceres::HuberLoss(options.huber_parameter),
//                            Rotation_, Translation_, &depth_ref.at<double>(u, v));
//                }
//
//                //				problem.SetParameterLowerBound(&depth_ref.at<double>(u,v), 0, depth_lower_bound);
//                //				problem.SetParameterUpperBound(&depth_ref.at<double>(u, v), 0, depth_upper_bound);
//
//                if (!options.optimize_pose) {
//                    problem.SetParameterBlockConstant(Rotation_);
//                    problem.SetParameterBlockConstant(Translation_);
//                }
//                if (options.optimize_depth) {
////					if (inliers_filter.count(u) != 0 && inliers_filter[u] == v) {
//                    //std::cerr<<"optimized  depth: "<< u<<","<< v<<endl;
//                    problem.SetParameterBlockVariable(&depth_ref.at<double>(u, v));
////					}
//                } else {
//                    problem.SetParameterBlockConstant(&depth_ref.at<double>(u, v));
//                }
//            }
//        }
//        // Solve
//
//
//
//        std::cout << "\n Showing number of point used in directBA: " <<num_points_used<< endl;
//        std::cout << "\n Solving ceres directBA ... " << endl;
//        ceres::Solver::Options ceres_options;
//        ceres_options.max_num_iterations = 300;
//
//        ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
//        ceres_options.num_threads = std::thread::hardware_concurrency();
//        ceres_options.minimizer_progress_to_stdout = true;
//        ceres::Solver::Summary summary;
//
//        Solve(ceres_options, &problem, &summary);
//        switch (options.verbosity_level) {
//            // 0: silent
//            case 1:
//                std::cout << summary.BriefReport() << std::endl;
//                break;
//            case 2:
//                std::cout << summary.FullReport() << std::endl;
//                break;
//        }





    }


}// namespace DSONL