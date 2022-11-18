#ifndef PCLOpt_H
#define PCLOpt_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/mls.h>
//#include <omp.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <pcl/search/search.h>

using namespace cv;

template <typename T> class NormalEstimator {

  using Mat3T = Eigen::Matrix<T, 3, 3>;

  // problem parameters
  int width_, height_;
  Mat3T K_;
  cv::Size2i window_size_;
  // values needed for normal computation
  Mat_<T> x0_, y0_;
  Mat_<T> x0_n_sq_inv_, y0_n_sq_inv_, n_sq_inv_;
  Mat_<T> Q11_, Q12_, Q13_, Q22_, Q23_, Q33_;

  // similar to Matlab's meshgrid function
  template <typename U>
  void pixelgrid(U shift_x, U shift_y, Mat_<U> &u, Mat_<U> &v) {

    std::vector<U> row, col;
    for (int k = 0; k < width_; ++k)
      row.push_back(k - shift_x);
    for (int k = 0; k < height_; ++k)
      col.push_back(k - shift_y);
    Mat Row(row), Col(col);
    repeat(Row.reshape(1, 1), Col.total(), 1, u);
    repeat(Col.reshape(1, 1).t(), 1, Row.total(), v);
  }

  // compute values needed for normal estimation only once
  void cache() {

    Eigen::Matrix<double, 3, 3> K;
    K = K_.template cast<double>();

    const double fx_inv = 1. / K(0, 0);
    const double fy_inv = 1. / K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);

    Mat_<double> x0, y0, x0_sq, y0_sq, x0_y0, n_sq;
    Mat_<double> x0_n_sq_inv, y0_n_sq_inv, n_sq_inv;
    Mat_<double> M11, M12, M13, M22, M23, M33, det, det_inv;
    Mat_<double> Q11, Q12, Q13, Q22, Q23, Q33;

    pixelgrid<double>(cx, cy, x0, y0);

    x0 = fx_inv * x0;
    x0_sq = x0.mul(x0);
    y0 = fy_inv * y0;
    y0_sq = y0.mul(y0);
    x0_y0 = x0.mul(y0);

    n_sq = 1. + x0_sq + y0_sq;
    divide(1., n_sq, n_sq_inv);
    x0_n_sq_inv = x0.mul(n_sq_inv);
    y0_n_sq_inv = y0.mul(n_sq_inv);

    boxFilter(x0_sq.mul(n_sq_inv), M11, -1, window_size_, Point(-1, -1), false);
    boxFilter(x0_y0.mul(n_sq_inv), M12, -1, window_size_, Point(-1, -1), false);
    boxFilter(x0_n_sq_inv, M13, -1, window_size_, Point(-1, -1), false);
    boxFilter(y0_sq.mul(n_sq_inv), M22, -1, window_size_, Point(-1, -1), false);
    boxFilter(y0_n_sq_inv, M23, -1, window_size_, Point(-1, -1), false);
    boxFilter(n_sq_inv, M33, -1, window_size_, Point(-1, -1), false);

    det =
        M11.mul(M22.mul(M33)) + 2 * M12.mul(M23.mul(M13)) -
        (M13.mul(M13.mul(M22)) + M12.mul(M12.mul(M33)) + M23.mul(M23.mul(M11)));
    divide(1., det, det_inv);

    Q11 = det_inv.mul(M22.mul(M33) - M23.mul(M23));
    Q12 = det_inv.mul(M13.mul(M23) - M12.mul(M33));
    Q13 = det_inv.mul(M12.mul(M23) - M13.mul(M22));
    Q22 = det_inv.mul(M11.mul(M33) - M13.mul(M13));
    Q23 = det_inv.mul(M12.mul(M13) - M11.mul(M23));
    Q33 = det_inv.mul(M11.mul(M22) - M12.mul(M12));

    // TODO: write in more concise way!!
    if (std::is_same<T, float>::value) {
      x0.convertTo(x0_, CV_32F);
      y0.convertTo(y0_, CV_32F);
      x0_n_sq_inv.convertTo(x0_n_sq_inv_, CV_32F);
      y0_n_sq_inv.convertTo(y0_n_sq_inv_, CV_32F);
      n_sq_inv.convertTo(n_sq_inv_, CV_32F);
      Q11.convertTo(Q11_, CV_32F);
      Q12.convertTo(Q12_, CV_32F);
      Q13.convertTo(Q13_, CV_32F);
      Q22.convertTo(Q22_, CV_32F);
      Q23.convertTo(Q23_, CV_32F);
      Q33.convertTo(Q33_, CV_32F);
    } else {
      x0_ = x0;
      y0_ = y0;
      x0_n_sq_inv_ = x0_n_sq_inv;
      y0_n_sq_inv_ = y0_n_sq_inv;
      n_sq_inv_ = n_sq_inv;
      Q11_ = Q11;
      Q12_ = Q12;
      Q13_ = Q13;
      Q22_ = Q22;
      Q23_ = Q23;
      Q33_ = Q33;
    }
  }

public:
  NormalEstimator(int width, int height, Eigen::Matrix<float, 3, 3> K,
                  Size window_size)
      : width_(width), height_(height), K_(K.cast<T>()),
        window_size_(window_size) {
    cache();
  }

  NormalEstimator(int width, int height, Eigen::Matrix<double, 3, 3> K,
                  Size window_size)
      : width_(width), height_(height), K_(K.cast<T>()),
        window_size_(window_size) {
    cache();
  }

  ~NormalEstimator() {}

  // compute normals
  void compute(const Mat &depth, Mat &nx, Mat &ny, Mat &nz) const {

    // workaround to only divide by depth where it is non-zero
    // not needed for OpenCV versions <4
    Mat_<T> tmp;
    divide(1., depth, tmp);
    Mat z_inv = Mat::zeros(tmp.size(), tmp.type());
    Mat mask = (depth != 0);
    tmp.copyTo(z_inv, mask);

    Mat_<T> b1, b2, b3, norm_n;

    boxFilter(x0_n_sq_inv_.mul(z_inv), b1, -1, window_size_, Point(-1, -1),
              false);
    boxFilter(y0_n_sq_inv_.mul(z_inv), b2, -1, window_size_, Point(-1, -1),
              false);
    boxFilter(n_sq_inv_.mul(z_inv), b3, -1, window_size_, Point(-1, -1), false);

    nx = b1.mul(Q11_) + b2.mul(Q12_) + b3.mul(Q13_);
    ny = b1.mul(Q12_) + b2.mul(Q22_) + b3.mul(Q23_);
    nz = b1.mul(Q13_) + b2.mul(Q23_) + b3.mul(Q33_);

    sqrt(nx.mul(nx) + ny.mul(ny) + nz.mul(nz), norm_n);

    divide(nx, norm_n, nx);
    divide(ny, norm_n, ny);
    divide(nz, norm_n, nz);
  }

  Mat *x0_ptr() { return &x0_; }

  Mat *y0_ptr() { return &y0_; }

  Mat *n_sq_inv_ptr() { return &n_sq_inv_; }
};

void comp_accurate_normals(std::vector<Eigen::Vector3d> cloud_eigen,
                           cv::Mat &init_normal_map);

void resamplePts_and_compNormal(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered,
    std::vector<Eigen::Vector3d> cloud_eigen, cv::Mat &init_normal_map);

int iterate_all_pts(int n_eigen_current,
                    const std::vector<Eigen::Vector3d> &cloud_eigen,
                    Eigen::Vector3d pt_mls, bool &is_match);

using namespace cv;
using namespace std;

class normalMap_MLS {
private:
  // member function to pad the image before convolution
  Mat padding(Mat img, int k_width, int k_height, string type) {
    Mat scr;
    img.convertTo(scr, CV_64FC1);
    int pad_rows, pad_cols;
    pad_rows = (k_height - 1) / 2;
    pad_cols = (k_width - 1) / 2;
    Mat pad_image(Size(scr.cols + 2 * pad_cols, scr.rows + 2 * pad_rows),
                  CV_64FC1, Scalar(0));
    scr.copyTo(pad_image(Rect(pad_cols, pad_rows, scr.cols, scr.rows)));
    // mirror padding
    if (type == "mirror") {
      for (int i = 0; i < pad_rows; i++) {
        scr(Rect(0, pad_rows - i, scr.cols, 1))
            .copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
        scr(Rect(0, (scr.rows - 1) - pad_rows + i, scr.cols, 1))
            .copyTo(pad_image(
                Rect(pad_cols, (pad_image.rows - 1) - i, scr.cols, 1)));
      }

      for (int j = 0; j < pad_cols; j++) {
        pad_image(Rect(2 * pad_cols - j, 0, 1, pad_image.rows))
            .copyTo(pad_image(Rect(j, 0, 1, pad_image.rows)));
        pad_image(
            Rect((pad_image.cols - 1) - 2 * pad_cols + j, 0, 1, pad_image.rows))
            .copyTo(pad_image(
                Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
      }

      return pad_image;
    }
    // replicate padding
    else if (type == "replicate") {
      for (int i = 0; i < pad_rows; i++) {
        scr(Rect(0, 0, scr.cols, 1))
            .copyTo(pad_image(Rect(pad_cols, i, scr.cols, 1)));
        scr(Rect(0, (scr.rows - 1), scr.cols, 1))
            .copyTo(pad_image(
                Rect(pad_cols, (pad_image.rows - 1) - i, scr.cols, 1)));
      }

      for (int j = 0; j < pad_cols; j++) {
        pad_image(Rect(pad_cols, 0, 1, pad_image.rows))
            .copyTo(pad_image(Rect(j, 0, 1, pad_image.rows)));
        pad_image(Rect((pad_image.cols - 1) - pad_cols, 0, 1, pad_image.rows))
            .copyTo(pad_image(
                Rect((pad_image.cols - 1) - j, 0, 1, pad_image.rows)));
      }
      // zero padding
      return pad_image;
    } else {
      return pad_image;
    }
  }

  // member function to define kernels for convolution
  Mat define_kernel(int k_width, int k_height, string type) {
    // box kernel
    if (type == "box") {
      Mat kernel(k_height, k_width, CV_64FC1,
                 Scalar(1.0 / (k_width * k_height)));
      return kernel;
    }
    // gaussian kernel
    else if (type == "gaussian") {
      // I will assume k = 1 and sigma = 1
      int pad_rows = (k_height - 1) / 2;
      int pad_cols = (k_width - 1) / 2;
      Mat kernel(k_height, k_width, CV_64FC1);
      for (int i = -pad_rows; i <= pad_rows; i++) {
        for (int j = -pad_cols; j <= pad_cols; j++) {
          kernel.at<double>(i + pad_rows, j + pad_cols) =
              exp(-(i * i + j * j) / 2.0);
        }
      }

      kernel = kernel / sum(kernel).val[0];
      return kernel;
    }
  }

public:
  void storeNormal(pcl::PointCloud<pcl::Normal>::Ptr cloud_mls_normal,
                   int pointIdx, Mat &dst, int u, int v) {

    double n_x = (*cloud_mls_normal)[pointIdx].normal_x;
    double n_y = (*cloud_mls_normal)[pointIdx].normal_y;
    double n_z = (*cloud_mls_normal)[pointIdx].normal_z;

    Vec3f normal_new(n_x, n_y, n_z);
    normal_new = normalize(normal_new);

    Vec3f principal_axis(0, 0, 1);
    if (normal_new.dot(principal_axis) < 0) {
      normal_new = -normal_new;
    }
    dst.at<Vec3f>(u, v)[0] = normal_new.val[2];
    dst.at<Vec3f>(u, v)[1] = normal_new.val[1];
    dst.at<Vec3f>(u, v)[2] = normal_new.val[0];
  }

  void convolve(Mat &src, Mat &dst, int &k_w, int &k_h, string &paddingType,
                const Eigen::Matrix<double, 3, 3> &K_) {
    double fx = K_(0, 0), cx = K_(0, 2), fy = K_(1, 1), cy = K_(1, 2);
    Mat pad_img, kernel;
    pad_img = padding(src, k_w, k_h, paddingType);
    //		kernel = define_kernel(k_w, k_h, filterType);

    //		Mat output = Mat::zeros(scr.size(), CV_64FC1);
    int pad_rows, pad_cols;
    pad_rows = (k_h - 1) / 2;
    pad_cols = (k_w - 1) / 2;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointNormal> mls_points;
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    //		pcl::MovingLeastSquaresOMP<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setComputeNormals(true);

    // scr is the depth map
    pcl::PointXYZ currentPoint;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    for (int u = 0; u < src.rows; u += 2) {
      for (int v = 0; v < src.cols; v += 2) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_patch(
            new pcl::PointCloud<pcl::PointXYZ>());

        for (int i = u - pad_rows; i <= u + pad_rows; i++) // row
        {
          for (int j = v - pad_cols; j <= v + pad_cols; j++) // col
          {

            if (i >= 0 && i <= src.rows && j >= 0 && j <= src.cols) {
              float depth_val = src.at<float>(i, j);
              Vec3f point(depth_val * Vec3f((j - cx) / fx, (i - cy) / fy, 1.0));
              if (i == u && j == v) {
                currentPoint.x = point.val[0];
                currentPoint.y = point.val[1];
                currentPoint.z = point.val[2];
              }

              cloud_patch->push_back(
                  pcl::PointXYZ(point.val[0], point.val[1], point.val[2]));
            }
            //						Vec3f point(p_3d_no_d_path.at<Vec3f>(i + pad_rows, j +
            //pad_cols).val[0], p_3d_no_d_path.at<Vec3f>(i + pad_rows, j +
            //pad_cols).val[1], p_3d_no_d_path.at<Vec3f>(i + pad_rows, j +
            //pad_cols).val[2]);
          }
        }

        mls.setInputCloud(cloud_patch);
        mls.setPolynomialOrder(2);
        mls.setSearchMethod(tree);
        //				mls.setNumberOfThreads(4);
        mls.setSearchRadius(1.0); // 307200 pts  tune the parameter
                                  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        mls.process(mls_points);

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_mls(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::Normal>::Ptr cloud_mls_normal(
            new pcl::PointCloud<pcl::Normal>());
        for (int nIndex = 0; nIndex < mls_points.size(); ++nIndex) {
          cloud_mls->push_back(pcl::PointXYZ(mls_points.points[nIndex].x,
                                             mls_points.points[nIndex].y,
                                             mls_points.points[nIndex].z));
          cloud_mls_normal->push_back(
              pcl::Normal(mls_points.points[nIndex].normal_x,
                          mls_points.points[nIndex].normal_y,
                          mls_points.points[nIndex].normal_z));
        }
        if (cloud_mls->size() == 0 || isnan(cloud_mls->points[0].x)) {
          continue;
        }
        kdtree_.setInputCloud(cloud_mls);
        int K = 1;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);

        if (kdtree_.nearestKSearch(currentPoint, K, pointIdxKNNSearch,
                                   pointKNNSquaredDistance) > 0) {

          //					double n_x = (*cloud_mls_normal)[
          //pointIdxKNNSearch[0] ].normal_x; 					double n_y = (*cloud_mls_normal)[
          //pointIdxKNNSearch[0] ].normal_y; 					double n_z =(*cloud_mls_normal)[
          //pointIdxKNNSearch[0] ].normal_z;
          //
          //					Vec3f normal_new(n_x, n_y, n_z);
          //					normal_new =
          //normalize(normal_new);
          //
          //					Vec3f principal_axis(0, 0, 1);
          //					if(normal_new.dot(principal_axis)<0)
          //					{
          //						normal_new =
          //-normal_new;
          //					}
          //					dst.at<Vec3f>(u, v)[0] =
          //normal_new.val[2]; 					dst.at<Vec3f>(u, v)[1] = normal_new.val[1];
          //					dst.at<Vec3f>(u, v)[2] =
          //normal_new.val[0];
          int index_ = pointIdxKNNSearch[0];
          storeNormal(cloud_mls_normal, index_, dst, u, v);
          storeNormal(cloud_mls_normal, index_ + 1, dst, u, v + 1);
          //					if
          //(v+1<=640){storeNormal(cloud_mls_normal,index_+1,dst,u,v+1);} 					if
          //(v+2<=640){storeNormal(cloud_mls_normal,index_+2,dst,u,v+2 );}

          //					storeNormal(cloud_mls_normal,index_+3,dst,u,v+3
          //);

          storeNormal(cloud_mls_normal, index_ + 7, dst, u + 1, v);
          storeNormal(cloud_mls_normal, index_ + 8, dst, u + 1, v + 1);
          //					if (v+1<=640)
          //{storeNormal(cloud_mls_normal,index_+8,dst,u+1,v+1);} 					if (v+1<=640)
          //{storeNormal(cloud_mls_normal,index_+9,dst,u+1,v+2);}
          //					storeNormal(cloud_mls_normal,index_+10,dst,u+1,v+3);

          //					storeNormal(cloud_mls_normal,index_+14,dst,u+1,v);
          //					storeNormal(cloud_mls_normal,index_+15,dst,u+1,v+1);
          //					if (v+1<=640)
          //{storeNormal(cloud_mls_normal,index_+15,dst,u+1,v+1);} 					if (v+1<=640)
          //{storeNormal(cloud_mls_normal,index_+16,dst,u+1,v+2);}
          //					storeNormal(cloud_mls_normal,index_+17,dst,u+1,v+3);
          //
          //					storeNormal(cloud_mls_normal,index_+21,dst,u+1,v);
          //					storeNormal(cloud_mls_normal,index_+22,dst,u+1,v+1);
          //					storeNormal(cloud_mls_normal,index_+23,dst,u+1,v+2);
          //					storeNormal(cloud_mls_normal,index_+24,dst,u+1,v+3);
        }
      }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    cout << "Reconstruct normals: " << time_used.count() << " seconds." << endl;

    //		   output.convertTo(dst, CV_8UC1);
    //		dst=output;
  }
};

#endif // PCLOpt_H
