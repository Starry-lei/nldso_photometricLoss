//
// Created by cheng on 13.09.22.
//
#pragma once


//#include "settings/common.h"
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <cmath>
#include <pangolin/pangolin.h>
#include <matplot/matplot.h>
#include "gnuplot-iostream/gnuplot-iostream.h"


//#include <ceres/ceres.h>
//#include <ceres/cubic_interpolation.h>
//#include <ceres/loss_function.h>

//#include <omp.h>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/surface/mls.h>

#include <pcl/io/io.h>
#include <pcl/point_types.h>

//#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/filters/radius_outlier_removal.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/features/normal_3d.h>


namespace DSONL {

    using namespace cv;
    using namespace std;
    const double DEG_TO_ARC = 0.0174532925199433;

    void showPointCloud(const Sophus::SE3d inputPoseGT,const Sophus::SE3d inputPose,const Sophus::SE3d pose1,const Sophus::SE3d pose2,const Sophus::SE3d pose, const std::vector<cv::Point3f> points3D,
                        const float fx, const float fy, const float cx, const float cy) {
        Sophus::SE3d worldFrame = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0,0,0));

        pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
        );

        pangolin::View &d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        while (pangolin::ShouldQuit() == false) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            glPointSize(2);
            glBegin(GL_POINTS);

            for (auto &p: points3D) {
                glColor3f(0.0, 0.0, 0.0);
                glVertex3d(p.x, p.y, p.z);
            }

            glEnd();


            // 绘制坐标系
            glLineWidth(3);
            glBegin (GL_LINES);

            // axis x
            glColor3f ( 0.8f,0.f,0.f );
            glVertex3f( 0,0, 0 );
            glVertex3f( 1,0, 0 );
            // axis y
            glColor3f( 0.f,0.8f,0.f);
            glVertex3f( 0,0, 0 );
            glVertex3f( 0,1,0 );
            // axis z
            glColor3f( 0.2f,0.2f,1.f);
            glVertex3f( 0,0, 0 );
            glVertex3f( 0,0,1);

            glEnd();

            // draw poses
            float sz = 0.1;
            int width = 640, height = 480;

            {
                glPushMatrix();
                Sophus::Matrix4f m = inputPoseGT.inverse().matrix().cast<float>();
                glMultMatrixf((GLfloat *) m.data());
                glColor3f(0, 1, 1);
                glLineWidth(2);

                glLineWidth(3);
                glBegin (GL_LINES);

                // axis x
                glColor3f ( 0.8f,0.f,0.f );
                glVertex3f( 0,0, 0 );
                glVertex3f( 1,0, 0 );
                // axis y
                glColor3f( 0.f,0.8f,0.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,1,0 );
                // axis z
                glColor3f( 0.2f,0.2f,1.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,0,1);

                glEnd();

                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glEnd();
                glPopMatrix();
            }


            {
                glPushMatrix();
                Sophus::Matrix4f m = inputPose.inverse().matrix().cast<float>();
                glMultMatrixf((GLfloat *) m.data());
                glColor3f(1, 1, 0);
                glLineWidth(2);

                glLineWidth(3);
                glBegin (GL_LINES);

                // axis x
                glColor3f ( 0.8f,0.f,0.f );
                glVertex3f( 0,0, 0 );
                glVertex3f( 1,0, 0 );
                // axis y
                glColor3f( 0.f,0.8f,0.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,1,0 );
                // axis z
                glColor3f( 0.2f,0.2f,1.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,0,1);

                glEnd();

                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glEnd();
                glPopMatrix();
            }


            {
                glPushMatrix();
                Sophus::Matrix4f m = pose1.inverse().matrix().cast<float>();
                glMultMatrixf((GLfloat *) m.data());
                glColor3f(0, 1, 0);
                glLineWidth(2);

                glLineWidth(3);
                glBegin (GL_LINES);

                // axis x
                glColor3f ( 0.8f,0.f,0.f );
                glVertex3f( 0,0, 0 );
                glVertex3f( 1,0, 0 );
                // axis y
                glColor3f( 0.f,0.8f,0.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,1,0 );
                // axis z
                glColor3f( 0.2f,0.2f,1.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,0,1);

                glEnd();

                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glEnd();
                glPopMatrix();
            }


            {
                glPushMatrix();
                Sophus::Matrix4f m = pose2.inverse().matrix().cast<float>();
                glMultMatrixf((GLfloat *) m.data());
                glColor3f(1, 0, 0);
                glLineWidth(2);

                glLineWidth(3);
                glBegin (GL_LINES);

                // axis x
                glColor3f ( 0.8f,0.f,0.f );
                glVertex3f( 0,0, 0 );
                glVertex3f( 1,0, 0 );
                // axis y
                glColor3f( 0.f,0.8f,0.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,1,0 );
                // axis z
                glColor3f( 0.2f,0.2f,1.f);
                glVertex3f( 0,0, 0 );
                glVertex3f( 0,0,1);

                glEnd();


                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glEnd();
                glPopMatrix();
            }



            {
                glPushMatrix();
                Sophus::Matrix4f m = pose.inverse().matrix().cast<float>();
                glMultMatrixf((GLfloat *) m.data());
                glColor3f(1, 0, 0);
                glLineWidth(2);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glEnd();
                glPopMatrix();
            }

            {
                glPushMatrix();
                Sophus::Matrix4f m = worldFrame.inverse().matrix().cast<float>();
                glMultMatrixf((GLfloat *) m.data());
                glColor3f(0, 1, 0);
                glLineWidth(3);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(0, 0, 0);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
                glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
                glEnd();
                glPopMatrix();
            }

            pangolin::FinishFrame();
            usleep(5000);   // sleep 5 ms
        }
        return;
    }




    void readUV(string blhTxtName, Mat& coordMask )
    {
        ifstream infile(blhTxtName);
        if (!infile)
        {
            cout << "error!" << endl;
            return;
        }
//        double x, y, z;
//        while (infile>>x>>y>>z)
//        {
//            cout << x << " " << y << " " << z << endl;
//        }
//        infile.close();
        int u, v;
        while (infile>>u>>v)
        {
            coordMask.at<uchar>(u,v)=255;
//            cout <<"check vals of coord:"<< u<< "," << v << endl;
        }
        infile.close();

    }

    bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean = 0.0, double StdDev = 10.0) {
        if (mSrc.empty()) {
            cout << "[Error]! Input Image Empty!";
            return 0;
        }

        Mat mSrc_32FC1;
        Mat mGaussian_noise = Mat(mSrc.size(), CV_64FC1);
        randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));

        mSrc.convertTo(mSrc_32FC1, CV_64FC1);
        addWeighted(mSrc_32FC1, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_32FC1);
        mSrc_32FC1.convertTo(mDst, mSrc.type());

        return true;
    }

    bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst, double Mean, double StdDev, float *statusMap) {
        if (mSrc.empty()) {
            cout << "[Error]! Input Image Empty!";
            return 0;
        }

        Mat mSrc_32FC1;
        Mat mGaussian_noise = Mat(mSrc.size(), CV_64FC1);
        randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));
        mSrc.convertTo(mSrc_32FC1, CV_64FC1);
        for (int u = 0; u < mSrc.rows; u++)// colId, cols: 0 to 480
        {
            for (int v = 0; v < mSrc.cols; v++)// rowId,  rows: 0 to 640
            {
                if (statusMap != NULL && statusMap[u * mSrc.cols + v] != 0) {
                    mSrc_32FC1.at<double>(u, v) += mGaussian_noise.at<double>(u, v);
                }
            }
        }
        //		addWeighted(mSrc_32FC1, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_32FC1);
        mSrc_32FC1.convertTo(mDst, mSrc.type());

        return true;
    }


    // show image function
    void imageInfo(Mat im_, Eigen::Vector2i &position) {

        //position,  fst is rowIdx, snd is colIdx
        Mat im = im_;
        int img_depth = im.depth();
        switch (img_depth) {
            case 0:
                cout << "The data type of current image is CV_8U. \n"
                     << endl;
                break;
            case 1:
                cout << "The data type of current image is CV_8S. \n"
                     << endl;
                break;
            case 2:
                cout << "The data type of current image is CV_16U. \n"
                     << endl;
                break;
            case 3:
                cout << "The data type of current image is CV_16S. \n"
                     << endl;
                break;
            case 4:
                cout << "The data type of current image is CV_32S. \n"
                     << endl;
                break;
            case 5:
                cout << "The data type of current image is CV_32F. \n"
                     << endl;
                break;
            case 6:
                cout << "The data type of current image is CV_64F. \n"
                     << endl;
                break;
            case 7:
                cout << "The data type of current image is CV_USRTYPE1. \n"
                     << endl;
                break;
        }

        cout << "\n show Image depth:\n"
             << im.depth() << "\n show Image channels :\n " << im.channels() << endl;
        imshow("Image", im);

        double min_v, max_v;
        cv::minMaxLoc(im, &min_v, &max_v);
        cout << "\n show Image min, max:\n"
             << min_v << "," << max_v << endl;

        //		double fstChannel, sndChannel,thrChannel;

        if (static_cast<int>(im.channels()) == 1) {
            //若为灰度图，显示鼠标点击的坐标以及灰度值
            cout << "at(" << position.x() << "," << position.y() << ")value is:"
                 << static_cast<float>(im.at<float>(position.x(), position.y())) << endl;
        } else if (static_cast<int>(im.channels() == 3)) {
            //若图像为彩色图像，则显示鼠标点击坐标以及对应的B, G, R值
            cout << "at (" << position.x() << ", " << position.y() << ")"
                 << "  R value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[2])
                 << "  G value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[1])
                 << "  B value is: " << static_cast<float>(im.at<Vec3f>(position.x(), position.y())[0])
                 << endl;
        }

        waitKey(0);
    }


    template<typename T>
    T rotationErr(Eigen::Matrix<T, 3, 3> rotation_gt, Eigen::Matrix<T, 3, 3> rotation_rs) {


        T compare1 = max(acos(std::min(std::max(rotation_gt.col(0).dot(rotation_rs.col(0)), -1.0), 1.0)),
                         acos(std::min(std::max(rotation_gt.col(1).dot(rotation_rs.col(1)), -1.0), 1.0)));

        return max(compare1, acos(std::min(std::max(rotation_gt.col(2).dot(rotation_rs.col(2)), -1.0), 1.0))) * 180.0 /
               M_PI;
    }

    template<typename T>
    T translationErr(Eigen::Matrix<T, 3, 1> translation_gt, Eigen::Matrix<T, 3, 1> translation_es) {
        if (translation_gt.norm() == 0){ return (translation_es).norm(); }
        return (translation_gt - translation_es).norm() / translation_gt.norm();
    }


    Scalar depthErr(const Mat &depth_gt, const Mat &depth_es) {

        if (depth_es.depth() != depth_gt.depth()) { std::cerr << "the depth image type are different!" << endl; }
        return cv::sum(cv::abs(depth_gt - depth_es)) / cv::sum(depth_gt);
    }

    template<typename T>
    Eigen::Matrix<T, 3, 3> rotation_pertabation(const T pertabation_x, const T pertabation_y, const T pertabation_z,
                                                const Eigen::Matrix<T, 3, 3> &Rotation, double &roErr) {

        Eigen::Matrix<T, 3, 3> R;

//		T roll = pertabation_x / 180.0 * M_PI;
//		T yaw = pertabation_y / 180.0 * M_PI;
//		T pitch = pertabation_z / 180.0 * M_PI;

        T roll = pertabation_z / 180.0 * M_PI;
        T yaw = pertabation_y / 180.0 * M_PI;
        T pitch = pertabation_x / 180.0 * M_PI;


        Eigen::AngleAxis<T> rollAngle(roll, Eigen::Matrix<T, 3, 1>::UnitZ());
        Eigen::AngleAxis<T> yawAngle(yaw, Eigen::Matrix<T, 3, 1>::UnitY());
        Eigen::AngleAxis<T> pitchAngle(pitch, Eigen::Matrix<T, 3, 1>::UnitX());

        Eigen::Quaternion<T> q = rollAngle * yawAngle * pitchAngle;

        R = q.matrix();
        Eigen::Matrix<T, 3, 3> updatedRotation;
        updatedRotation.setZero();
        updatedRotation = R * Rotation;
        //		cout<<" ----------------R------------:"<< R<< endl;
        //		cout<<" ----------------Eigen::Matrix<T,3,1>::UnitX()-----------:"<< Eigen::Matrix<T,3,1>::UnitX()<< endl;
        //		cout<<"Show the rotation loss:"<<updatedRotation<< endl;
        roErr = rotationErr(Rotation, updatedRotation);

        return updatedRotation;
    }

    template<typename T>
    Eigen::Matrix<T, 3, 1> translation_pertabation(const T pertabation_x, const T pertabation_y, const T pertabation_z,
                                                   const Eigen::Matrix<T, 3, 1> &translation, double &roErr) {

        Eigen::Matrix<T, 3, 1> updated_translation;
        updated_translation.setZero();
        updated_translation.x() = translation.x() * (1.0 + pertabation_x);
        updated_translation.y() = translation.y() * (1.0 + pertabation_y);
        updated_translation.z() = translation.z() * (1.0 + pertabation_z);
        roErr = translationErr(translation, updated_translation);
        return updated_translation;
    }

    template<typename T>
    Sophus::SE3<T> posePerturbation(Eigen::Matrix<T, 6, 1> se3, const Sophus::SE3<T> &pose_GT, double &roErr,
                                    const Eigen::Matrix<T, 3, 3> &Rotation, double &trErr,
                                    const Eigen::Matrix<T, 3, 1> &translation) {
        Sophus::SE3<T> SE3_updated = Sophus::SE3<T>::exp(se3) * pose_GT;
        trErr = translationErr(translation, SE3_updated.translation());
        roErr = rotationErr(Rotation, SE3_updated.rotationMatrix());
        return SE3_updated;
    }

    void downscale(Mat &image, const Mat &depth, Eigen::Matrix3f &K, int &level, Mat &image_d, Mat &depth_d,
                   Eigen::Matrix3f &K_d) {

        if (level <= 1) {
            image_d = image;
            // remove negative gray values
            image_d = cv::max(image_d, 0.0);
            depth_d = depth;
            // set all nan zero
            Mat mask = Mat(depth_d != depth_d);
            depth_d.setTo(0.0, mask);
            K_d = K;
            return;
        }

        // downscale camera intrinsics

        K_d << K(0, 0) / 2.0, 0, (K(0, 2) + 0.5) / 2.0 - 0.5,
                0, K(1, 1) / 2.0, (K(1, 2) + 0.5) / 2 - 0.5,
                0, 0, 1;
        pyrDown(image, image_d, Size(image.cols / 2, image.rows / 2));
        pyrDown(depth, depth_d, Size(depth.cols / 2, depth.rows / 2));
        // remove negative gray values
        image_d = cv::max(image_d, 0.0);
        // set all nan zero
        Mat mask = Mat(depth_d != depth_d);
        depth_d.setTo(0.0, mask);

        level -= 1;
        downscale(image_d, depth_d, K_d, level, image_d, depth_d, K_d);
    }


    void showScaledImage(const Mat &org_GT, const Mat &GT, const Mat &ES) {

        double max_orig, min_orig;
        cv::minMaxLoc(GT, &min_orig, &max_orig);
        double max_adj, min_adj;
        cv::minMaxLoc(ES, &min_adj, &max_adj);

        //	 double max_real= max(max_adj, max_orig);
        //	 double min_real=min(min_adj, min_orig);
        double max_real = 0.1;
        double min_real = 0.001;



        Mat GT_for_show = GT * (1.0 / (max_real - min_real)) + (-min_real * (1.0 / (max_real - min_real)));
        Mat ES_for_show = ES * (1.0 / (max_real - min_real)) + (-min_real * (1.0 / (max_real - min_real)));
        Mat GT_orig_for_show = org_GT * (1.0 / (max_real - min_real)) + (-min_real * (1.0 / (max_real - min_real)));

        //org_GT


//		imshow("GT_NS_for_show", GT_orig_for_show);
//		imshow("GT_for_show", GT_for_show);
//		imshow("ES_for_show", ES_for_show);

        //	 GT_for_show.convertTo(GT_for_show, CV_32FC1);
        //	 ES_for_show.convertTo(ES_for_show, CV_32FC1);
        //
        //	 imwrite("GT_for_show.exr",GT_for_show); //inv_depth_ref
        //	 imwrite("ES_for_show.exr",ES_for_show);

        waitKey(0);
    }

    void projection_K(double uj, double vj, double iDepth, Sophus::SO3d &Rotation,
                      Eigen::Vector3d &Translation, Eigen::Matrix<double, 2, 1> &pt2d) {

        cout << "======================show Rotation projection============:" << Rotation.matrix() << endl;

        Eigen::Matrix<double, 3, 3> K;

//		K << 1361.1, 0, 320,
//             0, 1361.1, 240,
//             0,     0,    1;

//        K <<   577.8705, 0, 320,
//                0, 577.8705, 240,
//                0, 0, 1;

        K  <<   574.540648625183, 0, 320,
                0, 574.540648625183, 240,
                0, 0, 1;


        double fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);
        Eigen::Matrix<double, 3, 1> p_3d_no_d;
        p_3d_no_d << (uj - cx) / fx, (vj - cy) / fy, (double) 1.0;
        Eigen::Matrix<double, 3, 1> p_c1;
        p_c1 << p_3d_no_d.x() / iDepth, p_3d_no_d.y() / iDepth, p_3d_no_d.z() / iDepth;

        cout << "show iDepth:" << iDepth << endl;
        cout << "show parameter using:<<\n"
             << Rotation.matrix() << "," << Translation << endl;
        Eigen::Matrix<double, 3, 1> p1 = Rotation * p_c1 + Translation;
        Eigen::Matrix<double, 3, 1> p_3d_transformed2, point_K;
        p_3d_transformed2 = p1;
        point_K = K * p_3d_transformed2;

        pt2d.x() = point_K.x() / point_K.z();
        pt2d.y() = point_K.y() / point_K.z();
    }

    template<typename T>
    bool checkImageBoundaries(const Eigen::Matrix<T, 2, 1> &pixel, int width, int height) {
        return (pixel[0] > 1.1 && pixel[0] < width - 2.1 && pixel[1] > 1.1 && pixel[1] < height - 2.1);
    }

    template<typename T>
    void kmeans(vector<T> &sample, int k)
    {
        int len = sample.size();
        vector<T> meanValue(k, 0);

        //初始化均值,取待分类的前k个，作为初始化值
        for (int i = 0; i < k; ++i)
            meanValue[i] = sample[i];

        while (1)
        {
            vector<vector<T> > C(k, vector<T>(len, 0));  //用于存储类别，//定义了k个vector<int>(k, 0)向量；vector<int>(len, 0)定义了len个整型元素的向量，且给出每个元素的初值为0
            vector<T> numC(k, 0);//定义了k个整型元素的向量,且给出每个元素的初值为0

            //计算每个样本与各个均值的距离
            for (int i = 0; i < len; ++i)
            {
                T minDis = abs(sample[i] - meanValue[0]);//与第一个类别的差值
                int minDisIndex = 0;

                //更新最小差值，以及index
                for (int j = 1; j < k; ++j)
                {
                    T dis = abs(sample[i] - meanValue[j]);
                    if (dis < minDis)
                    {
                        minDis = dis;
                        minDisIndex = j;
                    }
                }

                //每个样本属于哪个类：C[minDistInex]：当前类，C[minDisIndex][numC[minDisIndex]]当前类的元素
                C[minDisIndex][numC[minDisIndex]] = sample[i];
                numC[minDisIndex]++;
            }

            //均值更新
            int ifBreak = 0;
            for (int i = 0; i < k; ++i)
            {
                T Sum = 0;
                for (int j = 0; j < numC[i]; ++j)
                {
                    Sum += C[i][j];
                }
                T lastMeanValue = meanValue[i];

                //当前类所有元素求均值，更新均值
                meanValue[i] = Sum / numC[i];

                //如果更新后均值未改变，说明当前类已更新完毕、未再变动
                if (lastMeanValue == meanValue[i])  ifBreak++;
            }
            //判断均值是否被更新
            if (ifBreak == k)
            {
                T maxNum = 0;
                int maxNumIndex = 0;
                for (int i = 0; i < k; ++i)
                {
                    if (numC[i] > maxNum)
                    {
                        maxNum = numC[i];
                        maxNumIndex = i;
                    }
                }
                cout << meanValue[maxNumIndex] << endl;

                for (int m = 0; m < k; m++)
                {
                    cout << m << ":\t";
                    for (int n = 0; n < numC[m]; n++)
                    {
                        cout << C[m][n] << "\t";
                    }
                    cout << endl;
                }

                //cout << " Break" << endl;
                break;
            }

        }
    }


    void  drawClusters(Mat clusterImage_sparse, Mat dsoSelectedPointMask_C3 , string name){


        // refine the point selector
        Mat inputImage= dsoSelectedPointMask_C3.clone();
        Mat inputImage_copy_1= inputImage.clone();
        Mat inputImage_copy_2= inputImage.clone();
        Mat inputImage_copy_3= inputImage.clone();

        int counter_1=0;
        int counter_2=0;
        int counter_3=0;
        for (int i = 0; i < dsoSelectedPointMask_C3.rows; i++) {
            for (int j = 0; j < dsoSelectedPointMask_C3.cols; j++)
            {
                if (clusterImage_sparse.at<int>(i, j) == 1 )
                {
                    counter_1++;
                    circle(inputImage_copy_1, Point(j, i), 1, Scalar(255, 0, 0), 1, 8, 0);
                    circle(inputImage, Point(j, i), 1, Scalar(255, 0, 0), 1, 8, 0);
                }
                else if (clusterImage_sparse.at<int>(i,j)==2){
                    counter_2++;
                    circle(inputImage_copy_2, Point(j, i), 1, Scalar(0, 255, 0), 1, 8, 0);
                    circle(inputImage, Point(j, i), 1, Scalar(0, 255, 0), 1, 8, 0);
                }else if (clusterImage_sparse.at<int>(i,j)==3){
                    counter_3++;
                    circle(inputImage_copy_3, Point(j, i), 1, Scalar(0, 0, 255), 1, 8, 0);
                    circle(inputImage, Point(j, i), 1, Scalar(0, 0, 255), 1, 8, 0);
                }
            }
        }
        string  inputImage_name = name + "_inputImage.png";
        imshow(inputImage_name,inputImage);
        string inputImage_copy_1_name = name + "_inputImage_copy_1.png";
        imshow(inputImage_copy_1_name, inputImage_copy_1);
        string inputImage_copy_2_name = name + "_inputImage_copy_2.png";
        imshow(inputImage_copy_2_name, inputImage_copy_2);
        string inputImage_copy_3_name = name + "_inputImage_copy_3.png";
        imshow(inputImage_copy_3_name, inputImage_copy_3);

        cout<<"counter_1: "<<counter_1<<endl;
        cout<<"counter_2: "<<counter_2<<endl;
        cout<<"counter_3: "<<counter_3<<endl;
    }


    bool project(double uj, double vj, double iDepth, int width, int height,
                 Eigen::Matrix<double, 2, 1> &pt2d, Sophus::SO3d &Rotation,
                 Eigen::Vector3d &Translation) {
        Eigen::Matrix<double, 2, 1> point_2d_K;
        //		cout<<"show parameter later:<<\n"<<Rotation.matrix()<<","<<Translation<<endl;
        projection_K(uj, vj, iDepth, Rotation, Translation, point_2d_K);
        return checkImageBoundaries(point_2d_K, width, height);
    }

    template<typename T>
    bool project(T uj, T vj, T iDepth, int width, int height,
                 const Eigen::Matrix<T, 3, 3> &KRKinv, const Eigen::Matrix<T, 3, 1> &Kt,
                 Eigen::Matrix<T, 2, 1> &pt2d) {
        // transform and project
        const Eigen::Matrix<T, 3, 1> pt = KRKinv * Eigen::Matrix<T, 3, 1>(uj, vj, 1) + Kt * iDepth;

        // rescale factor
        const T rescale = 1 / pt[2];

        // if the point was in the range [0, Inf] in camera1
        // it has to be also in the same range in camera2
        // This allows using negative inverse depth values
        // i.e. same iDepth sign in both cameras
        if (!(rescale > 0)) return false;

        // inverse depth in new image
        //		newIDepth = iDepth*rescale;

        // normalize
        pt2d[0] = pt[0] * rescale;
        pt2d[1] = pt[1] * rescale;

        // check image boundaries
        return checkImageBoundaries(pt2d, width, height);
    }


    void showMinus(Mat &minus_original, Mat &minus_adjust, Mat &minus_mask) {

        double max_orig, min_orig;
        cv::minMaxLoc(minus_original, &min_orig, &max_orig);
        double max_adj, min_adj;
        cv::minMaxLoc(minus_adjust, &min_adj, &max_adj);

        double max_real = max(max_adj, max_orig);
        double min_real = min(min_adj, min_orig);

        Mat adj_show(minus_original.rows, minus_original.cols, CV_32FC3, Scalar(0, 0, 0));
        Mat orig_show(minus_original.rows, minus_original.cols, CV_32FC3, Scalar(0, 0, 0));


        for (int x = 0; x < minus_original.rows; ++x) {
            for (int y = 0; y < minus_original.cols; ++y) {
                if (minus_mask.at<uchar>(x, y) == 1) {

                    float minus_orig_val = minus_original.at<float>(x, y);

                    minus_orig_val *= (1.0 / (max_real - min_real)) + (-min_real * (1.0 / (max_real - min_real)));

                    float minus_adj_val = minus_adjust.at<float>(x, y);
                    minus_adj_val *= (1.0 / (max_real - min_real)) + (-min_real * (1.0 / (max_real - min_real)));
                    if (isnan(minus_orig_val) || isnan(minus_adj_val)) {
                        orig_show.at<Vec3f>(x, y)[0] = 1;
                        orig_show.at<Vec3f>(x, y)[1] = 0;
                        orig_show.at<Vec3f>(x, y)[2] = 0;

                        adj_show.at<Vec3f>(x, y)[0] = 1;
                        adj_show.at<Vec3f>(x, y)[1] = 0;
                        adj_show.at<Vec3f>(x, y)[2] = 0;
                        continue;
                    }
                    orig_show.at<Vec3f>(x, y)[0] = minus_orig_val;
                    orig_show.at<Vec3f>(x, y)[1] = minus_orig_val;
                    orig_show.at<Vec3f>(x, y)[2] = minus_orig_val;

                    adj_show.at<Vec3f>(x, y)[0] = minus_adj_val;
                    adj_show.at<Vec3f>(x, y)[1] = minus_adj_val;
                    adj_show.at<Vec3f>(x, y)[2] = minus_adj_val;

                } else {

                    orig_show.at<Vec3f>(x, y)[0] = 0;
                    orig_show.at<Vec3f>(x, y)[1] = 0;
                    orig_show.at<Vec3f>(x, y)[2] = 1;

                    adj_show.at<Vec3f>(x, y)[0] = 0;
                    adj_show.at<Vec3f>(x, y)[1] = 0;
                    adj_show.at<Vec3f>(x, y)[2] = 1;
                }
            }
        }

        imshow("orig_show", orig_show);
        imshow("adj_show", adj_show);

        //          imwrite("orig_show.png",orig_show);
        //          imwrite("adj_show.png",adj_show);

        //	    waitKey(0);
    }


    Mat colorMap(Mat &deltaMap, float upper, float lower) {
        Mat deltaMap_Color(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0, 0, 0));


        Vec2i boundingBoxUpperLeft(83, 76);
        Vec2i boundingBoxBotRight(240, 320);


        for (int x = 0; x < deltaMap.rows; ++x) {
            for (int y = 0; y < deltaMap.cols; ++y) {


                if ( (y<boundingBoxUpperLeft.val[1] || y>boundingBoxBotRight.val[1]) || (x< boundingBoxUpperLeft.val[0] ||  x> boundingBoxBotRight.val[0])){ continue;}



                float delta = deltaMap.at<float>(x, y);
                if (delta < upper && delta > lower) {

                    delta = delta * (1.0 / (upper - lower)) + (-lower * (1.0 / (upper - lower)));
                    deltaMap_Color.at<Vec3f>(x, y)[0] = delta;
                    deltaMap_Color.at<Vec3f>(x, y)[1] = delta;
                    deltaMap_Color.at<Vec3f>(x, y)[2] = delta;

                } else if (delta > upper) {
                    deltaMap_Color.at<Vec3f>(x, y)[0] = 0;
                    deltaMap_Color.at<Vec3f>(x, y)[1] = 1;
                    deltaMap_Color.at<Vec3f>(x, y)[2] = 0;
                } else if (delta < lower && delta != -1) {
                    deltaMap_Color.at<Vec3f>(x, y)[0] = 1;
                    deltaMap_Color.at<Vec3f>(x, y)[1] = 0;
                    deltaMap_Color.at<Vec3f>(x, y)[2] = 1;
                } else if (delta == -1) {
                    deltaMap_Color.at<Vec3f>(x, y)[0] = 0;
                    deltaMap_Color.at<Vec3f>(x, y)[1] = 0;
                    deltaMap_Color.at<Vec3f>(x, y)[2] = 1;

                } else {
                    deltaMap_Color.at<Vec3f>(x, y)[0] = 1;
                    deltaMap_Color.at<Vec3f>(x, y)[1] = 0;
                    deltaMap_Color.at<Vec3f>(x, y)[2] = 0;
                }
            }
        }

        return deltaMap_Color;
    }


    void readEnvMapCtrlPointPose(string fileName, vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> &pose) {

        ifstream trajectory(fileName);
        if (!trajectory.is_open()) {
            cout << "No controlPointPose data!" << fileName << endl;
            return;
        }

        float qw, qx, qy, qz, tx, ty, tz;
        string line;
        while (getline(trajectory, line)) {
            stringstream lineStream(line);
            lineStream >> qw >> qx >> qy >> qz >> tx >> ty >> tz;

            Eigen::Vector3f t(tx, ty, tz);
            Eigen::Quaternionf q = Eigen::Quaternionf(qw, qx, qy, qz).normalized();
            Sophus::SE3f SE3_qt(q, t);
            pose.push_back(SE3_qt);
        }

    }

    void drawResidualPerPixels(vector<float>& residuals, float step_size, string  title_str,  int rows, int cols){

        // convert the residuals vector to a cv::Mat of the same size as the image of size 307200 * 8
        cv::Mat residuals_mat = cv::Mat::zeros(rows*cols, 1, CV_32F);

        vector<float> residuals_mat_vector;

        // convert vector into a matrix
//        for (int r = 0; r < rows*cols; r++){
//            float sum_val= 0.0;
//            for (int c = 0; c < 8; c++){
//                sum_val += abs(residuals[r*8 + c]);
//            }
//            residuals_mat_vector.push_back(sum_val/8.0);
//        }
        cout<< "residuals.size(): " << residuals.size() << endl;
        // convert vector into an image
        cv::Mat residuals_mat_reshaped = cv::Mat::zeros(rows, cols, CV_32F);
        //copy vector to mat
        memcpy(residuals_mat_reshaped.data, residuals.data(), residuals_mat_vector.size()*sizeof(float));
        // output the range of the residuals
        double min, max;
        cv::minMaxLoc(residuals_mat_reshaped, &min, &max);
        std::cout << "min: " << min << std::endl;
        std::cout << "max: " << max << std::endl;

        // specify the range of residuals_mat_reshaped
        cv::Mat residuals_mat_normalized;
        cv::normalize(residuals_mat_reshaped, residuals_mat_normalized, 0, 1, cv::NORM_MINMAX, CV_32F);

//        auto [X, Y] = matplot::meshgrid(std::iota(0, step_size, 640.0),std::iota(0, step_size, 480.0));

//        auto Z = transform(X, Y, [&residuals_mat_normalized](float x, float y) { return residuals_mat_normalized.at<float>(y,x); });
//
//
//        mesh(X, Y, Z)->hidden_3d(false);



        matplot::title(title_str);

        matplot::show();



    }


    void drawResidualDistribution(vector<double>& residuals, string  title_str, int rows, int cols){

        vector<float> residuals_mat_vector;
        cout<< "residuals.size(): " << residuals.size() << endl;
        // convert vector into an image
        auto h = matplot::hist(residuals);
        std::cout << "Histogram with " << h->num_bins() << " bins" << std::endl;
        matplot::title(title_str);
        matplot::show();
    }

    void  drawEnvMapPoints(const string envMapPosePath, const Mat EnvLightWorkMap, int canvasId,  Eigen::Matrix3f K,  Sophus::SE3f cameraExtrinsics){

        std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> controlPointPoses;

        readEnvMapCtrlPointPose(envMapPosePath, controlPointPoses);
        // output the envMapPosePath
        cout << "envMapPosePath = " << envMapPosePath << endl;

        // output number of control points
        cout << "controlPointPoses.size() = " << controlPointPoses.size() << endl;

        float fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);

        // draw control points

        for (size_t i = 1; i <= controlPointPoses.size(); i++) {
            Sophus::SE3f envMapPose_world = controlPointPoses[i - 1].cast<float>();
            cv::Point3f pointBase = Vec3f(controlPointPoses[i - 1].translation().x(),
                                          controlPointPoses[i - 1].translation().y(),
                                          controlPointPoses[i - 1].translation().z());

            Eigen::Vector3f envMap_point_c1 = cameraExtrinsics.inverse() * Eigen::Vector3f(pointBase.x, pointBase.y, pointBase.z);
            // project envMap_point to image plane
            float pixel_x = (fx * envMap_point_c1.x()) / envMap_point_c1.z() + cx;
            float pixel_y = (fy * envMap_point_c1.y()) / envMap_point_c1.z() + cy;
            cv::Point2i point(round(pixel_x),round(pixel_y));
            cv::circle(EnvLightWorkMap, point, 2, cv::Scalar(0, 255, 255), -1); // 255,255,0 : yellow


        }



        // convert the name of EnvLightWorkMap to string
        std::string ctrlPointUsed= "ctrlPointOnCanvas_" + std::to_string(canvasId);

        imwrite(ctrlPointUsed + ".png", EnvLightWorkMap);

        imshow(ctrlPointUsed, EnvLightWorkMap);


    }

    Mat intensityCorrespondenceChangeGT(Mat &Img_left, Mat &depth_left, Mat &Img_right, Mat &depth_right, const Eigen::Matrix<double, 3, 3> &K_,double &thres,
                                        const Sophus::SE3d &ExtrinsicPose,Mat pointOfInterest) {

        Mat correspInLeft(pointOfInterest.rows, pointOfInterest.cols, CV_8UC1, Scalar(0));
        Mat correspInRight(pointOfInterest.rows, pointOfInterest.cols, CV_8UC1, Scalar(0));





        Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(0));// default value 1.0
        float radius = thres;
        //		Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(1)); // default value 1.0
        double fx = K_(0, 0), cx = K_(0, 2), fy = K_(1, 1), cy = K_(1, 2);
        pcl::PCDWriter writer;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rig(new pcl::PointCloud<pcl::PointXYZ>);
        for (int x = 0; x < depth_left.rows; ++x) {
            for (int y = 0; y < depth_left.cols; ++y) {
                // IOA setting
                double d_r = depth_right.at<float>(x, y);
                // Mark: skip depth=15
                // if (round(d_r) == 15.0f) {continue;}
                Eigen::Matrix<double, 3, 1> p_3d_no_d_r;
                p_3d_no_d_r << (y - cx) / fx, (x - cy) / fy, 1.0;
                Eigen::Matrix<double, 3, 1> p_c2 = d_r * p_3d_no_d_r;

                cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
            }
        }
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud_rig);
        double max, min;
        cv::minMaxLoc(Img_left, &min, &max);
        cout << "\n show max and min of Img_left:\n"<< max << "," << min << endl;
        cv::minMaxLoc(Img_right, &min, &max);
        cout << "\n show max and min of Img_right:\n"<< max << "," << min << endl;
        std::unordered_map<int, int> inliers_filter;
        std::unordered_map<int, int> inliers_filter_sixPoints;
        //new image
        inliers_filter.emplace(411, 439);
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        Vec2i boundingBoxUpperLeft_AoI(145,180);
        Vec2i boundingBoxBotRight_AoI(173,242);
        imshow("pointOfInterestArea",pointOfInterest)   ;

        for (int x = 0; x < depth_left.rows; ++x) {
            for (int y = 0; y < depth_left.cols; ++y) {

//								if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//								if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~

//                    if ( (y<boundingBoxUpperLeft_AoI.val[1] || y>boundingBoxBotRight_AoI.val[1]) || (x< boundingBoxUpperLeft_AoI.val[0] ||  x> boundingBoxBotRight_AoI.val[0])){ continue;}
//                    if (statusMap!=NULL && static_cast<int>(statusMap[x * depth_left.cols + y])!= 255){ continue;}

                if ( ((int)pointOfInterest.at<uchar>(x,y)) !=255){ continue;}

                // calculate 3D point of left camera
                double d = depth_left.at<float>(x, y);

                Eigen::Matrix<double, 3, 1> p_3d_no_d;
                p_3d_no_d << (y - cx) / fx, (x - cy) / fy, 1.0;
                Eigen::Matrix<double, 3, 1> p_c1 = d * p_3d_no_d;
                Eigen::Vector3d point_Trans = ExtrinsicPose.rotationMatrix() * p_c1 + ExtrinsicPose.translation();
                cloud->push_back(pcl::PointXYZ(point_Trans.x(), point_Trans.y(), point_Trans.z()));
                pcl::PointXYZ searchPoint(point_Trans.x(), point_Trans.y(), point_Trans.z());
                if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
//                    for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
//                        std::cout << "\n------"  <<   (*cloud_rig)[ pointIdxRadiusSearch[0] ].x
//                                  << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].y
//                                  << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].z
//                                  << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;

                    correspInLeft.at<uchar>(x,y)=255;


                    float left_intensity;
                    if (Img_left.type() == CV_8UC1)
                        left_intensity =static_cast<float>(Img_left.at<uchar>(x, y));
                    else if (Img_left.type() == CV_32FC1)
                        left_intensity =static_cast<float>(Img_left.at<float>(x, y));

//                    float left_intensity =static_cast<float>(Img_left.at<float>(x, y));



                    float pointCorres_x = (*cloud_rig)[pointIdxRadiusSearch[0]].x;
                    float pointCorres_y = (*cloud_rig)[pointIdxRadiusSearch[0]].y;
                    float pointCorres_z = (*cloud_rig)[pointIdxRadiusSearch[0]].z;
                    float pixel_x = (fx * pointCorres_x) / pointCorres_z + cx;
                    float pixel_y = (fy * pointCorres_y) / pointCorres_z + cy;


                    float right_intensity;
                    if (Img_right.type() == CV_8UC1) {
                        right_intensity = static_cast<float>(Img_right.at<uchar>(round(pixel_y), round(pixel_x)));
                    } else if (Img_right.type() == CV_32FC1) {
                        right_intensity = static_cast<float>(Img_right.at<float>(round(pixel_y), round(pixel_x)));
                    }
//                    float right_intensity = static_cast<float>(Img_right.at<float>(round(pixel_y), round(pixel_x)));

                    correspInRight.at<uchar>(round(pixel_y),round(pixel_x))=255;



//                    float delta = right_intensity / left_intensity;
                    float delta= right_intensity-left_intensity;

//                    cout<<"\n IN deltaMapGT func, checking radiance vals:"<< "left Coord: u:"<<x<<", v:"<<y<<" left_intensity:"<< left_intensity
//                        << ", and right_intensity at row:"<<pixel_y <<", at col:"<< pixel_x<< "is:"<<  right_intensity
//                        <<"show intensity difference: "<<left_intensity-right_intensity <<"show GT delta: "<<delta <<endl;
//
                    float diff_residual = std::abs(right_intensity - left_intensity);
                    deltaMapGT.at<float>(x, y) = delta;
                }
            }
        }
        // draw the circle one left and right image
        for (int x = 0; x < depth_left.rows; ++x) {
            for (int y = 0; y < depth_left.cols; ++y) {
                if (correspInLeft.at<uchar>(x,y)==255){
                    circle(Img_left, cv::Point2d(y,x), 1, Scalar(255), 1, 8, 0);
                }
                if (correspInRight.at<uchar>(x,y)==255){
                    circle(Img_right, cv::Point2d(y,x),1, Scalar(255), 1, 8, 0);
                }
            }
        }
        imshow("Img_left", Img_left);
        imshow("Img_right", Img_right);
        waitKey(0);
//		 writer.write("PointCloud_Transformed_06.pcd",*cloud, false);// do we need the sensor acquisition origin?
//		 writer.write("PointCloud_right_HD_06.pcd",*cloud_rig, false);// do we need the sensor acquisition origin?
        double max_n, min_n;
        cv::minMaxLoc(deltaMapGT, &min_n, &max_n);
        //deltaMapGT=deltaMapGT*(1.0/(upper-buttom))+(-buttom*(1.0/(upper-buttom)));
        cout << "\n show max and min of deltaMapGT:<---------------"<< max_n << "," << min_n << endl;
        return deltaMapGT;
    }



    void saveArray(vector<double>& DistributionMap){

        ofstream delta_comp;
        delta_comp.open ("DistributionArray.txt");
        // iterate through the vector and print the values
        for (int i = 0; i < DistributionMap.size(); i++)
        {
//            cout << DistributionMap[i] << " ";

            delta_comp <<DistributionMap[i]<<"\n";
        }

        delta_comp.close();

    }

    Mat deltaMapGT(Mat &Img_left, Mat &depth_left, Mat &Img_right, Mat &depth_right, const Eigen::Matrix<double, 3, 3> &K_, double &thres,
                   const Sophus::SE3d &ExtrinsicPose, float &upper, float &buttom, Mat &pred_deltaMap,
                   float *statusMap, Mat pointOfInterest
            , string envMapPosePath
            , Sophus::SE3f cameraExtrinsics, Mat normal_left
    ) {



        Mat deltda_map_allchannels;
        cvtColor(pred_deltaMap, deltda_map_allchannels, CV_RGB2GRAY);

        Mat envMapWorkMap(pointOfInterest.rows, pointOfInterest.cols, CV_8UC3, Scalar(0,0,0));
        Mat correspInLeft(pointOfInterest.rows, pointOfInterest.cols, CV_8UC1, Scalar(0));
        Mat correspInRight(pointOfInterest.rows, pointOfInterest.cols, CV_8UC1, Scalar(0));



        for (int x = 0; x < pointOfInterest.rows; ++x) {
            for (int y = 0; y < pointOfInterest.cols; ++y) {
                if (pointOfInterest.at<uchar>(x, y) == 255) {
                    envMapWorkMap.at<Vec3b>(x, y)[0] = 255;
                    envMapWorkMap.at<Vec3b>(x, y)[1] = 255;
                    envMapWorkMap.at<Vec3b>(x, y)[2] = 255;
                }
            }
        }

        Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(0));// default value 1.0
        //		Mat deltaMapGT(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(1)); // default value 1.0
        double fx = K_(0, 0), cx = K_(0, 2), fy = K_(1, 1), cy = K_(1, 2);
        pcl::PCDWriter writer;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rig(new pcl::PointCloud<pcl::PointXYZ>);


        for (int x = 0; x < depth_left.rows; ++x) {
            for (int y = 0; y < depth_left.cols; ++y) {
                // IOA setting
                double d_r = depth_right.at<float>(x, y);
                // Mark: skip depth=15
                // if (round(d_r) == 15.0f) {continue;}
                Eigen::Matrix<double, 3, 1> p_3d_no_d_r;
                p_3d_no_d_r << (y - cx) / fx, (x - cy) / fy, 1.0;
                Eigen::Matrix<double, 3, 1> p_c2 = d_r * p_3d_no_d_r;

                cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
            }
        }

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud_rig);
        double max, min;
        cv::minMaxLoc(Img_left, &min, &max);
        cout << "\n show max and min of Img_left:\n"<< max << "," << min << endl;
        cv::minMaxLoc(Img_right, &min, &max);
        cout << "\n show max and min of Img_right:\n"<< max << "," << min << endl;
        std::unordered_map<int, int> inliers_filter;
        std::unordered_map<int, int> inliers_filter_sixPoints;
        //new image


        inliers_filter.emplace(411, 439);



        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        float radius = thres;
        Mat minus_original(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(0));
        Mat minus_adjust(depth_left.rows, depth_left.cols, CV_32FC1, Scalar(0));
        Mat minus_mask(depth_left.rows, depth_left.cols, CV_8UC1, Scalar(0));

//        Vec2i boundingBoxUpperLeft(83, 76);
//        Vec2i boundingBoxBotRight(240, 320);

        Vec2i boundingBoxUpperLeft_AoI(145,180);
        Vec2i boundingBoxBotRight_AoI(173,242);

        ofstream delta_comp;
        ofstream delta_comp_sixPoints;
        delta_comp.open ("delta_comp_sparsed.txt");
        delta_comp_sixPoints.open ("delta_comp_sixPoints.txt");

        for (int x = 0; x < depth_left.rows; ++x) {
            for (int y = 0; y < depth_left.cols; ++y) {

//								if(inliers_filter.count(x)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//								if(inliers_filter[x]!=y ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~

//                    if ( (y<boundingBoxUpperLeft_AoI.val[1] || y>boundingBoxBotRight_AoI.val[1]) || (x< boundingBoxUpperLeft_AoI.val[0] ||  x> boundingBoxBotRight_AoI.val[0])){ continue;}
//                    if (statusMap!=NULL && static_cast<int>(statusMap[x * depth_left.cols + y])!= 255){ continue;}

                if ( ((int)pointOfInterest.at<uchar>(x,y)) !=255){ continue;}

//                if (deltda_map_allchannels.at<float>(x, y)==0.0) { continue; }



                // calculate 3D point of left camera
                double d = depth_left.at<float>(x, y);
                // Mark: skip depth=15
                //if (round(d) == 15.0) { continue; }
                // projection
                Eigen::Matrix<double, 3, 1> p_3d_no_d;
                p_3d_no_d << (y - cx) / fx, (x - cy) / fy, 1.0;
                Eigen::Matrix<double, 3, 1> p_c1 = d * p_3d_no_d;
                Eigen::Vector3d point_Trans = ExtrinsicPose.rotationMatrix() * p_c1 + ExtrinsicPose.translation();
                cloud->push_back(pcl::PointXYZ(point_Trans.x(), point_Trans.y(), point_Trans.z()));
                pcl::PointXYZ searchPoint(point_Trans.x(), point_Trans.y(), point_Trans.z());
                if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                    for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
                        std::cout << "\n------"  <<   (*cloud_rig)[ pointIdxRadiusSearch[0] ].x
                                  << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].y
                                  << " " << (*cloud_rig)[ pointIdxRadiusSearch[0] ].z
                                  << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;
                    //


                    correspInLeft.at<uchar>(x,y)=255;

                    float left_intensity;
                    if (Img_left.type() == CV_8UC1)
                        left_intensity =static_cast<float>(Img_left.at<uchar>(x, y));
                    else if (Img_left.type() == CV_32FC1)
                        left_intensity =static_cast<float>(Img_left.at<float>(x, y));


//                  float left_intensity =static_cast<float>(Img_left.at<float>(x, y));
                    float pointCorres_x = (*cloud_rig)[pointIdxRadiusSearch[0]].x;
                    float pointCorres_y = (*cloud_rig)[pointIdxRadiusSearch[0]].y;
                    float pointCorres_z = (*cloud_rig)[pointIdxRadiusSearch[0]].z;
                    float pixel_x = (fx * pointCorres_x) / pointCorres_z + cx;
                    float pixel_y = (fy * pointCorres_y) / pointCorres_z + cy;

                    float right_intensity;
                    if (Img_right.type() == CV_8UC1) {
                        right_intensity = static_cast<float>(Img_right.at<uchar>(round(pixel_y), round(pixel_x)));
                    } else if (Img_right.type() == CV_32FC1) {
                        right_intensity = static_cast<float>(Img_right.at<float>(round(pixel_y), round(pixel_x)));
                    }
//                    float right_intensity = static_cast<float>(Img_right.at<float>(round(pixel_y), round(pixel_x)));

                    correspInRight.at<uchar>(round(pixel_y),round(pixel_x))=255;



//                    float delta = right_intensity / left_intensity;
                    float delta= right_intensity-left_intensity;

                    cout<<"\n IN deltaMapGT func, checking radiance vals:"<< "left Coord: u:"<<x<<", v:"<<y<<" left_intensity:"<< left_intensity
                        << ", and right_intensity at row:"<<pixel_y <<", at col:"<< pixel_x<< "is:"<<  right_intensity
                        <<"show intensity difference: "<<left_intensity-right_intensity <<"show GT delta: "<<delta <<endl;

                    float diff_orig = std::abs(left_intensity - right_intensity);
                    minus_original.at<float>(x, y) = diff_orig;
                    float delta_pred_b = pred_deltaMap.at<Vec3f>(x, y)[0];
                    float delta_pred_g = pred_deltaMap.at<Vec3f>(x, y)[1];
                    float delta_pred_r = pred_deltaMap.at<Vec3f>(x, y)[2];

                    if (delta_pred_b==1.0 && delta_pred_g ==1.0 && delta_pred_r==1.0){ continue;}
                    float diff_residual = std::abs(right_intensity - left_intensity);
                    float diff_adj = std::abs(right_intensity - delta_pred_g * left_intensity);
                    delta_comp <<x<<" "<<y<<" "<<diff_residual<<" "<<diff_adj<<"\n";
                    // use the inlier filter
                    if (inliers_filter_sixPoints.count(x) != 0 ) {
                        delta_comp_sixPoints <<x<<" "<<y<<" "<<diff_residual<<" "<<diff_adj<<" "<<left_intensity<<" "<<right_intensity<<"\n";
                    }




                    if (diff_residual>diff_adj){
                        envMapWorkMap.at<Vec3b>(x,y)[0]=0;
                        envMapWorkMap.at<Vec3b>(x,y)[1]=255 * (diff_residual-diff_adj)/diff_residual;
                        envMapWorkMap.at<Vec3b>(x,y)[2]=0;

                    }else if (diff_residual<diff_adj){
                        envMapWorkMap.at<Vec3b>(x,y)[0]=0;
                        envMapWorkMap.at<Vec3b>(x,y)[1]=0;
                        envMapWorkMap.at<Vec3b>(x,y)[2]=255* (diff_adj-diff_residual)/diff_residual;
                    }

//                    if (diff_residual>diff_adj){
//                        envMapWorkMap.at<Vec3b>(x,y)[0]=0;
//                        envMapWorkMap.at<Vec3b>(x,y)[1]=255 ;
//                        envMapWorkMap.at<Vec3b>(x,y)[2]=0;
//
//                    }else if (diff_residual<diff_adj){
//                        envMapWorkMap.at<Vec3b>(x,y)[0]=0;
//                        envMapWorkMap.at<Vec3b>(x,y)[1]=0;
//                        envMapWorkMap.at<Vec3b>(x,y)[2]=255;
//                    }


                    minus_adjust.at<float>(x, y) = diff_adj;
                    minus_mask.at<uchar>(x, y) = 1;

                    deltaMapGT.at<float>(x, y) = delta;
                }


                //				double left_intensity=Img_left.at<double>(x,y);
                //				for(int x_ = 0; x_ < depth_left.rows; ++x_) {
                //					for (int y_ = 0; y_ < depth_left.cols; ++y_) {
                //
                //						double d_r= depth_right.at<double>(x_,y_);
                //						Eigen::Matrix<double,3,1> p_3d_no_d_r;
                //						p_3d_no_d_r<< (y_-cx)/fx, (x_-cy)/fy,1.0;
                //						Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
                //						cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
                //						double distance= ((point_Trans-p_c2).norm());
                //						if ( distance< thres){
                //							double rigth_intensity=Img_right.at<double>(x_,y_);
                //							deltaMapGT.at<double>(x,y)=left_intensity/rigth_intensity;
                //						}
                //					}
                //				}


                //				double d_r= depth_right.at<double>(x,y);
                //				Eigen::Matrix<double,3,1> p_3d_no_d_r;
                //				p_3d_no_d_r<< (y-cx)/fx, (x-cy)/fy,1.0;
                //				Eigen::Matrix<double, 3,1> p_c2=d_r*p_3d_no_d_r;
                //				cloud_rig->push_back(pcl::PointXYZ(p_c2.x(), p_c2.y(), p_c2.z()));
            }
        }


        delta_comp.close();
        delta_comp_sixPoints.close();


        // draw the circle one left and right image

        //  cv::Point2i point_r(round(pixel_x),round(pixel_y));
        // cv::circle(Img_right, point_r, 2, cv::Scalar(255), -1);

        for (int x = 0; x < depth_left.rows; ++x) {
            for (int y = 0; y < depth_left.cols; ++y) {
                if (correspInLeft.at<uchar>(x,y)==255){
                    circle(Img_left, cv::Point2d(y,x), 1, Scalar(255), 1, 8, 0);
                }
                if (correspInRight.at<uchar>(x,y)==255){
                    circle(Img_right, cv::Point2d(y,x),1, Scalar(255), 1, 8, 0);
                }
            }
        }




        imshow("Img_left", Img_left);
        imshow("Img_right", Img_right);
        showMinus(minus_original, minus_adjust, minus_mask);
        imshow("envMapWorkMap",envMapWorkMap);
        // draw envmap points on the envMapWorkMap
        drawEnvMapPoints(envMapPosePath,envMapWorkMap, 1, K_.cast<float>(), cameraExtrinsics);
        drawEnvMapPoints(envMapPosePath, normal_left, 2, K_.cast<float>(), cameraExtrinsics);
        // save the image
        imwrite("envMapWorkMap.png",envMapWorkMap);
//		 writer.write("PointCloud_Transformed_06.pcd",*cloud, false);// do we need the sensor acquisition origin?
//		 writer.write("PointCloud_right_HD_06.pcd",*cloud_rig, false);// do we need the sensor acquisition origin?
        double max_n, min_n;
        cv::minMaxLoc(deltaMapGT, &min_n, &max_n);
        //deltaMapGT=deltaMapGT*(1.0/(upper-buttom))+(-buttom*(1.0/(upper-buttom)));
        cout << "\n show max and min of deltaMapGT:<---------------"<< max_n << "," << min_n << endl;
        return deltaMapGT;
    }


}// namespace DSONL