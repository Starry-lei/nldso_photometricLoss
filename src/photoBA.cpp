//
// Created by lei on 28.04.23.
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "PhotometricBA/pba.h"
#include "PhotometricBA/utils.h"
#include "PhotometricBA/debug.h"
#include "PhotometricBA/dataset.h"
#include "PhotometricBA/imgproc.h"
#include "PhotometricBA/pose_utils.h"
#include <signal.h>

#include <atomic>
#include <iostream>
#include <thread>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

void readCtrlPointPoseData(string fileName, vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& pose) {

    ifstream trajectory(fileName);
    if (!trajectory.is_open()) {
        cout << "No controlPointPose data!" << fileName << endl;
        return;
    }

    float  qw, qx, qy, qz, tx, ty, tz;
    string line;
    while (getline(trajectory, line)) {
        stringstream lineStream(line);
        lineStream >> qw >> qx >> qy >> qz>> tx >> ty >> tz;

        Eigen::Vector3f t(tx, ty, tz);
        Eigen::Quaternionf q = Eigen::Quaternionf(qw, qx, qy, qz).normalized();
        Sophus::SE3f SE3_qt(q, t);
        pose.push_back(SE3_qt);
    }

}



bool gStop = false;
bool show_gui = true;
void sigHandler(int) { gStop = true; }

int main(int argc, char** argv)
{
    signal(SIGINT, sigHandler);

    pbaUtils::ProgramOptions options;
    options
            ("output,o", "refined_poses_es.txt", "trajectory output file")
            ("config,c", "../config/tum_rgbd.cfg", "config file")
            .parse(argc, argv);

    pbaUtils::ConfigFile cf(options.get<std::string>("config"));
    UniquePointer<Dataset> dataset = Dataset::Create(options.get<std::string>("config"));

    PoseList T_init = loadPosesTumRGBDFormat(cf.get<std::string>("trajectory"));
    if(T_init.empty()) {
        std::cerr<<"Failed to load poses from %s\n", cf.get<std::string>("trajectory").c_str();
        return -1;
    }

    // load environment light path
    std::string EnvMapPath=cf.get<std::string>("EnvMapPath");
    std::string EnvMapPosePath=cf.get<std::string>("EnvMapPosePath");

    // convert environment light pose the coordinate system of the first camera in PBA sequence
    std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> trajectoryPoses;
    string fileName = "/home/lei/Documents/Dataset/dataSetPBA/sequences/02/poses.txt";
    readCtrlPointPoseData(fileName, trajectoryPoses);
    cout<<"trajectoryPoses size: "<<trajectoryPoses.size()<<endl;
    Sophus::SE3f frontCamPose_w (trajectoryPoses[0]);


    PhotometricBundleAdjustment::Result result;
    PhotometricBundleAdjustment photoba(dataset->calibration(), dataset->imageSize(), {cf});
    photoba.EnvMapPath=EnvMapPath;
//    photoba.EnvMapPosePath=EnvMapPosePath;
    PBANL::envLightLookup  *EnvLightLookup= new PBANL::envLightLookup(argc, argv, EnvMapPath,EnvMapPosePath,frontCamPose_w);
    photoba.EnvLightLookup= EnvLightLookup;

    EigenAlignedContainer_<Mat44> T_opt;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PCDWriter writer;

    cv::Mat_<float> zmap;
    UniquePointer<DatasetFrame> frame;
    for(int f_i = 0; (frame = dataset->getFrame(f_i)) && !gStop; ++f_i) {
        printf("Frame %05d\n", f_i);
        const uint8_t* I = frame->image().ptr<const uint8_t>();
        float* Z =frame->depth().ptr<float>();
        const Vec3f* N = frame->normal().ptr<Vec3f>();
        const float* R= frame->roughness().ptr<float>();

        photoba.addFrame(I, Z, N, R, T_init[f_i],  &result);
        if(!result.refinedPoints.empty()) {
            for (size_t i = 0; i < result.refinedPoints.size(); ++i) {
                pcl::PointXYZ p(result.refinedPoints[i].x(),result.refinedPoints[i].y(),result.refinedPoints[i].z());
                cloud->push_back(p);
            }
        }
    }

    string output_fn = options.get<std::string>("output");
    Info("Writing refined poses to %s\n", output_fn.c_str());
    writePosesTumRGBDFormat(output_fn, result.poses, dataset->getTimestamp());
    // save the point cloud
    writer.write("refined_points.pcd",*cloud ,false );




    return 0;
}

// code note:
// show frame
//        imshow("image", frame->image());
//        cv::waitKey(0);
//        std::cerr<<"current frame name: "<<frame->filename()<<std::endl;
//        int num_nonZero_Depth= cv::countNonZero(frame->depth());
//        std::cout<<"num_nonZero_Depth: "<<num_nonZero_Depth<<std::endl;
//        imshow("depth", frame->depth());
//        std::cout<<"depth map type: "<<frame->depth().type()<<std::endl;
//        double min_v, max_v;
//        cv::minMaxLoc(frame->depth(), &min_v, &max_v);
//        std::cout << "\n show depth min, max:\n"<< min_v << "," << max_v <<std::endl;