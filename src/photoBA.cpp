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

bool gStop = false;

void sigHandler(int) { gStop = true; }

int main(int argc, char** argv)
{
    signal(SIGINT, sigHandler);

    utils::ProgramOptions options;
    options
            ("output,o", "refined_poses_test_00.txt", "trajectory output file")
            ("config,c", "../config/tum_rgbd.cfg", "config file")
            .parse(argc, argv);

    utils::ConfigFile cf(options.get<std::string>("config"));
    UniquePointer<Dataset> dataset = Dataset::Create(options.get<std::string>("config"));




    // traverse the dataset and refine the poses
    //    UniquePointer<DatasetFrame> frame_t;
    //    for(int f_i = 0; (frame_t = dataset->getFrame(f_i)) && !gStop; ++f_i) {
    //
    //        imshow("image", frame_t->image());
    //        imshow("depth", frame_t->depth());
    //        cv::waitKey(0);
    //    }
    auto Bf = dataset->calibration().b() * dataset->calibration().fx();


    PoseList T_init = loadPosesTumRGBDFormat(cf.get<std::string>("trajectory"));
    if(T_init.empty()) {
        std::cerr<<("Failed to load poses from %s\n", cf.get<std::string>("trajectory").c_str());
        return -1;
    }

//    for (int i = 0; i < T_init.size(); ++i) {
//
//        std::cout<< "pose: \n"<<T_init[i] <<"\n"<< std::endl;
//
//    }


    PhotometricBundleAdjustment::Result result;
    PhotometricBundleAdjustment photoba(dataset->calibration(), dataset->imageSize(), {cf});

    EigenAlignedContainer_<Mat44> T_opt;

    cv::Mat_<float> zmap;
    UniquePointer<DatasetFrame> frame;
    for(int f_i = 0; (frame = dataset->getFrame(f_i)) && !gStop; ++f_i) {
        printf("Frame %05d\n", f_i);

        const uint8_t* I = frame->image().ptr<const uint8_t>();
        float* Z =frame->depth().ptr<float>();
        // show frame name
        std::cerr<<"current frame name: "<<frame->filename()<<std::endl;
//        int num_nonZero_Depth= cv::countNonZero(frame->depth());
//        std::cout<<"num_nonZero_Depth: "<<num_nonZero_Depth<<std::endl;

//        imshow("depth", frame->depth());
        std::cout<<"depth map type: "<<frame->depth().type()<<std::endl;
        double min_v, max_v;
        cv::minMaxLoc(frame->depth(), &min_v, &max_v);
        std::cout << "\n show depth min, max:\n"<< min_v << "," << max_v <<std::endl;
        photoba.addFrame(I, Z, T_init[f_i],  &result);
        if(!result.refinedPoints.empty()) {
            // store the refinement points if you'd like
        }
    }

    auto output_fn = options.get<std::string>("output");
    Info("Writing refined poses to %s\n", output_fn.c_str());
    writePosesTumRGBDFormat(output_fn, result.poses, dataset->getTimestamp());

    return 0;
}
