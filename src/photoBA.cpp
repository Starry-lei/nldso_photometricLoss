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
            ("output,o", "refined_poses_absoluteCheckPose_00.txt", "trajectory output file")
            ("config,c", "../config/kitti_stereo.cfg", "config file")
            .parse(argc, argv);

    utils::ConfigFile cf(options.get<std::string>("config"));
    auto dataset = Dataset::Create(options.get<std::string>("config"));
    auto Bf = dataset->calibration().b() * dataset->calibration().fx();
    auto T_init = loadPosesKittiFormat(cf.get<std::string>("trajectory"));
	std::cout<< "trajectory name: " << cf.get<std::string>("trajectory")<< std::endl;

    PhotometricBundleAdjustment::Result result;
    PhotometricBundleAdjustment photoba(dataset->calibration(), dataset->imageSize(), {cf});

	// ----------debugging--------------------------
	Trajectory check_trajectory, check_trajectory_2;
	std::string absolutePose= "../data/kitti_init_poor/11.txt";
	auto T_debugging_init = loadPosesKittiFormat(absolutePose);
	// ----------debugging--------------------------



    EigenAlignedContainer_<Mat44> T_opt;
    cv::Mat_<float> zmap;
    UniquePointer<DatasetFrame> frame;
    for(int f_i = 0; (frame = dataset->getFrame(f_i)) && !gStop; ++f_i) {
        printf("Frame %05d\n", f_i);
        disparityToDepth(frame->disparity(), Bf, zmap);
        auto I = frame->image().ptr<const uint8_t>();
        auto Z = zmap.ptr<float>();
		// temporary test code
		// photoba.addFrame4CheckInputPose(I, Z, T_init[f_i],  &result);

		// protected code:
        photoba.addFrame(I, Z, T_init[f_i], T_debugging_init[f_i], &result);
        if(!result.refinedPoints.empty()) {
            // store the refinement points if you'd like
        }
//				if (f_i == 50){
//					break ;
//				}
    }
	// protected code:
	auto output_fn = options.get<std::string>("output");
    Info("Writing refined poses to %s\n", output_fn.c_str());
    writePosesKittiFormat(output_fn, result.poses);

	// temporary test code
	//writePosesKittiFormat("refined_poses_relative2absoluteCheckPose_00.txt", photoba._trajectory.poses());
//	writePosesKittiFormat("checkPose_allFrames.txt", photoba.te.poses());
//	writePosesKittiFormat("checkPose_absFrames.txt", check_trajectory_2.poses());
    return 0;
}
