//
// Created by lei on 28.04.23.
//

#ifndef NLDSO_PHOTOMETRICLOSS_PBA_H
#define NLDSO_PHOTOMETRICLOSS_PBA_H

#include <ceres/iteration_callback.h>

#include "types.h"
#include "trajectory.h"
#include "calibration.h"

#include <iosfwd>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <boost/circular_buffer.hpp>
#include <map>

#include "pixelSelector.h"

namespace pbaUtils {
    class ConfigFile;
};  // utils

extern bool  optimizeSignal;

class PhotometricBundleAdjustment
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     */
    struct Options
    {
        /** maximum number of points to intialize from a new frame */
        int maxNumPoints =4096 ;// 4096;

        /** number of frames in the sliding window */
        int slidingWindowSize = 5;//7; //5;

        /** radius of the image patch */
        int patchRadius = 2;//4;//2;

        /** radius (side length) of an area to prevent intializing new points when a
         * new frame is added */
        int maskBlockRadius = 1;

        /** Maximum distance (age) to keep a scene point in the system */
        int maxFrameDistance = 1;

        /** number of threads to use in the solve (-1 means maximum threads) */
        int numThreads = -1;

        /** optional gaussian weighting to focus on the center of the patch */
        bool doGaussianWeighting = false;

        /** print information about the optimization */
        bool verbose = false;

        /** minimum score to verify if a scene point exists in a new frame. This is
         * the ZNCC score which is [-1, 1] */
        double minScore =  0.70; // 0.60; // 0.75

        /** threshold to use for a HuberLoss (if > 0) */
        double robustThreshold = 20; //0.05;

        /** minimum depth to use */
        double minValidDepth = 0.01;

        /** maximum depth to use */
        double maxValidDepth = 20.0;

        /** non-maxima suppression radius for pixel selection */
        int nonMaxSuppRadius = 1;

        enum class DescriptorType
        {
            Intensity, // single channel image patch is only intensities
            IntensityAndGradient, // 3 channels, {I, Ix, Iy}
            BitPlanes // 8 channel BitPlanes
        };

        /** type of the patch/descriptor */
        DescriptorType descriptorType;

        Options() {}

        Options(const pbaUtils::ConfigFile& cf);

    private:
        friend std::ostream& operator<<(std::ostream&, const Options&);
    }; //  Options


    struct Result
    {
        /** refined world poses */
        EigenAlignedContainer_<Mat44> poses;

        /** refined world points */
        EigenAlignedContainer_<Vec3> refinedPoints;

        /** the original points for comparison */
        EigenAlignedContainer_<Vec3> originalPoints;

        //
        // optimization statistics
        //
        double initialCost  = -1.0; //< objective at the first start
        double finalCost    = -1.0; //< objective at termination
        double fixedCost    = -1.0; //< fixed cost not included in optimization

        int numSuccessfulStep = 0; //< number of successfull optimizer steps
        int numResiduals      = 0; //< number of residuals in the problem

        double totalTime = -1.0;   //< total optimization run time in seconds

        std::string message; //< optimizer message

        /**
         * Iteration details from ceres
         * iterationSummary.size() is the total number of iterations
         */
        std::vector<ceres::IterationSummary> iterationSummary;

        /**
         * Writes detailed results to a file in a binary format
         *
         * Requires cereal library
         */
        struct Writer
        {
            Writer(std::string prefix = "./")
                    : _counter(0), _prefix(prefix) {}

            bool add(const Result&);

        private:
            int _counter;
            std::string _prefix;
        }; // Writer

        static Result FromFile(std::string);

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    }; // Result


public:
    /**
     * \param the camera calibration (pinhole model)
     * \param the image size
     * \param options for the algorithm
     */
    PhotometricBundleAdjustment(const Calibration&, const ImageSize&, const Options& = Options());

    /**
     */
    ~PhotometricBundleAdjustment();

    /**
     * \param image pointer to the image
     * \param depth_map pointer to the depth map
     * \param T pose initialization for this frame
     * \param result, if not null we store the optmization results in it
     */
    void addFrame(const uint8_t* image, const cv::Mat& grayImg, const float* depth_map, const Mat44& T, Result* = nullptr);

	Trajectory initial_trajectory;
	class DescriptorFrame;
	typedef UniquePointer<DescriptorFrame> DescriptorFramePointer ;
	typedef boost::circular_buffer<DescriptorFramePointer> DescriptorFrameBuffer;
	DescriptorFrameBuffer _frame_buffer;



	class ImageSrcMap;
	typedef UniquePointer<ImageSrcMap> ImageSrcMapPointer;
	typedef boost::circular_buffer<ImageSrcMapPointer> ImageSrcMapBuffer;
	ImageSrcMapBuffer _image_src_map_buffer;



	static std::map<uint32_t, Eigen::Map<const Image_<uint8_t>, Eigen::Aligned>> _image_buffer;

	ImageSize   _image_size;
	ImageSize _image_size_orig;

	void setImage_size(int lvl);

	Calibration _calib;
	Image_<uint16_t> _mask;
	Image_<float> _saliency_map;
	Mat33 _K_inv;
	Trajectory _trajectory;
	int lvl=0;
	uint32_t _frame_id = 0;




protected:
    void optimize(Result*);

private:
    struct ScenePoint;
    typedef UniquePointer<ScenePoint>      ScenePointPointer;
    typedef std::vector<ScenePointPointer> ScenePointPointerList;

    /** removes scene points whose frame id == id */
    ScenePointPointerList removePointsAtFrame(uint32_t id);

//    class DescriptorFrame;
//    typedef UniquePointer<DescriptorFrame>                 DescriptorFramePointer;
//    typedef boost::circular_buffer<DescriptorFramePointer> DescriptorFrameBuffer;

    /** \return the frame data at the given id */
    const DescriptorFrame* getFrameAtId(uint32_t id) const;
	const ImageSrcMap* getSrcImageAtId(uint32_t id) const;

    class DescriptorError;

private:
//    uint32_t _frame_id = 0;
//    Calibration _calib;
//    ImageSize   _image_size;
    Options     _options;
//    Trajectory _trajectory;
	std::map<uint32_t, const uint8_t*> I_ptr_map;

//    DescriptorFrameBuffer _frame_buffer;
    ScenePointPointerList _scene_points;
//    Image_<uint16_t> _mask;
//    Image_<float> _saliency_map;

//    Mat33 _K_inv;
}; // PhotometricBundleAdjustment


#endif //NLDSO_PHOTOMETRICLOSS_PBA_H
