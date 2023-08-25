//
// Created by lei on 28.04.23.
//

#include "pba.h"
#include <ceres/cubic_interpolation.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <ostream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>

#include <string>
#include <vector>
#include <map>
#include "debug.h"
//#include "types.h"
#include "imgproc.h"
#include "sample_eigen.h"
#include "utils.h"

#if defined(WITH_CEREAL)
#include "ceres_cereal.h"
#include "eigen_cereal.h"
#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#endif

#include <cassert>
#include <cmath>
#include <type_traits>
#include <iterator>
#include <algorithm>
#include <map>

// this is just for YCM to stop highlighting openmp as error (there is no openmp
// in clang3.5)
#define HAS_OPENMP __GNUC__ >= 4 && __clang__ == 0

#if HAS_OPENMP
#include <omp.h>
#endif

#include <Eigen/Geometry>

bool optimizeSignal = false;
const size_t PATTERN_SIZE = 8;
const int PATTERN_OFFSETS[PATTERN_SIZE][2] = {{0, 0}, {1, -1}, {-1, 1}, {-1, -1},
                                              {2, 0}, {0, 2},  {-2, 0}, {0, -2}};


static PhotometricBundleAdjustment::Options::DescriptorType
DescriptorTypeFromString(std::string s)
{
    if(pbaUtils::icompare("Intensity", s))
        return PhotometricBundleAdjustment::Options::DescriptorType::Intensity;
    else if(pbaUtils::icompare("IntensityAndGradient", s))
        return PhotometricBundleAdjustment::Options::DescriptorType::IntensityAndGradient;
    else if(pbaUtils::icompare("BitPlanes", s))
        return PhotometricBundleAdjustment::Options::DescriptorType::BitPlanes;
    else {
        Warn("Unknown descriptorType '%s'\n", s.c_str());
        return PhotometricBundleAdjustment::Options::DescriptorType::Intensity;
    }
}

PhotometricBundleAdjustment::Result
PhotometricBundleAdjustment::Result::FromFile(std::string filename)
{
#if defined(WITH_CEREAL)
    std::ifstream ifs(filename);
  if(ifs.is_open()) {
    cereal::BinaryInputArchive ar(ifs);
    Result ret;
    ar(ret);
    return ret;
  } else {
    Fatal("Failed to open %s\n", filename.c_str());
  }
#else
    UNUSED(filename);
    Fatal("compile WITH_CEREAL\n");
#endif
}

bool PhotometricBundleAdjustment::Result::Writer::add(const Result& result)
{
    bool ret = false;

#if defined(WITH_CEREAL)
    std::ofstream ofs(utils::Format("%s/%05d.out", _prefix.c_str(), _counter++));
  if(ofs.is_open()) {
    cereal::BinaryOutputArchive ar(ofs);
    ar(result);
    ret = true;
  }
#else
    pbaUtils::UNUSED(result);
#endif

    return ret;
}

PhotometricBundleAdjustment::Options::Options(const pbaUtils::ConfigFile& cf)
        : maxNumPoints(cf.get<int>("maxNumPoints", 4096)),
          slidingWindowSize(cf.get<int>("slidingWindowSize", 5)),
          patchRadius(cf.get<int>("patchRadius", 2)),
          maskBlockRadius(cf.get<int>("maskBlockRadius", 1)),
          maxFrameDistance(cf.get<int>("maxFrameDistance", 1)),
          numThreads(cf.get<int>("numThreads", -1)),
          doGaussianWeighting((bool) cf.get<int>("doGaussianWeighting", 0)),
          verbose((bool) cf.get<int>("verbose", 1)),
          minScore(cf.get<double>("minScore", 0.75)),
          robustThreshold(cf.get<double>("robustThreshold", 0.05)),
          minValidDepth(cf.get<double>("minValidDepth", 0.01)),
          maxValidDepth(cf.get<double>("maxValidDepth", 1000.0)),
          nonMaxSuppRadius(cf.get<int>("nonMaxSuppRadius", 1)),
          descriptorType(DescriptorTypeFromString(cf.get<std::string>("descriptorType", "Intensity")))
{}

/**
 * simple class to store the image gradient
 */
class ImageGradient
{
public:
    typedef Image_<float> ImageT;

public:
    ImageGradient() = default;
    ImageGradient(const ImageT& Ix, const ImageT& Iy)
            : _Ix(Ix), _Iy(Iy) {}

    inline const ImageT& Ix() const { return _Ix; }
    inline const ImageT& Iy() const { return _Iy; }

    inline ImageT absGradientMag() const
    {
        return _Ix.array().abs() + _Iy.array().abs();
    }

    template <class InputImage>
    inline void compute(const InputImage& I)
    {
        static_assert(std::is_same<typename InputImage::Scalar, uint8_t>::value ||
                      std::is_same<typename InputImage::Scalar, float>::value,
                      "type mismatch, input image must be uint8_t or float");

        optResize(I.rows(), I.cols());
        imgradient(I.data(), ImageSize(I.rows(), I.cols()), _Ix.data(), _Iy.data());
    }

private:
    ImageT _Ix;
    ImageT _Iy;

    void optResize(int rows, int cols)
    {
        if(_Ix.rows() != rows || _Ix.cols() != cols) {
            _Ix.resize(rows, cols);
            _Iy.resize(rows, cols);
        }
    }
}; // ImageGradient


class PhotometricBundleAdjustment:: ImageSrcMap{
	public:
	    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	    typedef Image_<uint8_t> SrcMap_teamp;
	    uint32_t _src_frame_id =0;
	    Eigen::Isometry3d _pose_w;
	    const SrcMap_teamp &_src_map;
	public:
	    ImageSrcMap() = default;
	    ImageSrcMap(const SrcMap_teamp& src_map, uint32_t src_frame_id, const Eigen::Isometry3d& pose_w)
	            : _src_map(src_map), _src_frame_id(src_frame_id) {
		    _pose_w = pose_w;
	    }

	    static inline ImageSrcMap* Create(uint32_t id, const Image_<uint8_t>& image, const Eigen::Isometry3d& pose_w){
		       return new ImageSrcMap(SrcMap_teamp(image), id, pose_w);
	    }

	    inline uint32_t id() const { return _src_frame_id; }
	    inline Eigen::Isometry3d pose_w() const { return _pose_w; }

};

class PhotometricBundleAdjustment::DescriptorFrame
{
public:
    typedef EigenAlignedContainer_<Image_<float>> Channels;
    typedef EigenAlignedContainer_<ImageGradient> ImageGradientList;
public:
    /**
     * \param frame_id the frame number (unique per image)
     * \param I        grayscale input image
     * \param gx       list of x-gradients per channel
     * \param gy       list of y-gradients per channel
     */
    inline DescriptorFrame(uint32_t frame_id, const Channels& channels)
            : _frame_id(frame_id), _channels(channels)
    {
        assert( !_channels.empty() );

        _max_rows = _channels[0].rows() - 1;
        _max_cols = _channels[0].cols() - 1;

		// vectorize the image
		image_tar_vectorized.reserve(_channels[0].rows() *_channels[0].cols());  // Reserve space for efficiency

		for (int row = 0; row <_channels[0].rows(); ++row) {
			for (int col = 0; col < _channels[0].cols(); ++col) {
				image_tar_vectorized.push_back(static_cast<double>(_channels[0](row, col)));
			}
		}

        _gradients.resize( _channels.size() );
        for(size_t i = 0; i < _channels.size(); ++i) {
            _gradients[i].compute(_channels[i]);
        }
    }


    DescriptorFrame(const DescriptorFrame&) = delete;
    DescriptorFrame& operator=(const DescriptorFrame&) = delete;

    inline uint32_t id() const { return _frame_id; }

    inline bool operator<(const DescriptorFrame& other) const
    {
        return _frame_id < other._frame_id;
    }

    inline size_t numChannels() const { return _channels.size(); }

    inline const Image_<float>& getChannel(size_t i) const
    {
        assert(i < _channels.size());
        return _channels[i];
    }

    inline const ImageGradient& getChannelGradient(size_t i) const
    {
        assert(i < _gradients.size());
        return _gradients[i];
    }

    /**
     * \return true of point projects to the image
     */
    template <class ProjType> inline
    bool isProjectionValid(const ProjType& x) const
    {
        return x[0] >= 0.0 && x[0] < _max_cols &&
               x[1] >= 0.0 && x[1] < _max_rows;
    }

    void computeSaliencyMap(Image_<float>& smap) const
    {
        assert( !_gradients.empty() );

        smap.array() = _gradients[0].absGradientMag();
        for(size_t i = 1; i < _gradients.size(); ++i) {
            smap.array() += _gradients[i].absGradientMag().array();
        }
    }

	std::vector<double> image_tar_vectorized;


public:
    static inline DescriptorFrame* Create(uint32_t id, const Image_<uint8_t>& image,
                                          PhotometricBundleAdjustment::Options::DescriptorType type)
    {
        Channels channels;

        switch(type) {
            case PhotometricBundleAdjustment::Options::DescriptorType::Intensity:
                channels.push_back( image.cast<Channels::value_type::Scalar>());
				// std::cout<<"channels size: "<<channels.size()<<std::endl;
				// std::cout<<"Intensity : "<<std::endl;
                break;
            case PhotometricBundleAdjustment::Options::DescriptorType::IntensityAndGradient: {
                channels.push_back( image.cast<Channels::value_type::Scalar>() );
                channels.push_back( Image_<float>(image.rows(), image.cols()) );
                channels.push_back( Image_<float>(image.rows(), image.cols()) );

                imgradient(image.data(), ImageSize(image.rows(), image.cols()),
                           channels[1].data(), channels[2].data());

            } break;
            case PhotometricBundleAdjustment::Options::DescriptorType::BitPlanes: {
                computeBitPlanes(image.data(), ImageSize(image.rows(), image.cols()), channels);
            } break;
        }

        return new DescriptorFrame(id, channels);
    }

private:
    uint32_t _frame_id;
    uint32_t _max_rows;
    uint32_t _max_cols;
    Channels _channels;
    ImageGradientList _gradients;




}; // DescriptorFrame

/**
 * \return bilinearly interpolated pixel value at subpixel location (xf,yf)
 */
template <class Image, class T> inline
T interp2(const Image& I, T xf, T yf, T fillval = 0.0, T offset = 0.0)
{
    const int max_cols = I.cols() - 1;
    const int max_rows = I.rows() - 1;

    xf += offset;
    yf += offset;

    int xi = (int) std::floor(xf);
    int yi = (int) std::floor(yf);

    xf -= xi;
    yf -= yi;

    if( xi >= 0 && xi < max_cols && yi >= 0 && yi < max_rows )
    {
        const T wx = 1.0 - xf;
        return (1.0 - yf) * ( I(yi,   xi)*wx + I(yi,   xi+1)*xf )
               +  yf  * ( I(yi+1, xi)*wx + I(yi+1, xi+1)*xf );
    } else
    {
        if( xi == max_cols && yi < max_rows )
            return ( xf > 0 ) ? fillval : (1.0-yf)*I(yi,xi) + yf*I(yi+1, xi);
        else if( yi == max_rows && xi < max_cols )
            return ( yf > 0 ) ? fillval : (1.0-xf)*I(yi,xi) + xf*I(yi, xi+1);
        else if( xi == max_cols && yi == max_rows )
            return ( xf > 0 || yf > 0 ) ? fillval : I(yi, xi);
        else
            return fillval;
    }

}

template <int N> constexpr int square() { return N*N; }

template <int R, class ImageType, class ProjType, typename T = double>
void interpolateFixedPatch(Vec_<T, square<2*R+1>()>& dst,
                           const ImageType& I, const ProjType& p,
                           const T& fillval = T(0), const T& offset = T(0))
{
    const T x = static_cast<T>( p[0] + offset );
    const T y = static_cast<T>( p[1] + offset );

    auto d_ptr = dst.data();
    for(int r = -R; r <= R; ++r) {
        for(int c = -R; c <= R; ++c) {
            *d_ptr++ = interp2(I, c + x, r + y, fillval);
        }
    }
}


template <int R, typename T = float>
class ZnccPatch_
{
    static_assert(std::is_floating_point<T>::value, "T must be floating point");

public:
    static constexpr int Radius    = R;
    static constexpr int Dimension = (2*R+1) * (2*R+1);

public:
    inline ZnccPatch_() {}

    template <class ImageType, class ProjType> inline
    ZnccPatch_(const ImageType& image, const ProjType& uv) { set(image, uv); }

    template <class ImageType, class ProjType> inline
    const ZnccPatch_& set(const ImageType& I, const ProjType& uv)
    {
        interpolateFixedPatch<R>(_data, I, uv, T(0.0), T(0.0));
        T mean = _data.array().sum() / (T) _data.size();
        _data.array() -= mean;
        _norm = _data.norm();
        return *this;
    }

    template <class ImageType, class ProjType>
    inline static ZnccPatch_ FromImage(const ImageType& I, const ProjType& p)
    {
        ZnccPatch_ ret;
        ret.set(I, p);
        return ret;
    }

    inline T score(const ZnccPatch_& other) const
    {
        T d = _norm * other._norm;

        return d > 1e-6 ? _data.dot(other._data) / d : -1.0;
    }

private:
    Vec_<T, Dimension> _data;
    T _norm;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // ZnccPatch


/**
 */
struct PhotometricBundleAdjustment::ScenePoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    typedef std::vector<uint32_t>        VisibilityList;
    typedef EigenAlignedContainer_<Vec2> ProjectionList;
    typedef ZnccPatch_<2, float>         ZnccPatchType;

    /**
     * Create a scene point with position 'X' and reference frame number 'f_id'
     *
     * We also store the original point for later comparision
     */
    inline ScenePoint(const Vec3& X, uint32_t f_id)
            : _X(X), _X_original(X)
    {
        _f.reserve(8);
        _f.push_back(f_id);
    }

    /**
     * \return true if the scene point has 'f_id' it is visibility list
     */
    inline bool hasFrame(uint32_t f_id) const {
        return std::find(_f.begin(), _f.end(), f_id) != _f.end();
    }

    /**
     * \return the visibility list
     */
    inline const VisibilityList& visibilityList() const { return _f; }

    /** \return the reference frame number (also the first in the list) */
    inline const uint32_t& refFrameId() const { return _f.front(); }

    /** \return the last frame number, most recent observation */
    inline const uint32_t& lastFrameId() const { return _f.back(); }

    /** \return the 3D point associated with the ScenePoint */
    inline const Vec3& X() const { return _X; }
    inline       Vec3& X()       { return _X; }

    /** \return the original 3D point */
    inline const Vec3& getOriginalPoint() const { return _X_original; }

    /** \return the associated patch */
    inline const ZnccPatchType& patch() const { return _patch; }

    inline void addFrame(uint32_t f) { _f.push_back(f); }

    template <class ImageType, class ProjType> inline
    void setZnccPach(const ImageType& I, const ProjType& x)
    {
        _patch.set(I, x);
    }

    inline const std::vector<double>& descriptor() const { return _descriptor; }
    inline       std::vector<double>& descriptor()       { return _descriptor; }

    inline void setSaliency(double v) { _saliency = v; }
    inline const double& getSaliency() const { return _saliency; }

    inline void setRefined(bool v) { _was_refined = v; }
    inline const bool& wasRefined() const { return _was_refined; }

    inline size_t numFrames() const { return _f.size(); }

    inline void setFirstProjection(const Vec_<int,2>& x) { _x = x; }
    inline const Vec_<int,2>& getFirstProjection() const { return _x; }

    Vec3 _X;
    Vec3 _X_original;
    VisibilityList _f;
    ZnccPatchType _patch;
    std::vector<double> _descriptor;

	std::map<uint32_t,Vec3> specularitySequence; //frame number, specularity ={{0,Vec3(0,0,0)}}


    double _saliency  = 0.0;
    bool _was_refined = false;

    Vec_<int,2> _x;
	double  ori_depth;
	double  inv_depth;
	double depth_tar_coeff;
}; // ScenePoint


PhotometricBundleAdjustment::PhotometricBundleAdjustment(
        const Calibration& calib, const ImageSize& image_size, const Options& options)
        : _calib(calib), _image_size(image_size), _options(options),
          _frame_buffer(options.slidingWindowSize),_image_src_map_buffer(options.slidingWindowSize)
{

	_calib._K_orig=_calib.K();
	_image_size_orig=_image_size;


    _mask.resize(_image_size.rows, _image_size.cols);
    _saliency_map.resize(_image_size.rows, _image_size.cols);
    _K_inv = calib.K().inverse();
}

PhotometricBundleAdjustment::~PhotometricBundleAdjustment() {}

static inline int PatchSizeFromRadius(int r) { return (2*r+1) * (2*r+1); }

static inline int PatchRadiusFromLength(int l) { return std::sqrt(l)/2; }

template <typename T, class Image> static inline
void ExtractPatch(T* dst, const Image& I, const Vec_<int,2>& uv, int radius)
{
    int max_cols = I.cols() - radius - 1,
            max_rows = I.rows() - radius - 1;

    for(int r = -radius, i=0; r <= radius; ++r) {
        int r_i = std::max(radius, std::min(uv[1] + r, max_rows));
        for(int c = -radius; c <= radius; ++c, ++i) {
            int c_i = std::max(radius, std::min(uv[0] + c, max_cols));
            dst[i] = static_cast<T>( I(r_i, c_i) );
        }
    }
}

//static Vec_<double,6> PoseToParams_test(const Mat44& T)
//{
//	Vec_<double,6> ret;
//	const Mat_<double,3,3> R = T.block<3,3>(0,0);
//	ceres::RotationMatrixToAngleAxis(ceres::ColumnMajorAdapter3x3(R.data()), ret.data());
//
//	// translation
//	ret[3] = T(0,3);
//	ret[4] = T(1,3);
//	ret[5] = T(2,3);
//	return ret;
//}

//static Mat_<double,4,4> ParamsToPose_test(const double* p)
//{
//	Mat_<double,3,3> R;
//	ceres::AngleAxisToRotationMatrix(p, ceres::ColumnMajorAdapter3x3(R.data()));
//
//	Mat_<double,4,4> ret(Mat_<double,4,4>::Identity());
//	ret.block<3,3>(0,0) = R;
//	ret.block<3,1>(0,3) = Vec_<double,3>(p[3], p[4], p[5]);
//	return ret;
//}


void PhotometricBundleAdjustment::specularityCalcualtion(const ScenePointPointer& pt,
                                                         const Eigen::Isometry3d& T_w_beta_prime,
                                                         const Vec3f* N_ptr,
                                                         const float* R_ptr,
                                                         PBANL::IBL_Radiance* ibl_Radiance
                                                         ){
	int width= _image_size.cols;
	int height= _image_size.rows;
	int r= pt->_x[1];
	int c= pt->_x[0];
	int B = std::max(_options.maskBlockRadius, std::max(2, _options.patchRadius));
	int max_rows = (int) height - B - 1, max_cols = (int) width -  B - 1;


	Eigen::Vector3f point_w= pt->X().cast<float>(); // point in world coordinate
//	Eigen::Isometry3d T_w(_trajectory.atId(pt->refFrameId()).matrix());
	Eigen::Isometry3d T_c(T_w_beta_prime.inverse());
	Eigen::Vector3f point_c= (T_c * pt->X()).cast<float>();
	const Vec3f& normal_pixel = N_ptr[r * width + c];
	Vec3f normal =normalize(normal_pixel); //normal in camera coordinate
	const float roughness_pixel= R_ptr[r * width + c];          // roughness
	float reflectance = 1.0f;                                   // reflectance

	int num_K = 6;                                              // K nearest neighbor search
	std::unordered_map<int, PBANL::pointEnvlight> envLightMap_cur;// maintain envMaps of current frame here
	float image_metallic=1e-3;                                  // metallic
	Eigen::Matrix<float, 3, 1> beta= -point_c;
	Vec3f baseColor(0.0, 0.0, 0.0);                             //base color
	Vec3f N_(normal(0), normal(1), normal(2));
	Vec3f View_beta(beta(0), beta(1), beta(2));
	std::string renderedEnvLight_path=EnvMapPath;


	// ===================================search for Env Light from control points===================
	pcl::PointXYZ searchPoint(point_w.x(), point_w.y(), point_w.z());
	std::vector<int> pointIdxKNNSearch(num_K);
	std::vector<float> pointKNNSquaredDistance(num_K);
	Vec3f key4Search;

	if ( EnvLightLookup->kdtree.nearestKSearch(searchPoint, num_K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
	{
		float disPoint2Env_min= 10.0f;
		int targetEnvMapIdx=-1;
		Vec3f targetEnvMapVal;

		// find the closest control point which domain contains the current point
		for (std::size_t i = 0; i < pointIdxKNNSearch.size (); ++i) {
			Vec3f envMap_point((*(EnvLightLookup->ControlpointCloud))[pointIdxKNNSearch[i]].x, (*(EnvLightLookup->ControlpointCloud))[pointIdxKNNSearch[i]].y,
			                   (*(EnvLightLookup->ControlpointCloud))[pointIdxKNNSearch[i]].z);

			//                        std::cout << "\n------"<<envMap_point.val[0]<< " " << envMap_point.val[1]<< " " << envMap_point.val[2]
			//                                  << " (squared distance: " << pointKNNSquaredDistance[i] << ")" << std::endl;
			// 0.004367 is the squared distance of the closest control point
			if (pointKNNSquaredDistance[i] > 0.004367) { continue; }


			// calculate control point normal
			// transform envMap_point to camera coordinate system
			Eigen::Vector3f envMap_point_c1 = T_c.cast<float>() * Eigen::Vector3f(envMap_point.val[0], envMap_point.val[1], envMap_point.val[2]);
			// project envMap_point to image plane
			// float pixel_x = (fx * envMap_point_c1.x()) / envMap_point_c1.z() + cx;
			// float pixel_y = (fy * envMap_point_c1.y()) / envMap_point_c1.z() + cy;

			Vec2 uv = _calib.project(envMap_point_c1.cast<double>());
			int r_n = std::round(uv[1]), c_n = std::round(uv[0]);
			// check if the projected point is in the image
			if(r_n < B || r_n >= max_rows || c_n < B || c_n >= max_cols) { continue; }

			Vec3f ctrlPointNormal =  N_ptr[r_n * width + c_n];//newNormalMap.at<Vec3f>(r, c);
			ctrlPointNormal=cv::normalize(ctrlPointNormal);

			float angle_consine = ctrlPointNormal.dot(N_);
			if (angle_consine<0.9962){ continue;} // 0.9848 is the cos(10 degree), 0.9962 is the cos(5 degree)
			float disPoint2Env=  pointKNNSquaredDistance[i]/(ctrlPointNormal.dot(N_));
			if (disPoint2Env<disPoint2Env_min){
				disPoint2Env_min=disPoint2Env;
				targetEnvMapIdx=i;
				targetEnvMapVal=envMap_point;
			}


		}
		if (targetEnvMapIdx!=-1){
			key4Search.val[0] = targetEnvMapVal.val[0];
			key4Search.val[1] = targetEnvMapVal.val[1];
			key4Search.val[2] = targetEnvMapVal.val[2];
		}

	}
	// if no envMap point is found, skip this point
	if (key4Search.dot(key4Search)==0){ return ;}

	int ctrlIndex= EnvLightLookup->envLightIdxMap[key4Search];
	if ( EnvLightLookup->envLightIdxMap.size()==0){std::cerr<<"Error in EnvLight->envLightIdxMap! "<<endl;}

	if (envLightMap_cur.count(ctrlIndex)==0){
		stringstream ss;
		string img_idx_str;
		ss << ctrlIndex;
		ss >> img_idx_str;
		string name_prefix = "/envMap";

		string renderedEnvLightfolder =renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLight";
		string renderedEnvLightDiffuse =renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLightDiffuse";
		string envMapDiffuse = renderedEnvLightDiffuse + "/envMapDiffuse_" + img_idx_str + ".pfm";
		PBANL::pointEnvlight pEnv;
		DSONL::EnvMapLookup *EnvMapLookup = new DSONL::EnvMapLookup();
		EnvMapLookup->makeMipMap(pEnv.EnvmapSampler,renderedEnvLightfolder); // index_0: prefiltered Env light
		delete EnvMapLookup;

		//            diffuseMap *diffuseMap = new DSONL::diffuseMap;
		//            diffuseMap->makeDiffuseMap(pEnv.EnvmapSampler, envMapDiffuse); // index_1: diffuse
		//            delete diffuseMap;

		envLightMap_cur.insert(make_pair(ctrlIndex, pEnv));
		//                    cout<<"show size of envLightMap_cur:"<<envLightMap_cur.size()<<endl;
	}

	DSONL::prefilteredEnvmapSampler= & ( envLightMap_cur[ctrlIndex].EnvmapSampler[0]);
	DSONL::brdfSampler_ = & (EnvLightLookup->brdfSampler[0]);
	//diffuseSampler = & (envLightMap_cur[ctrlIndex].EnvmapSampler[1]);
	// ===================================RADIANCE-COMPUTATION====================================
	//PBANL::IBL_Radiance *ibl_Radiance = new PBANL::IBL_Radiance;
	Sophus::SO3f enterPanoroma;
	Sophus::SE3d T_c2w(T_w_beta_prime.rotation(), T_w_beta_prime.translation());
	// TODO: convert the envmap to current "world coordinate" in advance !!!!!! done
	Vec3f radiance_beta = ibl_Radiance->solveForRadiance(View_beta, N_, roughness_pixel, image_metallic,
	                                                     reflectance, baseColor, T_c2w.rotationMatrix(),
	                                                     enterPanoroma.inverse());

	Vec3 radiance_beta_(radiance_beta[0],radiance_beta[1],radiance_beta[2]);

	//	cout<<"radiance_beta: "<<radiance_beta<<endl;

	if(radiance_beta_.x()<0 || radiance_beta_.y()<0 || radiance_beta_.z()<0 || radiance_beta_.x()>1e5 || radiance_beta_.y()>1e5 || radiance_beta_.z()>1e5)
	{
		cout<<"radiance_beta: "<<radiance_beta<<endl;

		cout<<"radiance_beta_.x()<0 || radiance_beta_.y()<0 || radiance_beta_.z()<0"<<endl;
//		exit(0);

		return ;
	}





	pt->specularitySequence.insert(make_pair(_frame_id,Vec3(radiance_beta[0],radiance_beta[1],radiance_beta[2])));

//	pt->specularitySequence[_frame_id]=radiance_beta_;



}


void PhotometricBundleAdjustment::
        addFrame(const uint8_t* I_ptr, const float* Z_ptr, const Vec3f* N_ptr, const float* R_ptr,  const Mat44& T, Result* result)
{
    _trajectory.push_back(T, _frame_id);
    const Eigen::Isometry3d T_w(_trajectory.back());
    const Eigen::Isometry3d T_c(T_w.inverse());
	PBANL::IBL_Radiance *ibl_Radiance = new PBANL::IBL_Radiance;
    typedef Eigen::Map<const Image_<uint8_t>, Eigen::Aligned> SrcMap;
    auto I = SrcMap(I_ptr, _image_size.rows, _image_size.cols);
	I_ptr_map[_frame_id] = I_ptr;
    typedef Eigen::Map<const Image_<float>, Eigen::Aligned> SrcDepthMap;
    auto Z = SrcDepthMap(Z_ptr, _image_size.rows, _image_size.cols);
    // check the depth map
	std::cout<<"show options.descriptorType"<<static_cast<int>(_options.descriptorType)<<std::endl;
	DescriptorFrame* frame = DescriptorFrame::Create(_frame_id, I, _options.descriptorType);

    int B = std::max(_options.maskBlockRadius, std::max(2, _options.patchRadius));
    int max_rows = (int) I.rows() - B - 1,
            max_cols = (int) I.cols() - B - 1,
            radius = _options.patchRadius,
            patch_length = PatchSizeFromRadius(radius),
            descriptor_dim = (int) frame->numChannels() * patch_length,
            mask_radius = _options.maskBlockRadius;

    //
    // Establish "correspondences" with the old data. This is the visibility list
    // computation
    //
    _mask.setOnes();
    std::cout<<"show scene point size:"<<_scene_points.size() <<std::endl;
    int num_updated = 0, max_num_to_update = 0;
    for(size_t i = 0; i < _scene_points.size(); ++i) {
        const auto& pt = _scene_points[i];
        int f_dist = _frame_id - pt->lastFrameId();
        if(f_dist <= _options.maxFrameDistance) {
			// do not go too far back
            //
            // If the point projects to the current frame and it zncc score is
            // sufficiently highly, we'll add the current image to its visibility list
			//
            Vec2 uv = _calib.project(T_c * pt->X());
            ++max_num_to_update;
            int r = std::round(uv[1]), c = std::round(uv[0]);

            if(r >= B && r < max_rows && c >= B && c <= max_cols) {
                typename ScenePoint::ZnccPatchType other_patch(I, uv);
                auto score = pt->patch().score( other_patch );
                if(score > _options.minScore) {
                    num_updated++;
																	// old TODO update the patch for the new frame data
                    pt->addFrame(_frame_id);


					// calculate specularity
					specularityCalcualtion(pt, T_w, N_ptr, R_ptr, ibl_Radiance);

                    //
                    // block an area in the mask to prevent initializing redundant new
                    // scene points
                    //
					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    for(int r_i = -mask_radius; r_i <= mask_radius; ++r_i)
                        for(int c_i = -mask_radius; c_i <= mask_radius; ++c_i)
                            _mask(r+r_i, c+c_i) = 0;
                }
            }


//			// back projection on previous frames, check and add previous frame to visualization frame list added by lei
//			// note: Image_<uint8_t>& image
//			uint32_t frame_id_start = _frame_buffer.front()->id(),frame_id_end=_frame_id;
//			for(uint32_t id = frame_id_start; id <frame_id_end; ++id) {
//				if (id==pt->refFrameId()){continue;} // skip the point itself
////				std::cout<<"show and check the id: "<<id<<std::endl;
//				Eigen::Isometry3d T_c_id(Eigen::Isometry3d(_trajectory.atId(id)).inverse().matrix());
//				Vec2 uv = _calib.project(T_c_id * pt->X());
//				int r = std::round(uv[1]), c = std::round(uv[0]);
//				if(r >= B && r < max_rows && c >= B && c <= max_cols) {
//					typedef Eigen::Map<const Image_<uint8_t>, Eigen::Aligned> SrcMap;
//					auto I_prev = SrcMap(I_ptr_map[id], _image_size.rows, _image_size.cols);
//
//					typename ScenePoint::ZnccPatchType other_patch(I_prev, uv);
//					float score = pt->patch().score( other_patch );
//					if(score > _options.minScore) {
//
//											// check if the id is already in the visibility list: pt->visibilityList(), which is of type std::vector<uint32_t>
//						// if not, add it to the visibility list
//						auto it = std::find(pt->visibilityList().begin(), pt->visibilityList().end(), id);
//						if (it == pt->visibilityList().end()) {
//							pt->addFrame(id);
//						}
//
//
//
//					}
//				}
//			}

		}
    }


    //
    // Add new scene points
    //
    decltype(_scene_points) new_scene_points;
    new_scene_points.reserve( max_rows * max_cols * 0.5 );
    frame->computeSaliencyMap(_saliency_map);
    // print the saliency map

	cv::Mat salientImage(_saliency_map.rows(), _saliency_map.cols(), CV_32F, _saliency_map.data());

    // Display the image using OpenCV
    int count_selectedPoint = 0;
    salientImage.convertTo(salientImage, CV_8UC1);

//	cv::imshow("salientImage:"+std::to_string(_frame_id),salientImage);
//	cv::waitKey(0);

    typedef IsLocalMax_<decltype(_saliency_map), decltype(_mask)> IsLocalMax;
    const IsLocalMax is_local_max(_saliency_map, _mask, _options.nonMaxSuppRadius);


	// pixel selector
    for(int y = B; y < max_rows; ++y) {
        for(int x = B; x < max_cols; ++x) {

            double z = Z(y,x);
            if(z >= _options.minValidDepth && z <= _options.maxValidDepth) {

                if(is_local_max(y, x)) {
                    Vec3 X = T_w * (z * _K_inv * Vec3(x, y, 1.0)); // X in the world frame
                    std::unique_ptr<ScenePoint> p = std::make_unique<ScenePoint>(X, _frame_id);// associate a new scene point with its frame id
                    Vec_<int,2> xy(x, y);
                    p->setZnccPach( I, xy );
                    p->descriptor().resize(descriptor_dim);
                    p->setSaliency( _saliency_map(y,x) );
                    p->setFirstProjection(xy);
					p->ori_depth = z; // added by lei
					p->inv_depth = 1.0f/z; // added by lei

                    new_scene_points.push_back(std::move(p));
                }
            }


        }
    }

    std::cout<<"new scene points size: "<<new_scene_points.size()<<std::endl;
	// keep the best N points
    if(new_scene_points.size() > (size_t) _options.maxNumPoints) {
        auto nth = new_scene_points.begin() + _options.maxNumPoints;
        std::nth_element(new_scene_points.begin(), nth, new_scene_points.end(),
                         [&](const ScenePointPointer& a, const ScenePointPointer& b) {
                             return a->getSaliency() > b->getSaliency();
                         });
        new_scene_points.erase(nth, new_scene_points.end());
    }

	std::cout<<"show final new scene points size: "<<new_scene_points.size()<<std::endl;


	for(size_t i = 0; i < new_scene_points.size(); ++i) {

		const auto& pt = new_scene_points[i];
		specularityCalcualtion(pt,T_w,N_ptr,R_ptr,ibl_Radiance);

	}

	delete ibl_Radiance;










    //
    // extract the descriptors
    //
    const int num_channels = frame->numChannels(), num_new_points = (int) new_scene_points.size();
	// use Info print out the number of new points and the number of channels
	Info("new points %d channels %d\n", num_new_points, num_channels);
    Info("updated %d [%0.2f%%] max %d new %d\n", num_updated, 100.0 * num_updated / _scene_points.size(), max_num_to_update, num_new_points);
    for(int k = 0; k < num_channels; ++k) {
        const auto& channel = frame->getChannel(k);
        for(int i = 0; i < num_new_points; ++i) {
            auto ptr = new_scene_points[i]->descriptor().data() + k*patch_length;
            ExtractPatch(ptr, channel, new_scene_points[i]->getFirstProjection(), radius);
        }
    }
    _scene_points.reserve(_scene_points.size() + new_scene_points.size());
    std::move(new_scene_points.begin(), new_scene_points.end(), std::back_inserter(_scene_points));
    _frame_buffer.push_back(DescriptorFramePointer(frame));


    if(_frame_buffer.full()) {


		std::cout<<"show Scene point size in current round of optimiation : "<<_scene_points.size()<<std::endl;


		// define a file to save the optimized points and corresponding pixel values
        uint32_t frame_id_start = _frame_buffer.front()->id(),
                frame_id_end   = _frame_buffer.back()->id();
        int num_selected_points = 0;

		//count the number of points in each frame in the current round of optimization
		int counter_frame1 = 0;
		int counter_frame2 = 0;
		int counter_frame3 = 0;
		int counter_frame4 = 0;
		int counter_frame5 = 0;
		for(auto& pt : _scene_points) {

			if(pt->numFrames() >= 3 && pt->refFrameId() >= frame_id_start) {

				if (pt->refFrameId()== frame_id_start){
					counter_frame1++;
				}else if(pt->refFrameId()== frame_id_start+1){
					counter_frame2++;
				}else if(pt->refFrameId()== frame_id_start+2){
					counter_frame3++;
				}else if(pt->refFrameId()== frame_id_start+3){
					counter_frame4++;
				}else if(pt->refFrameId()== frame_id_start+4){
					counter_frame5++;
				}


			}
		}

		// print the number of points in each frame in the current round of optimization
		std::cout<<"number of points in frame 0: "<<counter_frame1<<std::endl;
		std::cout<<"number of points in frame 1: "<<counter_frame2<<std::endl;
		std::cout<<"number of points in frame 2: "<<counter_frame3<<std::endl;
		std::cout<<"number of points in frame 3: "<<counter_frame4<<std::endl;
		std::cout<<"number of points in frame 4: "<<counter_frame5<<std::endl;


//        for(auto& pt : _scene_points) {
//            if(pt->numFrames() >= 3 && pt->refFrameId() >= frame_id_start) {
//                num_selected_points++;
//				for(auto id : pt->visibilityList()) {
//					if(id >= frame_id_start && id <= frame_id_end) {
////						auto camera_pose = camera_params_test[id].data();
////						// convert camera pose to Eigen::Isometry3d
////						Mat44 camera_pose_eigen = ParamsToPose_test(camera_pose);
////						Vec3 xyz = pt->X();
////						// transform the point from world coordinate to camera coordinate using block matrix multiplication
////						Vec3 xyz_cam = camera_pose_eigen.block<3,3>(0,0)*xyz + camera_pose_eigen.block<3,1>(0,3);
////						Vec2 uv = _calib.project(xyz_cam);
//						// save the optimized points and corresponding pixel values
////						myfile << pt->refFrameId()<<" "<<pt->_x[0]<<" "<<pt->_x[1]<<" "<<id<<" "<<std::round(uv[0])<<" "<<std::round(uv[1])<<std::endl;
//
//						myfile << pt->refFrameId()<<" "<<id<<std::endl;
//
//					}
//				}
//            }
//        }
//		myfile.close();
//      Info("!!! myfile1 saved and show num_selected_points: %d\n", num_selected_points);


		// use image pyramid to optimize the points







		int countSame=0;
		int countDiff=0;
		int counter_size_1=0;
		int counter_size_2=0;
		int counter_size_3=0;
		int counter_size_4=0;
		int counter_size_5=0;
		int counter_specu_size_1=0;
		int counter_specu_size_2=0;
		int counter_specu_size_3=0;
		int counter_specu_size_4=0;
		int counter_specu_size_5=0;
		for(auto& pt : _scene_points) {
			// statistics of the visibilityList size of all point
			if (pt->visibilityList().size()==1){
				counter_size_1++;
			}else if (pt->visibilityList().size()==2){
				counter_size_2++;
			}else if (pt->visibilityList().size()==3){
				counter_size_3++;
			}else if (pt->visibilityList().size()==4){
				counter_size_4++;
			}else if (pt->visibilityList().size()==5){
				counter_size_5++;
			}
		}

		for (auto &pt : _scene_points) {
//			for (auto& ele :pt->specularitySequence) {
//				cout<<"show pt->specularitySequence["<<ele.first<<"]: \n "<<ele.second<<endl;
//
//			}
			if (pt->specularitySequence.size()==1){
				counter_specu_size_1++;
			}else if (pt->specularitySequence.size()==2){
				counter_specu_size_2++;
			}else if (pt->specularitySequence.size()==3){
				counter_specu_size_3++;
			}else if (pt->specularitySequence.size()==4){
				counter_specu_size_4++;
			}else if (pt->specularitySequence.size()==5){
				counter_specu_size_5++;
			}
		}

		// check if visibility has the same length of specularity sequence
		for(auto& pt : _scene_points) {
			if (pt->visibilityList().size()==pt->specularitySequence.size()){
				countSame++;
			}else{
				countDiff++;
			}
		}

		Info("\n counter_size_1:  %d", counter_size_1);
		Info("\n counter_size_2:  %d", counter_size_2);
		Info("\n counter_size_3:  %d", counter_size_3);
		Info("\n counter_size_4:  %d", counter_size_4);
		Info("\n counter_size_5:  %d", counter_size_5);
		Info("\n countSame:  %d", countSame);
		Info("\n countDiff:  %d", countDiff);
		Info("\n counter_specu_size_1:  %d", counter_specu_size_1);
		Info("\n counter_specu_size_2:  %d", counter_specu_size_2);
		Info("\n counter_specu_size_3:  %d", counter_specu_size_3);
		Info("\n counter_specu_size_4:  %d", counter_specu_size_4);
		Info("\n counter_specu_size_5:  %d", counter_specu_size_5);




        optimize(result);
		optimizeSignal=true;

		// save the optimized points and corresponding pixel values
		//		std::ofstream myfile2;
		//		std::string filename2 = "optimized_points" +std::to_string(_frame_id) +"_after_optimization.txt";
		//		myfile2.open (filename2);
		//		std::map<uint32_t, Vec_<double,6>> camera_params_after_optimization;
		//		for(uint32_t id = frame_id_start; id <= frame_id_end; ++id) {
		//			camera_params_after_optimization[id] =PoseToParams_test(Eigen::Isometry3d(_trajectory.atId(id)).inverse().matrix());
		//		}
		//
		//		// project the refined points to the image plane
		//
		//		for(auto& pt : _scene_points) {
		//			if(pt->numFrames() >= 2 && pt->refFrameId() >= frame_id_start) {
		//				num_selected_points++;
		//				for(auto id : pt->visibilityList()) {
		//					if(id >= frame_id_start && id <= frame_id_end) {
		//						auto camera_pose = camera_params_after_optimization[id].data();
		//						// convert camera pose to Eigen::Isometry3d
		//						Mat44 camera_pose_eigen = ParamsToPose_test(camera_pose);
		//						Vec3 xyz = pt->X();
		//						Vec3 xyz_cam = camera_pose_eigen.block<3,3>(0,0)*xyz + camera_pose_eigen.block<3,1>(0,3);
		//						Vec2 uv = _calib.project(xyz_cam);
		//						// save the optimized points and corresponding pixel values
		//						myfile2 << pt->refFrameId()<<" "<<pt->_x[0]<<" "<<pt->_x[1]<<" "<<id<<" "<<std::round(uv[0])<<" "<<std::round(uv[1])<<std::endl;
		//					}
		//				}
		//			}
		//		}
		//
		//		myfile2.close();
		//		Info("!!! myfile2 saved and show num_selected_points: %d\n", num_selected_points);


    }

    ++_frame_id;
}

static inline std::vector<double>
MakePatchWeights(int radius, bool do_gaussian, double s_x = 1.0,
                 double s_y = 1.0, double a = 1.0)
{
    int n = (2*radius + 1) * (2*radius + 1);

    if(do_gaussian) {
        std::vector<double> ret(n);
        double sum = 0.0;
        for(int r = -radius, i = 0; r <= radius; ++r) {
            const double d_r = (r*r) / s_x;
            for(int c = -radius; c <= radius; ++c, ++i) {
                const double d_c = (c*c) / s_y;
                const double w = a * std::exp( -0.5 * (d_r + d_c) );
                ret[i] = w;
                sum += w;
            }
        }

        for(int i = 0; i < n; ++i) {
            ret[i] /= sum;
        }

        return ret;
    } else {
        return std::vector<double>(n, 1.0);
    }
}



static inline float specularityWeight( Vec3 refSpecularity, Vec3 tarSpecularity)
{
	Vec3 deltaSpecularity = (refSpecularity - tarSpecularity).normalized() ; // RGB

//	Vec3 deltaSpecularity = (refSpecularity - tarSpecularity); // RGB

	double sumWeight= 0.587* abs(deltaSpecularity.y())+0.114* abs(deltaSpecularity.x())+0.299* abs(deltaSpecularity.z());

	//	if (sumWeight<0.5f){
	//		cout<<"sumWeight great than 0.5 !!!: "<<sumWeight<<endl;
	//	}

	//	Info("CHECKING sumWeight %d\n", sumWeight);

	if (sumWeight==0.0f){
		return -1.0;}
	else{
		//float y = exp(-15*sumWeight);
//		float y = exp(-6.0f*sumWeight);

		float y = 10*exp(-5.0f*sumWeight);

		//		if (sumWeight<0.5f){
		//			cout<<"y value !!!: "<<y<<endl;
		//		}
		return y;
	}

}









static Vec_<double,7> PoseToParams(const Mat44& T)
{
    Vec_<double,7> ret;
    const Mat_eigen<double,3,3> R = T.block<3,3>(0,0);
//    ceres::RotationMatrixToAngleAxis(ceres::ColumnMajorAdapter3x3(R.data()), ret.data());
	ceres::RotationMatrixToQuaternion(ceres::ColumnMajorAdapter3x3(R.data()), ret.data());

//	Eigen::Quaterniond q(R);
//	q=q.normalized();
//	std::cout<<"q: "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<std::endl;
//	std::cout<<"show ret quaternion: " << ret[0]<<" "<<ret[1]<<" "<<ret[2]<<" "<<ret[3]<<std::endl;

	ret[4] = T(0,3);
	ret[5] = T(1,3);
	ret[6] = T(2,3);
	return ret;

}

static Mat_eigen<double,4,4> ParamsToPose(const double* p)
{
	Mat_eigen<double,3,3> R;
//    ceres::AngleAxisToRotationMatrix(p, ceres::ColumnMajorAdapter3x3(R.data()));
	ceres::QuaternionToRotation(p, ceres::ColumnMajorAdapter3x3(R.data()));

	Mat_eigen<double,4,4> ret(Mat_eigen<double,4,4>::Identity());
    ret.block<3,3>(0,0) = R;
    ret.block<3,1>(0,3) = Vec_<double,3>(p[4], p[5], p[6]);
    return ret;
}

class PhotometricBundleAdjustment::DescriptorError
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /**
     * \param radius  the patch radius
     * \param calib   the camera calibration
     * \param p0      the reference frame descriptor must have size (2*radius+1)^2
     * \param frame   descriptor data of the image we are matching against
     */
    DescriptorError(const Calibration& calib, const std::vector<double>& p0,
                    const DescriptorFrame* frame, const double w,
	                const size_t image_width,
	                const size_t image_height,
	                double mean_patch_value_ref,
	                std::vector<double>& patch,
	                double x_norm, double y_norm,
	                double * refCameraPose,
	                double depth_coeff
	                )
            : _radius(PatchRadiusFromLength(p0.size() / frame->numChannels())),
              _calib(calib), _p0(p0.data()), _frame(frame), img_width (image_width),
	                                                            img_height (image_height),
	                                                         mean_patch_value (mean_patch_value_ref),
	      depth_coeffe(depth_coeff)

    {



		image_grid.reset(new ceres::Grid2D<double, 1>(&_frame->image_tar_vectorized[0], 0, image_height, 0, image_width));
		compute_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> >(*image_grid));
		x_host_normalized = x_norm;
		y_host_normalized = y_norm;

		ref_camera = refCameraPose;

		if (mean_patch_value == 0.0){
			mean_patch_value = 1.0;
		}
		patch_values=patch;

		patch_weights_specularity=w;
        // TODO should just pass the config to get the radius value
        assert( p0.size() == w.size() );
    }
	std::vector<double> patch_values;
	double patch_weights_specularity;
	double mean_patch_value;
	double x_host_normalized;
	double y_host_normalized;
	double depth_coeffe;
	double * ref_camera=nullptr;
    static ceres::CostFunction* Create(const Calibration& calib,
                                       const std::vector<double>& p0,
                                       const DescriptorFrame* f,
                                       const double w,
	                                   size_t image_width,
	                                   size_t image_height,
	                                   double mean_patch_value,
	                                   std::vector<double>& patch,
	                                   double xnorm,
	                                   double ynorm,
	                                   double * ref_camera_ptr,
	                                   double depth_coeff
	                                   )
    {
//        return new ceres::AutoDiffCostFunction<DescriptorError, ceres::DYNAMIC, 6, 3>(
//                new DescriptorError(calib, p0, f, w), p0.size());

		return new ceres::AutoDiffCostFunction<DescriptorError, PATTERN_SIZE,7,7,1>(
		        new DescriptorError(calib, p0, f, w,image_width,image_height, mean_patch_value , patch , xnorm,ynorm, ref_camera_ptr, depth_coeff ));

    }

    template <class T> inline
    bool operator()(const T* const camera_src,const T* const camera, const T* const pidepth, T* presiduals) const
    {


		Eigen::Map<Eigen::Array<T, PATTERN_SIZE, 1>> residuals(presiduals);
		T quaternion[4] = {camera[0], camera[1], camera[2], camera[3]};
		T quaternion_ref_camera[4] = { T(camera_src[0]), T(camera_src[1]), T(camera_src[2]), T(camera_src[3])}; // mark here
		const T& depth(*pidepth);



		Eigen::Array<T, PATTERN_SIZE, 1> patch_values_target;
		// define R
		Eigen::Matrix<T, 3, 3> R, R_ref2W;
		Eigen::Matrix<T, 3, 1> t_ref2W;
		ceres::QuaternionToRotation(quaternion_ref_camera, ceres::ColumnMajorAdapter3x3(R.data()));
		R_ref2W= R.transpose();
		t_ref2W.x() = -T(camera_src[4]);
		t_ref2W.y() = -T(camera_src[5]);
		t_ref2W.z() = -T(camera_src[6]);
		t_ref2W= R_ref2W * t_ref2W;

//		T p_c[3];
//		ceres::UnitQuaternionRotatePoint(quaternion_ref_camera, point, p_c);
//		p_c[0] += ref_camera[4];
//		p_c[1] += ref_camera[5];
//		p_c[2] += ref_camera[6]; // depth of the point in the reference frame


		for (size_t i = 0; i < PATTERN_SIZE; i++){
			int du = PATTERN_OFFSETS[i][0];
			int dv = PATTERN_OFFSETS[i][1];

			T   p_host_normalized[3] = {T(x_host_normalized), T(y_host_normalized), T(1.f)};
				p_host_normalized[0] += T(du * 1.0 / _calib.fx());
				p_host_normalized[1] += T(dv * 1.0 / _calib.fy());
			// apply ref_camera to transform the point to the target frame
			Eigen::Matrix<T, 3, 1> p_ref_normalized;
//			T p_ref_normalized[3];
//			p_ref_normalized[0] = depth * p_host_normalized[0];
//			p_ref_normalized[1] = depth * p_host_normalized[1];
//			p_ref_normalized[2] = depth * p_host_normalized[2];
//
			p_ref_normalized.x() =   T(depth_coeffe)*depth * p_host_normalized[0];
			p_ref_normalized.y() =   T(depth_coeffe)*depth * p_host_normalized[1];
			p_ref_normalized.z() =   T(depth_coeffe)*depth * p_host_normalized[2];
			p_ref_normalized = R_ref2W * p_ref_normalized + t_ref2W;


			T p_ref_normalized_array[3] = {p_ref_normalized.x(), p_ref_normalized.y(), p_ref_normalized.z()};


			T p[3];
			ceres::UnitQuaternionRotatePoint(quaternion, p_ref_normalized_array, p);

			p[0] += (camera[4]);
			p[1] += (camera[5]);
			p[2] += (camera[6]);

			T u = T(_calib.fx())*(p[0] / p[2]) + T(_calib.cx());
			T v = T(_calib.fy())*(p[1] / p[2]) + T(_calib.cy());

			compute_interpolation->Evaluate(v, u, &patch_values_target[i]);
		}

//		for (size_t i = 0; i < PATTERN_SIZE; i++){
//			int du = PATTERN_OFFSETS[i][0];
//			int dv = PATTERN_OFFSETS[i][1];
//			T u_new = u_w + T(du);
//			T v_new = v_w + T(dv);
//			// print out u_w and v_w
//			compute_interpolation->Evaluate(v_new, u_new, &patch_values_target[i]);
//
//		}



//		T xw[3];
//		T point[3];
//        ceres::AngleAxisRotatePoint(camera, point, xw);
//
//		xw[0] += camera[3];
//        xw[1] += camera[4];
//        xw[2] += camera[5];
//
//        T u_w, v_w;
//        _calib.project(xw, u_w, v_w);

//        for(size_t k = 0, i=0; k < _frame->numChannels(); ++k) {
//            const auto& I = _frame->getChannel(k);
//            const auto& G = _frame->getChannelGradient(k);
//            const auto& Gx = G.Ix();
//            const auto& Gy = G.Iy();
//            for(int y = -_radius, j = 0; y <= _radius; ++y) {
//                const T v = v_w + T(y);
//                for(int x = -_radius; x <= _radius; ++x, ++i, ++j) {
//                    const T u = u_w + T(x);
//                    const T i0 = T(_p0[i]);
//                    const T i1 = SampleWithDerivative(I, Gx, Gy, u, v);
//                    residuals[i] = _patch_weights[j] * (i0 - i1);
//                }
//            }
//        }

		// sum of patch_values_target
		T mean_patch_value_target = patch_values_target.sum() /T(PATTERN_SIZE);

//		T mean_patch_value_target = patch_values_target/(PATTERN_SIZE);
		for (size_t i = 0; i < PATTERN_SIZE; i++){
//			residuals[i] = patch_values_target[i] - (mean_patch_value_target / mean_patch_value)*T(patch_values[i]);
//			residuals[i] = T(patch_weights_specularity)*(patch_values_target[i] -  (mean_patch_value_target / mean_patch_value)*T(patch_values[i]));
			residuals[i] = T(patch_weights_specularity)*(patch_values_target[i] -  T(patch_values[i]));

		}

        // maybe we should return false if the point goes out of the image!  done!
        return true;
    }


private:
    const int _radius;
    const Calibration& _calib;
    const double* const _p0;
    const DescriptorFrame* _frame;


	size_t img_width;
	size_t img_height;

	std::unique_ptr<ceres::Grid2D<double, 1> > image_grid;
	std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_interpolation;



}; // DescriptorError








static inline ceres::Solver::Options
GetSolverOptions(int num_threads, bool verbose = false, double tol = 1e-6)
{
    ceres::Solver::Options options;

    options.linear_solver_type            = ceres::SPARSE_SCHUR;

    options.minimizer_type                = ceres::TRUST_REGION;
    options.trust_region_strategy_type    = ceres::LEVENBERG_MARQUARDT;

    options.preconditioner_type           = ceres::CLUSTER_JACOBI;
    options.visibility_clustering_type    = ceres::SINGLE_LINKAGE;
    options.minimizer_progress_to_stdout  = verbose;
    options.max_num_iterations            = 500;

    options.num_threads = std::thread::hardware_concurrency();
//    options.num_linear_solver_threads = options.num_threads;

    options.function_tolerance  = tol;
    options.gradient_tolerance  = tol;
    options.parameter_tolerance = tol;

    return options;
}


void PhotometricBundleAdjustment::optimize(Result* result)
{
    uint32_t frame_id_start = _frame_buffer.front()->id(),
            frame_id_end   = _frame_buffer.back()->id();

    auto patch_weights = MakePatchWeights(_options.patchRadius, _options.doGaussianWeighting);

    //
    // collect the camera poses in a single map for easy access
    //
    std::map<uint32_t, Vec_<double,7>> camera_params;
    for(uint32_t id = frame_id_start; id <= frame_id_end; ++id) {
        // TODO:NOTE camera parameters are inverted
        camera_params[id] = PoseToParams(Eigen::Isometry3d(_trajectory.atId(id)).inverse().matrix());
        std::cout<<"before optimize: _trajectory.atId(it.first:"  <<id <<"\n"<<_trajectory.atId(id)<<std::endl;
    }
    Info("_trajectory.size() = %d, frame_id_start=  %d", _trajectory.size(), frame_id_start);
    Info("_scene_points.size() = %d", _scene_points.size());



    //-----------------------------------------optimization----- start-------------------------------------------------------------------------
//     get the points that we *should* optimize. They must have a large enough
//     visibility list

    ceres::Problem problem;
	ceres::LocalParameterization* camera_parameterization = new ceres::ProductParameterization(new ceres::QuaternionParameterization(),
	                                                                                           new ceres::IdentityParameterization(3));



    int num_selected_points = 0;


	int depth_counter=0;

	for(auto& pt : _scene_points) {

        if(pt->numFrames() >= 3 && pt->refFrameId() >= frame_id_start) {
            num_selected_points++;
			// get the patch values for the ref frame
			const DescriptorFrame* ref = getFrameAtId(pt->refFrameId());
//			std::cout<<"show ref-frame id: "<<pt->refFrameId()<<std::endl;
			std::unique_ptr<ceres::Grid2D<double, 1>> image_grid;
			std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > > compute_interpolation;
			// test value
																																							//			cv::imshow("Reconstructed Channel", channel);
																																									//			cv::waitKey(0);
			image_grid.reset(new ceres::Grid2D<double, 1>(&ref->image_tar_vectorized[0], 0, _image_size.rows, 0, _image_size.cols));
			compute_interpolation.reset(new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> >(*image_grid));

			std::vector<double> patch(PATTERN_SIZE, 0.0);
//			std::vector<double> patch_sepcularity_weight(PATTERN_SIZE, 1.0);
			for (size_t i = 0; i < PATTERN_SIZE; i++){
				int du = PATTERN_OFFSETS[i][0];
				int dv = PATTERN_OFFSETS[i][1];
				float u_new = pt->_x[0] + du; // col
				float v_new = pt->_x[1] + dv; // row
				compute_interpolation->Evaluate(v_new, u_new, &patch[i]);
			}
			// print patch
//			std::cout<<"show patch: "<<std::endl;
//			for (int i = 0; i < patch.size(); ++i) {
//				std::cout<<"show patch content:"<<patch[i]<<" ";
//			}

			// Remark by lei: the original code optimize the point in world coordinate and the abs camera pose, which is not better because the point in
			// world coordinate is not accurate enough, because it is calcualate using the abs camera pose. Hence we decide to optimize the depth and
			// estimated abs camera pose
			// the key is to use projected patch to calculate the residual


            for(auto id : pt->visibilityList()) {

                if(id >= frame_id_start && id <= frame_id_end) {
					// print out id

					if(pt->ori_depth < 1e-3) { continue; }
					if (id==pt->refFrameId()){ continue ;}
					double x_norm = (pt->_x[0] - _calib.cx()) / _calib.fx();
					double y_norm = (pt->_x[1] - _calib.cy()) / _calib.fy();

                    pt->setRefined(true);
					float specularity_weight = 1.0;

					if(pt->specularitySequence.find(id) != pt->specularitySequence.end()){
						specularity_weight = specularityWeight(pt->specularitySequence[pt->refFrameId()],pt->specularitySequence[id]);

						if (specularity_weight==-1){
							specularity_weight=1.0f;
						}
						if (isnan(specularity_weight)){
//							specularity_weight=1.0f;

							continue;
						}
					}
//					else {
////						continue;
//						// if the point has no specularity, continue
//						specularity_weight = 1.0;
//					}

					if (specularity_weight!=1.0){
						cout<<"check patch_sepcularity_weight: "<<specularity_weight<<endl;
					}



					double * depth_ptr = & pt->ori_depth;
					double * depth_coeff_ptr = & pt->depth_tar_coeff;
					double * ref_camera_ptr = camera_params[pt->refFrameId()].data();
                    double * camera_ptr = camera_params[id].data();
                    double * xyz = pt->X().data();
					double mean_patch_value= 0;
					for (int i = 0; i < patch.size(); ++i) { mean_patch_value += patch[i];}
					mean_patch_value /= patch.size();


					const auto huber_t = _options.robustThreshold;
                    auto* loss = huber_t > 0.0 ? new ceres::HuberLoss(huber_t) : nullptr;
                    ceres::CostFunction* cost = nullptr;
                    cost = DescriptorError::Create(_calib, pt->descriptor(), getFrameAtId(id), specularity_weight, size_t(_image_size.cols), size_t(_image_size.rows),
					                                mean_patch_value, patch, x_norm, y_norm,ref_camera_ptr,1);

					problem.AddResidualBlock(cost, loss,ref_camera_ptr,camera_ptr, depth_ptr);
					problem.SetParameterBlockConstant(depth_ptr);


					problem.SetParameterization(ref_camera_ptr, camera_parameterization);
					problem.SetParameterization(camera_ptr, camera_parameterization);




                }
            }
        }
    }


//	for(uint32_t id = frame_id_start; id <= frame_id_end; ++id) {
//		double * camera_ptr_para = camera_params[id].data();
//		problem.SetParameterization(camera_ptr_para, camera_parameterization);
//	}

    // set the first camera constant
    {

        auto p = camera_params[frame_id_start].data();
        Info("first camera id %d\n", frame_id_start);
        if(problem.HasParameterBlock(p)) {
            Info("setting first camera constant" );
            problem.SetParameterBlockConstant(p);
        } else {
            Warn("first camera is not in bundle\n");
        }
    }
    Info("Using %d points (%d residual blocks) [id start %d]\n",num_selected_points, problem.NumResidualBlocks(), frame_id_start);



	std::cout<<"depth_counter: "<<depth_counter<<std::endl; // depth_counter: 24433


    ceres::Solver::Summary summary;
	#if HAS_OPENMP
		int num_threads = _options.numThreads > 0 ? _options.numThreads : std::min(omp_get_max_threads(), 4);
	#else
		int num_threads = 4;
	#endif

    ceres::Solve(GetSolverOptions(num_threads, _options.verbose), &problem, &summary);
	std::cout << summary.BriefReport() << std::endl;


	//-----------------------------------------optimization----- end-------------------------------------------------------------------------



    //
    // TODO: run another optimization pass over residuals with small error
    // (eliminate the outliers)
    //
    //
    // put back the refined camera poses
    //
    for(auto& it : camera_params) {
        _trajectory.atId(it.first) = Eigen::Isometry3d( ParamsToPose(it.second.data())).inverse().matrix();
        std::cout<<"after optimize: _trajectory.atId(it.first:"  <<it.first <<"\n"<<_trajectory.atId(it.first)<<std::endl;
    }


	// put back the refined depth values

		for(auto& pt : _scene_points) {
			if(pt->numFrames() >= 3 && pt->refFrameId() >= frame_id_start) {
				for(auto id : pt->visibilityList()) {
					if(id >= frame_id_start && id <= frame_id_end) {
					const Eigen::Isometry3d refined_camera_pose_w(_trajectory.atId(pt->refFrameId()));
//						Vec3 X = refined_camera_pose_w * ((1.0/pt->inv_depth) * _K_inv * Vec3(pt->_x[0], pt->_x[1], 1.0)); // X in the world frame
					Vec3 X = refined_camera_pose_w * ((pt->ori_depth) * _K_inv * Vec3(pt->_x[0], pt->_x[1], 1.0)); // X in the world frame

					pt->X() = X;
					}
				}
			}
		}






    // set a side the old points. Since we are doing a sliding window, all points
    // at frame_id_start should go out
    //
	//		ScenePointPointerList points_to_remove;
	//		if (_frame_id!=1){
	//			 points_to_remove = removePointsAtFrame(frame_id_start);
	//		}

    auto points_to_remove = removePointsAtFrame(frame_id_start);
    printf("removing %zu old points\n", points_to_remove.size());

    //
    // check if we should return a result to the user
    //
    if(result) {
        result->poses = _trajectory.poses();
        const auto npts = points_to_remove.size();
        result->refinedPoints.resize(npts);
        result->originalPoints.resize(npts);
        for(size_t i = 0; i < npts; ++i) {
            result->refinedPoints[i] = points_to_remove[i]->X();
            result->originalPoints[i] = points_to_remove[i]->getOriginalPoint();
        }
        result->initialCost = summary.initial_cost;
        result->finalCost   = summary.final_cost;
        result->fixedCost   = summary.fixed_cost;
        result->numSuccessfulStep = summary.num_successful_steps;
        result->totalTime = summary.total_time_in_seconds;
        result->numResiduals = summary.num_residuals;
        result->message = std::string(summary.message);
        result->iterationSummary = summary.iterations;
    }


}

auto PhotometricBundleAdjustment::getFrameAtId(uint32_t id) const -> const DescriptorFrame*
{
    for(const auto& f : _frame_buffer)
        if(f->id() == id) {
            return f.get();
        }

    throw std::runtime_error("could not find frame id!");
}

auto PhotometricBundleAdjustment::getSrcImageAtId(uint32_t id) const -> const ImageSrcMap*
{
	for(const auto& f : _image_src_map_buffer)
		if(f->id() == id) {
			return f.get();
		}

	throw std::runtime_error("could not find Image   id!");
}

auto PhotometricBundleAdjustment::removePointsAtFrame(uint32_t id) -> ScenePointPointerList
{
    using namespace std;

    decltype(_scene_points) points_to_keep, points_to_remove;

    points_to_keep.reserve(_scene_points.size());
    points_to_remove.reserve( 0.5 * _scene_points.size() );

    partition_copy(make_move_iterator(begin(_scene_points)),
                   make_move_iterator(end(_scene_points)),
                   back_inserter(points_to_remove),
                   back_inserter(points_to_keep),
                   [&](const ScenePointPointer& p) { return p->refFrameId() <= id; });

    _scene_points.swap(points_to_keep);
    return points_to_remove;
}
void PhotometricBundleAdjustment::setImage_size(int lvl) {

	_image_size=_image_size_orig;

	if (lvl==0){
		return ;
	}else if (lvl==1){
		_image_size.rows = _image_size.rows/2;
		_image_size.cols = _image_size.cols/2;
	}else if (lvl==2){
		_image_size.rows = _image_size.rows/4;
		_image_size.cols = _image_size.cols/4;
	}else if (lvl==3){
		_image_size.rows = _image_size.rows/8;
		_image_size.cols = _image_size.cols/8;
	}
}
