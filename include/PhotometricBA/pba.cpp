//
// Created by lei on 28.04.23.
//

#include "pba.h"


#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "debug.h"
#include "types.h"
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


public:
    static inline DescriptorFrame* Create(uint32_t id, const Image_<uint8_t>& image,
                                          PhotometricBundleAdjustment::Options::DescriptorType type)
    {
        Channels channels;
        switch(type) {
            case PhotometricBundleAdjustment::Options::DescriptorType::Intensity:
                channels.push_back( image.cast<Channels::value_type::Scalar>() );
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
//    Vec3 specularity;
    VisibilityList _f;
    ZnccPatchType _patch;
    std::vector<double> _descriptor;
    std::map<uint32_t,Vec3> specularitySequence;
    double _saliency  = 0.0;
    bool _was_refined = false;

    Vec_<int,2> _x;
}; // ScenePoint


PhotometricBundleAdjustment::PhotometricBundleAdjustment(
        const Calibration& calib, const ImageSize& image_size, const Options& options)
        : _calib(calib), _image_size(image_size), _options(options),
          _frame_buffer(options.slidingWindowSize)
{
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

// after selecting some new points,we calculate specularity for them firstly. When the new frame comes, we calculate the specularity for all scene points onto their visible frames.
void PhotometricBundleAdjustment::addFrame(const uint8_t* I_ptr, const float* Z_ptr, const Vec3f* N_ptr, const float* R_ptr,  const Mat44& T, Result* result)
{

    _trajectory.push_back(T, _frame_id);
    const Eigen::Isometry3d T_w(_trajectory.back());
    const Eigen::Isometry3d T_c(T_w.inverse());

    int width= _image_size.cols;
    int height= _image_size.rows;
    PBANL::IBL_Radiance *ibl_Radiance = new PBANL::IBL_Radiance;


    typedef Eigen::Map<const Image_<uint8_t>, Eigen::Aligned> SrcMap;
    auto I = SrcMap(I_ptr, _image_size.rows, _image_size.cols);
    typedef Eigen::Map<const Image_<float>, Eigen::Aligned> SrcDepthMap;
    auto Z = SrcDepthMap(Z_ptr, _image_size.rows, _image_size.cols);

    // check the depth map
    DescriptorFrame* frame = DescriptorFrame::Create(_frame_id, I, _options.descriptorType);
    // show the selected points
    // std::cout <<"number of channels:" << frame->numChannels()<<std::endl;


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
        if(f_dist <= _options.maxFrameDistance) { // do not go too far back
            //
            // If the point projects to the current frame and it zncc score is
            // sufficiently highly, we'll add the current image to its visibility list
            Vec2 uv = _calib.project(T_c * pt->X());
            ++max_num_to_update;

            int r = std::round(uv[1]), c = std::round(uv[0]);
            if(r >= B && r < max_rows && c >= B && c <= max_cols) {
                typename ScenePoint::ZnccPatchType other_patch(I, uv);// use this uv to calculate the specularity!!!!!!!!!!!!!!!!!!!!!!!!!!!
                auto score = pt->patch().score( other_patch );
                if(score > _options.minScore) {
                    num_updated++;
                    // TODO update the patch for the new frame data
                    pt->addFrame(_frame_id);
                    // TODO: calculate specularity

                    Eigen::Vector3f point_w= pt->X().cast<float>();             // point: at world coordinate
                    Eigen::Vector3f point_c= (T_c * pt->X()).cast<float>();     // point: at camera coordinate
                    const Vec3f& normal_pixel = N_ptr[r * width + c];
                    Vec3f normal =normalize(normal_pixel);                      // normal
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
                        for (std::size_t i = 0; i < pointIdxKNNSearch.size (); ++i)
                        {
                            Vec3f envMap_point((*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[i] ].x,
                                               (*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[i] ].y,
                                               (*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[i] ].z);

//                        std::cout << "\n------"<<envMap_point.val[0]<< " " << envMap_point.val[1]<< " " << envMap_point.val[2]
//                                  << " (squared distance: " << pointKNNSquaredDistance[i] << ")" << std::endl;
                            // 0.004367 is the squared distance of the closest control point
                            if (pointKNNSquaredDistance[i]>0.004367){continue;}


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

                            // calculate the euclidean distance between shader point and envMap point
                            //float disPoint2EnvPoint_squa= powf(envMap_point.val[0]-p_c1_w.x(),2)+powf(envMap_point.val[1]-p_c1_w.y(),2)+powf(envMap_point.val[2]-p_c1_w.z(),2);

                            float disPoint2Env=  pointKNNSquaredDistance[i]/(ctrlPointNormal.dot(N_));
                            if (disPoint2Env<disPoint2Env_min){
                                disPoint2Env_min=disPoint2Env;
                                targetEnvMapIdx=i;
                                targetEnvMapVal=envMap_point;
                            }

//
//                        Eigen::Vector3f ctrlPointNormal_w_ = Camera1_extrin.rotationMatrix()* ctrlPointNormal_c;
//                        Vec3f ctrlPointNormal_w(ctrlPointNormal_w_.x(), ctrlPointNormal_w_.y(), ctrlPointNormal_w_.z());
//                        // normalize normal vector
//                        ctrlPointNormal_w = cv::normalize(ctrlPointNormal_w);
//                        std::cout << "ctrlPointNormal_w: " << ctrlPointNormal_w << std::endl;
//
//                        // calculate normal distance
//                        Vec3f vector_point2Env(searchPoint.x-envMap_point.val[0], searchPoint.y-envMap_point.val[1],
//                                               searchPoint.z-envMap_point.val[2]);
//                        vector_point2Env=cv::normalize(vector_point2Env);
//                        std::cout << "vector_point2Env: " << vector_point2Env << std::endl;
//                        std::cout << "ctrlPointNormal_w.dot(vector_point2Env): " << ctrlPointNormal_w.dot(vector_point2Env) << std::endl;
//
//                        // calculate angle between ctrlPointNormal_w and vector_point2Env
//                        float angle_cos_val = ctrlPointNormal_w.dot(vector_point2Env);
//
//
//                        // cosine(100/180*PI) = -0.173648
//                        cout<<"-------------->check idx:"<<i<<" and check angle:"<<angle_cos_val<<endl;
//
//                        if (angle_cos_val>=-0.173648 && angle_cos_val<=1.0){
//                            key4Search.val[0] = envMap_point.val[0];
//                            key4Search.val[1] = envMap_point.val[1];
//                            key4Search.val[2] = envMap_point.val[2];
//                            break;
//                        }

//                        if (vector_point2Env.dot(N_)>=0){
//                            key4Search.val[0] = envMap_point.val[0];
//                            key4Search.val[1] = envMap_point.val[1];
//                            key4Search.val[2] = envMap_point.val[2];
//                            break;
//                        }

                        }

                        if (targetEnvMapIdx!=-1){
                            key4Search.val[0] = targetEnvMapVal.val[0];
                            key4Search.val[1] = targetEnvMapVal.val[1];
                            key4Search.val[2] = targetEnvMapVal.val[2];
                        }

                    }
                    // if no envMap point is found, skip this point
                    if (key4Search.dot(key4Search)==0){ continue;}
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
                    Sophus::SE3d T_c2w(T_w.rotation(), T_w.translation());
                    // TODO: convert the envmap to current "world coordinate" in advance !!!!!! done
                    Vec3f radiance_beta = ibl_Radiance->solveForRadiance(View_beta, N_, roughness_pixel, image_metallic,
                                                                         reflectance, baseColor, T_c2w.rotationMatrix(),
                                                                         enterPanoroma.inverse());

                    Vec3 radiance_beta_(radiance_beta[0],radiance_beta[1],radiance_beta[2]);
                    pt->specularitySequence[_frame_id]=radiance_beta_;



                    //
                    // block an area in the mask to prevent initializing redundant new
                    // scene points
                    //
                    for(int r_i = -mask_radius; r_i <= mask_radius; ++r_i)
                        for(int c_i = -mask_radius; c_i <= mask_radius; ++c_i)
                            _mask(r+r_i, c+c_i) = 0;
                }
            }
        }
    }
    //
    // ==================================Add new scene points using new frame=============================================
    //
    decltype(_scene_points) new_scene_points;
    new_scene_points.reserve( max_rows * max_cols * 0.5 );
    frame->computeSaliencyMap(_saliency_map);


    // Convert the Image_<T> to a cv::Mat
    cv::Mat salientImage(_saliency_map.rows(), _saliency_map.cols(), CV_32F, _saliency_map.data());
	//cv::Mat salientImage(_saliency_map.rows(), _saliency_map.cols(), CV_32F, _saliency_map.data(), _saliency_map.stride());

    cv::Mat cmuSelectedPointMask(salientImage.rows,  salientImage.cols, CV_8UC1, cv::Scalar(0));
    cv::Mat cmuSelectedPointMask2(salientImage.rows,  salientImage.cols, CV_8UC1, cv::Scalar(0));
    // Display the image using OpenCV
    int count_selectedPoint = 0;
    salientImage.convertTo(salientImage, CV_8UC1);

    typedef IsLocalMax_<decltype(_saliency_map), decltype(_mask)> IsLocalMax;
    const IsLocalMax is_local_max(_saliency_map, _mask, _options.nonMaxSuppRadius);

    for(int y = B; y < max_rows; ++y) {
        for(int x = B; x < max_cols; ++x) {
            double z = Z(y,x);// TODO:maybe use inverse depth later
            if(z >= _options.minValidDepth && z <= _options.maxValidDepth) {
                if(is_local_max(y, x)) {
                    Vec3 X = T_w * (z * _K_inv * Vec3(x, y, 1.0)); // unproject the uv to 3D space  , X in the world frame, to project it on the current frame record the frame id if it is visible
                    std::unique_ptr<ScenePoint> p = std::make_unique<ScenePoint>(X, _frame_id);// associate a new scene point with its visible frame id
                    Vec_<int,2> xy(x, y);
                    p->setZnccPach( I, xy );
                    p->descriptor().resize(descriptor_dim);
                    p->setSaliency( _saliency_map(y,x) );
                    p->setFirstProjection(xy);

                    cmuSelectedPointMask.at<uchar>(y,x)= 255;
                    new_scene_points.push_back(std::move(p));
                }
            }
        }
    }
    std::cout<<"new scene points size: "<<new_scene_points.size()<<std::endl;
//
// keep the best N points
//
    if(new_scene_points.size() > (size_t) _options.maxNumPoints) {
        auto nth = new_scene_points.begin() + _options.maxNumPoints;
        std::nth_element(new_scene_points.begin(), nth, new_scene_points.end(),
                         [&](const ScenePointPointer& a, const ScenePointPointer& b) {
                             return a->getSaliency() > b->getSaliency();
                         });
        new_scene_points.erase(nth, new_scene_points.end());
    }

	std::cout<<"new scene points size erased : "<<new_scene_points.size()<<std::endl;

    //
    // calculate the specularity value for selected points
    //

    // TODO: calcualte specularity for new frame selected scene points
    for(size_t i = 0; i < new_scene_points.size(); ++i) {

        const auto& pt = new_scene_points[i];

        int r= pt->_x[1], c= pt->_x[0];                             // pixel: row, col
        Eigen::Vector3f point_w= pt->X().cast<float>();             // point: at world coordinate
        Eigen::Vector3f point_c= (T_c * pt->X()).cast<float>();     // point: at camera coordinate
        const Vec3f& normal_pixel = N_ptr[r * width + c];
        Vec3f normal =normalize(normal_pixel);                      // normal
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
        if ( EnvLightLookup->kdtree.nearestKSearch(searchPoint, num_K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {

            float disPoint2Env_min= 10.0f;
            int targetEnvMapIdx=-1;
            Vec3f targetEnvMapVal;

            // find the closest control point which domain contains the current point
            for (std::size_t i = 0; i < pointIdxKNNSearch.size (); ++i)
            {
                Vec3f envMap_point((*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[i] ].x,
                                   (*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[i] ].y,
                                   (*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[i] ].z);

//                        std::cout << "\n------"<<envMap_point.val[0]<< " " << envMap_point.val[1]<< " " << envMap_point.val[2]
//                                  << " (squared distance: " << pointKNNSquaredDistance[i] << ")" << std::endl;
                // 0.004367 is the squared distance of the closest control point
                if (pointKNNSquaredDistance[i]>0.004367){continue;}


                // calculate control point normal
                // transform envMap_point to camera coordinate system
                Eigen::Vector3f envMap_point_c1 = T_c.cast<float>() * Eigen::Vector3f(envMap_point.val[0], envMap_point.val[1], envMap_point.val[2]);
                // project envMap_point to image plane
//                float pixel_x = (fx * envMap_point_c1.x()) / envMap_point_c1.z() + cx;
//                float pixel_y = (fy * envMap_point_c1.y()) / envMap_point_c1.z() + cy;

                Vec2 uv = _calib.project(envMap_point_c1.cast<double>());
                int r_n = std::round(uv[1]), c_n = std::round(uv[0]);

				// check if the projected point is in the image
				if(r_n < B || r_n >= max_rows || c_n < B || c_n >= max_cols) { continue; }

                Vec3f ctrlPointNormal =  N_ptr[r_n * width + c_n];//newNormalMap.at<Vec3f>(r, c);
                ctrlPointNormal=cv::normalize(ctrlPointNormal);

                float angle_consine = ctrlPointNormal.dot(N_);
                if (angle_consine<0.9962){ continue;} // 0.9848 is the cos(10 degree), 0.9962 is the cos(5 degree)

                // calculate the euclidean distance between shader point and envMap point
                //float disPoint2EnvPoint_squa= powf(envMap_point.val[0]-p_c1_w.x(),2)+powf(envMap_point.val[1]-p_c1_w.y(),2)+powf(envMap_point.val[2]-p_c1_w.z(),2);

                float disPoint2Env=  pointKNNSquaredDistance[i]/(ctrlPointNormal.dot(N_));
                if (disPoint2Env<disPoint2Env_min){
                    disPoint2Env_min=disPoint2Env;
                    targetEnvMapIdx=i;
                    targetEnvMapVal=envMap_point;
                }

//
//                        Eigen::Vector3f ctrlPointNormal_w_ = Camera1_extrin.rotationMatrix()* ctrlPointNormal_c;
//                        Vec3f ctrlPointNormal_w(ctrlPointNormal_w_.x(), ctrlPointNormal_w_.y(), ctrlPointNormal_w_.z());
//                        // normalize normal vector
//                        ctrlPointNormal_w = cv::normalize(ctrlPointNormal_w);
//                        std::cout << "ctrlPointNormal_w: " << ctrlPointNormal_w << std::endl;
//
//                        // calculate normal distance
//                        Vec3f vector_point2Env(searchPoint.x-envMap_point.val[0], searchPoint.y-envMap_point.val[1],
//                                               searchPoint.z-envMap_point.val[2]);
//                        vector_point2Env=cv::normalize(vector_point2Env);
//                        std::cout << "vector_point2Env: " << vector_point2Env << std::endl;
//                        std::cout << "ctrlPointNormal_w.dot(vector_point2Env): " << ctrlPointNormal_w.dot(vector_point2Env) << std::endl;
//
//                        // calculate angle between ctrlPointNormal_w and vector_point2Env
//                        float angle_cos_val = ctrlPointNormal_w.dot(vector_point2Env);
//
//
//                        // cosine(100/180*PI) = -0.173648
//                        cout<<"-------------->check idx:"<<i<<" and check angle:"<<angle_cos_val<<endl;
//
//                        if (angle_cos_val>=-0.173648 && angle_cos_val<=1.0){
//                            key4Search.val[0] = envMap_point.val[0];
//                            key4Search.val[1] = envMap_point.val[1];
//                            key4Search.val[2] = envMap_point.val[2];
//                            break;
//                        }

//                        if (vector_point2Env.dot(N_)>=0){
//                            key4Search.val[0] = envMap_point.val[0];
//                            key4Search.val[1] = envMap_point.val[1];
//                            key4Search.val[2] = envMap_point.val[2];
//                            break;
//                        }

            }

            if (targetEnvMapIdx!=-1){
                key4Search.val[0] = targetEnvMapVal.val[0];
                key4Search.val[1] = targetEnvMapVal.val[1];
                key4Search.val[2] = targetEnvMapVal.val[2];
            }

        }
        // if no envMap point is found, skip this point
        if (key4Search.dot(key4Search)==0){ continue;}
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
        Sophus::SE3d T_c2w(T_w.rotation(), T_w.translation());
        // TODO: convert the envmap to current "world coordinate" in advance !!!!!!
        Vec3f radiance_beta = ibl_Radiance->solveForRadiance(View_beta, N_, roughness_pixel, image_metallic,
                                                             reflectance, baseColor, T_c2w.rotationMatrix(),
                                                             enterPanoroma.inverse());

        Vec3 radiance_beta_(radiance_beta[0],radiance_beta[1],radiance_beta[2]);
        // TODO concatenate the visibility vector  to the specularitySequence
        pt->specularitySequence[pt->refFrameId()]= radiance_beta_;

    }
    delete ibl_Radiance;
	//    imshow("cmuSelectedPointMask2", cmuSelectedPointMask2);
	//    cv::imwrite("cmuSelectedPointMask2"+std::to_string(_frame_id)+".png", cmuSelectedPointMask2);
	//    cv::waitKey(0);
	//    int num_selected_cmuSelectedPointMask2 = cv::countNonZero(cmuSelectedPointMask2);
	//    std::cout<<"num_nonZero_selected_cmuSelectedPointMask2: "<<num_selected_cmuSelectedPointMask2<<std::endl;
	//    cv::waitKey(0);
    //
    // extract the descriptors
    //
    const int num_channels = frame->numChannels(),num_new_points = (int) new_scene_points.size();
    Info("updated %d [%0.2f%%] max %d new %d\n",num_updated, 100.0 * num_updated / _scene_points.size(),max_num_to_update, num_new_points);
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


        // check if visibility has the same length of specularity sequence

        // iterate over all points and update the visibility
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



//
//            if (pt->visibilityList().size()==pt->specularitySequence.size()){
//                countSame++;
//                Info("\n visibilityList.size()==specularitySequence.size():  %d", pt->visibilityList().size());
//            }else{
//                countDiff++;
//                Info("\n visibilityList.size()!=specularitySequence.size():  %d", pt->visibilityList().size());
//            }
        }

        for (auto &pt : _scene_points) {
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

//        Info("!!! show num_selected_points: %d\n", num_selected_points);
        optimize(result);
    }

    ++_frame_id;
}

static inline std::vector<double> MakePatchWeights(int radius, bool do_gaussian, double s_x = 1.0,
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

    double sumWeight= 0.587* abs(deltaSpecularity.y())+0.114* abs(deltaSpecularity.x())+0.299* abs(deltaSpecularity.z());

    if (sumWeight==0.0f){return -1.0;}
    else{
        float y = exp(-15*sumWeight);
        return y;
    }

}



static Vec_<double,6> PoseToParams(const Mat44& T)
{
    Vec_<double,6> ret;
    const Mat_eigen<double,3,3> R = T.block<3,3>(0,0);
    ceres::RotationMatrixToAngleAxis(ceres::ColumnMajorAdapter3x3(R.data()), ret.data());

    ret[3] = T(0,3);
    ret[4] = T(1,3);
    ret[5] = T(2,3);
    return ret;
}

static Mat_eigen<double,4,4> ParamsToPose(const double* p)
{
    Mat_eigen<double,3,3> R;
    ceres::AngleAxisToRotationMatrix(p, ceres::ColumnMajorAdapter3x3(R.data()));

    Mat_eigen<double,4,4> ret(Mat_eigen<double,4,4>::Identity());
    ret.block<3,3>(0,0) = R;
    ret.block<3,1>(0,3) = Vec_<double,3>(p[3], p[4], p[5]);
    return ret;
}

class PhotometricBundleAdjustment::DescriptorError
{
public:
    /**
     * \param radius  the patch radius
     * \param calib   the camera calibration
     * \param p0      the reference frame descriptor must have size (2*radius+1)^2
     * \param frame   descriptor data of the image we are matching against
     */
    DescriptorError(const Calibration& calib, const std::vector<double>& p0,
                    const DescriptorFrame* frame, const std::vector<double>& w)
            : _radius(PatchRadiusFromLength(p0.size() / frame->numChannels())),
              _calib(calib), _p0(p0.data()), _frame(frame), _patch_weights(w.data())
    {
        // TODO should just pass the config to get the radius value
        assert( p0.size() == w.size() );
    }

    static ceres::CostFunction* Create(const Calibration& calib,
                                       const std::vector<double>& p0,
                                       const DescriptorFrame* f,
                                       const std::vector<double>& w)
    {
        return new ceres::AutoDiffCostFunction<DescriptorError, ceres::DYNAMIC, 6, 3>(
                new DescriptorError(calib, p0, f, w), p0.size());
    }

    template <class T> inline
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        T xw[3];
        ceres::AngleAxisRotatePoint(camera, point, xw);
        xw[0] += camera[3];
        xw[1] += camera[4];
        xw[2] += camera[5];

        T u_w, v_w;
        _calib.project(xw, u_w, v_w);

        for(size_t k = 0, i=0; k < _frame->numChannels(); ++k) {
            const auto& I = _frame->getChannel(k);
            const auto& G = _frame->getChannelGradient(k);
            const auto& Gx = G.Ix();
            const auto& Gy = G.Iy();

            for(int y = -_radius, j = 0; y <= _radius; ++y) {
                const T v = v_w + T(y);
                for(int x = -_radius; x <= _radius; ++x, ++i, ++j) {
                    const T u = u_w + T(x);
                    const T i0 = T(_p0[i]);
                    const T i1 = SampleWithDerivative(I, Gx, Gy, u, v);
                    residuals[i] = _patch_weights[j] * (i0 - i1);
                }
            }
        }

        // maybe we should return false if the point goes out of the image!
        return true;
    }


private:
    const int _radius;
    const Calibration& _calib;
    const double* const _p0;
    const DescriptorFrame* _frame;
    const double* const _patch_weights;
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

    options.num_threads = num_threads;
//    options.num_linear_solver_threads = options.num_threads;

    options.function_tolerance  = tol;
    options.gradient_tolerance  = tol;
    options.parameter_tolerance = tol;

    return options;
}


void PhotometricBundleAdjustment::optimize(Result* result)
{
    uint32_t frame_id_start = _frame_buffer.front()->id(), frame_id_end   = _frame_buffer.back()->id();
    _options.doGaussianWeighting = false;
    std::vector<double> patch_weights = MakePatchWeights(_options.patchRadius, _options.doGaussianWeighting);
    //
    // collect the camera poses in a single map for easy access
    //
    std::map<uint32_t, Vec_<double,6>> camera_params;
    for(uint32_t id = frame_id_start; id <= frame_id_end; ++id) {
        // NOTE camera parameters are inverted
        camera_params[id] = PoseToParams(Eigen::Isometry3d(_trajectory.atId(id)).inverse().matrix());
        std::cout<<"before optimize: _trajectory.atId(it.first:"  <<id <<"\n"<<_trajectory.atId(id)<<std::endl;
    }
    Info("_trajectory.size() = %d, frame_id_start=  %d", _trajectory.size(), frame_id_start);
    Info("_scene_points.size() = %d", _scene_points.size());

//    for(auto& pt : _scene_points) {
//        if (pt->specularitySequence.size()==0){
//            continue;
//        }
//        Info("\n show specularitySequence size of point: %d ", pt->specularitySequence.size());
//
//        for (auto& specularity : pt->specularitySequence){
//            Info("\n show specularitySequence of frame id: %d ", specularity.first);
//            cout<<"\n show specularitySequence of point: "<<specularity.second<<endl;
//        }
//    }

    //
    // get the points that we *should* optimize. They must have a large enough
    // visibility list
    //
    ceres::Problem problem;
    int num_selected_points = 0;
    for(auto& pt : _scene_points) {
        // it is enough to check the visibility list length, because we will remove
        // points as soon as they leave the optimization window
        if(pt->numFrames() >= 3 && pt->refFrameId() >= frame_id_start) { // triangle stability
            num_selected_points++;
            for(auto id : pt->visibilityList()) {

                if(id >= frame_id_start && id <= frame_id_end) {
                    pt->setRefined(true);



                    // TODO:assign specularity weight to patch_weights
                    float specularity_weight = 1.0;
                    if(pt->specularitySequence.find(id) != pt->specularitySequence.end()){
                        specularity_weight = specularityWeight(pt->specularitySequence[pt->refFrameId()],pt->specularitySequence[id]);
                        // repalce the weight of patch_weights
                        for(int i = 0; i < patch_weights.size(); i++){patch_weights[i] = specularity_weight;}
                    } else {
//                        continue;// if the point has no specularity, continue
//								//specularity_weight = 1.0;
                    }




                    auto* camera_ptr = camera_params[id].data();
                    double * xyz = pt->X().data();
                    const double huber_t = _options.robustThreshold;
                    auto* loss = huber_t > 0.0 ? new ceres::HuberLoss(huber_t) : nullptr;

                    ceres::CostFunction* cost = nullptr;
                    cost = DescriptorError::Create(_calib, pt->descriptor(), getFrameAtId(id), patch_weights);
                    problem.AddResidualBlock(cost, loss, camera_ptr, xyz);
                }
            }
        }
    }

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

    Info("Using %d points (%d residual blocks) [id start %d]\n",
         num_selected_points, problem.NumResidualBlocks(), frame_id_start);

    ceres::Solver::Summary summary;

#if HAS_OPENMP
    int num_threads = _options.numThreads > 0 ? _options.numThreads : std::min(omp_get_max_threads(), 4);
#else
    int num_threads = 4;
#endif

    ceres::Solve(GetSolverOptions(num_threads, _options.verbose), &problem, &summary);
    if(_options.verbose)
//        std::cout << summary.FullReport() << std::endl;
        std::cout << summary.BriefReport() << std::endl;

    //
    // TODO: run another optimization pass over residuals with small error
    // (eliminate the outliers)
    //

    //
    // put back the refined camera poses
    //
    for(auto& it : camera_params) {
        _trajectory.atId(it.first) = Eigen::Isometry3d(
                ParamsToPose(it.second.data())).inverse().matrix();

        std::cout<<"after optimize: _trajectory.atId(it.first:"  <<it.first <<"\n"<<_trajectory.atId(it.first)<<std::endl;
    }




    //
    // set a side the old points. Since we are doing a sliding window, all points
    // at frame_id_start should go out
    //
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




// notes

//    std::cout<<"count_selectedPoint: "<<count_selectedPoint<<std::endl;
//    imshow("saliency_map_drawed", salientImage);
//    imshow("cmuSelectedPointMask", cmuSelectedPointMask);
////    cv::imshow("saliency_map", salientImage);
//    cv::imwrite("cmuSelectedPointMask"+std::to_string(_frame_id)+".png", cmuSelectedPointMask);
//     cv::waitKey(0);
//    int num_selected = cv::countNonZero(cmuSelectedPointMask);
//    std::cout<<"num_nonZero_selected: "<<num_selected<<std::endl;

//      std::cerr<<"show normal_pixel:"<<normal_pixel<<"at:"<<r<<"and:"<<c<<endl;