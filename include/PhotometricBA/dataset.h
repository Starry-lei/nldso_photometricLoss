#ifndef PHOTOBUNDLE_DATASET_H
#define PHOTOBUNDLE_DATASET_H

#include "types.h"
#include "file_loader.h"
#include "calibration.h"
#include <opencv2/core/core.hpp>

#include "dataLoader/PFMReadWrite.h"

namespace pbaUtils {
class ConfigFile;
}; // pbaUtils

enum class DatasetType
{
  Stereo,    /** data is left and right stereo pairs */
  Disparity, /** data is left image and a disparity */
  Depth      /** RGBD, or image and depth */
}; // DataseType

struct DatasetFrame
{
  virtual const cv::Mat& image() const = 0;
  virtual  cv::Mat& depth()  = 0;
  virtual const cv::Mat& normal() const = 0;
  virtual const cv::Mat& roughness() const = 0;
  virtual std::string filename() const { return ""; }

  virtual ~DatasetFrame() {}
};

class Dataset
{
 public:
  Dataset() {}
  virtual ~Dataset() {}

  /**
   * Get the i-th frame
   */
  virtual UniquePointer<DatasetFrame> getFrame(int f_i) const = 0;

    /**
     * Get the timestamp of the i-th frameme
     */
    virtual std::vector<std::string> getTimestamp() const = 0;

  /**
   * \return the image size
   */
  virtual ImageSize imageSize() const = 0;

  /**
   * \return the calibration
   */
  virtual Calibration calibration() const = 0;

  /**
   * \return the type of the dataset
   */
  virtual DatasetType type() const = 0;

  /**
   * \return the name of the dataset
   */
  virtual std::string name() const = 0;


  /**
   * create a dataset loader from a config file
   */
  static UniquePointer<Dataset> Create(std::string conf_fn);

  /**
   * get the image size from a frame
   */
  static inline ImageSize GetImageSize(const DatasetFrame* f)
  {
    return ImageSize(f->image().rows, f->image().cols);
  }

 protected:
  /** the index of the first frame when doing a printf style image loading */
  int _first_frame_index = 0;
}; // Dataset


class RGBDDataset : public Dataset
{
public:
    struct MonoFrame : DatasetFrame
    {
        std::string fn;
        cv::Mat I_orig; //< original rgb image
        cv::Mat I;      //< grayscale image
        cv::Mat D;      //< depth as float
        cv::Mat N;      //< normal as float
        cv::Mat R;      //< roughness as float

        inline const cv::Mat& image() const { return I; }
        inline cv::Mat& depth()  { return D; }
        inline const cv::Mat& normal() const { return N; }
        inline const cv::Mat& roughness() const { return R; }
        inline std::string filename() const { return fn; }

        virtual ~MonoFrame() {}
    }; // singleFrame

public:
    RGBDDataset(std::string conf_fn);
    virtual ~RGBDDataset();

    virtual Calibration calibration() const = 0;
    virtual std::string name() const = 0;

    inline DatasetType type() const { return DatasetType::Depth; }
    inline ImageSize imageSize() const { return _image_size; }

    UniquePointer<DatasetFrame> getFrame(int f_i) const;
    std::vector<std::string> getTimestamp() const;

    cv::Mat loadNormal(int f_i) const;
    cv::Mat loadRoughness(int f_i) const;


protected:
    ImageSize _image_size;
    int _scale_by = 1;
    std::vector<std::string>  _rgb;
    std::vector<std::string>  _depth;
    std::vector<std::string> _normal;
    std::vector<std::string>  _roughness;

    float _depth_scale= 5000.0f;
    std::vector<std::string> timestamps;

    virtual bool init(const pbaUtils::ConfigFile&);
}; // RGBDDataset

class tumRGBDDataset : public RGBDDataset
{

public:
    tumRGBDDataset(std::string conf_fn);
    virtual ~tumRGBDDataset();

    inline std::string name() const { return "tum_rgbd"; }
    inline Calibration calibration() const { return _calib; }

protected:
    Calibration _calib;

    bool init(const pbaUtils::ConfigFile&);
    bool loadCalibration(std::string calib_fn);


}; // tumRGBDDataset

#endif // PHOTOBUNDLE_DATASET_H
