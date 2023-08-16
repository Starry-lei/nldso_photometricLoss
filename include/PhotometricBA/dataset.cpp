#include "dataset.h"
#include "utils.h"
#include "debug.h"
//#include "stereo_algorithm.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <vector>

using namespace utils;

static void toGray(const cv::Mat& src, cv::Mat& ret)
{
  switch( src.type() )
  {
    case CV_8UC1: ret = src; break;
	case CV_8UC3: { cv::cvtColor(src, ret, cv::COLOR_BGR2GRAY); } break;
	case CV_8UC4: { cv::cvtColor(src, ret, cv::COLOR_BGR2GRAY); } break;
    default: THROW_ERROR("unsupported image format");
  }
}



//StereoDataset::StereoDataset(std::string conf_fn)
//: _stereo_alg(new StereoAlgorithm(ConfigFile(conf_fn)))
//    , _scale_by( ConfigFile(conf_fn).get<int>("ScaleBy", 1) )
//{
//  THROW_ERROR_IF( !this->init(conf_fn), "failed to initialize StereoDataset" );
//}

//StereoDataset::~StereoDataset() {}

RGBDDataset::RGBDDataset(std::string conf_fn)
:_scale_by( ConfigFile(conf_fn).get<int>("ScaleBy", 1) )
{
  ConfigFile cf(conf_fn);

  THROW_ERROR_IF( !this->init(cf), "failed to initialize RGBDDataset" );
}

RGBDDataset::~RGBDDataset() {}


//UniquePointer<DatasetFrame> StereoDataset::getFrame(int f_i) const
//{
//  THROW_ERROR_IF( _left_filenames == nullptr || _right_filenames == nullptr,
//                 "has not been initialized" );
//
//  auto image_fn = _left_filenames->operator[](f_i);
//  StereoFrame frame;
//  frame.I_orig[0] = cv::imread(image_fn, cv::IMREAD_UNCHANGED);
//  frame.I_orig[1] = cv::imread(_right_filenames->operator[](f_i), cv::IMREAD_UNCHANGED);
//
//  if(frame.I_orig[0].empty() || frame.I_orig[1].empty())
//  {
//    dprintf("nore more images?\nleft:%s\nright:%s",
//            _left_filenames->operator[](f_i).c_str(),
//            _right_filenames->operator[](f_i).c_str());
//
//    return nullptr;
//  }
//
//  toGray(frame.I_orig[0], frame.I[0]);
//  toGray(frame.I_orig[1], frame.I[1]);
//
//  if(_scale_by > 1) {
//    double s = 1.0 / _scale_by;
//    cv::resize(frame.I[0], frame.I[0], cv::Size(), s, s);
//    cv::resize(frame.I[1], frame.I[1], cv::Size(), s, s);
//  }
//
//  frame.fn = image_fn;
//
//  _stereo_alg->run(frame.I[0], frame.I[1], frame.D);
//  return UniquePointer<DatasetFrame>(new StereoFrame(frame));
//}

UniquePointer<DatasetFrame> RGBDDataset::getFrame(int f_i) const
{

  THROW_ERROR_IF( _rgb.size() == 0 || _depth.size() ==0,
                 "has not been initialized" );

  std::string image_fn = _rgb[f_i];
  if (image_fn.empty()){
	return nullptr;
  }
  MonoFrame frame;
  frame.I_orig = cv::imread(image_fn, cv::IMREAD_UNCHANGED);
  std::string depth_fn = _depth[f_i];
  frame.D= cv::imread(depth_fn, cv::IMREAD_UNCHANGED);
  frame.D.convertTo(frame.D, CV_32FC1);  // scale depth by factor 5000.0f
  frame.D = frame.D/this->_depth_scale;


  // read normal and roughness later

//  imshow("frame.I_orig", frame.I_orig);
//  imshow("frame.D", frame.D);
//  cv::waitKey(0);

  if(frame.I_orig.empty())
  {
    dprintf("no more images?\nrgb:%s\ndepth:%s",
            _rgb[f_i].c_str(),
            _depth[f_i].c_str());
    return nullptr;
  }

  toGray(frame.I_orig, frame.I);

  if(_scale_by > 1) {
    double s = 1.0 / _scale_by;
    cv::resize(frame.I, frame.I, cv::Size(), s, s);
  }

  frame.fn = image_fn;

  return UniquePointer<DatasetFrame>(new MonoFrame(frame));
}

//const StereoAlgorithm* StereoDataset::stereo() const { return _stereo_alg.get(); }

//bool StereoDataset::init(const ConfigFile& cf)
//{
//  try
//  {
//    auto root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
//    THROW_ERROR_IF(!fs::exists(root_dir), "DataSetRootDirectory does not exist");
//
//    //
//    // we'll allow subclasses to do this part
//    //
//    auto left_fmt = cf.get<std::string>("LeftImageFormat", "");
//    auto right_fmt = cf.get<std::string>("RightImageFormat", "");
//    auto frame_start = cf.get<int>("FirstFrameNumber", 0);
//
//    if(!left_fmt.empty())
//    {
//      _left_filenames = std::make_unique<FileLoader>(root_dir, left_fmt, frame_start);
//      _right_filenames = std::make_unique<FileLoader>(root_dir, right_fmt, frame_start);
//
//      auto frame = this->getFrame(0);
//      THROW_ERROR_IF( nullptr == frame, "failed to load frame" );
//      _image_size = Dataset::GetImageSize(frame.get());
//    }
//  } catch(const std::exception& ex)
//  {
//    Warn("Error %s\n", ex.what());
//    return false;
//  }
//
//  return true;
//}

//std::vector<std::string> StereoDataset::getTimestamp() const {
//    return std::vector<std::string>();
//}

bool RGBDDataset::init(const utils::ConfigFile & cf) {

    try
    {
        std::string root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
        THROW_ERROR_IF(!fs::exists(root_dir), "DataSetRootDirectory does not exist");

        std::string sequenceFolder = cf.get<std::string>("SequenceFolder", "");
        int sequence = cf.get<int>("SequenceNumber");
        sequenceFolder= root_dir + Format("/sequences/%02d", sequence);
        THROW_ERROR_IF(!fs::exists(sequenceFolder), "SequenceFolder does not exist");
        std::string  strAssociationFilename = sequenceFolder + Format("/associated.txt");

        std::ifstream fAssociation;
        fAssociation.open(strAssociationFilename.c_str());
        if (!fAssociation)
        {
            printf("please ensure that you have the associate file\n");
            return -1;
        }
        while (!fAssociation.eof())
        {
            std::string s;
            std::getline(fAssociation, s);
            if (!s.empty())
            {
                std::stringstream ss;
                ss << s;
                std::string t;
                std::string sRGB, sDepth;
//                std::string sRGB, sDepth, sMetallic, sBasecolor, sNormal, sRoughness;
                // readin rgb file

                ss >> t;
                timestamps.push_back(t);
                ss >> sRGB;
                sRGB = sequenceFolder + "/" + sRGB;
                _rgb.push_back(sRGB);
                // readin depth file
                ss >> t;
                ss >> sDepth;
                sDepth = sequenceFolder + "/" + sDepth;
                _depth.push_back(sDepth);
                // readin Normal file
                //                ss >> t;
                //                ss >> sNormal;
                //                sNormal = dir + "/" + sNormal;
                //                normalfiles.push_back(sNormal);
                //                // readin Roughness file
                //                ss >> t;
                //                ss >> sRoughness;
                //                sRoughness = dir + "/" + sRoughness;
                //                roughnessfiles.push_back(sRoughness);
            }
        }
        auto frame = this->getFrame(0);
        THROW_ERROR_IF( nullptr == frame, "failed to load frame" );
        _image_size = Dataset::GetImageSize(frame.get());

    } catch(const std::exception& ex)
    {
        Warn("Error %s\n", ex.what());
        return false;
    }
    return true;
}

std::vector<std::string> RGBDDataset::getTimestamp() const {
    return timestamps;
}

namespace {

    static inline Mat_<double,3,4> set_kitti_camera_from_line(std::string line)
{
  auto tokens = utils::splitstr(line);
  THROW_ERROR_IF( tokens.empty() || tokens[0].empty() || tokens[0][0] != 'P',
                 "invalid calibration line");
  THROW_ERROR_IF( tokens.size() != 13, "wrong line length" );

  std::vector<float> vals;
  for(size_t i = 1; i < tokens.size(); ++i)
    vals.push_back(str2num<float>(tokens[i]));

  Mat_<double,3,4> ret;
  for(int r = 0, i = 0; r < ret.rows(); ++r)
    for(int c = 0; c < ret.cols(); ++c, ++i)
      ret(r,c) = vals[i];

  return ret;
}

} // namespace

tumRGBDDataset::tumRGBDDataset(std::string conf_fn)
    : RGBDDataset(conf_fn)
{
  THROW_ERROR_IF( !this->init(conf_fn), "failed to initialize tumRGBDDataset" );
}

//KittiDataset::KittiDataset(std::string conf_fn)
//    : StereoDataset(conf_fn)
//{
//  THROW_ERROR_IF( !this->init(conf_fn), "failed to initialize KittiDataset" );
//}

//KittiDataset::~KittiDataset(){}
tumRGBDDataset::~tumRGBDDataset(){}

//bool KittiDataset::init(const utils::ConfigFile& cf)
//{
//  try
//  {
//    std::string root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
//    int sequence = cf.get<int>("SequenceNumber");
//
//    auto left_fmt = Format("sequences/%02d/image_0/%s.png", sequence, "%06d");
//    auto right_fmt = Format("sequences/%02d/image_1/%s.png", sequence, "%06d");
//    auto frame_start = cf.get<int>("FirstFrameNumber", 0);
//
//    this->_left_filenames = std::make_unique<FileLoader>(root_dir, left_fmt, frame_start);
//    this->_right_filenames = std::make_unique<FileLoader>(root_dir, right_fmt, frame_start);
//
//    auto frame = this->getFrame(0);
//    THROW_ERROR_IF( nullptr == frame, "failed to load frame" );
//    this->_image_size = Dataset::GetImageSize(frame.get());
//
//    auto calib_fn = Format("%s/sequences/%02d/calib.txt", root_dir.c_str(), sequence);
//    THROW_ERROR_IF(!fs::exists(calib_fn), "calibration file does not exist");
//    return loadCalibration(calib_fn);
//
//  } catch(std::exception& ex)
//  {
//    Warn("Error %s\n", ex.what());
//    return false;
//  }
//
//  return true;
//}

bool tumRGBDDataset::init(const utils::ConfigFile &cf) {

    try
    {

        std::string root_dir = fs::expand_tilde(cf.get<std::string>("DataSetRootDirectory"));
        THROW_ERROR_IF(!fs::exists(root_dir), "DataSetRootDirectory does not exist");

        std::string sequenceFolder = cf.get<std::string>("SequenceFolder", "");
        int sequence = cf.get<int>("SequenceNumber");
        sequenceFolder= root_dir + Format("/sequences/%02d", sequence);
        THROW_ERROR_IF(!fs::exists(sequenceFolder), "SequenceFolder does not exist");

        std::string strCalibFilename = sequenceFolder + Format("/calibration.txt");
        printf("loading calibration from %s!\n", strCalibFilename.c_str());

        auto frame = this->getFrame(0);
        THROW_ERROR_IF( nullptr == frame, "failed to load frame" );
        this->_image_size = Dataset::GetImageSize(frame.get());

        return loadCalibration(strCalibFilename);

    } catch(std::exception& ex)
    {
        Warn("Error %s\n", ex.what());
        return false;
    }

    return true;
}


//bool KittiDataset::loadCalibration(std::string filename)
//{
//  std::ifstream ifs(filename);
//  THROW_ERROR_IF( !ifs.is_open(), "failed to open calib.txt" );
//
//  Mat_<double,3,4> P1, P2;
//  std::string line;
//
//  // the first camera
//  std::getline(ifs, line);
//  P1 = set_kitti_camera_from_line(line);
//
//  std::getline(ifs, line);
//  P2 = set_kitti_camera_from_line(line);
//
//  _calib.K() = P1.block<3,3>(0,0);
//  _calib.baseline() =  -P2(0,3) / P2(0,0);
//
//  if(this->_scale_by > 1) {
//    printf("scaling the calibration by %d\n", this->_scale_by);
//    float s = 1.0f / _scale_by;
//    _calib.K() *= s;
//    _calib.K()(2,2) = 1.0f;
//    _calib.baseline() /= s;
//  }
//
//  return true;
//}

bool tumRGBDDataset::loadCalibration(std::string filename) {

    std::ifstream f(filename.c_str());
    THROW_ERROR_IF( !f.is_open(), "failed to open calib.txt" );
    std::string l1;
    std::getline(f, l1);
    f.close();
    double fx, fy, cx, cy; // only pinhole model supported
    int num_fields = std::sscanf(l1.c_str(), "%lf %lf %lf %lf", &fx, &fy, &cx, &cy);
    _calib.K() = Eigen::Matrix3d::Identity();
    _calib.K()(0,0) = fx;
    _calib.K()(1,1) = fy;
    _calib.K()(0,2) = cx;
    _calib.K()(1,2) = cy;
    _calib.K()(2,2) = 1.0f;
    std::cout<<"\n num_fields: "<<num_fields<<" \n show intrinsics:"<<fx<<" "<<fy<<" "<<cx<<" "<<cy<<std::endl;
    return true;

}

UniquePointer<Dataset> Dataset::Create(std::string conf_fn)
{
  ConfigFile cf(conf_fn);

  auto name = cf.get<std::string>("Dataset");

//  if(icompare("kitti", name))
//    return UniquePointer<Dataset>( new KittiDataset(conf_fn) );
//  else if (icompare("tumRGBD",name))
//      return UniquePointer<Dataset>( new tumRGBDDataset(conf_fn) );

  if (icompare("tumRGBD",name))
		return UniquePointer<Dataset>( new tumRGBDDataset(conf_fn) );

  THROW_ERROR(Format("unknown dataset '%s'\n", name.c_str()).c_str());
}





