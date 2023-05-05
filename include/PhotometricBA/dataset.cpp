#include "dataset.h"
#include "utils.h"
#include "debug.h"
//#include "stereo_algorithm.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <vector>

using namespace pbaUtils;

static void toGray(const cv::Mat& src, cv::Mat& ret)
{
  switch( src.type() )
  {
    case CV_8UC1: ret = src; break;
    case CV_8UC3: { cv::cvtColor(src, ret, CV_BGR2GRAY); } break;
    case CV_8UC4: { cv::cvtColor(src, ret, CV_BGRA2GRAY); } break;
    default: THROW_ERROR("unsupported image format");
  }
}


RGBDDataset::RGBDDataset(std::string conf_fn)
:_scale_by( ConfigFile(conf_fn).get<int>("ScaleBy", 1) )
{
  ConfigFile cf(conf_fn);

  THROW_ERROR_IF( !this->init(cf), "failed to initialize RGBDDataset" );
}

RGBDDataset::~RGBDDataset() {}

UniquePointer<DatasetFrame> RGBDDataset::getFrame(int f_i) const
{
  THROW_ERROR_IF( _rgb.size() == 0 || _depth.size() ==0,
                 "has not been initialized" );

  std::string image_fn = _rgb[f_i];
  MonoFrame frame;
  frame.I_orig = cv::imread(image_fn, cv::IMREAD_UNCHANGED);
  std::string depth_fn = _depth[f_i];
  frame.D= cv::imread(depth_fn, cv::IMREAD_UNCHANGED);
  frame.D.convertTo(frame.D, CV_32FC1);  // scale depth by factor 5000.0f
  frame.D = frame.D/this->_depth_scale;

    if(frame.I_orig.empty())
    {
        dprintf("no more images?\nrgb:%s\ndepth:%s",
                _rgb[f_i].c_str(),
                _depth[f_i].c_str(),
                _normal[f_i].c_str(),
                _roughness[f_i].c_str());

        return nullptr;
    }
  frame.N= loadNormal(f_i);
  frame.R= loadRoughness(f_i);




  toGray(frame.I_orig, frame.I);

  if(_scale_by > 1) {
    double s = 1.0 / _scale_by;
    cv::resize(frame.I, frame.I, cv::Size(), s, s);
  }

  frame.fn = image_fn;

  return UniquePointer<DatasetFrame>(new MonoFrame(frame));
}

bool RGBDDataset::init(const pbaUtils::ConfigFile & cf) {

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
                std::string sRGB, sDepth,sNormal, sRoughness;
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


                // readin normal file
                ss >> t;
                ss >> sNormal;
                sNormal = sequenceFolder + "/" + sNormal;
                _normal.push_back(sNormal);

                // readin roughness file
                ss >> t;
                ss >> sRoughness;
                sRoughness = sequenceFolder + "/" + sRoughness;
                _roughness.push_back(sRoughness);
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

cv::Mat RGBDDataset::loadNormal(int f_i) const {
    std::string normal_fn = _normal[f_i];
    cv::Mat normal = loadPFM(normal_fn.c_str());
//    std::cout<<"show N file name:"<<normal_fn<<std::endl;
    // Scale the pixel values to the range [-1, 1]
    cv::Mat normals_c_oneView = 2 * normal - 1;
    // Swap the R and B channels and invert the G and B channels
    cv::Mat normals;
    cv::cvtColor(normals_c_oneView, normals, cv::COLOR_BGR2RGB);
    cv::Mat channels[3];
    cv::split(normals, channels);
    channels[1] = -channels[1];
    channels[2] = -channels[2];
    cv::merge(channels, 3, normals);

    return normals;
}

cv::Mat RGBDDataset::loadRoughness(int f_i) const {
    std::string roughness_fn = _roughness[f_i];
    cv::Mat roughness = loadPFM(roughness_fn.c_str());
//    std::cout<<"show R file name:"<<roughness_fn<<std::endl;
//    std::cout<<"show roughness channel"<<roughness.channels()<<std::endl;
//    int channelIdx=1;
//    cv::Mat roughness_C1;
//    extractChannel(roughness, roughness_C1, channelIdx);
    return roughness;
}

namespace {

    static inline Mat_eigen<double,3,4> set_kitti_camera_from_line(std::string line)
{
  auto tokens = pbaUtils::splitstr(line);
  THROW_ERROR_IF( tokens.empty() || tokens[0].empty() || tokens[0][0] != 'P',
                 "invalid calibration line");
  THROW_ERROR_IF( tokens.size() != 13, "wrong line length" );

  std::vector<float> vals;
  for(size_t i = 1; i < tokens.size(); ++i)
    vals.push_back(str2num<float>(tokens[i]));

  Mat_eigen<double,3,4> ret;
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




tumRGBDDataset::~tumRGBDDataset(){}


bool tumRGBDDataset::init(const pbaUtils::ConfigFile &cf) {

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
    std::cout<<"show"<<num_fields<<"intrinsics:"<<fx<<" "<<fy<<" "<<cx<<" "<<cy<<std::endl;
    return true;
}

UniquePointer<Dataset> Dataset::Create(std::string conf_fn)
{
  ConfigFile cf(conf_fn);

  auto name = cf.get<std::string>("Dataset");

  if (icompare("tumRGBD",name))
      return UniquePointer<Dataset>( new tumRGBDDataset(conf_fn) );

  THROW_ERROR(Format("unknown dataset '%s'\n", name.c_str()).c_str());
}





