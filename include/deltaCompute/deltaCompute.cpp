//
// Created by cheng on 05.12.22.
//

#include "deltaCompute.h"

namespace DSONL
{

    float dot(const Eigen::Vector3f x, const Eigen::Vector3f y)
    {
        float ans = x.dot(y);
        return ans;
    }

    float _dot(Vec3f&  fst, Vec3f&  snd){ return max( (float)0.0, fst.dot(snd));}

    float mod(const float numer, const float denom)
    {
        return std::fmod(numer, denom);
    }

    float clamp(const float x, const float min_v, const float max_v)
    {
        return std::min(std::max(x, min_v), max_v);
    }

    float pow(const float x, const float y)
    {
        return std::pow(x, y);
    }

    Eigen::Vector3f pow(const float x, const Eigen::Vector3f y_vec)
    {
        Eigen::Vector3f out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = pow(x, y_vec[i]);
        }
        return out_vec;
    }

    Eigen::Vector3f pow(const Eigen::Vector3f x, const float y)
    {
        return pow(y, x);
    }

    Eigen::Vector3f pow(const Eigen::Vector3f x_vec, const Eigen::Vector3f y_vec)
    {
        Eigen::Vector3f out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = pow(x_vec[i], y_vec[i]);
        }
        return out_vec;
    }

    Eigen::Vector3f normalize(const Eigen::Vector3f vec)
    {
        Eigen::Vector3f out_vec(vec);
        float sum_quare(0);
        for (int i = 0; i < 3; i++)
        {
            sum_quare += vec[i] * vec[i];
        }
        if (sum_quare < 1e-10f)
        {
            printf("[WARNING] The norm of vector is too small, return zeros vector.\n");
            return out_vec; // too small norm
        }
        sum_quare = std::sqrt(sum_quare);
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] /= sum_quare;
        }
        return out_vec;
    }

    float mix(const float x, const float y, const float a)
    {
        if (a > 1 || a < 0)
            throw std::invalid_argument("received value a not in interval(0,1)");
        return x * (1 - a) + y * a;
    }
    Eigen::Vector3f mix(const Eigen::Vector3f x_vec, const Eigen::Vector3f y_vec, const float a)
    {
        Eigen::Vector3f out_vec;
        for (int i = 0; i < 3; i++)
        {
            out_vec[i] = mix(x_vec[i], y_vec[i], a);
        }
        return out_vec;
    }

    /*
        Reflect — calculate the reflection direction for an incident vector
        I: Specifies the incident vector.
        N: Specifies the normal vector.
    */
    Vec3f reflect(Vec3f I, Vec3f N)
    {
      Vec3f N_norm = normalize(N);
      Vec3f out_vec = I - 2 * _dot(N_norm, I) * N_norm;
      return out_vec;
    }


    IBL_Radiance::IBL_Radiance() {}
    IBL_Radiance::~IBL_Radiance() {}


    Vec2f IBL_Radiance::directionToSphericalEnvmap(Vec3f dir) {
      float s = 1.0 - mod(1.0 / (2.0*M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);
      float t = 1.0 / (M_PI) * acos(-dir.val[2]);
      return Vec2f(s, t);
    }


    Vec3f IBL_Radiance::specularIBL(Vec3f F0, float roughness, Vec3f N,Vec3f V) {

      float NoV = clamp(_dot(N, V), 0.0, 1.0);
      Vec3f R = reflect(-V, N);
      Vec2f uv = directionToSphericalEnvmap(R);

      Vec3f prefiltered_Color=prefilteredColor(uv.val[0], uv.val[1], roughness*float(mipCount));

      Vec2f brdf_val= brdfIntegration(NoV,roughness);

      F0*=brdf_val.val[0];
      F0.val[0]+=brdf_val.val[1];
      F0.val[1]+=brdf_val.val[1];
      F0.val[2]+=brdf_val.val[1];

      prefiltered_Color.val[0]*=F0.val[0];
      prefiltered_Color.val[1]*=F0.val[1];
      prefiltered_Color.val[2]*=F0.val[2];

      return prefiltered_Color;
    }



    Vec3f IBL_Radiance::diffuseIBL(Vec3f normal) {

      Vec2f uv = directionToSphericalEnvmap(normal);

//      return texture(diffuseMap, uv).rgb;
      return cv::Vec3f();
    }





    Vec3f IBL_Radiance::fresnelSchlick(float cosTheta, Vec3f F0) {
      return cv::Vec3f();
    }

    Vec3f IBL_Radiance::prefilteredColor(float u, float v, float level) {

      gli::vec4 Sample_val = prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(u, v), level); // transform the texture coordinate

      std::cout << "\n============Sample_val val(BGRA):\n" << Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << std:: endl;
      return cv::Vec3f(Sample_val.r, Sample_val.g, Sample_val.b);
    }



    Vec2f IBL_Radiance::brdfIntegration(float NoV,float roughness ) {

//      if ()
     gli::vec4 SampleBrdf = brdfSampler->texture_lod(gli::fsampler2D::normalized_type(NoV, roughness), 0.0f); // transform the texture coordinate
     std::cout << "\n============SampleBrdf val(BGRA):\n"
          << SampleBrdf.b << "," << SampleBrdf.g << "," << SampleBrdf.r << "," << SampleBrdf.a << std::endl;


     return Vec2f (SampleBrdf.b, SampleBrdf.g);   // x, y

    }

    //    void updateDelta(
//        const Sophus::SE3d& CurrentT,
//        const Eigen::Matrix3d & K,
//        const Mat& image_baseColor,
//        const Mat depth_map,
//        const Mat& image_metallic,
//        const Mat& image_roughnes,
//        const Eigen::Vector3d & light_source,
//        Mat& deltaMap,
//        Mat& newNormalMap,
//        float& upper_b,
//        float& lower_b
//
//    ){
//      double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2);
//
//      std::unordered_map<int, int> inliers_filter;
//      //new image
//
//      //		inliers_filter.emplace(173,333); //yes
//      //		inliers_filter.emplace(378,268); //yes
//      inliers_filter.emplace(213,295); //yes
//      //		inliers_filter.emplace(370,488); //yes
//
//      //		pcl::PLYWriter writer;
//      //		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
//
//      for (int u = 0; u< depth_map.rows; u++) // colId, cols: 0 to 480
//      {
//        for (int v = 0; v < depth_map.cols; v++) // rowId,  rows: 0 to 640
//        {
//          //				if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//          //				if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~
//          //				cout<<"show delta:"<<deltaMap.at<double>(u,v)<<endl;
//
//          Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id
//          double d=depth_map.at<double>(u,v);
//          // calculate 3D point coordinate
//          Eigen::Vector3d p_3d_no_d((pixelCoord(0)-cx)/fx, (pixelCoord(1)-cy)/fy,1.0);
//          Eigen::Vector3d p_c1=d*p_3d_no_d;
//
//
//          // record point cloud
//          //				cloud->push_back(pcl::PointXYZ(p_c1.x(), p_c1.y(), p_c1.z()));
//          // calculate normal for each point
//
//
//          Eigen::Matrix<double,3,1> normal;
//          normal.x()= newNormalMap.at<Vec3d>(u,v)[0];
//          normal.y()= newNormalMap.at<Vec3d>(u,v)[1];
//          normal.z()= newNormalMap.at<Vec3d>(u,v)[2];
//
//          // calculate alpha_1
//          Eigen::Matrix<double,3,1> alpha_1= light_C1(light_source) -p_c1;
//          alpha_1=alpha_1.normalized();
//          // calculate beta and beta_prime;
//          Eigen::Matrix<double,3,1> beta,beta_prime;
//          beta=-p_c1;
//          beta=beta.normalized();
//          beta_prime=-CurrentT.rotationMatrix().transpose()*CurrentT.translation()-p_c1;
//          beta_prime=beta_prime.normalized();
//
//          // baseColor_bgr[3],metallic, roughness
//          float metallic=image_metallic.at<float>(u,v);
//          float roughness =image_roughnes.at<float>(u,v);
//          //Instantiation of BRDF object
//          //				if ( image_baseColor.at<Vec3f>(u,v)[1]<0.01 ){ continue;}
//          Vec3f baseColor(image_baseColor.at<Vec3f>(u,v)[2],image_baseColor.at<Vec3f>(u,v)[1],image_baseColor.at<Vec3f>(u,v)[0]);
//
//          Vec3f L_(alpha_1(0),alpha_1(1),alpha_1(2));
//          Vec3f N_(normal(0),normal(1),normal(2));
//          Vec3f View_beta(beta(0),beta(1),beta(2));
//          Vec3f View_beta_prime(beta_prime(0),beta_prime(1),beta_prime(2));
//
//
//          BrdfMicrofacet radiance_beta_vec(L_,N_,View_beta,(float )roughness,(float)metallic,baseColor);
//          Vec3f radiance_beta= radiance_beta_vec.brdf_value_c3;
//
//          BrdfMicrofacet radiance_beta_prime_vec(L_,N_,View_beta_prime,(float )roughness,(float)metallic,baseColor);
//          Vec3f radiance_beta_prime= radiance_beta_prime_vec.brdf_value_c3;
//          // right intensity / left intensity
//          float delta_r= radiance_beta_prime.val[0]/radiance_beta.val[0];
//          float delta_g= radiance_beta_prime.val[1]/radiance_beta.val[1];
//          float delta_b= radiance_beta_prime.val[2]/radiance_beta.val[2];
//
//
//          deltaMap.at<float>(u,v)= delta_g;
//
//
//
//        }
//      }
//      double  max_n, min_n;
//      cv::minMaxLoc(deltaMap, &min_n, &max_n);
//      cout<<"------->show max and min of estimated deltaMap:"<< max_n <<","<<min_n<<endl;
//
//      //		colorDeltaMap
//      //		Mat coloredDeltaMap(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0,0,0)); // default value 1.0
//      //		colorDeltaMap(deltaMap, coloredDeltaMap,upper_b, lower_b);
//
//      ////		writer.write("PointCloud.ply",*cloud, false);//
//      //		Mat mask = cv::Mat(deltaMap != deltaMap);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------remove nan value --------------
//      //		deltaMap.setTo(1.0, mask);
//
//      //		deltaMap=deltaMap*(1.0/(upper_b-lower_b))+(-lower_b*(1.0/(upper_b-lower_b)));
//    }
//
//



















































    }
