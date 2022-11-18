//
// Created by cheng on 19.09.22.
//

#pragma once


#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>


#include <iostream>
#include <vector>
#include <ultils.h>
#include <sophus/se3.hpp>

// change functions here to a brdf class

namespace DSONL {

	#define RECIPROCAL_PI 0.3183098861837907
	using namespace std;
	using namespace cv;
	//helper functions
	float lerp_1d(float fst, float sec, float by){return fst*(1-by)+ sec*by;}//????????????????????
//	float lerp_1d(float fst, float sec, float by){return fst*by+ sec*(1-by);}
	Vec3f lerp_3d( Vec3f firstVector, Vec3f secondVector, float by){

		float retX = lerp_1d(firstVector.val[0], secondVector.val[0], by);
		float retY = lerp_1d(firstVector.val[1], secondVector.val[1], by);
		float retZ = lerp_1d(firstVector.val[2], secondVector.val[2], by);
		Vec3f u(retX, retY, retZ);
		return  u;
	}
	template <class T> T clamp(T x, T min, T max){if (x>max){ return max;}if (x<min){return  min;}return x;}
	float SchlickFresnel(float i){
		float x = clamp(1.0-i, 0.0, 1.0);
		float x2 = x*x;
		return x2*x2*x;
	}


	class BrdfMicrofacet {
	private:
		Vec3f L; // light direction in world space
		Vec3f cameraPosition; // camera position in world space
		Vec3f N; // normal
		Vec3f H; // halfDirection
		Vec3f V;
		Vec3f baseColor;
		float IOR=0.04; //Fresnel IOR (index of refraction), _Ior("Ior",  Range(0,4)) = 1.5
		float metallicValue;
		float roughnessValue; //  roughness = 1 - _Glossiness;??????????
		float LdotH, NdotH, NdotL, NdotV;
//		Vec3f baseColorTexture; // base color
		//helper functions
		float _squared(float x){return x*x;}
		float _dot(Vec3f&  fst, Vec3f&  snd){ return max( (float)0.0, fst.dot(snd));}
	public:
		BrdfMicrofacet(const Vec3f&  L_, const Vec3f&  N_, const Vec3f& view_beta,
					   const float& roughnessValue_,
					   const float& metallicValue_,
					   const Vec3f& baseColor_ // RGB order,,
					   ){
			L=L_;
			V=view_beta; float shiftAmount = N_.dot(view_beta);
			N= shiftAmount < 0.0f ? (N_ + view_beta * (-shiftAmount + 1e-5f) ): N_; //normal direction calculations
			H= normalize(view_beta+L_);
			LdotH= _dot(L,H);
			NdotH= _dot(N,H);
			NdotL= _dot(N,L);
			NdotV= _dot(N,V);
			baseColor=baseColor_;
			roughnessValue=roughnessValue_;
			metallicValue=metallicValue_;
			specColor_= specColor(baseColor, metallicValue);
//			D= GGXNormalDistribution(roughnessValue, NdotH);// calculate the normal distribution function result
			D= GGXNormalDistribution(roughnessValue, NdotH);

            F=NewSchlickFresnelFunction(IOR,baseColor, LdotH,metallicValue);// calculate the Fresnel reflectance
			G=AshikhminShirleyGeometricShadowingFunction( NdotL,  NdotV,  LdotH);
			specularityVec=specularity(F,D,G, NdotL, NdotV);
			diffuseColorVec=diffuseColor(baseColor, metallicValue);
			brdf_value_c3= brdfMicrofacet(baseColor,metallicValue, specularityVec, NdotL);

//			brdf_value=0.299*lightingModel.val[0]+0.587*lightingModel.val[1]+0.114*lightingModel.val[2];// 0R,1G,2B
		}
		~BrdfMicrofacet(){};
        Vec3f D;
		Vec3f F;
		float G;
		Vec3f specularityVec;
		Vec3f diffuseColorVec;
		Vec3f brdf_value_c3;
		Vec3f specColor_;


		Vec3f GGXNormalDistribution(float roughness, float NdotH);
		Vec3f NewSchlickFresnelFunction(float ior, Vec3f Color, float LdotH, float Metallicness);
		float AshikhminShirleyGeometricShadowingFunction (float NdotL, float NdotV, float LdotH);
		Vec3f specularity( Vec3f FresnelFunction,Vec3f SpecularDistribution, float GeometricShadow, float NdotL,float NdotV );
		Vec3f diffuseColor ( Vec3f baseColor ,float _Metallic );
		Vec3f brdfMicrofacet (  Vec3f Color_rgb ,float _Metallic, Vec3f specularity, float  NdotL);
		Vec3f specColor( Vec3f baseColor, float  metallic);

	};

	Vec3f BrdfMicrofacet:: specColor( Vec3f baseColor, float  metallic){
		Vec3f _SpecularColor (1.0,1.0,1.0);// ???????????????????????????????????????????????????????????????????????????????
		return lerp_3d(_SpecularColor , baseColor , metallic * 0.5);
	}

//	float BrdfMicrofacet::GGXNormalDistribution(float roughness, float NdotH)
//	{
//
//		float roughnessSqr = roughness*roughness;
//
//		float NdotHSqr = NdotH*NdotH;
//
//		float TanNdotHSqr = (1-NdotHSqr)/NdotHSqr;
//
//		return (1.0/3.1415926535) * _squared(roughness/(NdotHSqr * (roughnessSqr + TanNdotHSqr)));
//
//
//	}

	Vec3f BrdfMicrofacet::GGXNormalDistribution(float roughness, float NdotH)
	{
//		NdotH=0.992188;

		float roughnessSqr = roughness*roughness;

		float NdotHSqr = NdotH*NdotH;

		Vec3f res;
		float TanNdotHSqr = (1.0-NdotHSqr)/NdotHSqr;
		float GGX=(1.0/3.1415926535) * _squared(roughness/(NdotHSqr * (roughnessSqr + TanNdotHSqr)));
		res.val[0]= specColor_.val[0]*GGX;
		res.val[1]= specColor_.val[1]*GGX;
		res.val[2]= specColor_.val[2]*GGX;
		return res ;


	}
	Vec3f BrdfMicrofacet:: NewSchlickFresnelFunction(float ior, Vec3f Color, float LdotH, float Metallicness){

		Vec3f f0 = Vec3f(0.16*ior*ior,0.16*ior*ior,0.16*ior*ior);
		Vec3f F0 = lerp_3d(f0,Color,Metallicness);
		Vec3f one(1.0,1.0,1.0);
		return F0 + (one - F0) * SchlickFresnel(LdotH);

	}

	float BrdfMicrofacet:: AshikhminShirleyGeometricShadowingFunction (float NdotL, float NdotV, float LdotH){
		float Gs = NdotL*NdotV/(LdotH*max(NdotL,NdotV));
		return  (Gs);
	}

	Vec3f BrdfMicrofacet::specularity( Vec3f FresnelFunction,Vec3f SpecularDistribution, float GeometricShadow, float NdotL,float NdotV ){

		FresnelFunction.val[0] *= SpecularDistribution.val[0];
		FresnelFunction.val[1] *= SpecularDistribution.val[1];
		FresnelFunction.val[2] *= SpecularDistribution.val[2];


		return   (FresnelFunction * GeometricShadow )/ (4.0 * (  NdotL * NdotV));// multiplication!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
	}

    //  Color_rgb is the texture base color
	Vec3f BrdfMicrofacet:: diffuseColor ( Vec3f baseColor ,float _Metallic ){

		return baseColor * (1.0 - _Metallic)*RECIPROCAL_PI;
	}

	Vec3f BrdfMicrofacet:: brdfMicrofacet (  Vec3f baseColor ,float _Metallic, Vec3f specularity, float  NdotL){

		Vec3f diffuseColor_=diffuseColor(baseColor, _Metallic);
		Vec3f brdf_Val = specularity+diffuseColor_;
//		Vec3f brdf_Val = specularity;
//		brdf_Val *= NdotL;

		return brdf_Val;
	}

	void colorDeltaMap(Mat& src, Mat& output, float & upper, float & buttom ){

		std::vector<Mat> channels;
		channels.push_back(src);
		channels.push_back(src);
		channels.push_back(src);
		merge(channels, output);
		for(int x = 0; x < src.rows; ++x)
		{
			for(int y = 0; y < src.cols; ++y)
			{

				float delta =output.at<Vec3f>(x,y)[0];

				if(delta> upper){
					output.at<Vec3f>(x,y)[0] =0;
					output.at<Vec3f>(x,y)[1] =upper;
					output.at<Vec3f>(x,y)[2] =0;
					continue;
				}
				if(delta< buttom){
					output.at<Vec3f>(x,y)[0] =0;
					output.at<Vec3f>(x,y)[1] = buttom;
					output.at<Vec3f>(x,y)[2] =0;
					continue;
				}
				if (isnan(delta)){
//					output.at<Vec3f>(x,y)[0] =upper;
//					output.at<Vec3f>(x,y)[1] =0;
//					output.at<Vec3f>(x,y)[2] =0;
					continue;

				}
//				output.at<Vec3f>(x,y)[1] =0;
//				output.at<Vec3f>(x,y)[0] =0;


			}
		}
		output=output*(1.0/(upper-buttom))+(-buttom*(1.0/(upper-buttom)));
	}


	void updateDelta(
			         const Sophus::SE3d& CurrentT,
					 const Eigen::Matrix3d & K,
					 const Mat& image_baseColor,
					 const Mat depth_map,
					 const Mat& image_metallic,
					 const Mat& image_roughnes,
					 const Eigen::Vector3d & light_source,
					 Mat& deltaMap,
					 Mat& newNormalMap,
			         float& upper_b,
			         float& lower_b

					 ){
		double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2);

		std::unordered_map<int, int> inliers_filter;
		//new image

//		inliers_filter.emplace(173,333); //yes
//		inliers_filter.emplace(378,268); //yes
		inliers_filter.emplace(213,295); //yes
//		inliers_filter.emplace(370,488); //yes

//		pcl::PLYWriter writer;
//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

		for (int u = 0; u< depth_map.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < depth_map.cols; v++) // rowId,  rows: 0 to 640
			{
//				if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
//				if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~
//				cout<<"show delta:"<<deltaMap.at<double>(u,v)<<endl;

				Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id
				double d=depth_map.at<double>(u,v);
				// calculate 3D point coordinate
				Eigen::Vector3d p_3d_no_d((pixelCoord(0)-cx)/fx, (pixelCoord(1)-cy)/fy,1.0);
				Eigen::Vector3d p_c1=d*p_3d_no_d;


				// record point cloud
//				cloud->push_back(pcl::PointXYZ(p_c1.x(), p_c1.y(), p_c1.z()));
				// calculate normal for each point


				Eigen::Matrix<double,3,1> normal;
				normal.x()= newNormalMap.at<Vec3d>(u,v)[0];
				normal.y()= newNormalMap.at<Vec3d>(u,v)[1];
				normal.z()= newNormalMap.at<Vec3d>(u,v)[2];

				// calculate alpha_1
				Eigen::Matrix<double,3,1> alpha_1= light_C1(light_source) -p_c1;
				alpha_1=alpha_1.normalized();
				// calculate beta and beta_prime;
				Eigen::Matrix<double,3,1> beta,beta_prime;
				beta=-p_c1;
				beta=beta.normalized();
				beta_prime=-CurrentT.rotationMatrix().transpose()*CurrentT.translation()-p_c1;
				beta_prime=beta_prime.normalized();

				// baseColor_bgr[3],metallic, roughness
				float metallic=image_metallic.at<float>(u,v);
				float roughness =image_roughnes.at<float>(u,v);
				//Instantiation of BRDF object
//				if ( image_baseColor.at<Vec3f>(u,v)[1]<0.01 ){ continue;}
				Vec3f baseColor(image_baseColor.at<Vec3f>(u,v)[2],image_baseColor.at<Vec3f>(u,v)[1],image_baseColor.at<Vec3f>(u,v)[0]);

				Vec3f L_(alpha_1(0),alpha_1(1),alpha_1(2));
				Vec3f N_(normal(0),normal(1),normal(2));
				Vec3f View_beta(beta(0),beta(1),beta(2));
				Vec3f View_beta_prime(beta_prime(0),beta_prime(1),beta_prime(2));


				BrdfMicrofacet radiance_beta_vec(L_,N_,View_beta,(float )roughness,(float)metallic,baseColor);
				Vec3f radiance_beta= radiance_beta_vec.brdf_value_c3;

				BrdfMicrofacet radiance_beta_prime_vec(L_,N_,View_beta_prime,(float )roughness,(float)metallic,baseColor);
				Vec3f radiance_beta_prime= radiance_beta_prime_vec.brdf_value_c3;
				// right intensity / left intensity
				float delta_r= radiance_beta_prime.val[0]/radiance_beta.val[0];
				float delta_g= radiance_beta_prime.val[1]/radiance_beta.val[1];
				float delta_b= radiance_beta_prime.val[2]/radiance_beta.val[2];


				deltaMap.at<float>(u,v)= delta_g;



			}
		}
		double  max_n, min_n;
		cv::minMaxLoc(deltaMap, &min_n, &max_n);
		cout<<"------->show max and min of estimated deltaMap:"<< max_n <<","<<min_n<<endl;

//		colorDeltaMap
//		Mat coloredDeltaMap(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0,0,0)); // default value 1.0
//		colorDeltaMap(deltaMap, coloredDeltaMap,upper_b, lower_b);

////		writer.write("PointCloud.ply",*cloud, false);//
//		Mat mask = cv::Mat(deltaMap != deltaMap);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------remove nan value --------------
//		deltaMap.setTo(1.0, mask);

//		deltaMap=deltaMap*(1.0/(upper_b-lower_b))+(-lower_b*(1.0/(upper_b-lower_b)));
	}














}