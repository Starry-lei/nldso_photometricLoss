//
// Created by cheng on 05.12.22.
//

#include "deltaCompute.h"



namespace DSONL {




	float dot(const Eigen::Vector3f x, const Eigen::Vector3f y) {
		float ans = x.dot(y);
		return ans;
	}

	float clamp(const float x, const float min_v, const float max_v) {
		return std::min(std::max(x, min_v), max_v);
	}

	Vec3f  clamp_vec3f(Vec3f x){

		Vec3f res;
		res.val[0]= clamp(x.val[0], 0.0f, 1.0f);
		res.val[1]= clamp(x.val[1], 0.0f, 1.0f);
		res.val[2]= clamp(x.val[2], 0.0f, 1.0f);

		return res;
	}


	float _dot(Vec3f &fst, Vec3f &snd) { return max((float) 0.0, fst.dot(snd)); }
	float mod(const float numer, const float denom) {
		return std::fmod(numer, denom);
	}
	float glslmod(const float numer, const float denom) {
		return numer - denom * std::floor(numer / denom);
	}


	Vec3f pow(const float x, Vec3f y_vec) {
		Vec3f out_vec;
		for (int i = 0; i < 3; i++) {
			out_vec[i] = std::pow(x, y_vec[i]);
		}
		return out_vec;
	}
	Vec3f pow(const Vec3f x_vec, const Vec3f y_vec) {
		Vec3f out_vec;
		for (int i = 0; i < 3; i++) {
			out_vec[i] = std::pow(x_vec[i], y_vec[i]);
		}
		return out_vec;
	}
	float mix(const float x, const float y, const float a) {
		if (a > 1 || a < 0)
			throw std::invalid_argument("received value a not in interval(0,1)");
		return x * (1 - a) + y * a;
	}
	Vec3f mix(const Vec3f x_vec, const Vec3f y_vec, const float a) {
		Vec3f out_vec;
		for (int i = 0; i < 3; i++) {
			out_vec[i] = mix(x_vec[i], y_vec[i], a);
		}
		return out_vec;
	}

	/*
        Reflect — calculate the reflection direction for an incident vector
        I: Specifies the incident vector.
        N: Specifies the normal vector.
    */
	Vec3f reflect(Vec3f I, Vec3f N) {


		Vec3f N_norm = cv::normalize(N);
		Vec3f out_vec = I - 2.0f * N_norm.dot(I) * N_norm;

		return out_vec;
	}

	IBL_Radiance::IBL_Radiance() {}
	IBL_Radiance::~IBL_Radiance() {}

	Vec2f IBL_Radiance::directionToSphericalEnvmap(Vec3f dir) {

		float s = 1.0 - glslmod(1.0 / (2.0 * M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);
		//      float s = 1.0 - mod(1.0 / (2.0*M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);

		float t = 1.0 / (M_PI) *acos(-dir.val[2]);
		if (s > 1.0 || t > 1.0) { std::cerr << "UV coordinates overflow!" << std::endl; }

		return Vec2f(s, t);
	}

	Vec3f IBL_Radiance::specularIBL(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w) {

		float NoV = clamp(_dot(N, V), 0.0, 1.0);

		// Vec3f R =reflect(-V, N);
		// std::cout<<"show the norm of R"<<norm(R)<<std::endl;
		// Vec2f uv = directionToSphericalEnvmap(R);

		Vec3f R_c = reflect(-V, N);//
		// -0.927108467, 0.132981807, -0.350408018
		Eigen::Matrix<float, 3, 1> R_c_(R_c.val[0], R_c.val[1], R_c.val[2]);
		// convert coordinate system
		Eigen::Vector3f R_w = Camera1_c2w.cast<float>() * R_c_;

		Vec2f uv = directionToSphericalEnvmap(Vec3f(R_w.x(), R_w.y(), R_w.z()));


		if (uv.val[0] > 1.0 || uv.val[1] > 1.0) {
			std::cerr << "\n===specularIBL=======Show UV=================:" << uv << std::endl;
		}


		//      std::cout<<"uv:"<<uv<<"show roughness*float(mipCount)"<<roughness*float(mipCount)<<std::endl;

		Vec3f prefiltered_Color = prefilteredColor(uv.val[0], uv.val[1], roughness * float(mipCount));

		//      std::cout<<"prefiltered_Color:"<<prefiltered_Color<<std::endl;

		//
		//    uv:[0.179422, 0.400616]show roughness*float(mipCount)1.5
		//          prefiltered_Color:[0.112979, 0.0884759, 0.0929931]

		//      show image_ref_path_PFM  of GSN(293,476)And values: [5.17454, 4.71557, 0.0619548]


		Vec3f prefiltered_Color_check = prefiltered_Color;

		Vec2f brdf_val = brdfIntegration(NoV, roughness);
		// TODO: use matrix operator here
		F0.val[0] *= brdf_val.val[0];
		F0.val[1] *= brdf_val.val[0];
		F0.val[2] *= brdf_val.val[0];

		F0.val[0] += brdf_val.val[1];
		F0.val[1] += brdf_val.val[1];
		F0.val[2] += brdf_val.val[1];
		prefiltered_Color.val[0] *= F0.val[0];
		prefiltered_Color.val[1] *= F0.val[1];
		prefiltered_Color.val[2] *= F0.val[2];
		return prefiltered_Color;
	}

//	Vec3f IBL_Radiance::specularIBLCheck(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w) {
//		float NoV = clamp(_dot(N, V), 0.0, 1.0);
//		Vec3f R_c = reflect(-V, N);//
//
//		// -0.927108467, 0.132981807, -0.350408018
//
//		Eigen::Matrix<float, 3, 1> R_c_(R_c.val[0], R_c.val[1], R_c.val[2]);
//		// convert coordinate system
//		Eigen::Vector3f R_w = Camera1_c2w.cast<float>() * R_c_;
//
//		Vec2f uv = directionToSphericalEnvmap(Vec3f(R_w.x(), R_w.y(), R_w.z()));
//
//		R_c = cv::normalize(R_c);
//		return R_c;
//	}


	Vec3f IBL_Radiance::diffuseIBL(Vec3f normal) {

		Vec2f uv = directionToSphericalEnvmap(normal);

		if (uv.val[0] > 1.0 || uv.val[1] > 1.0) {
			std::cerr << "\n===diffuseIBL=======Show UV=================:" << uv << std::endl;
		}


		gli::vec4 Sample_val = diffuseSampler->texture_lod(gli::fsampler2D::normalized_type(uv.val[0], 1 - uv.val[1]), 0.0f);// transform the texture coordinate
		                                                                                                                     //      std::cout << "\n============diffuseIBL val(BGRA):\n"<<"diffuseIBL:"<< uv<<"diffuse val:"<< Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << std:: endl;
		return cv::Vec3f((float) Sample_val.b, (float) Sample_val.g, (float) Sample_val.r);
	}

	Vec3f IBL_Radiance::fresnelSchlick(float cosTheta, Vec3f F0) {

		return (F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * std::pow(1.0f - cosTheta, 5.0f));
	}

	Vec3f IBL_Radiance::prefilteredColor(float u, float v, float level) {

		gli::vec4 Sample_val = prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(u, 1 - v), level);// transform the texture coordinate
		//      std::cout << "\n============Sample_val val(BGRA):\n" << Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << std:: endl;
		return cv::Vec3f((float) Sample_val.b, (float) Sample_val.g, (float) Sample_val.r);
	}

	Vec2f IBL_Radiance::brdfIntegration(float NoV, float roughness) {
		gli::vec4 SampleBrdf = brdfSampler_->texture_lod(gli::fsampler2D::normalized_type(NoV, 1 - roughness), 0.0f);// transform the texture coordinate
		                                                                                                            //     std::cout << "\n============SampleBrdf val(BGRA)!!!!!!!:\n" << SampleBrdf.b << "," << SampleBrdf.g << "," << SampleBrdf.r << "," << SampleBrdf.a << std::endl;
		return Vec2f(SampleBrdf.b, SampleBrdf.g);                                                                   // x, y  SampleBrdf.b, SampleBrdf.g
	}

	Vec3f IBL_Radiance::solveForRadiance(Vec3f viewDir, Vec3f normal,
	                                     const float &roughnessValue,
	                                     const float &metallicValue,
	                                     const float &reflectance,
	                                     const Vec3f &baseColorValue,
	                                     const Eigen::Matrix3d Camera1_c2w) {

		// !!!!!!!!  vec3 baseCol = pow(texture(baseColorTexture, texScale*tc).rgb, vec3(2.2)); // this is gamma correction!

		Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);
		Vec3f f0 = 0.16 * (reflectance * reflectance) * One;
		f0 = mix(f0, baseColorValue, metallicValue);
		Vec3f F = fresnelSchlick(_dot(normal, viewDir), f0);
		Vec3f kS = F;
		Vec3f kD = One - kS;
		kD = kD.mul(One - metallicValue * One);
		Vec3f specular = specularIBL(f0, roughnessValue, normal, viewDir, Camera1_c2w);


		//convert from camera to world
		Eigen::Vector3d normal_c(normal.val[0], normal.val[1], normal.val[2]);
		Vec3f normal_w((Camera1_c2w * normal_c).x(), (Camera1_c2w * normal_c).y(), (Camera1_c2w * normal_c).z());//

		Vec3f diffuse = diffuseIBL(normal_w);
		// shading front-facing
		Vec3f color = pow(kD.mul(baseColorValue.mul(diffuse)) + specular, 1.0 / 2.2 * One);
		//      Vec3f color = specular;
		//      Vec3f color=diffuse;


		// shading back-facing????????????????????????????????
		if (viewDir.dot(normal) < -0.1) {
			//discard;
			color = 0.1 * baseColorValue.mul(diffuse);
			//        std::cerr<<"_dot(viewDir, normal) < -0.1:"<< color<<std::endl;
		}

		return color;
	}

	Vec3f IBL_Radiance::RRTAndODTFit(Vec3f v) {
		Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);

		Vec3f a = v .mul(v + 0.0245786f*One ) - 0.000090537f*One;
		Vec3f b = v.mul (0.983729f * v + 0.4329510f*One) + 0.238081f*One;

		Vec3f inv_v= Vec3f (1.0f/b.val[0],  1.0f/b.val[1],1.0f/b.val[2] );

		return a.mul(inv_v);

	}

	Vec3f IBL_Radiance::ACESFilm(Vec3f radiance) {

		// https://computergraphics.stackexchange.com/questions/11018/how-to-change-a-rgb-value-using-a-radiance-value
		// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
		// method 1

		float adapted_lum=1.0;
		radiance*= adapted_lum;
		Eigen::Matrix3f ACESInputMat, ACESOutputMat;
		Eigen::Vector3f r1, r2;
		ACESInputMat <<  0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777;
		ACESOutputMat <<  1.60475, -0.53108, -0.07367 , -0.10208,  1.10813, -0.00605 ,-0.00327, -0.07276,  1.07602;

		r1= ACESInputMat* Eigen::Vector3f(radiance.val[0], radiance.val[1],radiance.val[2] );
		radiance=RRTAndODTFit(Vec3f(r1.x(),r1.y(),r1.z()));
		r2= ACESOutputMat*Eigen::Vector3f(radiance.val[0], radiance.val[1],radiance.val[2] );

		radiance=Vec3f(r2.x(),r2.y(),r2.z());
		// method 2
		//			float a = 2.51f;
		//			float b = 0.03f;
		//			float c = 2.43f;
		//			float d = 0.59f;
		//			float e = 0.14f;
		//		    Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);
		//		    Vec3f sndVecor= Vec3f(1.0/(radiance.mul(c*radiance+d*One)+e*One).val[0],
		//		                           1.0/(radiance.mul(c*radiance+d*One)+e*One).val[1],
		//		                           1.0/(radiance.mul(c*radiance+d*One)+e*One).val[2]);
		//		return  clamp_vec3f((radiance.mul(a*radiance+b*One)) .mul(sndVecor));

		// method 1 output
		return clamp_vec3f(radiance);

	}





	void updateDelta(
			envLight* EnvLight,
	        Eigen::Matrix3d Camera1_c2w,
//	        const Sophus::SE3d &CurrentT,
	        Sophus::SO3d& Rotation,
	        Eigen::Matrix<double, 3, 1>& Translation,
	        const Eigen::Matrix3f &K,
	        const Mat &image_baseColor,
	        const Mat depth_map,
	        const Mat &image_metallic_,
	        const Mat &image_roughnes_,
	        Mat &deltaMap,
	        Mat &newNormalMap,
	        float &upper_b,
	        float &lower_b) {

		// ===================================RENDERING PARAMETERS:====================================
		float fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);
		float reflectance = 1.0f;
		//      vec3 normal = normalize(wfn);
		//      vec3 viewDir = normalize(cameraPos - vertPos);
		std::unordered_map<int, int> inliers_filter;
		inliers_filter.emplace(290, 130);//


		Mat radianceMap_left(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));

        // TODO(Binghui): to be parallelized

		for (int u = 0; u < depth_map.rows; u++)// colId, cols: 0 to 480
		{
			for (int v = 0; v < depth_map.cols; v++)// rowId,  rows: 0 to 640
			{

				// if(inliers_filter.count(u)==0){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~~~~~~~~~~
				// if(inliers_filter[u]!=v ){continue;}// ~~~~~~~~~~~~~~Filter~~~~~~~~~~~~~~
				//cout<<"show delta:"<<deltaMap.at<double>(u,v)<<endl;


                // get image_roughnes

                float image_roughnes= image_roughnes_.at<float>(u,v);
                float image_metallic= image_metallic_.at<float>(u,v);


				// ===================================PROJECTION====================================
				Eigen::Vector2f pixelCoord((float) v, (float) u);//  u is the row id , v is col id
				float iDepth = depth_map.at<double>(u, v);
//				if (round(1.0 / iDepth) == 15.0) { continue; }



				Eigen::Vector3f p_3d_no_d((pixelCoord(0) - cx) / fx, (pixelCoord(1) - cy) / fy, (float) 1.0);
				Eigen::Vector3f p_c1;
				p_c1 << p_3d_no_d.x() / iDepth, p_3d_no_d.y() / iDepth, p_3d_no_d.z() / iDepth;

				// record point cloud
				//cloud->push_back(pcl::PointXYZ(p_c1.x(), p_c1.y(), p_c1.z()));
				// calculate normal for each point Transformation_wc

				// ===================================NORMAL====================================
				Eigen::Matrix<float, 3, 1> normal;
				normal.x() = newNormalMap.at<Vec3f>(u, v)[0];
				normal.y() = newNormalMap.at<Vec3f>(u, v)[1];
				normal.z() = newNormalMap.at<Vec3f>(u, v)[2];
				// convert normal vector from camera coordinate system to world coordinate system
				normal = normal.normalized();

				// ===================================VIEW-DIRECTION====================================
				Eigen::Matrix<float, 3, 1> beta, beta_prime;
				beta = -p_c1;
				beta = beta.normalized();
				beta_prime = - Rotation.matrix().transpose().cast<float>() * Translation.cast<float>() - p_c1;
				beta_prime = beta_prime.normalized();


				// ===================================BASE-COLOR=============================================
				Vec3f baseColor(image_baseColor.at<Vec3f>(u, v)[2], image_baseColor.at<Vec3f>(u, v)[1], image_baseColor.at<Vec3f>(u, v)[0]);
				Vec3f N_(normal(0), normal(1), normal(2));
				Vec3f View_beta(beta(0), beta(1), beta(2));
				Vec3f View_beta_prime(beta_prime(0), beta_prime(1), beta_prime(2));

				// ===================================search for Env Light from control points===================

//				Vec3f firstPoint(0.660882, -0.334907, 0.799127);
//				pcl::PointXYZ searchPoint(firstPoint.val[0], firstPoint.val[1], firstPoint.val[2]);

				pcl::PointXYZ searchPoint(p_c1.x(), p_c1.y(), p_c1.z());
				float radius = 0.2;
				std::vector<int> pointIdxRadiusSearch;
				std::vector<float> pointRadiusSquaredDistance;

				Vec3f key4Search;
				if ( EnvLight->kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
					for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
						std::cout << "\n------"<<       (*(EnvLight->ControlpointCloud))[ pointIdxRadiusSearch[0] ].x
								  << " " << (*(EnvLight->ControlpointCloud))[ pointIdxRadiusSearch[0] ].y
								  << " " << (*(EnvLight->ControlpointCloud))[ pointIdxRadiusSearch[0] ].z
								  << " (squared distance: " << pointRadiusSquaredDistance[0] << ")" << std::endl;

					float pointCorres_x = (*(EnvLight->ControlpointCloud))[pointIdxRadiusSearch[0]].x;
					float pointCorres_y = (*(EnvLight->ControlpointCloud))[pointIdxRadiusSearch[0]].y;
					float pointCorres_z = (*(EnvLight->ControlpointCloud))[pointIdxRadiusSearch[0]].z;

					key4Search.val[0]= pointCorres_x;
					key4Search.val[1]= pointCorres_y;
					key4Search.val[2]= pointCorres_z;
				}

				//    std::cerr<<"\n show count of envLightMap"<<  EnvLight->envLightMap.count(key4Search)<<endl;
				brdfSampler_ = & (EnvLight->brdfSampler[0]);
				prefilteredEnvmapSampler= & (EnvLight->envLightMap[key4Search].EnvmapSampler[0]);
				diffuseSampler = & (EnvLight->envLightMap[key4Search].EnvmapSampler[1]);






				// ===================================RADIANCE-COMPUTATION====================================
				IBL_Radiance *ibl_Radiance = new IBL_Radiance;
				Vec3f radiance_beta = ibl_Radiance->ACESFilm(ibl_Radiance->solveForRadiance(View_beta, N_, image_roughnes, image_metallic, reflectance, baseColor, Camera1_c2w));
				Vec3f radiance_beta_prime = ibl_Radiance->ACESFilm(ibl_Radiance->solveForRadiance(View_beta_prime, N_, image_roughnes, image_metallic, reflectance, baseColor, Camera1_c2w));

				// ===================================SAVE-RADIANCE===========================================
				radianceMap_left.at<Vec3f>(u, v) = radiance_beta;

				//  ===================================TONE MAPPING===========================================
				// remark: Yes, you need to multiply by exposure before the tone mapping and do the gamma correction after.
// TODO: use a good tone mapping so that we can get closer to GT delta map like clamp in [0,1.2]


				// right intensity / left intensity
				float delta_r = radiance_beta_prime.val[0] / radiance_beta.val[0];
				float delta_g = radiance_beta_prime.val[1] / radiance_beta.val[1];
				float delta_b = radiance_beta_prime.val[2] / radiance_beta.val[2];
				deltaMap.at<float>(u, v) = delta_g;
				//deltaMap.at<float>(u, v) = delta_b;
			}
		}


		double max_n, min_n;
		cv::minMaxLoc(deltaMap, &min_n, &max_n);
		std::cout << "------->show max and min of estimated deltaMap<-----------------:" << max_n << "," << min_n << std::endl;
		cvtColor(radianceMap_left, radianceMap_left, COLOR_BGR2RGB);


		double max_n_radiance, min_n_radiance;
		cv::minMaxLoc(radianceMap_left, &min_n_radiance, &max_n_radiance);
		std::cout << "------->show max and min of estimated radianceMap_left<-----------------:" << max_n_radiance << "," << min_n_radiance << std::endl;

		imshow("radianceMap_left", radianceMap_left);
		//	 DiffuseMAP
		//	 show image_ref_path_PFM range:0.0321633,2.18064
		//	 min_depth_val0.0313726max_depth_val2.17949

		// queried val
		// ianceMap_left<-----------------:2.17949,0
		//	  show image_ref_path_PFM range:0,2.1785

		//		writer.write("PointCloud.ply",*cloud, false);//
		//		Mat mask = cv::Mat(deltaMap != deltaMap);// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------remove nan value --------------
		//		deltaMap.setTo(1.0, mask);

		//      waitKey(0);
	}


}// namespace DSONL
