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

	Vec3f IBL_Radiance::specularIBL(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w,
                                    Sophus::SO3f enterEnv_Rotatio_inv
//                                    ,
//                                    pointEnvlight pointEnvlight_cur
                                    ) {

		float NoV = clamp(_dot(N, V), 0.0, 1.0);

		// Vec3f R =reflect(-V, N);
		// std::cout<<"show the norm of R"<<norm(R)<<std::endl;
		// Vec2f uv = directionToSphericalEnvmap(R);

		Vec3f R_c = reflect(-V, N);//
		// -0.927108467, 0.132981807, -0.350408018
		Eigen::Matrix<float, 3, 1> R_c_(R_c.val[0], R_c.val[1], R_c.val[2]);
		// convert coordinate system
		Eigen::Vector3f R_w =  enterEnv_Rotatio_inv.matrix()*Camera1_c2w.cast<float>() * R_c_;


		Vec2f uv = directionToSphericalEnvmap(Vec3f(R_w.x(), R_w.y(), R_w.z()));
		if (uv.val[0] > 1.0 || uv.val[1] > 1.0) {
			std::cerr << "\n===specularIBL=======Show UV=================:" << uv << std::endl;
		}


		//      std::cout<<"uv:"<<uv<<"show roughness*float(mipCount)"<<roughness*float(mipCount)<<std::endl;
		Vec3f prefiltered_Color = prefilteredColor(uv.val[0], uv.val[1], roughness * float(mipCount)
//                                                   ,pointEnvlight_cur
        );

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

//        cout<<"show T BRDF>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<F0<<endl;
//        cout<<"\n show T u  and 1-v >>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<uv.val[0]<<","<< uv.val[1]<<":"<<" 1- uv.val[1]"<<endl;

		prefiltered_Color.val[0] *= F0.val[0];
		prefiltered_Color.val[1] *= F0.val[1];
		prefiltered_Color.val[2] *= F0.val[2];

//        cout<<"show T prefiltered_Color>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<<prefiltered_Color<<endl;
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

		                   // new formula                                                                                                  //      std::cout << "\n============diffuseIBL val(BGRA):\n"<<"diffuseIBL:"<< uv<<"diffuse val:"<< Sample_val.b << "," << Sample_val.g << "," << Sample_val.r << ","   << Sample_val.a << std:: endl;
		cv::Vec3f radiance_diff= cv::Vec3f((float) Sample_val.b, (float) Sample_val.g, (float) Sample_val.r);

        return radiance_diff;
	}

	Vec3f IBL_Radiance::fresnelSchlick(float cosTheta, Vec3f F0) {

		return (F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * std::pow(1.0f - cosTheta, 5.0f));
	}

	Vec3f IBL_Radiance::prefilteredColor(float u, float v, float level
//                                         ,pointEnvlight pointEnvlight_cur
                                         ) {
		gli::vec4 Sample_val =  prefilteredEnvmapSampler->texture_lod(gli::fsampler2D::normalized_type(u, 1 - v), level);// transform the texture coordinate
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
	                                     const Eigen::Matrix3d Camera1_c2w,
                                         Sophus::SO3f enterEnv_Rotation_inv
//                                         ,
//                                         pointEnvlight pointEnvlight_cur
                                         ) {

		// !!!!!!!!  vec3 baseCol = pow(texture(baseColorTexture, texScale*tc).rgb, vec3(2.2)); // this is gamma correction!

		Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);
		Vec3f f0 = 0.16 * (reflectance * reflectance) * One;
		f0 = mix(f0, baseColorValue, metallicValue);
		Vec3f F = fresnelSchlick(_dot(normal, viewDir), f0);
		Vec3f kS = F;
		Vec3f kD = One - kS;
		kD = kD.mul(One - metallicValue * One);
		Vec3f specular = specularIBL(f0, roughnessValue, normal, viewDir,
                                     Camera1_c2w,enterEnv_Rotation_inv
//                                     ,
//                                     pointEnvlight_cur
                                     );

        Specularity =specular;
		//convert from camera to world
		Eigen::Vector3d normal_c(normal.val[0], normal.val[1], normal.val[2]);
		Vec3f normal_w((Camera1_c2w * normal_c).x(), (Camera1_c2w * normal_c).y(), (Camera1_c2w * normal_c).z());//
		Vec3f diffuse = diffuseIBL(normal_w);
        diffusity=diffuse;


//        diffuse=Vec3f(0.0,0.0,0.0);
		// shading front-facing
        Vec3f color = pow(kD.mul(baseColorValue.mul(diffuse)) + specular, 1.0 / 2.2 * One);



//        cout<<"\n Checking vals:"<<"kD: "<<kD <<","<<"baseColorValue: "<< baseColorValue<< ","<<"diffuse: "<<diffuse<<","<<"specular: "<<specular<<endl;
//        cout<<"Checking color:"<<color<<endl;
		//      Vec3f color = specular;
		//      Vec3f color=diffuse;





//		// shading back-facing????????????????????????????????
//		if (viewDir.dot(normal) < -0.1) {
//			//discard;
//			color = 0.1 * baseColorValue.mul(diffuse);
//			//        std::cerr<<"_dot(viewDir, normal) < -0.1:"<< color<<std::endl;
//		}

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
            Sophus::SE3d& Camera1_c2w,
            envLightLookup* EnvLightLookup,
            float *statusMap,
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
	        float &lower_b
            , Mat pointOfInterest
            ) {

		// ===================================RENDERING PARAMETERS:====================================
		float fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);
		float reflectance = 1.0f;
		//      vec3 normal = normalize(wfn);
		//      vec3 viewDir = normalize(cameraPos - vertPos);
		std::unordered_map<int, int> inliers_filter, inliers_filter_i;

//		inliers_filter.emplace(108, 97 );//cabinet
//        inliers_filter.emplace(125, 102);//table

        inliers_filter.emplace( 112, 130); // 112, 130

//        inliers_filter_i.emplace(108,97);
//        inliers_filter_i.emplace(105,119);
//        inliers_filter_i.emplace(105,117);
//        inliers_filter_i.emplace(98,202);
//        inliers_filter_i.emplace(95,242);
//        inliers_filter_i.emplace(125,102);





        Mat ctrlPointMask(deltaMap.rows, deltaMap.cols, CV_8UC3, Scalar(0,0,0));

		Mat radianceMap_left(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));
        Mat radianceMap_right(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));

        Mat radianceMap_leftSave(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));
        Mat radianceMap_rightSave(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));

        Mat specularityMap(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));
        Mat DiffuseMap(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));

        // TODO: to be parallelized
        // K nearest neighbor search
        int num_K = 1;
        Vec2i boundingBoxUpperLeft(83, 76);
        Vec2i boundingBoxBotRight(240, 320);


        Vec2i boundingBoxUpperLeft_AoI( 145,180);
        Vec2i boundingBoxBotRight_AoI(173,242);

		for (int u = 0; u < depth_map.rows; u++)// colId, cols: 0 to 480
		{

            for (int v = 0; v < depth_map.cols; v++)// rowId,  rows: 0 to 640
			{

                //=====================================Inliers Filter=====================================
                //				 if(inliers_filter.count(u)==0){continue;}
                //				 if(inliers_filter[u]!=v ){continue;}


                //=====================================Area of interest Filter=====================================
//                                if ( (v<boundingBoxUpperLeft_AoI.val[1] || v>boundingBoxBotRight_AoI.val[1]) || (u< boundingBoxUpperLeft_AoI.val[0] ||  u> boundingBoxBotRight_AoI.val[0])){ continue;}


                // ====================================use DSO pixel selector================================================
                //                if (statusMap!=NULL && statusMap[u*depth_map.cols+v]==0 ){ continue;}

                // =====================================use non lambertian point selector================================
//                                if (statusMap!=NULL && static_cast<int>(statusMap[u * depth_map.cols + v])!= 255){ continue;}

                if (pointOfInterest.at<uchar>(u,v)!=255){ continue;}



                // cout<<"show current index:"<< u<<","<<v<<endl;
                // get image_roughnes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                float image_roughnes= image_roughnes_.at<float>(u,v);
                float image_metallic= image_metallic_.at<float>(u,v);
                //                float image_roughnes= 0.1;
                //                float image_metallic= 1.0;

				// ===================================PROJECTION====================================
				Eigen::Vector2f pixelCoord((float) v, (float) u);//  u is the row id , v is col id
				float iDepth = depth_map.at<double>(u, v);
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

                // envMapPose_world
				// ===================================BASE-COLOR=============================================
//                 Vec3f baseColor(image_baseColor.at<Vec3f>(u, v)[2], image_baseColor.at<Vec3f>(u, v)[1], image_baseColor.at<Vec3f>(u, v)[0]);
                Vec3f baseColor(std::pow(image_baseColor.at<Vec3f>(u, v)[2], 2.2), std::pow(image_baseColor.at<Vec3f>(u, v)[1], 2.2), std::pow(image_baseColor.at<Vec3f>(u, v)[0], 2.2));

                // vec3 baseCol = pow(texture(baseColorTexture, texScale*tc).rgb, vec3(2.2)); //~~~
                //                pow(image_baseColor.at<Vec3f>(u, v)[2], 2.2);
                //                pow(image_baseColor.at<Vec3f>(u, v)[1], 2.2);
                //                pow(image_baseColor.at<Vec3f>(u, v)[0], 2.2);

// GT  Checking radiance vals:left Coord: u:172, v:207left_intensity:0.905309  and right_intensity at pixel_x:211, pixel_y:222is:        0.766174  show GT delta: 0.846311
// ES  Checking radiance vals:left Coord: u:172, v:207left_radiance:1.00905    and right_intensity at pixel_x:pixel_x, pixel_y:pixel_yis:1.00017   show delta_g: 0.991198




                Vec3f N_(normal(0), normal(1), normal(2));
				Vec3f View_beta(beta(0), beta(1), beta(2));
				Vec3f View_beta_prime(beta_prime(0), beta_prime(1), beta_prime(2));

				// ===================================search for Env Light from control points===================
                // coordinate system conversion
                Sophus::SE3f Camera1_extrin = Camera1_c2w.cast<float>();
                Eigen::Vector3f p_c1_w=Camera1_extrin* p_c1;
				pcl::PointXYZ searchPoint(p_c1_w.x(), p_c1_w.y(), p_c1_w.z());

                std::vector<int> pointIdxKNNSearch(num_K);
                std::vector<float> pointKNNSquaredDistance(num_K);
				Vec3f key4Search;
				if ( EnvLightLookup->kdtree.nearestKSearch(searchPoint, num_K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {


//					for (std::size_t i = 0; i < pointIdxKNNSearch.size (); ++i)
//                    {
//                        std::cout << "\n------"<<
//                                           (*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[0] ].x
//                                  << " " << (*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[0]].y
//                                  << " " << (*(EnvLightLookup->ControlpointCloud))[ pointIdxKNNSearch[0]].z
//                                  << " (squared distance: " << pointKNNSquaredDistance[0] << ")" << std::endl;
//                    }

                    key4Search.val[0] = (*(EnvLightLookup->ControlpointCloud))[pointIdxKNNSearch[0]].x;
                    key4Search.val[1] = (*(EnvLightLookup->ControlpointCloud))[pointIdxKNNSearch[0]].y;
                    key4Search.val[2] = (*(EnvLightLookup->ControlpointCloud))[pointIdxKNNSearch[0]].z;
				}
//                cout<<"\n Show current shader point:\n"<<p_c1_w<<"\n show nearst envMap point coordinate:\n"<<key4Search<<endl;
//                cout<<"\n show count of envLightMap"<<  EnvLightLookup->envLightIdxMap.count(key4Search)<<endl;
//                cout<<"show EnvLight size:"<< EnvLightLookup->envLightIdxMap.size()<<endl;
//
                int ctrlIndex= EnvLightLookup->envLightIdxMap[key4Search];
//                cout<<"\n show ctrlIndex :"<< ctrlIndex<<endl;

                if ( EnvLightLookup->envLightIdxMap.size()==0){std::cerr<<"Error in EnvLight->envLightIdxMap! "<<endl;}


                // Get pyramid
                std::string renderedEnvLight_path="/home/lei/Documents/Research/envMapData/maskedSelector";
                stringstream ss;
                string img_idx_str;
                ss << ctrlIndex;
                ss >> img_idx_str;
                string name_prefix = "/envMap";


                string renderedEnvLightfolder =renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLight";
                string renderedEnvLightDiffuse =renderedEnvLight_path + "/envMap" + img_idx_str + "/renderedEnvLightDiffuse";
                string envMapDiffuse = renderedEnvLightDiffuse + "/envMapDiffuse_" + img_idx_str + ".pfm";

                pointEnvlight pEnv;

                EnvMapLookup *EnvMapLookup = new DSONL::EnvMapLookup();
                EnvMapLookup->makeMipMap(pEnv.EnvmapSampler,renderedEnvLightfolder); // index_0: prefiltered Env light
                delete EnvMapLookup;

                diffuseMap *diffuseMap = new DSONL::diffuseMap;
                diffuseMap->makeDiffuseMap(pEnv.EnvmapSampler, envMapDiffuse); // index_1: diffuse
                delete diffuseMap;





                prefilteredEnvmapSampler= & (pEnv.EnvmapSampler[0]);
                brdfSampler_ = & (EnvLightLookup->brdfSampler[0]);
				diffuseSampler = & (pEnv.EnvmapSampler[1]);


//                int numberColor= 12;
//                if (ctrlIndex%numberColor==0){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 255;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 0;
//                } else if(ctrlIndex%numberColor==1){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 255;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 0;
//                } else if(ctrlIndex%numberColor==2){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 255;
//                } else if(ctrlIndex%numberColor==3){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 255;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 255;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 0;
//                } else if(ctrlIndex%numberColor==4){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 255;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 255;
//                }else if (ctrlIndex%numberColor==5){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 255;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 255;
//                }
//                else if (ctrlIndex%numberColor==6){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 125;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 0;
//                } else if(ctrlIndex%numberColor==7){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 125;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 0;
//                } else if(ctrlIndex%numberColor==8){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 125;
//                } else if(ctrlIndex%numberColor==9){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 125;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 125;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 0;
//                } else if(ctrlIndex%numberColor==10){
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 125;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 125;
//                }else {
//                    ctrlPointMask.at<Vec3b>(u,v)[0]= 125;
//                    ctrlPointMask.at<Vec3b>(u,v)[1]= 0;
//                    ctrlPointMask.at<Vec3b>(u,v)[2]= 125;
//                }


				// ===================================RADIANCE-COMPUTATION====================================
				IBL_Radiance *ibl_Radiance = new IBL_Radiance;
//				Vec3f radiance_beta = ibl_Radiance->ACESFilm(ibl_Radiance->solveForRadiance(View_beta, N_, image_roughnes, image_metallic, reflectance, baseColor, Camera1_c2w.rotationMatrix()));
//				Vec3f radiance_beta_prime = ibl_Radiance->ACESFilm(ibl_Radiance->solveForRadiance(View_beta_prime, N_, image_roughnes, image_metallic, reflectance, baseColor, Camera1_c2w.rotationMatrix()));


//                Sophus::SO3f enterPanoroma = EnvLight->envLightMap[key4Search].envMapPose_world.rotationMatrix();
                Sophus::SO3f enterPanoroma ;

                Vec3f radiance_beta = ibl_Radiance->solveForRadiance(View_beta, N_, image_roughnes, image_metallic,
                                                                     reflectance, baseColor, Camera1_c2w.rotationMatrix(),
                                                                     enterPanoroma.inverse()
//                                                                     ,EnvLight->envLightMap[key4Search]
                                                                     );
                Vec3f radiance_beta_prime = ibl_Radiance->solveForRadiance(View_beta_prime, N_, image_roughnes, image_metallic,
                                                                           reflectance, baseColor, Camera1_c2w.rotationMatrix(),
                                                                           enterPanoroma.inverse()
//                                                                           ,EnvLight->envLightMap[key4Search]
                                                                           );

				// ===================================SAVE-RADIANCE===========================================
				radianceMap_left.at<Vec3f>(u, v) = radiance_beta;
                radianceMap_right.at<Vec3f>(u,v)= radiance_beta_prime;


                radianceMap_leftSave.at<Vec3f>(u, v) =ibl_Radiance->ACESFilm( radiance_beta);
                radianceMap_rightSave.at<Vec3f>(u, v) = ibl_Radiance->ACESFilm( radiance_beta_prime);


                specularityMap.at<Vec3f>(u,v)=ibl_Radiance->Specularity;
                DiffuseMap.at<Vec3f>(u,v)=ibl_Radiance->diffusity;



				//  ===================================TONE MAPPING===========================================
				// remark: Yes, you need to multiply by exposure before the tone mapping and do the gamma correction after.
                // TODO: use a good tone mapping so that we can get closer to GT delta map like clamp in [0,1.2]

				// right intensity / left intensity
				float delta_r = radiance_beta_prime.val[0] / radiance_beta.val[0];
				float delta_g = radiance_beta_prime.val[1] / radiance_beta.val[1];
				float delta_b = radiance_beta_prime.val[2] / radiance_beta.val[2];
				deltaMap.at<float>(u, v) = delta_g;

//                cout<<"\n Checking radiance vals:"<< "left Coord: u:"<<u<<", v:"<<v<<"left_radiance:"<< radiance_beta.val[1]
//                    << "and right_intensity at pixel_x:"<<"pixel_x"<<", pixel_y:"<< "pixel_y"<< "is:"<<  radiance_beta_prime.val[1]
//                    <<"  show delta_g: "<<delta_g <<endl;



                //deltaMap.at<float>(u, v) = delta_b;
			}
		}


		double max_n, min_n;
		cv::minMaxLoc(deltaMap, &min_n, &max_n);
		std::cout << "------->show max and min of estimated deltaMap<-----------------:" << max_n << "," << min_n << std::endl;
//        cvtColor(radianceMap_left, radianceMap_left, COLOR_RGB2BGR);
//        cvtColor(radianceMap_leftSave, radianceMap_leftSave, COLOR_RGB2BGR);

        cvtColor(radianceMap_left, radianceMap_left, COLOR_BGR2RGB);
        cvtColor(radianceMap_leftSave, radianceMap_leftSave, COLOR_BGR2RGB);


//        imwrite("radianceMap_left.png", radianceMap_left*255.0);
//        imwrite("radianceMap_leftSave.png", radianceMap_leftSave*255.0);
//        imwrite("radianceMap_right.png", radianceMap_right*255.0);
//        imwrite("radianceMap_rightSave.png", radianceMap_rightSave*255.0);
//        imwrite("ctrlPointMask.png",ctrlPointMask);




		double max_n_radiance, min_n_radiance;
		cv::minMaxLoc(radianceMap_left, &min_n_radiance, &max_n_radiance);
		std::cout << "------->show max and min of estimated radianceMap_left<-----------------:" << max_n_radiance << "," << min_n_radiance << std::endl;

		imshow("radianceMap_left", radianceMap_left);
        imshow("specularityMap", specularityMap);
        imshow("DiffuseMap", DiffuseMap);
        imshow("ctrlPointMask",ctrlPointMask);

//
//        for (int u = 0; u < depth_map.rows; u++)// colId, cols: 0 to 480
//        {
//            for (int v = 0; v < depth_map.cols; v++)// rowId,  rows: 0 to 640
//            {
//                if ( (v<boundingBoxUpperLeft.val[1] || v>boundingBoxBotRight.val[1]) || (u< boundingBoxUpperLeft.val[0] ||  u> boundingBoxBotRight.val[0])){ continue;}
////                cout<<"show =====================ctrlPointMask point value :"<< ctrlPointMask.at<Vec3f>(u, v);
//
//            }}



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
