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

//		float s = 1.0 - glslmod(1.0 / (2.0 * M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);
        float s =  glslmod(1.0 / (2.0 * M_PI) * atan2(-dir.val[1], dir.val[0]), 1.0);
		//      float s = 1.0 - mod(1.0 / (2.0*M_PI) * atan2(dir.val[1], dir.val[0]), 1.0);
		float t = 1.0 / (M_PI) *acos(-dir.val[2]);
		if (s > 1.0 || t > 1.0) { std::cerr << "UV coordinates overflow!" << std::endl; }

		return Vec2f(s, t);
	}

	Vec3f IBL_Radiance::specularIBL(Vec3f F0, float roughness, Vec3f N, Vec3f V, const Eigen::Matrix3d Camera1_c2w,
                                    Sophus::SO3f enterEnv_Rotatio_inv
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

//		cout<<"R_w:"<<R_w<<endl;
//		cout<<"Camera1_c2w:"<<Camera1_c2w.matrix()<<endl;
//		cout<<"enterEnv_Rotatio_inv:"<<enterEnv_Rotatio_inv.matrix()<<endl;
//		cout<<"R_c_:"<<R_c_<<endl;



		Vec2f uv = directionToSphericalEnvmap(Vec3f(R_w.x(), R_w.y(), R_w.z()));
		if (uv.val[0] > 1.0 || uv.val[1] > 1.0) {
			std::cerr << "\n===specularIBL=======Show UV=================:" << uv << std::endl;
		}


//		std::cout<<"uv:"<<uv<<"show roughness*float(mipCount)"<<roughness*float(mipCount)<<std::endl;
		Vec3f prefiltered_Color = prefilteredColor(uv.val[0], uv.val[1], roughness * float(mipCount)
//                                                   ,pointEnvlight_cur
        );

		//    uv:[0.179422, 0.400616]show roughness*float(mipCount)1.5
		//          prefiltered_Color:[0.112979, 0.0884759, 0.0929931]

		//      show image_ref_path_PFM  of GSN(293,476)And values: [5.17454, 4.71557, 0.0619548]

//		cout<<"show prefiltered_Color!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1"<<prefiltered_Color<<endl;


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
                                         Sophus::SO3f enterEnv_Rotation_inv) {

		// !!!!!!!!  vec3 baseCol = pow(texture(baseColorTexture, texScale*tc).rgb, vec3(2.2)); // this is gamma correction!

		Vec3f One = Vec3f(1.0f, 1.0f, 1.0f);
		Vec3f f0 = 0.16 * (reflectance * reflectance) * One;
		f0 = mix(f0, baseColorValue, metallicValue);
		Vec3f F = fresnelSchlick(_dot(normal, viewDir), f0);
		Vec3f kS = F;
		Vec3f kD = One - kS;
		kD = kD.mul(One - metallicValue * One);
//		Vec3f specular = specularIBL(f0, roughnessValue, normal, viewDir,Camera1_c2w,enterEnv_Rotation_inv);
        Vec3f specular = specularIBL(F, roughnessValue, normal, viewDir,Camera1_c2w,enterEnv_Rotation_inv);// !!!!!
        Specularity = specular;
		//convert from camera to world
		Eigen::Vector3d normal_c(normal.val[0], normal.val[1], normal.val[2]);
		Vec3f normal_w((Camera1_c2w * normal_c).x(), (Camera1_c2w * normal_c).y(), (Camera1_c2w * normal_c).z());
		Vec3f diffuse = diffuseIBL(normal_w);
        diffusity=diffuse;
        // only focus on specular property
        diffuse=Vec3f(0.0,0.0,0.0);
//        cout<<"Checking kD:"<<kD<<endl;
		// shading front-facing
        Vec3f color = pow(kD.mul(baseColorValue.mul(diffuse)) + specular, 1.0 / 2.2 * One);

//        cout<<"\n Checking vals:"<<"kD: "<<kD <<","<<"baseColorValue: "<< baseColorValue<< ","<<"diffuse: "<<diffuse<<","<<"specular: "<<specular<<endl;
//        cout<<"Checking color:"<<color<<endl;

		//      Vec3f color=diffuse;


//		// shading back-facing
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


   float calculateCosDis(Vec3f vec1, Vec3f vec2){
       return vec1.dot(vec2);
    }

    template<typename T>
    bool checkImageBoundaries(const Eigen::Matrix<T, 2, 1> &pixel, int width, int height) {
        return (pixel[0] > 1.1 && pixel[0] < width - 2.1 && pixel[1] > 1.1 && pixel[1] < height - 2.1);
    }


	void updateDelta(
            Sophus::SE3d& Camera1_c2w,
            envLightLookup* EnvLightLookup,
            float *statusMap,
	        Sophus::SO3d& Rotation,
	        Eigen::Matrix<double, 3, 1>& Translation,
	        const Eigen::Matrix3f &K,
	        const Mat depth_map,
	        const Mat &image_roughnes_,
	        Mat &deltaMap,
	        Mat &newNormalMap,
            Mat pointOfInterest,
            string renderedEnvMapPath,
            Mat envMapWorkMask,
            Mat& specularityMap_1,
            Mat& specularityMap_2
    ) {

		// ===================================RENDERING PARAMETERS:====================================
		float fx = K(0, 0), cx = K(0, 2), fy = K(1, 1), cy = K(1, 2);
        float reflectance = 1.0f;
        std::unordered_map<cv::Point2i, float, hash2d<cv::Point2i>, equalTo2D<cv::Point2i>> pixelDepthMap;

        // vec3 normal = normalize(wfn);
		// vec3 viewDir = normalize(cameraPos - vertPos);
		std::unordered_map<int, int> inliers_filter, inliers_filter_i;
        // 446,356 floor point
        // 360,435  // 446,356
        inliers_filter.emplace(411, 439);


        Mat ctrlPointMask(deltaMap.rows, deltaMap.cols, CV_8UC3, Scalar(0,0,0));
		Mat radianceMap_left(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));


        Mat radianceMap_right(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));

        Mat radianceMap_leftSave(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));
        Mat radianceMap_rightSave(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));

        Mat specularityMap(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));
        Mat specularityMap_right(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));
        Mat DiffuseMap(deltaMap.rows, deltaMap.cols, CV_32FC3, Scalar(0));

        // K nearest neighbor search
//        int num_K = 1;
        int num_K = 6;
        float lift=0.002f; // 2mm

        // maintain envMaps of current frame here
        std::unordered_map<int, pointEnvlight> envLightMap_cur;
		for (int u = 0; u < depth_map.rows; u++)// colId, cols: 0 to 480
		{
            for (int v = 0; v < depth_map.cols; v++)// rowId,  rows: 0 to 640
			{

                //=====================================Inliers Filter=====================================
//                				 if(inliers_filter.count(u)==0){continue;}
//                				 if(inliers_filter[u]!=v ){continue;}

                //=====================================Area of interest Filter=====================================
//                                if ( (v<boundingBoxUpperLeft_AoI.val[1] || v>boundingBoxBotRight_AoI.val[1]) || (u< boundingBoxUpperLeft_AoI.val[0] ||  u> boundingBoxBotRight_AoI.val[0])){ continue;}

                // ====================================use DSO pixel selector================================================
                //                if (statusMap!=NULL && statusMap[u*depth_map.cols+v]==0 ){ continue;}

                // =====================================use non lambertian point selector================================
//                                if (statusMap!=NULL && static_cast<int>(statusMap[u * depth_map.cols + v])!= 255){ continue;}

//                cout<<"show current index:"<< u<<","<<v<<"pointOfInterest.at<uchar>(u,v)"<<(int)pointOfInterest.at<uchar>(u,v)<<endl;
                if ( ((int)pointOfInterest.at<uchar>(u,v))!=255){ continue;}

//				if (!(u==414 && v==67)){ continue ;}

                // ===================================RENDERING PARAMETERS:====================================
                float image_roughnes= image_roughnes_.at<float>(u,v);
//                cout<<"image_roughnes:"<<image_roughnes<<endl;
                float image_metallic=1e-3;
                // float image_roughnes= 0.1;
                // float image_metallic= 1.0;

				// ===================================PROJECTION====================================
				Eigen::Vector2f pixelCoord((float) v, (float) u);//  u is the row id , v is col id
				float iDepth = depth_map.at<float>(u, v);
//                cout<<"iDepth:"<<iDepth<<endl;
				Eigen::Vector3f p_3d_no_d((pixelCoord(0) - cx) / fx, (pixelCoord(1) - cy) / fy, (float) 1.0);
				Eigen::Vector3f p_c1;
				p_c1 << p_3d_no_d.x() / iDepth, p_3d_no_d.y() / iDepth, p_3d_no_d.z() / iDepth;



                Eigen::Matrix<float, 3, 1> p1 = Rotation.cast<float>() * p_c1 + Translation.cast<float>();
                Eigen::Matrix<float, 3, 1>  point_K;
                point_K = K * p1;
                int pixel_col_right= std::round(point_K.x() / point_K.z());
                int pixel_row_right =std::round(point_K.y() / point_K.z());

//
//				cout<<"\n show current index:"<< u<<","<<v<<endl;
//				cout<<"pixel_col_right:"<<pixel_col_right<<endl;
//				cout<<"pixel_row_right:"<<pixel_row_right<<endl;

                Eigen::Matrix<int, 2, 1> pt2d(pixel_col_right,pixel_row_right );
                cv::Point2i pixel_coor (pixel_row_right, pixel_col_right);
                // for boundary problem
                if (!checkImageBoundaries(pt2d, depth_map.cols, depth_map.rows)){continue;}
                // for occlusion problem
                if (pixelDepthMap.count(pixel_coor)!=0 ){
                    if( point_K.z()<pixelDepthMap[pixel_coor]){
                        pixelDepthMap.insert(make_pair(pixel_coor,point_K.z()));
                    } else{
                        continue;
                    }
                }
                pixelDepthMap.insert(make_pair(pixel_coor,point_K.z()));


				// record point cloud
				// cloud->push_back(pcl::PointXYZ(p_c1.x(), p_c1.y(), p_c1.z()));
				// calculate normal for each point Transformation_wc

				// ===================================NORMAL====================================
				Eigen::Matrix<float, 3, 1> normal;
				normal.x() = newNormalMap.at<Vec3f>(u, v)[0];
				normal.y() = newNormalMap.at<Vec3f>(u, v)[1];
				normal.z() = newNormalMap.at<Vec3f>(u, v)[2];
				// convert normal vector from camera coordinate system to world coordinate system
				normal = normal.normalized();

//				cout<<"normal:"<<normal<<endl;

				// ===================================VIEW-DIRECTION====================================
				Eigen::Matrix<float, 3, 1> beta, beta_prime;
				beta = -p_c1;
				beta = beta.normalized();
				beta_prime = - Rotation.matrix().transpose().cast<float>() * Translation.cast<float>() - p_c1;
				beta_prime = beta_prime.normalized();

//				cout<<"beta:"<<beta<<endl;
//				cout<<"beta_prime:"<<beta_prime<<endl;

//
//			            beta: 0.309155
//				        -0.108533
//				        -0.944798
//				        beta_prime: 0.348181
//				        -0.132241
//				        -0.928053
//				        beta: 0.286594
//				        -0.109337
//				        -0.951793
//				        beta_prime:  0.32627
//				        -0.133434
//				        -0.935811
//
//				        show specularityMap_left:[0.0470487, 0.0411333, 0.0301246]
//
//				                                   show specularityMap_right:[0.0737205, 0.063109, 0.0497172]






				                                                               // envMapPose_world
				// ===================================BASE-COLOR=============================================
//                 Vec3f baseColor(image_baseColor.at<Vec3f>(u, v)[2], image_baseColor.at<Vec3f>(u, v)[1], image_baseColor.at<Vec3f>(u, v)[0]);
//                Vec3f baseColor(std::pow(image_baseColor.at<Vec3f>(u, v)[2], 2.2), std::pow(image_baseColor.at<Vec3f>(u, v)[1], 2.2), std::pow(image_baseColor.at<Vec3f>(u, v)[0], 2.2));

                Vec3f baseColor(0.0, 0.0, 0.0);

                // vec3 baseCol = pow(texture(baseColorTexture, texScale*tc).rgb, vec3(2.2)); //~~~
                //                pow(image_baseColor.at<Vec3f>(u, v)[2], 2.2);
                //                pow(image_baseColor.at<Vec3f>(u, v)[1], 2.2);
                //                pow(image_baseColor.at<Vec3f>(u, v)[0], 2.2);

                Vec3f N_(normal(0), normal(1), normal(2));
				Vec3f View_beta(beta(0), beta(1), beta(2));
				Vec3f View_beta_prime(beta_prime(0), beta_prime(1), beta_prime(2));

				// ===================================search for Env Light from control points===================
                // coordinate system conversion
                Sophus::SE3f Camera1_extrin = Camera1_c2w.cast<float>();
                Eigen::Vector3f p_c1_w=Camera1_extrin* p_c1;
				pcl::PointXYZ searchPoint(p_c1_w.x(), p_c1_w.y(), p_c1_w.z());

//                cout<<"searchPoint:"<<searchPoint.x<<","<<searchPoint.y<<","<<searchPoint.z<<endl;
                std::vector<int> pointIdxKNNSearch(num_K);
                std::vector<float> pointKNNSquaredDistance(num_K);
				Vec3f key4Search;
//                cout<<"show  EnvLightLookup->kdtree size"<<EnvLightLookup->envLightIdxMap.size()<<endl;

				if ( EnvLightLookup->kdtree.nearestKSearch(searchPoint, num_K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0) {
                    // std::unordered_map<int, float> consine_Dis;
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
                        Eigen::Vector3f envMap_point_c1 = Camera1_extrin.inverse().cast<float>() * Eigen::Vector3f(envMap_point.val[0], envMap_point.val[1], envMap_point.val[2]);
                        // project envMap_point to image plane
                        float pixel_x = (fx * envMap_point_c1.x()) / envMap_point_c1.z() + cx;
                        float pixel_y = (fy * envMap_point_c1.y()) / envMap_point_c1.z() + cy;
                        Vec3f ctrlPointNormal = newNormalMap.at<Vec3f>(round(pixel_y), round(pixel_x));
                        ctrlPointNormal=cv::normalize(ctrlPointNormal);

                        // transform normal vector from camera coordinate system to world coordinate system
                        //Eigen::Vector3f ctrlPointNormal_c(ctrlPointNormal.val[0], ctrlPointNormal.val[1], ctrlPointNormal.val[2]);
                        float angle_consine = ctrlPointNormal.dot(N_);
//                        cout << "angle_consine: " << angle_consine << endl;
                        if (angle_consine<0.9962){ continue;} // 0.9848 is the cos(10 degree), 0.9962 is the cos(5 degree)

                        // calculate the euclidean distance between shader point and envMap point
                        //float disPoint2EnvPoint_squa= powf(envMap_point.val[0]-p_c1_w.x(),2)+powf(envMap_point.val[1]-p_c1_w.y(),2)+powf(envMap_point.val[2]-p_c1_w.z(),2);

                        float disPoint2Env =  pointKNNSquaredDistance[i]/(ctrlPointNormal.dot(N_));
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
//                cout<<"\n Show current shader point:\n"<<p_c1_w<<"\n show nearst envMap point coordinate:\n"<<key4Search<<endl;
//                cout<<"\n show count of envLightMap"<<  EnvLightLookup->envLightIdxMap.count(key4Search)<<endl;
//                cout<<"show EnvLight size:"<< EnvLightLookup->envLightIdxMap.size()<<endl;
                // if no envMap point is found, skip this point
                if (key4Search.dot(key4Search)==0){ continue;}

                int ctrlIndex= EnvLightLookup->envLightIdxMap[key4Search];
//                cout<<"\n ========>>>>show ctrlIndex :"<< ctrlIndex<<"and show key4Search:"<<key4Search<<endl;
                if ( EnvLightLookup->envLightIdxMap.size()==0){std::cerr<<"Error in EnvLight->envLightIdxMap! "<<endl;}
                std::string renderedEnvLight_path=renderedEnvMapPath;


                if (envLightMap_cur.count(ctrlIndex)==0){
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

                    envLightMap_cur.insert(make_pair(ctrlIndex, pEnv));
//                    cout<<"show size of envLightMap_cur:"<<envLightMap_cur.size()<<endl;
                }


//				cout<<"show env ctrlIndex :"<<ctrlIndex<<endl;

//                prefilteredEnvmapSampler= & (pEnv.EnvmapSampler[0]);
//                brdfSampler_ = & (EnvLightLookup->brdfSampler[0]);
//				diffuseSampler = & (pEnv.EnvmapSampler[1]);

                prefilteredEnvmapSampler= & ( envLightMap_cur[ctrlIndex].EnvmapSampler[0]);
                brdfSampler_ = & (EnvLightLookup->brdfSampler[0]);
				diffuseSampler = & (envLightMap_cur[ctrlIndex].EnvmapSampler[1]);


//				cout<<"check normal and roughness:\n "<<N_<<", and "<<image_roughnes<<"and depth:"<<1/iDepth<<endl;
//				cout<<"show Camera1_c2w"<<Camera1_c2w.matrix()<<endl;

				// ===================================RADIANCE-COMPUTATION====================================
				IBL_Radiance *ibl_Radiance = new IBL_Radiance;
//				Vec3f radiance_beta = ibl_Radiance->ACESFilm(ibl_Radiance->solveForRadiance(View_beta, N_, image_roughnes, image_metallic, reflectance, baseColor, Camera1_c2w.rotationMatrix()));
//				Vec3f radiance_beta_prime = ibl_Radiance->ACESFilm(ibl_Radiance->solveForRadiance(View_beta_prime, N_, image_roughnes, image_metallic, reflectance, baseColor, Camera1_c2w.rotationMatrix()));


//                Sophus::SO3f enterPanoroma = EnvLight->envLightMap[key4Search].envMapPose_world.rotationMatrix();
                Sophus::SO3f enterPanoroma ;
                Vec3f radiance_beta = ibl_Radiance->solveForRadiance(View_beta, N_, image_roughnes, image_metallic,
                                                                     reflectance, baseColor, Camera1_c2w.rotationMatrix(),
                                                                     enterPanoroma.inverse());
//                cout<<"\n ========>>>>show LEFT data vals :"<< "ibl_Radiance->Specularity\n"
//                    <<ibl_Radiance->Specularity<< "ibl_Radiance->diffusity\n"<<ibl_Radiance->diffusity <<endl;

                specularityMap.at<Vec3f>(u,v)=ibl_Radiance->Specularity;

				Vec3f specularityMap_left=ibl_Radiance->Specularity;
                DiffuseMap.at<Vec3f>(u,v)=ibl_Radiance->diffusity;
                Vec3f radiance_beta_prime = ibl_Radiance->solveForRadiance(View_beta_prime, N_, image_roughnes, image_metallic,
                                                                           reflectance, baseColor, Camera1_c2w.rotationMatrix(),enterPanoroma.inverse());
//                cout<<"\n ========>>>>show RIGHT data vals :"<< "ibl_Radiance->Specularity\n"
//                <<ibl_Radiance->Specularity<< "ibl_Radiance->diffusity\n"<<ibl_Radiance->diffusity <<endl;

                // calculate the correspondent pixel coordinate of right specular image


                specularityMap_right.at<Vec3f>(pixel_row_right,pixel_col_right)=ibl_Radiance->Specularity;
				Vec3f specularityMap_right=ibl_Radiance->Specularity;


//				cout<<"\n show specularityMap_left:"<<specularityMap_left<<endl;
//				cout<<"\n show specularityMap_right:"<<specularityMap_right<<endl;

				// ===================================SAVE-RADIANCE===========================================
				radianceMap_left.at<Vec3f>(u, v) = radiance_beta;
                envMapWorkMask.at<uchar>(u, v) = 255;
                radianceMap_right.at<Vec3f>(u,v)= radiance_beta_prime;
                radianceMap_leftSave.at<Vec3f>(u, v) =ibl_Radiance->ACESFilm( radiance_beta);
                radianceMap_rightSave.at<Vec3f>(u, v) = ibl_Radiance->ACESFilm( radiance_beta_prime);

//                radiance_beta =ibl_Radiance->ACESFilm( radiance_beta);
//                radiance_beta_prime = ibl_Radiance->ACESFilm( radiance_beta_prime);

				//  ===================================TONE MAPPING===========================================
				// remark: Yes, you need to multiply by exposure before the tone mapping and do the gamma correction after.
                // TODO: use a good tone mapping so that we can get closer to GT delta map like clamp in [0,1.2]

				// right intensity / left intensity
//                if(radiance_beta.val[1]==0|| radiance_beta.val[0]==0|| radiance_beta.val[2]==0 ){continue;}
//				float delta_b = radiance_beta_prime.val[0] / radiance_beta.val[0];
//				float delta_g = radiance_beta_prime.val[1] / radiance_beta.val[1];
//				float delta_r = radiance_beta_prime.val[2] / radiance_beta.val[2];

                float delta_b = abs(specularityMap_left.val[0] - specularityMap_right.val[0]);
                float delta_g = abs(specularityMap_left.val[1] - specularityMap_right.val[1]);
                float delta_r = abs(specularityMap_left.val[2] - specularityMap_right.val[2]);

//                float delta_b = (radiance_beta_prime.val[0] - radiance_beta.val[0]);
//                float delta_g = (radiance_beta_prime.val[1] - radiance_beta.val[1]);
//                float delta_r = (radiance_beta_prime.val[2] - radiance_beta.val[2]);



                if (std::isnan(delta_g)){continue;}
                //if (abs(delta_g)>0.025f){continue;}
                //                if(std::abs(delta_g-1.0f)<1e-3){continue;}
                deltaMap.at<Vec3f>(u, v)[0] = delta_b;
                deltaMap.at<Vec3f>(u, v)[1] = delta_g;
                deltaMap.at<Vec3f>(u, v)[2] = delta_r;
//                cout<<"\n Checking radiance vals:"<< "left Coord: u:"<<u<<", v:"<<v<<"left_radiance:\n"<< radiance_beta
//                    << " and right_intensity at pixel_x:\n"<<"pixel_x"<<", pixel_y:"<< "pixel_y"<< "is:"<<  radiance_beta_prime
//                    << " and intensity difference:"<<radiance_beta-radiance_beta_prime<<"  show delta_g: "<<delta_g <<endl;
			}
		}


        // save deltaMap
        imwrite("deltaMap.png",deltaMap);
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



        specularityMap_1= specularityMap.clone();
        specularityMap_2= specularityMap_right.clone();

		double max_n_radiance, min_n_radiance;
		cv::minMaxLoc(radianceMap_left, &min_n_radiance, &max_n_radiance);
		std::cout << "------->show max and min of estimated radianceMap_left<-----------------:" << max_n_radiance << "," << min_n_radiance << std::endl;

//      imshow("newNormalMap", newNormalMap);
		imshow("radianceMap_left", radianceMap_left);
        imshow("specularityMap_left", specularityMap*255.0);
        imshow("specularityMap_right", specularityMap_right*255.0);


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


// note
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