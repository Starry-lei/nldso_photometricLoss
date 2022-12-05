
#include <sophus/se3.hpp>
#include <unordered_map>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "preFilter/preFilter.h"
#include "dataLoader/dataLoader.h"
#include "deltaCompute/deltaCompute.h"



using namespace cv;
using namespace std;
using namespace DSONL;


int main(int argc, char **argv){



//	dataLoader* dataLoader= new DSONL::dataLoader();
//	dataLoader->Init();
//	dataLoader->load_brdfIntegrationMap();





	float u= 0.5f, v= 0.25f;
	diffuseMapMask DiffuseMaskMap;
	DiffuseMaskMap.Init(argc,argv);
	DiffuseMaskMap.getDiffuseMask(u,v);

	



	diffuseMap getDiffuseMap;
	getDiffuseMap.Init(argc,argv);
	getDiffuseMap.getDiffuse(u, v);


	Mat final_diffuseMap;
	makeDiffuseMap(getDiffuseMap.diffuse_Map, DiffuseMaskMap.diffuse_Map_Mask, final_diffuseMap);

	getDiffuseMap.diffuse_Map=final_diffuseMap;
	cout<<"show final search value:"<<endl;
	getDiffuseMap.getDiffuse(u, v);





	// case 1
	// GLI 0.5, 0.5

	//	============SampleBrdf val(RGBA):
	//	1,0.837255,0.393137,1
	// GSN:
//	image_ref_path_PFM blue  green and red channel value: [0.394053, 0.837957, 1.81198]


// case 2

// GSN 0.5, 0.75
//	image_ref_path_PFM blue  green and red channel value:
//	[0.378213, 0.753771, 1.5108]

// GLI : 0.5, 0.25
//	============SampleDiffuse val(RGBA):
//	1,0.760294,0.382353,1






	imshow("show Diffuse Map", getDiffuseMap.diffuse_Map);

//	brdfIntegrationMap *brdfIntegrationMap;
//	brdfIntegrationMap= new DSONL::brdfIntegrationMap;
//	float NoV= 0.5f, roughness= 0.25f;
//	gli::vec4 test_brdf= brdfIntegrationMap->get_brdfIntegrationMap(NoV, roughness);
//
//
//
//
//	EnvMapLookup EnvMapLookup(argc,argv);
//	EnvMapLookup.makeMipMap();
//






//	for (int i = 0; i < 6; ++i) {
//		imshow("image"+ to_string(i), EnvMapLookup.image_pyramid[i]);
//
////		// show the min max val
////		double min_depth_val, max_depth_val;
////		cv::minMaxLoc( EnvMapLookup.image_pyramid[i], &min_depth_val, &max_depth_val);
////		cout<<"\n show  EnvMapLookup.image_pyramid[i] min, max:\n"<<min_depth_val<<","<<max_depth_val<<endl;
//	}
	waitKey(0);











	return 0;
}
