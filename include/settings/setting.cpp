//
// Created by cheng on 31.10.22.
//
#include "setting.h"




namespace DSONL{


	int pyrLevelsUsed = PYR_LEVELS;
	bool disableAllDisplay = false;
	int setting_minFrames = 5; // min frames in window.
	int setting_maxFrames = 7; // max frames in window.


	// parameters controlling pixel selection
	float setting_minGradHistCut = 0.5;
	float setting_minGradHistAdd = 7;
	float setting_gradDownweightPerLevel = 0.75;
	bool  setting_selectDirectionDistribution = true;
	int sparsityFactor = 5;	// not actually a setting, only some legacy stuff for coarse initializer.


	int wG[PYR_LEVELS], hG[PYR_LEVELS];
	float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
			cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
			cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];


	float wM3G;
	float hM3G;

	void setGlobalCalib(int w, int h,const Eigen::Matrix3f &K)
	{
		int wlvl=w;
		int hlvl=h;
		pyrLevelsUsed=1;
		while(wlvl%2==0 && hlvl%2==0 && wlvl*hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
		{
			wlvl /=2;
			hlvl /=2;
			pyrLevelsUsed++;
		}
		printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
		       pyrLevelsUsed-1, wlvl, hlvl);
		if(wlvl>100 && hlvl > 100)
		{
			printf("\n\n===============WARNING!===================\n "
			       "using not enough pyramid levels.\n"
			       "Consider scaling to a resolution that is a multiple of a power of 2.\n");
		}
		if(pyrLevelsUsed < 3)
		{
			printf("\n\n===============WARNING!===================\n "
			       "I need higher resolution.\n"
			       "I will probably segfault.\n");
		}

		wM3G = w-3;
		hM3G = h-3;

		wG[0] = w;
		hG[0] = h;
		KG[0] = K;
		fxG[0] = K(0,0);
		fyG[0] = K(1,1);
		cxG[0] = K(0,2);
		cyG[0] = K(1,2);
		KiG[0] = KG[0].inverse();
		fxiG[0] = KiG[0](0,0);
		fyiG[0] = KiG[0](1,1);
		cxiG[0] = KiG[0](0,2);
		cyiG[0] = KiG[0](1,2);

		for (int level = 1; level < pyrLevelsUsed; ++ level)
		{
			wG[level] = w >> level;
			hG[level] = h >> level;

			fxG[level] = fxG[level-1] * 0.5;
			fyG[level] = fyG[level-1] * 0.5;
			cxG[level] = (cxG[0] + 0.5) / ((int)1<<level) - 0.5;
			cyG[level] = (cyG[0] + 0.5) / ((int)1<<level) - 0.5;

			KG[level]  << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0;	// synthetic
			KiG[level] = KG[level].inverse();

			fxiG[level] = KiG[level](0,0);
			fyiG[level] = KiG[level](1,1);
			cxiG[level] = KiG[level](0,2);
			cyiG[level] = KiG[level](1,2);
		}
	}

//	void makeK( Eigen::Matrix<float, 3,3>& K)
//	{
//		w[0] = wG[0];
//		h[0] = hG[0];
//
//		fx[0] = HCalib->fxl();
//		fy[0] = HCalib->fyl();
//		cx[0] = HCalib->cxl();
//		cy[0] = HCalib->cyl();
//
//		for (int level = 1; level < pyrLevelsUsed; ++ level)
//		{
//			w[level] = w[0] >> level;
//			h[level] = h[0] >> level;
//			fx[level] = fx[level-1] * 0.5;
//			fy[level] = fy[level-1] * 0.5;
//			cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
//			cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
//		}
//
//		for (int level = 0; level < pyrLevelsUsed; ++ level)
//		{
//			K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
//			Ki[level] = K[level].inverse();
//			fxi[level] = Ki[level](0,0);
//			fyi[level] = Ki[level](1,1);
//			cxi[level] = Ki[level](0,2);
//			cyi[level] = Ki[level](1,2);
//		}
//	}


}