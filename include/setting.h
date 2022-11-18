//
// Created by cheng on 31.10.22.
//

#include "numType.h"
#pragma once

namespace DSONL{

	#define PYR_LEVELS 5
	#define patternPadding 2
	extern int pyrLevelsUsed;
	extern bool disableAllDisplay;
	extern int   setting_minFrames ; // min frames in window.
	extern int   setting_maxFrames ; // max frames in window.
	extern int wG[PYR_LEVELS], hG[PYR_LEVELS];
	extern float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
			cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	extern float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
			cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	extern Eigen::Matrix3f KG[PYR_LEVELS],KiG[PYR_LEVELS];

	extern float wM3G;
	extern float hM3G;


	extern Eigen::Matrix<float,3,3> K[PYR_LEVELS];			//!< camera参数
	extern Eigen::Matrix<float,3,3> Ki[PYR_LEVELS];
	extern double fx[PYR_LEVELS];
	extern double fy[PYR_LEVELS];
	extern double fxi[PYR_LEVELS];
	extern double fyi[PYR_LEVELS];
	extern double cx[PYR_LEVELS];
	extern double cy[PYR_LEVELS];
	extern double cxi[PYR_LEVELS];
	extern double cyi[PYR_LEVELS];






	void setGlobalCalib(int w, int h, const Eigen::Matrix3f &K );
//	void makeK( Eigen::Matrix<float, 3,3>& K);


	// parameters controlling pixel selection
	extern float setting_minGradHistCut ;
	extern float setting_minGradHistAdd ;
	extern float setting_gradDownweightPerLevel;
	extern bool  setting_selectDirectionDistribution;
	extern int sparsityFactor;



}