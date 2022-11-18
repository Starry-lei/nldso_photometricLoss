//
// Created by cheng on 30.10.22.
//

#include "numType.h"
#include <opencv2/core/mat.hpp>
#include "minimalImage.h"
#include <unordered_set>
#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "setting.h"
#pragma once
namespace DSONL{

	const float minUseGrad_pixsel = 10;
	enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};

	// Image  constant info & pre-calculated values
	struct FrameHessian{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		inline FrameHessian(){

		}


		//* 图像导数[0]:辐照度  [1]:x方向导数  [2]:y方向导数, （指针表示图像）
		Eigen::Vector3f* dI;				//!< 图像导数  // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
		Eigen::Vector3f* dIp[PYR_LEVELS];	//!< 各金字塔层的图像导数  // coarse tracking / coarse initializer. NAN in [0] only.
		float* absSquaredGrad[PYR_LEVELS];  //!< x,y 方向梯度的平方和 // only used for pixel select (histograms etc.). no NAN.
		float* img_pyr[PYR_LEVELS];  // !!!!! not needed???????????????????????????



		inline ~FrameHessian()
		{
			for(int i=0;i<pyrLevelsUsed;i++)
			{
				delete[] dIp[i];
				delete[]  absSquaredGrad[i];
				delete[] img_pyr[i];
			}
//			    delete[] selectionMap;
		};


		void makeImages(float* color) {

			// 每一层创建图像值, 和图像梯度的存储空间
			for(int i=0;i<pyrLevelsUsed;i++)
			{
				dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
				absSquaredGrad[i] = new float[wG[i]*hG[i]];
				img_pyr[i] = new float[wG[i]*hG[i]];
			}
			dI = dIp[0]; // 原来他们指向同一个地方


			// make d0
			int w=wG[0]; // 零层weight
			int h=hG[0]; // 零层height
			for(int i=0;i<w*h;i++){
				dI[i][0] = color[i];
				img_pyr[0][i] = color[i];
			}


			for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
			{
				int wl = wG[lvl], hl = hG[lvl]; // 该层图像大小

				Eigen::Vector3f* dI_l = dIp[lvl];
				float* dabs_l = absSquaredGrad[lvl];

				if(lvl>0)
				{
					int lvlm1 = lvl-1;
					int wlm1 = wG[lvlm1]; // 列数
					Eigen::Vector3f* dI_lm = dIp[lvlm1];


					// 像素4合1, 生成金字塔
					for(int y=0;y<hl;y++)
						for(int x=0;x<wl;x++)
						{
							dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
							                             dI_lm[2*x+1 + 2*y*wlm1][0] +
							                             dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
							                             dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
							img_pyr[lvl][x + y*wl] = dI_l[x + y*wl][0];
						}
				}

				for(int idx=wl;idx < wl*(hl-1);idx++) // 第二行开始
				{
					float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
					float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);


					if(!std::isfinite(dx)) dx=0;
					if(!std::isfinite(dy)) dy=0;

					dI_l[idx][1] = dx; // 梯度
					dI_l[idx][2] = dy;


					dabs_l[idx] = dx*dx+dy*dy; // 梯度平方

//					if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
//					{
//						//! 乘上响应函数, 变换回正常的颜色, 因为光度矫正时 I = G^-1(I) / V(x)
//						float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
//						dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
//					}
				}
			}




		}

	};

	void plot_img_pyramid( FrameHessian* fh, float * map_out , int idx);

	void plotImPyr(FrameHessian* fh,int i, std::string ImgName );

	class PixelSelector
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		int makeMaps(
				const FrameHessian* const fh,
				float* map_out, float density, int recursionsLeft=1, bool plot=false, float thFactor=1);

		PixelSelector(int w, int h);
		~PixelSelector();
		int currentPotential; 		//!< 当前选择像素点的潜力, 就是网格大小, 越大选点越少


		bool allowFast;
		void makeHists(const FrameHessian* const fh);
	private:

		Eigen::Vector3i select(const FrameHessian* const fh,
		                       float* map_out, int pot, float thFactor=1);


		unsigned char* randomPattern;


		int* gradHist;  			//!< 根号梯度平方和分布直方图, 0是所有像素个数
		float* ths;					//!< 平滑之前的阈值
		float* thsSmoothed;			//!< 平滑后的阈值
		int thsStep;
		const FrameHessian* gradHistFrame;
	};



	template<int pot>
	inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, float THFac)
	{

		memset(map_out, 0, sizeof(bool)*w*h);

		int numGood = 0;
		for(int y=1;y<h-pot;y+=pot)
		{
			for(int x=1;x<w-pot;x+=pot)
			{
				int bestXXID = -1;
				int bestYYID = -1;
				int bestXYID = -1;
				int bestYXID = -1;

				float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

				Eigen::Vector3f* grads0 = grads+x+y*w;
				for(int dx=0;dx<pot;dx++)
					for(int dy=0;dy<pot;dy++)
					{
						int idx = dx+dy*w;
						Eigen::Vector3f g=grads0[idx];
						float sqgd = g.tail<2>().squaredNorm();
						float TH = THFac*minUseGrad_pixsel * (0.75f);

						if(sqgd > TH*TH)
						{
							float agx = fabs((float)g[1]);
							if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

							float agy = fabs((float)g[2]);
							if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

							float gxpy = fabs((float)(g[1]-g[2]));
							if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

							float gxmy = fabs((float)(g[1]+g[2]));
							if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
						}
					}

				bool* map0 = map_out+x+y*w;

				if(bestXXID>=0)
				{
					if(!map0[bestXXID])
						numGood++;
					map0[bestXXID] = true;

				}
				if(bestYYID>=0)
				{
					if(!map0[bestYYID])
						numGood++;
					map0[bestYYID] = true;

				}
				if(bestXYID>=0)
				{
					if(!map0[bestXYID])
						numGood++;
					map0[bestXYID] = true;

				}
				if(bestYXID>=0)
				{
					if(!map0[bestYXID])
						numGood++;
					map0[bestYXID] = true;

				}
			}
		}

		return numGood;
	}

	inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, int pot, float THFac)
	{

		memset(map_out, 0, sizeof(bool)*w*h);

		int numGood = 0;
		for(int y=1;y<h-pot;y+=pot)
		{
			for(int x=1;x<w-pot;x+=pot)
			{
				int bestXXID = -1;
				int bestYYID = -1;
				int bestXYID = -1;
				int bestYXID = -1;

				float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

				Eigen::Vector3f* grads0 = grads+x+y*w;
				for(int dx=0;dx<pot;dx++)
					for(int dy=0;dy<pot;dy++)
					{
						int idx = dx+dy*w;
						Eigen::Vector3f g=grads0[idx];
						float sqgd = g.tail<2>().squaredNorm();
						float TH = THFac*minUseGrad_pixsel * (0.75f);

						if(sqgd > TH*TH)
						{
							float agx = fabs((float)g[1]);
							if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

							float agy = fabs((float)g[2]);
							if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

							float gxpy = fabs((float)(g[1]-g[2]));
							if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

							float gxmy = fabs((float)(g[1]+g[2]));
							if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
						}
					}

				bool* map0 = map_out+x+y*w;

				if(bestXXID>=0)
				{
					if(!map0[bestXXID])
						numGood++;
					map0[bestXXID] = true;

				}
				if(bestYYID>=0)
				{
					if(!map0[bestYYID])
						numGood++;
					map0[bestYYID] = true;

				}
				if(bestXYID>=0)
				{
					if(!map0[bestXYID])
						numGood++;
					map0[bestXYID] = true;

				}
				if(bestYXID>=0)
				{
					if(!map0[bestYXID])
						numGood++;
					map0[bestYXID] = true;

				}
			}
		}

		return numGood;
	}

	inline int makePixelStatus(Eigen::Vector3f* grads, bool* map, int w, int h, float desiredDensity, int recsLeft=5, float THFac = 1)
	{
		if(sparsityFactor < 1) sparsityFactor = 1;

		int numGoodPoints;


		if(sparsityFactor==1) numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
		else if(sparsityFactor==2) numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
		else if(sparsityFactor==3) numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
		else if(sparsityFactor==4) numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
		else if(sparsityFactor==5) numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
		else if(sparsityFactor==6) numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
		else if(sparsityFactor==7) numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
		else if(sparsityFactor==8) numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
		else if(sparsityFactor==9) numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
		else if(sparsityFactor==10) numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
		else if(sparsityFactor==11) numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
		else numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);


		/*
		 * #points is approximately proportional to sparsityFactor^2.
		 */

		float quotia = numGoodPoints / (float)(desiredDensity);

		int newSparsity = (sparsityFactor * sqrtf(quotia))+0.7f;


		if(newSparsity < 1) newSparsity=1;


		float oldTHFac = THFac;
		if(newSparsity==1 && sparsityFactor==1) THFac = 0.5;


		if((abs(newSparsity-sparsityFactor) < 1 && THFac==oldTHFac) ||
		   ( quotia > 0.8 &&  1.0f / quotia > 0.8) ||
		   recsLeft == 0)
		{

//		printf(" \n");
			//all good
			sparsityFactor = newSparsity;
			return numGoodPoints;
		}
		else
		{
//		printf(" -> re-evaluate! \n");
			// re-evaluate.
			sparsityFactor = newSparsity;
			return makePixelStatus(grads, map, w,h, desiredDensity, recsLeft-1, THFac);
		}
	}




// ==============================================Display====================================================

	namespace IOWrap
	{

		void displayImage(const char* windowName, const MinimalImageB* img, bool autoSize = false);
		void displayImage(const char* windowName, const MinimalImageB3* img, bool autoSize = false);
		void displayImage(const char* windowName, const MinimalImageF* img, bool autoSize = false);
		void displayImage(const char* windowName, const MinimalImageF3* img, bool autoSize = false);
		void displayImage(const char* windowName, const MinimalImageB16* img, bool autoSize = false);


		void displayImageStitch(const char* windowName, const std::vector<MinimalImageB*> images, int cc=0, int rc=0);
		void displayImageStitch(const char* windowName, const std::vector<MinimalImageB3*> images, int cc=0, int rc=0);
		void displayImageStitch(const char* windowName, const std::vector<MinimalImageF*> images, int cc=0, int rc=0);
		void displayImageStitch(const char* windowName, const std::vector<MinimalImageF3*> images, int cc=0, int rc=0);

		int waitKey(int milliseconds);
		void closeAllWindows();

	}

















}


