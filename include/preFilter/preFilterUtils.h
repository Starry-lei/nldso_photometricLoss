//
// Created by lei on 14.01.23.
//

#ifndef NLDSO_PHOTOMETRICLOSS_PREFILTERUTILS_H
#define NLDSO_PHOTOMETRICLOSS_PREFILTERUTILS_H


#include <GL/glew.h>
#include <GL/freeglut.h>
#include <memory>
#include <regex>
#define _USE_MATH_DEFINES
#include <Eigen/Core>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <dirent.h>


#include "preFilter/Renderer.h"
#include "preFilter/specular_mask_Renderer.h"

#include "diffuseMap/Renderer_diffuse.h"
#include "diffuseMap/diffuse_mask_Renderer.h"


#include "settings/preComputeSetting.h"

using namespace std;
using namespace gsn;
using namespace cv;

namespace DSONL {

    static Renderer *renderer = NULL;
    static specular_Mask_Renderer *renderer_specular = NULL;
    static Renderer_diffuse *render_diffuse = NULL;
    static diffuse_Mask_Renderer *render_diffuse_mask = NULL;

    static int window(0);
    static int window_specular_mask(0);
    static int window_diffuse(0);
    static int window_diffuse_mask(0);

    // glut static callbacks start
    static void glutResize(int w, int h) {
        if (renderer->roughness == (float) 0.0) {
            glutReshapeWindow(1024, 512);
        } else if (renderer->roughness == (float) 0.2) {
            glutReshapeWindow(512, 256);
        } else if (renderer->roughness == (float) 0.4) {
            glutReshapeWindow(256, 128);
        } else if (renderer->roughness == (float) 0.6) {
            glutReshapeWindow(128, 64);
        } else if (renderer->roughness == (float) 0.8) {
            glutReshapeWindow(64, 32);
        } else if (renderer->roughness == (float) 1.0) {
            glutReshapeWindow(32, 16);
        } else {
            std::cerr << "Wrong roughness!" << endl;
        }
    }

    static void glutResize_specular_mask(int w, int h) {
//        cout<<"in Use now!"<<endl;
        if (renderer_specular->roughness == (float) 0.0) {
            glutReshapeWindow(1024, 512);
        } else if (renderer_specular->roughness == (float) 0.2) {
            glutReshapeWindow(512, 256);
        } else if (renderer_specular->roughness == (float) 0.4) {
            glutReshapeWindow(256, 128);
        } else if (renderer_specular->roughness == (float) 0.6) {
            glutReshapeWindow(128, 64);
        } else if (renderer_specular->roughness == (float) 0.8) {
            glutReshapeWindow(64, 32);
        } else if (renderer_specular->roughness == (float) 1.0) {
            glutReshapeWindow(32, 16);
        } else {
            std::cerr << "Wrong roughness!" << endl;
        }
    }

    // glut static callbacks start
    static void glutResize_diffuse(int w, int h) { render_diffuse->resize(w, h); }

    static void glutResize_diffuse_mask(int w, int h) { render_diffuse_mask->resize(w, h); }

    static void glutDisplay() {
        renderer->display();
        glutSwapBuffers();
        glutReportErrors();
        glutDestroyWindow(window);
    }
    static void glutDisplay_specular_mask() {
        renderer_specular->display();
        glutSwapBuffers();
        glutReportErrors();
        glutDestroyWindow(window_specular_mask);
    }

    static void glutDisplay_diffuse() {
        render_diffuse->display();
        glutSwapBuffers();
        glutReportErrors();
        glutDestroyWindow(window_diffuse);
    }
    static void glutDisplay_diffuse_mask() {
        render_diffuse_mask->display();
        glutSwapBuffers();
        glutReportErrors();
        glutDestroyWindow(window_diffuse_mask);
    }

    //	render_diffuse
    static void glutClose() {
        renderer->dispose();
        delete renderer;
    }
    static void glutClose_specular_mask() {
        renderer_specular->dispose();
        delete renderer_specular;
    }

    static void glutClose_diffuse() {
        render_diffuse->dispose();
        delete render_diffuse;
    }
    static void glutClose_diffuse_mask() {
        render_diffuse_mask->dispose();
        delete render_diffuse_mask;
    }

    static void timer(int v) {
        float offset = 1.0f;
        renderer->t += offset;
        glutDisplay();
        glutTimerFunc(unsigned(20), timer, ++v);
    }

    static void timer_specular(int v) {
        float offset = 1.0f;
        renderer_specular->t += offset;
        glutDisplay_specular_mask();
        glutTimerFunc(unsigned(20), timer_specular, ++v);
    }

    static void timer_diffuse(int v) {
        float offset = 1.0f;
        render_diffuse->t += offset;
        glutDisplay_diffuse();
        glutTimerFunc(unsigned(20), timer_diffuse, ++v);
    }
    static void timer_diffuse_mask(int v) {
        float offset = 1.0f;
        render_diffuse_mask->t += offset;
        glutDisplay_diffuse_mask();
        glutTimerFunc(unsigned(20), timer_diffuse_mask, ++v);
    }

    static void glutKeyboard(unsigned char key, int x, int y) {
        bool redraw = false;
        std::string modeStr;
        std::stringstream ss;
        if (key >= '1' && key <= '9') { renderer->selectedOutput = int(key) - int('1'); }
    }
    static void glutKeyboard_specular(unsigned char key, int x, int y) {
        bool redraw = false;
        std::string modeStr;
        std::stringstream ss;
        if (key >= '1' && key <= '9') { renderer_specular->selectedOutput = int(key) - int('1'); }
    }

    static void glutKeyboard_diffuse(unsigned char key, int x, int y) {
        bool redraw = false;
        std::string modeStr;
        std::stringstream ss;
        if (key >= '1' && key <= '9') { render_diffuse->selectedOutput = int(key) - int('1'); }
    }

    static void glutKeyboard_diffuse_mask(unsigned char key, int x, int y) {
        bool redraw = false;
        std::string modeStr;
        std::stringstream ss;
        if (key >= '1' && key <= '9') { render_diffuse_mask->selectedOutput = int(key) - int('1'); }
    }




}



#endif //NLDSO_PHOTOMETRICLOSS_PREFILTERUTILS_H
