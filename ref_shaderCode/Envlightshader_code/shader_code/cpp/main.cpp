// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <GL/glew.h>
#include <GL/freeglut.h> // we use glut here as window manager
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "Renderer.h"
using namespace std;
using namespace gsn;

//this is a static pointer to a Renderer used in the glut callback functions
static Renderer* renderer;

//glut static callbacks start
static void glutResize(int w, int h)
{
  renderer->resize(w, h);
}

static void glutDisplay()
{
  renderer->display();
  glutSwapBuffers();
  glutReportErrors();
}

static void glutClose()
{
  renderer->dispose();
  delete renderer;
}

static void timer(int v)
{
  float offset = 1.0f;
  renderer->t += offset;
  glutDisplay();
  glutTimerFunc(unsigned(20), timer, ++v);
}

static void glutKeyboard(unsigned char key, int x, int y) {
  bool redraw = false;
  std::string modeStr;
  std::stringstream ss;
  if (key >= '1' && key <= '9') {
    renderer->selectedOutput = int(key) - int('1');
  }
}

int main(int argc, char** argv)
{
  renderer = new Renderer;

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(640, 640);

  glutCreateWindow("GSN Composer Shader Export");
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    fprintf(stderr, "Glew error: %s\n", glewGetErrorString(err));
  }
  glutDisplayFunc(glutDisplay);
  //glutIdleFunc(glutDisplay);
  glutReshapeFunc(glutResize);
  glutCloseFunc(glutClose);
  glutKeyboardFunc(glutKeyboard);

  renderer->init();
  glutTimerFunc(unsigned(20), timer, 0);

  glutMainLoop();
}