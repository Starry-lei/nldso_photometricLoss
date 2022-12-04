// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <string>
#include <vector>
#include <map>

#include "Mesh.h"

#define USE_OPENGL_BUFFERS
#ifdef USE_OPENGL_BUFFERS
#include <GL/glew.h>
#endif

using namespace std;
using namespace gsn;

void Mesh::clearAll()
{

#ifdef USE_OPENGL_BUFFERS
  if (true) {
    for (int i = 0; i < int(arrayBuffers.size()); i++) {
      unsigned buf = arrayBuffers[i].arrayBufferOpenGL;
      if (buf != 0) {
        glDeleteBuffers(1, &buf);
        arrayBuffers[i].arrayBufferOpenGL = 0;
      }
    }

    for (int i = 0; i < int(elementArrayBuffers.size()); i++) {
      unsigned buf = elementArrayBuffers[i].elementArrayBufferOpenGL;
      if (buf != 0) {
        glDeleteBuffers(1, &buf);
        elementArrayBuffers[i].elementArrayBufferOpenGL = 0;
      }
    }
  }
#endif

  // clear everything
  arrayBuffers.clear();
  elementArrayBuffers.clear();
  vertexBuffers.clear();
  triangleGroups.clear();
  quadGroups.clear();
}

void Mesh::updateTrianglesFromQuads() 
{
  for (int i = 0; i < int(quadGroups.size()); i++) {
    QuadGroup& qg = quadGroups[i];
    int triangleGroupID = -1;
    if (qg.triangleGroupID >= 0) {
      triangleGroupID = qg.triangleGroupID;
    }
    else {
      triangleGroupID = int(triangleGroups.size());
      triangleGroups.push_back(TriangleGroup());
      qg.triangleGroupID = triangleGroupID;
    }
    TriangleGroup& tg = triangleGroups[triangleGroupID];
    tg.groupName = qg.groupName + "_AutoCreate";
    tg.noOfTriangles = qg.noOfQuads * 2;
    int elementArrayBufferID = -1;
    if (tg.elementArrayBufferID >= 0) {
      elementArrayBufferID = tg.elementArrayBufferID;
    }
    else {
      elementArrayBufferID = int(elementArrayBuffers.size());
      elementArrayBuffers.push_back(ElementArrayBuffer());
      tg.elementArrayBufferID = elementArrayBufferID;
    }
    ElementArrayBuffer& eb = elementArrayBuffers[elementArrayBufferID];

    eb.elementArrayBuffer.resize(tg.noOfTriangles * 3);

    bool ebqDefined = false;
    if (qg.elementArrayBufferID >= 0) {
      ebqDefined = true;
    }

    int q0, q1, q2, q3;
    int k = 0;
    for (int q = 0; q < qg.noOfQuads; q++) {
      int qq = q + qg.offset;
      if (ebqDefined) {
        std::vector<unsigned int>& ebq = elementArrayBuffers[qg.elementArrayBufferID].elementArrayBuffer;
        q0 = ebq[qq * 4 + 0];
        q1 = ebq[qq * 4 + 1];
        q2 = ebq[qq * 4 + 2];
        q3 = ebq[qq * 4 + 3];
      }
      else {
        q0 = qq * 4 + 0;
        q1 = qq * 4 + 1;
        q2 = qq * 4 + 2;
        q3 = qq * 4 + 3;
      }
      eb.elementArrayBuffer[k++] = q0;
      eb.elementArrayBuffer[k++] = q1;
      eb.elementArrayBuffer[k++] = q2;
      eb.elementArrayBuffer[k++] = q0;
      eb.elementArrayBuffer[k++] = q2;
      eb.elementArrayBuffer[k++] = q3;
    }
  }
};

void Mesh::updateWireframeBuffers()
{
  map<int, bool> underQuadControl;

  for (int i = 0; i < int(quadGroups.size()); i++) {
    QuadGroup& qg = quadGroups[i];
    TriangleGroup& tg = triangleGroups[qg.triangleGroupID];
    underQuadControl[qg.triangleGroupID] = true;

    int ebIndex = -1;
    if (tg.wireframeElementArrayBufferID >= 0) {
      ebIndex = tg.wireframeElementArrayBufferID;
    }
    else {
      ebIndex = int(elementArrayBuffers.size());
      tg.wireframeElementArrayBufferID = ebIndex;
      elementArrayBuffers.push_back(ElementArrayBuffer());
    }
    ElementArrayBuffer& eb = elementArrayBuffers[ebIndex];
    eb.elementArrayBuffer.resize(qg.noOfQuads * 4 * 2);

    bool ebqDefined = false;
    if (qg.elementArrayBufferID >= 0) {
      ebqDefined = true;
    }

    int q0, q1, q2, q3;
    int k = 0;
    for (int q = 0; q < qg.noOfQuads; q++) {
      int qq = q + qg.offset;
      if (ebqDefined) {
        std::vector<unsigned int>& ebq = elementArrayBuffers[qg.elementArrayBufferID].elementArrayBuffer;
        q0 = ebq[qq * 4 + 0];
        q1 = ebq[qq * 4 + 1];
        q2 = ebq[qq * 4 + 2];
        q3 = ebq[qq * 4 + 3];
      }
      else {
        q0 = qq * 4 + 0;
        q1 = qq * 4 + 1;
        q2 = qq * 4 + 2;
        q3 = qq * 4 + 3;
      }
      eb.elementArrayBuffer[k++] = q0;
      eb.elementArrayBuffer[k++] = q1;
      eb.elementArrayBuffer[k++] = q1;
      eb.elementArrayBuffer[k++] = q2;
      eb.elementArrayBuffer[k++] = q2;
      eb.elementArrayBuffer[k++] = q3;
      eb.elementArrayBuffer[k++] = q3;
      eb.elementArrayBuffer[k++] = q0;
    }
  }

  for (int i = 0; i < int(triangleGroups.size()); i++) {
    if (underQuadControl.count(i) == 0) {
      TriangleGroup& tg = triangleGroups[i];
      int ebIndex = -1;
      if (tg.wireframeElementArrayBufferID >= 0) {
        ebIndex = tg.wireframeElementArrayBufferID;
      }
      else {
        ebIndex = int(elementArrayBuffers.size());
        tg.wireframeElementArrayBufferID = ebIndex;
        elementArrayBuffers.push_back(ElementArrayBuffer());
      }
      ElementArrayBuffer& eb = elementArrayBuffers[ebIndex];
      eb.elementArrayBuffer.resize(tg.noOfTriangles * 3 * 2);

      bool ebtDefined = false;
      if (tg.elementArrayBufferID >= 0) {
        ebtDefined = true;
      }

      int t0, t1, t2;
      int k = 0;
      for (int t = 0; t < tg.noOfTriangles; t++) {
        int tt = t + tg.offset;
        if (ebtDefined) {
          std::vector<unsigned int>& ebt = elementArrayBuffers[tg.elementArrayBufferID].elementArrayBuffer;
          t0 = ebt[tt * 3 + 0];
          t1 = ebt[tt * 3 + 1];
          t2 = ebt[tt * 3 + 2];
        }
        else {
          t0 = tt * 3 + 0;
          t1 = tt * 3 + 1;
          t2 = tt * 3 + 2;
        }
        eb.elementArrayBuffer[k++] = t0;
        eb.elementArrayBuffer[k++] = t1;
        eb.elementArrayBuffer[k++] = t1;
        eb.elementArrayBuffer[k++] = t2;
        eb.elementArrayBuffer[k++] = t2;
        eb.elementArrayBuffer[k++] = t0;
      }
    }
  }
}

void Mesh::updateOpenGLBuffers()
{
#ifdef USE_OPENGL_BUFFERS
  if (true) {
    for (int i = 0; i < int(arrayBuffers.size()); i++) {
      unsigned buf = arrayBuffers[i].arrayBufferOpenGL;
      if (buf != 0) {
        glDeleteBuffers(1, &buf);
        arrayBuffers[i].arrayBufferOpenGL = 0;
      }
      glGenBuffers(1, &buf);
      arrayBuffers[i].arrayBufferOpenGL = buf;
      glBindBuffer(GL_ARRAY_BUFFER, buf);
      glBufferData(GL_ARRAY_BUFFER, arrayBuffers[i].arrayBuffer.size() * sizeof(float),
        &arrayBuffers[i].arrayBuffer[0], GL_STATIC_DRAW);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    for (int i = 0; i < int(elementArrayBuffers.size()); i++) {
      unsigned buf = elementArrayBuffers[i].elementArrayBufferOpenGL;
      if (buf != 0) {
        glDeleteBuffers(1, &buf);
        elementArrayBuffers[i].elementArrayBufferOpenGL = 0;
      }
      glGenBuffers(1, &buf);
      elementArrayBuffers[i].elementArrayBufferOpenGL = buf;
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementArrayBuffers[i].elementArrayBuffer.size() * sizeof(int),
        &elementArrayBuffers[i].elementArrayBuffer[0], GL_STATIC_DRAW);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
  }
#endif
}


void Mesh::updateAllInternalData() {
  updateTrianglesFromQuads();
  updateWireframeBuffers();
  updateOpenGLBuffers();
}

void Mesh::createFromArrays(std::string groupName, int numberOfPrimitives, const std::vector<float>& positions,
  const std::vector<float>& normals, const std::vector<float>& texCoords, const std::vector<unsigned int>& indices, bool triGroupFlag)
{
  clearAll();

  // positions
  Mesh::ArrayBuffer abv;
  abv.arrayBuffer = positions;
  arrayBuffers.push_back(abv);

  Mesh::VertexBuffer vbv;
  vbv.semantic = "position";
  vbv.arrayBufferID = int(arrayBuffers.size()) - 1;
  vbv.stride = 3;
  vbv.offset = 0;
  vbv.elementSize = 3;
  vertexBuffers.push_back(vbv);


  // normals
  Mesh::ArrayBuffer abn;
  abn.arrayBuffer = normals;
  arrayBuffers.push_back(abn);

  Mesh::VertexBuffer vbn;
  vbn.semantic = "normal";
  vbn.arrayBufferID = int(arrayBuffers.size()) - 1;
  vbn.stride = 3;
  vbn.offset = 0;
  vbn.elementSize = 3;
  vertexBuffers.push_back(vbn);

  // texCoords
  Mesh::ArrayBuffer abt;
  abt.arrayBuffer = texCoords;
  arrayBuffers.push_back(abt);

  Mesh::VertexBuffer vbt;
  vbt.semantic = "texcoord";
  vbt.arrayBufferID = int(arrayBuffers.size()) - 1;
  vbt.stride = 2;
  vbt.offset = 0;
  vbt.elementSize = 2;
  vertexBuffers.push_back(vbt);

  // indices
  Mesh::ElementArrayBuffer eb;
  eb.elementArrayBuffer = indices;
  elementArrayBuffers.push_back(eb);

  if (!triGroupFlag) {
    Mesh::QuadGroup qg;
    qg.groupName = groupName;
    qg.elementArrayBufferID = int(elementArrayBuffers.size()) - 1;
    qg.noOfQuads = numberOfPrimitives;
    qg.offset = 0;
    quadGroups.push_back(qg);
  }
  else {
    Mesh::TriangleGroup tg;
    tg.groupName = groupName;
    tg.elementArrayBufferID = int(elementArrayBuffers.size()) - 1;
    tg.noOfTriangles = numberOfPrimitives;
    tg.offset = 0;
    triangleGroups.push_back(tg);
  }

  updateAllInternalData();
};

void Mesh::createQuad() {
  std::vector<float> positions = { -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0 };
  std::vector<float> normals = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
  std::vector<float> texCoords = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
  std::vector<unsigned int> indices = { 0, 1, 3, 2 };

  createFromArrays("Quad", 1, positions, normals, texCoords, indices, false);
}
