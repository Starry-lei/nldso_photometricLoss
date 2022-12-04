// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <string>
#include <iostream> 
#include <iomanip>
#include <fstream>
#include <sstream>
#include <tuple>
#include <map>
#include <math.h>

#include "LoadOBJ.h"
#include "StringTools.h"

using namespace std;
using namespace gsn;

#define gsn_ERROR( msg ) { cout << "ERROR: "   << msg << " (" << __FILE__ << ":"<< __LINE__ << ")" << endl; }  
#define gsn_WARN ( msg ) { cout << "WARNING: " << msg << " (" << __FILE__ << ":"<< __LINE__ << ")" << endl; }  
#define gsn_DEBUG( msg ) { cout << "DEBUG: "   << msg << " (" << __FILE__ << ":"<< __LINE__ << ")" << endl; } 

// some static helper functions
namespace gsn {
  int getIndex(const std::string str) {
    int value = StringTools::stringToInt(str);
    if (value < 1) {
      gsn_ERROR("OBJ parsing error: OBJ starts counting indices at 1");
    }
    return value - 1;
  }

  struct FaceGroup {
    string name;
    bool isQuad;
    std::vector<int> indices;
  };

  void computeNormalFromVerts(int t,int vc, std::vector<int>& fc, std::vector<float>& verts, std::vector<float>& normals) {
    int vi0 = fc[t * 3 * vc + 0 * 3 + 0]; // vertex index 0
    int vi1 = fc[t * 3 * vc + 1 * 3 + 0]; // vertex index 1
    int vi2 = fc[t * 3 * vc + 2 * 3 + 0]; // vertex index 2

    // get triangle coordinates (for quads the first three points are used)
    float v0x = verts[vi0 * 3 + 0];
    float v0y = verts[vi0 * 3 + 1];
    float v0z = verts[vi0 * 3 + 2];

    float v1x = verts[vi1 * 3 + 0];
    float v1y = verts[vi1 * 3 + 1];
    float v1z = verts[vi1 * 3 + 2];

    float v2x = verts[vi2 * 3 + 0];
    float v2y = verts[vi2 * 3 + 1];
    float v2z = verts[vi2 * 3 + 2];


    // compute normalized vectors a and b that span the triangle
    float ax = v2x - v1x;
    float ay = v2y - v1y;
    float az = v2z - v1z;

    float bx = v0x - v1x;
    float by = v0y - v1y;
    float bz = v0z - v1z;

    // compute cross product
    float cx = ay * bz - az * by;
    float cy = az * bx - ax * bz;
    float cz = ax * by - ay * bx;

    // normalize normal
    float lengthC = sqrt(cx * cx + cy * cy + cz * cz);
    if (lengthC > 0.0) {
      cx /= lengthC;
      cy /= lengthC;
      cz /= lengthC;
    }

    normals.push_back(cx);
    normals.push_back(cy);
    normals.push_back(cz);
  }

  void copyToMesh(std::vector<float>& verts, std::vector<float>& normals, std::vector<float>& texCoords, std::vector <FaceGroup>& faceGroups,
     Mesh& mesh) 
  {

    std::vector<float> tmpVerts;
    std::vector<float> tmpNormals;
    std::vector<float> tmpTexCoords;
    std::vector<unsigned int> tmpIndices;

    map<int, tuple<int, int, int>> vertsReuse; // key is index in verts; values are indices of normal and texCoords and the index in tmpVerts
    
    mesh.clearAll();

    std::vector<int> missingTexCoordIndex;
    
    for (int g = 0; g < int(faceGroups.size()); g++) {
      FaceGroup& fg = faceGroups[g];
      string name = fg.name;
      std::vector<int>& fc = fg.indices;
      bool isQuad = fg.isQuad;
      if (fc.size() > 0) {
        int vc = 4;
        if(!isQuad) {
          vc = 3;
        }
         
        mesh.elementArrayBuffers.push_back(Mesh::ElementArrayBuffer());
        Mesh::ElementArrayBuffer& eb = mesh.elementArrayBuffers.back();
        int numberOfFaces = 0;
        if (isQuad) {
          mesh.quadGroups.push_back(Mesh::QuadGroup());
          Mesh::QuadGroup& qg = mesh.quadGroups.back();
          numberOfFaces = int(fc.size() / 12);
          qg.noOfQuads = numberOfFaces;
          qg.groupName = name;
          qg.elementArrayBufferID = int(mesh.elementArrayBuffers.size()) - 1;
          qg.offset = 0;
          qg.triangleGroupID = int(mesh.triangleGroups.size());
          mesh.triangleGroups.push_back(Mesh::TriangleGroup());
        } else {
          mesh.triangleGroups.push_back(Mesh::TriangleGroup());
          Mesh::TriangleGroup& tg = mesh.triangleGroups.back();
          numberOfFaces = int(fc.size() / 9);
          tg.noOfTriangles = numberOfFaces;
          tg.groupName = name;
          tg.elementArrayBufferID = int(mesh.elementArrayBuffers.size()) - 1;
          tg.offset = 0;
          tg.wireframeElementArrayBufferID = -1;
        }

        tmpIndices.clear();

        // handle missing texture or normals
        for (int t = 0; t < numberOfFaces; t++) {
          if (fc[t * 3 * vc + 0 * 3 + 1] < 0) { // if texture coords are not provided
            if (int(missingTexCoordIndex.size()) < 4) {
              texCoords.push_back(0.0);
              texCoords.push_back(0.0);
              missingTexCoordIndex.push_back((int(texCoords.size()) / 2) - 1);
              texCoords.push_back(1.0);
              texCoords.push_back(0.0);
              missingTexCoordIndex.push_back((int(texCoords.size()) / 2) - 1);
              texCoords.push_back(1.0);
              texCoords.push_back(1.0);
              missingTexCoordIndex.push_back((int(texCoords.size()) / 2) - 1);
              texCoords.push_back(0.0);
              texCoords.push_back(1.0);
              missingTexCoordIndex.push_back((int(texCoords.size()) / 2) - 1);
            }
            for (int v = 0; v < vc; v++) {
              fc[t * 3 * vc + v * 3 + 1] = missingTexCoordIndex[v];
            }
          } // end if texture coords are not provided

          if (fc[t * 3 * vc + 0 * 3 + 2] < 0) {  // if normal coords are not provided
            computeNormalFromVerts(t, vc, fc, verts, normals);
            for (int v = 0; v < vc; v++) {
              fc[t * 3 * vc + v * 3 + 2] = (int(normals.size()) / 3) - 1;
            }
          } // end if normals are not provided
        }

        // loop over all triangles
        for (int t = 0; t < numberOfFaces; t++) {
          for (int v = 0; v < vc; v++) {

            int vi = fc[t * 3 * vc + v * 3 + 0]; // vertex index
            int vt = fc[t * 3 * vc + v * 3 + 1]; // texCoord index
            int vn = fc[t * 3 * vc + v * 3 + 2]; // normal index

            bool reuse = false;
            tuple<int, int, int> tu;
            if (vertsReuse.count(vi) > 0) {
              tu = vertsReuse.at(vi);
              if (std::get<0>(tu) == vt && std::get<1>(tu) == vn) {
                // if normals and texture index is the same
                reuse = true;
              }
            }

            if (reuse) {
              // reuse vert index
              tmpIndices.push_back(std::get<2>(tu));
            }
            else {
              for (int xyz = 0; xyz < 3; xyz++) {
                tmpVerts.push_back(verts[vi * 3 + xyz]);
              }
              for (int xy = 0; xy < 2; xy++)
                tmpTexCoords.push_back(texCoords[vt * 2 + xy]);
              for (int xyz = 0; xyz < 3; xyz++)
                tmpNormals.push_back(normals[vn * 3 + xyz]);
              // add index
              int idx = (int(tmpVerts.size()) / 3) - 1;
              tmpIndices.push_back(idx);

              // store vertex indices for possible reuse

              tuple<int, int, int> newtu;
              std::get<0>(newtu) = vt;
              std::get<1>(newtu) = vn;
              std::get<2>(newtu) = idx;
              vertsReuse.insert(make_pair(vi, newtu));
            }
          }
        }//end loop over all triangle

        eb.elementArrayBuffer = tmpIndices;
      }
    } // end for faceGroups

    // positions
    Mesh::ArrayBuffer abv;
    abv.arrayBuffer = tmpVerts;
    mesh.arrayBuffers.push_back(abv);
    
    Mesh::VertexBuffer vbv;
    vbv.semantic = "position";
    vbv.arrayBufferID = int(mesh.arrayBuffers.size()) - 1;
    vbv.stride = 3;
    vbv.offset = 0;
    vbv.elementSize = 3;
    mesh.vertexBuffers.push_back(vbv);

    // normals
    Mesh::ArrayBuffer abn;
    abn.arrayBuffer = tmpNormals;
    mesh.arrayBuffers.push_back(abn);

    Mesh::VertexBuffer vbn;
    vbn.semantic = "normal";
    vbn.arrayBufferID = int(mesh.arrayBuffers.size()) - 1;
    vbn.stride = 3;
    vbn.offset = 0;
    vbn.elementSize = 3;
    mesh.vertexBuffers.push_back(vbn);

    // texCoords
    Mesh::ArrayBuffer abt;
    abt.arrayBuffer = tmpTexCoords;
    mesh.arrayBuffers.push_back(abt);

    Mesh::VertexBuffer vbt;
    vbt.semantic = "texcoord";
    vbt.arrayBufferID = int(mesh.arrayBuffers.size()) - 1;
    vbt.stride = 2;
    vbt.offset = 0;
    vbt.elementSize = 2;
    mesh.vertexBuffers.push_back(vbt);

    mesh.updateAllInternalData();
  }
}

bool LoadOBJ::load(std::string filename, Mesh& mesh)
{
  std::vector<float> verts;
  std::vector<float> normals;
  std::vector<float> texCoords;

  string currentMaterial = "gsn_default_mat";
  std::vector <FaceGroup> faceGroups;
  FaceGroup triFace = { "Tris_" + currentMaterial, false, std::vector<int>()};
  FaceGroup quadFace = { "Quads_" + currentMaterial, true, std::vector<int>()};
  int currentFaceGroupTriId = int(faceGroups.size());
  faceGroups.push_back(triFace);
  int currentFaceGroupQuadId = int(faceGroups.size());
  faceGroups.push_back(quadFace);

  bool smoothGroupWarnedOnce = false;
  bool polyWarnedOnce = false;

  std::string line;
  unsigned lineNumber = 0;
  
  ifstream myfile(filename.c_str());
  if (!myfile.is_open()) {
    gsn_ERROR("Can not open file" << filename);
    return false;
  }

  cout << "LoadOBJ::load: " << filename << endl;

  while (myfile.good()) {
    std::getline(myfile, line);
    ++lineNumber;
    //gsn_DEBUG(line);

    std::vector <string> items = StringTools::split(line, " \n\r\t\f\v");

    if (items.size() > 0) {
      char firstChar = items[0].at(0);
      switch (firstChar) {
      case '#':
      {
        //gsn_DEBUG("Found comment:");
      }
      break;
      case 'v':
      {
        char secondChar = firstChar;
        if (items[0].length() > 1) {
          secondChar = items[0].at(1);
        }
        switch (secondChar) {
        case 'v': // vertex
        {
          if (items.size() >= 4) {
            verts.push_back(StringTools::stringToFloat(items[1]));
            verts.push_back(StringTools::stringToFloat(items[2]));
            verts.push_back(StringTools::stringToFloat(items[3]));
          }
          else {
            gsn_ERROR("OBJ parsing error:  not enought data");
          }
        }
        break;
        case 't': // texture vertex
        {
          if (items.size() >= 3) {
            texCoords.push_back(StringTools::stringToFloat(items[1]));
            texCoords.push_back(StringTools::stringToFloat(items[2]));
          }
          else {
            gsn_ERROR("OBJ parsing error: not enought data");
          }
        }
        break;
        case 'n': // normal
        {
          if (items.size() >= 4) {
            normals.push_back(StringTools::stringToFloat(items[1]));
            normals.push_back(StringTools::stringToFloat(items[2]));
            normals.push_back(StringTools::stringToFloat(items[3]));
          }
          else {
            gsn_ERROR("OBJ parsing error: not enought data");
          }
        }
        break;
        }
      }
      break;
      case 'u':
      {
        if (items.size() >= 2) {
          if (StringTools::toLower(items[0]) == "usemtl") {

            string matName = items[1];
            bool found = false;
            for (int i = 0; i < int(faceGroups.size()); i++) {
              if ("Tris_" + matName ==  faceGroups[i].name) {
                found = true;
                currentFaceGroupTriId = i;
              }
              if ("Quads_" + matName == faceGroups[i].name) {
                found = true;
                currentFaceGroupQuadId = i;
              }
            }
            if (!found) {
              currentMaterial = matName;
              FaceGroup newTriFace = { "Tris_" + currentMaterial, false, std::vector<int>() };
              FaceGroup newQuadFace = { "Quads_" + currentMaterial, true, std::vector<int>() };
              currentFaceGroupTriId = int(faceGroups.size());
              faceGroups.push_back(newTriFace);
              currentFaceGroupQuadId = int(faceGroups.size());
              faceGroups.push_back(newQuadFace);
              
            }
          }
        }
      }
      break;
      case 's':
      {
        if (!smoothGroupWarnedOnce) {
          smoothGroupWarnedOnce = true;
          gsn_ERROR("WarningSmooth groups are not supported. Please export your mesh with explicit normal information.");
        }
      }
      break;
      case 'f':
      {
        int nv = 0; // number of vertices per face
        if (items.size() > 1) {
          nv = int(items.size()) - 1;
        }

        std::vector <int> faceIdx(nv * 3); // fill triplets of pos, tex, normal indices per vertex
        for (int k = 0; k < nv; k++) {
          faceIdx[k * 3 + 0] = -1;
          faceIdx[k * 3 + 1] = -1;
          faceIdx[k * 3 + 2] = -1;
          std::vector <string> indices = StringTools::splitEvery(items[k + 1], "/");
          if (indices.size() == 1) { // position index
            faceIdx[k * 3 + 0] = getIndex(indices[0]);
          }
          if (indices.size() == 3) { //format is f v/vt/vn or v//vn
            if (indices[1].length() != 0) { //format is f v/vt/vn
              faceIdx[k * 3 + 0] = getIndex(indices[0]);
              faceIdx[k * 3 + 1] = getIndex(indices[1]);
              faceIdx[k * 3 + 2] = getIndex(indices[2]);
            }
            else { // format is f v//vn
              faceIdx[k * 3 + 0] = getIndex(indices[0]);
              faceIdx[k * 3 + 2] = getIndex(indices[2]);
            }
          }
          if (indices.size() == 2) { // format v/vt
            faceIdx[k * 3 + 0] = getIndex(indices[0]);
            faceIdx[k * 3 + 1] = getIndex(indices[1]);
          }
        }

        if (nv == 3) { // triangle
          for (int i = 0; i < 9; i++) {
            faceGroups[currentFaceGroupTriId].indices.push_back(faceIdx[i]);
          }
        }
        else {
          if (nv == 4 && faceIdx[3 * 3 + 0] != -1) { // quad
            for (int i = 0; i < 12; i++) {
              faceGroups[currentFaceGroupQuadId].indices.push_back(faceIdx[i]);
            }
          }
          else {
            if (nv > 4) { // poly
              if (!polyWarnedOnce) {
                polyWarnedOnce = true;
                gsn_ERROR("Warning ObjMeshParser: The mesh contains polygons with more than 4 vertices. These are converted to triangles.");
              }
              for (int t = 0; t < nv - 2; t++) {
                for (int k = 0; k < 3; k++) {
                  int kk = 0;
                  if (k == 1) {
                    kk = t + 1;
                  }
                  if (k == 2) {
                    kk = t + 2;
                  }
                  for (int i = 0; i < 3; i++) { // put first triangle
                    faceGroups[currentFaceGroupTriId].indices.push_back(faceIdx[kk * 3 + i]);
                  }
                }
              }
            }
          }
        }
      }
      break;
      }
    }
  }
  cout << "LoadOBJ::load: " << filename << " Info: vertices=" << verts.size() / 3 << " normals="  << normals.size() / 3 << " texCoords=" << texCoords.size() / 2 << endl;
  copyToMesh(verts, normals, texCoords, faceGroups, mesh);

  myfile.close();

  return true;
}
