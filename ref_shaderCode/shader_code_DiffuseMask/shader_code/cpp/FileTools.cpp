// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "FileTools.h"

using namespace std;
using namespace gsn;


bool FileTools::fileExists(const std::string& filename)
{
  ifstream myfile(filename.c_str());
  if (!myfile.is_open()) {
    return false;
  }
  myfile.close();
  return true;
}

std::string FileTools::findFile(const std::string& filename, int depth)
{
  int counter = 0;
  std::string path("");

  while (counter < depth) {
    if (FileTools::fileExists(path + filename)) return path + filename;
    path += "../";
    counter++;
  }
  return filename;
}

std::string FileTools::readTextFile(const std::string& filename) {
  std::ifstream is(filename);
  if (is.is_open()) {
    std::stringstream buffer;
    buffer << is.rdbuf();
    return buffer.str();
  }
  cerr << "Unable to open file " << filename << endl;
  exit(1);
}

std::string FileTools::getDirectory(const std::string& path) {
  size_t pos = path.find_last_of("\\/");
  if (pos != std::string::npos) {
    return path.substr(0, pos);
  }
  return path;
 }

std::string FileTools::getFileName(const std::string& path) {
  size_t pos = path.find_last_of("\\/");
  if (pos != std::string::npos) {
    return path.substr(pos + 1);
  }
  return path;
}


namespace gsn {
  void getNextStringPNMHeader(FILE* fp, char* line) // gets next string ignoring comments marked by "#"
  {
    int i;
    line[0] = '\0';
    while (line[0] == '\0') {
      fscanf(fp, "%s", line);
      i = -1;
      do {
        i++;
        if (line[i] == '#') {
          line[i] = '\0';
          while (fgetc(fp) != '\n');
        }
      } while (line[i] != '\0');
    }
  }
}

bool FileTools::loadPFM(const std::string& filename, int& width, int& height, int& channels, std::vector<float>& data)
{
  const int identiferLength = 1000;
  char identifer[identiferLength];

  FILE* file = fopen(filename.c_str(), "rb");

  if (file == NULL) {
    fprintf(stderr, "ERROR: [FileTools::loadPFM] could not open file %s", filename.c_str());
    return false;
  }

  // read "magic number" for identifying the file type.
  char letter;
  char letter2;
  getNextStringPNMHeader(file, identifer);
  sscanf(identifer, "%c%c", &letter, &letter2);

  if (letter != 'P') {
    fprintf(stderr, "ERROR: [LoadPFG::load] The input data is not PFM.\n");
    fclose(file);
    return false;
  }

  bool found = false;

  if (letter2 == 'f') {
    channels = 1;
    found = true;
  }
  if (letter2 == 'F') {
    channels = 3;
    found = true;
  }
  if (letter2 == 'g') {
    sscanf(identifer, "%c%c%d", &letter, &letter2, &channels);
    if (channels > 1 && channels < 256) {
      found = true;
    }
  }

  if (!found) {
    fprintf(stderr, "ERROR: [LoadPFG::load] The input data is not PFM.\n");
    fclose(file);
    return false;
  }

  // read width and height of the image
  getNextStringPNMHeader(file, identifer);
  sscanf(identifer, "%d", &width);
  getNextStringPNMHeader(file, identifer);
  sscanf(identifer, "%d", &height);
  float byteOrder;
  getNextStringPNMHeader(file, identifer);
  sscanf(identifer, "%f", &byteOrder);

  if (byteOrder >= 0.0) {
    fprintf(stderr, "ERROR: [FileTools::loadPFM] only little-endian byte order is supported\n");
    fclose(file);
    return false;
  }

  // A single whitespace character (usually a newline). 
  fgetc(file);

  data.resize(width * height * channels);
  fread(&data[0], sizeof(float), width * height * channels, file);
  
  float pixelmax = -1e37f;
  float pixelmin = 1e37f;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int z = 0; z < channels; z++) {
        float value = data[y * width * channels + x * channels + z];
        if (pixelmax < value) pixelmax = value;
        if (pixelmin > value) pixelmin = value;
      }
    }
  }

  cout << "FileTools::loadPFM: " << filename << " Info: width=" << width << " height=" << height << " channels=" << channels << " min=" << pixelmin << " max=" << pixelmax << endl; 
  fclose(file);
  return true;
}

bool FileTools::loadAlphaPFM(const std::string& filename, int& width, int& height, std::vector<float>& data)
{
  size_t pos = filename.find_last_of(".");
  string filenameAlpha;
  if (pos != std::string::npos) {
    filenameAlpha = filename.substr(0, pos) + "_alpha" + filename.substr(pos);
  }
  int widthRGB = 0;
  int heightRGB = 0;
  int channelsRGB = 0;
  std::vector <float> dataRGB;
  bool ret = FileTools::loadPFM(filename, widthRGB, heightRGB, channelsRGB, dataRGB);
  if (!ret) {
    return false;
  }
  int widthAlpha = 0;
  int heightAlpha = 0;
  int channelsAlpha = 0;
  std::vector <float> dataAlpha;
  ret = FileTools::loadPFM(filenameAlpha, widthAlpha, heightAlpha, channelsAlpha, dataAlpha);
  if (!ret) {
    return false;
  }
  if (widthRGB != widthAlpha || heightRGB != heightAlpha || channelsRGB != 3 || channelsAlpha < 1) {
    return false;
  }
  width = widthRGB;
  height = heightRGB;
  int noOfPixels = width * height;
  data.resize(noOfPixels * 4);
  int countRGB = 0;
  int countAlpha = 0;
  for (int i = 0; i < noOfPixels; i++) {
    data[i * 4 + 0] = dataRGB[countRGB++];
    data[i * 4 + 1] = dataRGB[countRGB++];
    data[i * 4 + 2] = dataRGB[countRGB++];
    data[i * 4 + 3] = dataAlpha[countAlpha++];
  }

  return true;
}