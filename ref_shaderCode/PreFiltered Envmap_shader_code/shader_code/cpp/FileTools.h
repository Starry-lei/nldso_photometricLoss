// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_FILETOOLS_H
#define DEF_FILETOOLS_H

#include <string>
#include <vector>

namespace gsn {
  /*!
  \class FileTools FileToolss.h
  \brief This class holds a collection of useful functions for files.
  */

  class FileTools {

  public:
    //! returns true if a file exists
    static bool fileExists(const std::string& filename);

    //! finds a file relative to the directory in which the application is started
    static std::string findFile(const std::string& filename, int depth = 5);

    //! returns the directory from a path
    static std::string getDirectory(const std::string& path);

    //! returns the file name from a path
    static std::string getFileName(const std::string& path);

    //! reads the file content as a std::string (exits if file can not be read)
    static std::string readTextFile(const std::string& filename);

    //! loads a PFM (PortableFloatMap) file
    static bool loadPFM(const std::string& filename, int& width, int& height, int& channels, std::vector<float>& data);

    //! loads a PFM (PortableFloatMap) file with a separate alpha channel PFM file
    static bool loadAlphaPFM(const std::string& filename, int& width, int& height, std::vector<float>& data);
  };

}

#endif
