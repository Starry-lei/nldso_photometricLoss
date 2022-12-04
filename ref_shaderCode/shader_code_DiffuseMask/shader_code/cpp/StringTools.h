// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#ifndef DEF_STRINGTOOLS_H
#define DEF_STRINGTOOLS_H

#include <string>
#include <vector>

namespace gsn {
  /*!
  \class StringTools StringTools.h
  \brief This class holds a collection of useful functions for the std::string class.
  */

  class StringTools {

  public:
    //! converts the given integer to a string
    static std::string intToString(const int i);

    //! converts the given unsigned integer to a string
    static std::string unsignedToString(const unsigned i, const int digitCount = 3);

    //! converts the given float value to a string
    static std::string floatToString(const float i, const int digitCount = 3);

    //! converts the given double value to a string
    static std::string doubleToString(const double i, const int digitCount = 6);

    //! converts the given bool value to a string
    static std::string boolToString(const bool val);

    //! converts the given string to an integer value
    static int stringToInt(const std::string& s);

    //! converts the given string to an unsigned integer value
    static unsigned stringToUnsigned(const std::string& s);

    //! converts the given string to a float value
    static float stringToFloat(const std::string& s);

    //! converts the given string to a double value
    static double stringToDouble(const std::string& s);

    //! converts the given string to a bool value
    static bool stringToBool(const std::string& s);

    //! tokenizes the given string with the given delimiters
    static std::vector<std::string> tokenize(const std::string& s, const std::string& delimiters);

    //! splits the given string with the given separators
    static std::vector<std::string> split(const std::string& s, const std::string separators = " \n\r\t\f\v");

    //! splits the given string with the given separators, returns also empty strings
    static std::vector<std::string> splitEvery(const std::string& s, const std::string separators = ",");

    //! returns a lower case version of the given string
    static std::string toLower(const std::string& s);

    //! returns a upper case version of the given string
    static std::string toUpper(const std::string& s);

    //! trim from left side
    static std::string trimLeft(const std::string& str, const std::string ws = " \n\r\t\f\v");

    //! trim from right side
    static std::string trimRight(const std::string& str, const std::string ws = " \n\r\t\f\v");

    //! trim string
    static std::string trim(const std::string& str, const std::string ws = " \n\r\t\f\v");
  };

}

#endif
