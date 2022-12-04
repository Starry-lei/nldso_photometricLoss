// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace std;

#include "StringTools.h"
using namespace gsn;

std::string StringTools::intToString(const int i) 
{
	stringstream ss;
	ss << i;
	return ss.str();
}

std::string StringTools::unsignedToString(const unsigned i, const int digitCount) 
{
    stringstream ss;
    ss << std::setw( digitCount ) << std::setfill( '0' ) << i;
    return ss.str();
}

std::string StringTools::floatToString(const float f, const int digitCount) 
{
	stringstream ss;
	ss.setf(ios::fixed);
	ss << setprecision(digitCount);
	ss << setfill(' ');
	ss << f;
	ss << setw(0);
	return ss.str();
}

std::string StringTools::doubleToString(const double f, const int digitCount) 
{
	stringstream ss;
	ss.setf(ios::fixed);
	ss << setprecision(digitCount);
	ss << setfill(' ');
	ss << f;
	ss << setw(0);
	return ss.str();
}


std::string StringTools::boolToString(const bool val)
{
    if(val) return "true"; else return "false";
}


float StringTools::stringToFloat(const std::string& s)
{
	return (float)atof(s.c_str());
}

double StringTools::stringToDouble(const std::string& s)
{
	return (double)atof(s.c_str());
}


int StringTools::stringToInt(const std::string& s)
{
	return atoi(s.c_str());
}

unsigned StringTools::stringToUnsigned(const std::string& s)
{
    int i = atoi(s.c_str());
    if(i < 0) i = 0;
	return unsigned(i);
}


bool StringTools::stringToBool(const std::string& s)
{
    if(toLower(s).compare("true") == 0) return true;
    if(toLower(s).compare("false") == 0) return false;
    if(s.compare("1") == 0) return true;
    if(s.compare("0") == 0) return false;
    return false;
}

std::vector<std::string> StringTools::tokenize(const std::string& str, const std::string& delimiters) 
{
	
	vector<string> tokens;

	// skip delimiters at beginning.
	string::size_type lastPos = str.find_first_not_of(delimiters, 0);

	// find first "non-delimiter".
	string::size_type pos = str.find_first_of(delimiters, lastPos);

	while (string::npos != pos || string::npos != lastPos)
	{
		// found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));

		// skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);

		// find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}

	return tokens;
}

std::vector<string> StringTools::split(const std::string& s, const std::string separators)
{
	vector <string> result;
	size_t pos = 0;
	size_t found = 0;
	do {
		found = s.find_first_of(separators, pos);

		if (found != string::npos) {
			if (found - pos > 0) {
				string item = s.substr(pos, found - pos);
				result.push_back(item);
			}
			pos = found + 1;
		}
	} while (found != string::npos);

	if (pos < s.size()) {
		string item = s.substr(pos);
		result.push_back(item);
	}

	return result;
}

std::vector<string> StringTools::splitEvery(const std::string& s, const std::string separators)
{
	vector <string> result;
	size_t pos = 0;
	size_t found = 0;
	do {
		found = s.find_first_of(separators, pos);

		if (found != string::npos) {
			if (found - pos >= 0) {
				string item = s.substr(pos, found - pos);
				result.push_back(item);
			}
			pos = found + 1;
		}
	} while (found != string::npos);

	if (pos < s.size()) {
		string item = s.substr(pos);
		result.push_back(item);
	}

	return result;
}

std::string StringTools::toLower(const std::string& s) {
	string s2 = s;
	for(int i = 0; i < (int)s.length(); i++) {
		s2[i] = tolower(s[i]);
	}
	return s2;
}

std::string StringTools::toUpper(const std::string& s) {
	string s2 = s;
	for(int i = 0; i < (int)s.length(); i++) {
		s2[i] = toupper(s[i]);
	}
	return s2;
}

std::string StringTools::trimRight(const std::string& str, const std::string ws) {
	std::string copy = str;
	copy.erase(copy.find_last_not_of(ws) + 1);
	return copy;
}

std::string StringTools::trimLeft(const std::string& str, const std::string ws) {
	std::string copy = str;
	copy.erase(0, copy.find_first_not_of(ws));
	return copy;
}

std::string StringTools::trim(const std::string& str, const std::string ws) {
	return trimLeft(trimRight(str, ws), ws);
}