// ===================
//  Author: Peize Lin
//  date: 2022.06.06
// ===================

#pragma once

#include "RI/ri/Label.h"
#include <string>
#include <array>
#include <unordered_map>
#include <fstream>

namespace CS_Matrix_Test
{
	const std::unordered_map<Label::ab, std::string> label_name = {
		{Label::ab::a   , "a"   },
		{Label::ab::b   , "b"   },
		{Label::ab::a0b0, "a0b0"},
		{Label::ab::a0b1, "a0b1"},
		{Label::ab::a0b2, "a0b2"},
		{Label::ab::a1b0, "a1b0"},
		{Label::ab::a1b1, "a1b1"},
		{Label::ab::a1b2, "a1b2"},
		{Label::ab::a2b0, "a2b0"},
		{Label::ab::a2b1, "a2b1"},
		{Label::ab::a2b2, "a2b2"}};

	template<typename T>
	void print_uplimits3(const std::string &file_name, const std::unordered_map<Label::ab,std::array<T,3>> &datas)
	{
		for(const auto & data : datas)
		{
			for(int i=0; i<3; ++i)
				if(!data.second[i].empty())
				{
					std::ofstream ofs(file_name+"_"+label_name.at(data.first)+"_"+std::to_string(i));
					ofs<<data.second[i]<<std::endl;
				}
		}
	}

	template<typename T>
	void print_uplimits2(const std::string &file_name, const std::unordered_map<Label::ab,T> &datas)
	{
		for(const auto & data : datas)
		{
			if(!data.second.empty())
			{
				std::ofstream ofs(file_name+"_"+label_name.at(data.first));
				ofs<<data.second<<std::endl;
			}
		}
	}
}