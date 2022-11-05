// ===================
//  Author: Peize Lin
//  Date:  2022.07.25
// ===================

#pragma once

#include <map>

namespace RI
{

namespace Map_Operator
{
	template<typename Tkey, typename Tvalue>
	std::map<Tkey,Tvalue> operator+(const std::map<Tkey,Tvalue> &m1, const std::map<Tkey,Tvalue> &m2)
	{
		std::map<Tkey,Tvalue> m;
		auto ptr1 = m1.begin();
		auto ptr2 = m2.begin();
		while(ptr1!=m1.end() && ptr2!=m2.end())
		{
			if(ptr1->first == ptr2->first)
			{
				m.emplace_hint(m.end(), ptr1->first, ptr1->second + ptr2->second);
				++ptr1;
				++ptr2;
			}
			else if(ptr1->first < ptr2->first)
			{
				m.emplace_hint(m.end(), ptr1->first, ptr1->second);
				++ptr1;
			}
			else
			{
				m.emplace_hint(m.end(), ptr2->first, ptr2->second);
				++ptr2;
			}
		}
		m.insert(ptr1, m1.end());
		m.insert(ptr2, m2.end());
		return m;
	}
}

}