// ===================
//  Author: Peize Lin
//  date: 2022.06.02
// ===================

#pragma once

#include <complex>

namespace Global_Func
{
	template<typename T> struct To_Real	{ using type=T; };
	template<typename T> struct To_Real<std::complex<T>> { using type=T; };
	template<typename T> using To_Real_t = typename Global_Func::To_Real<T>::type;
}