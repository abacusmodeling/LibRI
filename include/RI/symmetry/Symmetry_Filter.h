#include <array>
#include <map>
#include <set>
#include <tuple>
#include "../global/Array_Operator.h"
#include "../ri/Filter_Atom.h"
#define NO_SEC_RETURN_FALSE if(this->irreducible_sector_.empty()) return false;
#define FILTER_FOR1_A0B0(A1) \
switch (label)\
{\
case Lab::a0b0_a1b1: case Lab::a0b0_a2b1: case Lab::a0b0_a2b2:\
	return !this->is_I_in_irreducible_sector(A1);\
case Lab::a0b0_a1b2:\
	return !this->is_J_in_irreducible_sector(A1);\
default:\
	return false;\
}
namespace RI
{
	using namespace Array_Operator;
	template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
	class Symmetry_Filter : public Filter_Atom<TA, std::pair<TA, std::array<Tcell, Ndim>>>
	{
		using TC = std::array<Tcell, Ndim>;
		using TAC = std::pair<TA, TC>;

		using TIJ = std::pair<TA, TA>;
		using TIJR = std::pair<TIJ, TC>;
		using Tsec = std::map<TIJ, std::set<TC>>;

		using Lab = Label::ab_ab;

		bool in_irreducible_sector(const TA& Aa, const TAC& Ab) const
		{
			const TIJ& ap = { Aa, Ab.first };
			if (irreducible_sector_.find(ap) != irreducible_sector_.end())
				if (irreducible_sector_.at(ap).find(Ab.second % this->period) != irreducible_sector_.at(ap).end())
					return true;
			return false;
		}
		bool in_irreducible_sector(const TAC& Aa, const TAC& Ab) const
		{
			const TC dR = (Ab.second - Aa.second) % this->period;
			const std::pair<TA, TA> ap = { Aa.first, Ab.first };
			if (irreducible_sector_.find(ap) != irreducible_sector_.end())
				if (irreducible_sector_.at(ap).find(dR) != irreducible_sector_.at(ap).end())
					return true;
			return false;
		}
		bool is_I_in_irreducible_sector(const TA& Aa) const
		{
			for (const auto& apRs : irreducible_sector_)
				if (apRs.first.first == Aa)return true;
			return false;
		}
		bool is_J_in_irreducible_sector(const TA& Ab) const
		{
			for (const auto& apRs : irreducible_sector_)
				if (apRs.first.second == Ab)return true;
			return false;
		}
		// TIJR get_IJR(const TA& I, const TAC& J) const
		// {
		// 	return { {I,J.first}, J.second % this->period };
		// }
		// TIJR get_IJR(const TAC& I, const TAC& J) const
		// {
		// 	return { {I.first,J.first}, (J.second - I.second) % this->period };
		// }

		const Tsec& irreducible_sector_;
		const TC& period;

	public:
		Symmetry_Filter(const TC& period_in, const Tsec& irsec)
			:period(period_in), irreducible_sector_(irsec) {}
		virtual bool filter_for1(const Label::ab_ab& label, const TA& A1) const override
		{
			NO_SEC_RETURN_FALSE;
			FILTER_FOR1_A0B0(A1);
		}
		virtual bool filter_for1(const Label::ab_ab& label, const TAC& A1) const override
		{
			NO_SEC_RETURN_FALSE;
			FILTER_FOR1_A0B0(A1.first);
		}
		virtual bool filter_for32(const Label::ab_ab& label, const TA& A1, const TAC& A2, const TAC& A3) const override
		{
			NO_SEC_RETURN_FALSE;
			switch (label)
			{
			case Lab::a0b0_a2b1: case Lab::a0b0_a2b2:
				return !this->in_irreducible_sector(A1, A3);
			default:
				return false;
			}
		}
		virtual bool filter_for32(const Label::ab_ab& label, const TAC& A1, const TA& A2, const TAC& A3) const override
		{
			NO_SEC_RETURN_FALSE;
			switch (label)
			{
			case Lab::a0b0_a1b2:
				return !this->in_irreducible_sector(A3, A1);
			default:
				return false;
			}
		}
		virtual bool filter_for32(const Label::ab_ab& label, const TAC& A1, const TAC& A2, const TAC& A3) const override
		{
			NO_SEC_RETURN_FALSE;
			switch (label)
			{
			case Lab::a0b0_a1b1:
				return !this->in_irreducible_sector(A1, A3);
			default:
				return false;
			}
		}
	};

}