#include <array>
#include <map>
#include <set>
#include <tuple>
#define NO_SEC_RETURN_TRUE if(this->irreducible_sector_.empty()) return true;
#include "../global/Array_Operator.h"
namespace RI
{
	using namespace Array_Operator;
	template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
	class Symmetry_Filter
	{
		using TC = std::array<Tcell, Ndim>;
		using TAC = std::pair<TA, TC>;

		using TIJ = std::pair<TA, TA>;
		using TIJR = std::pair<TIJ, TC>;
		using Tsec = std::map<TIJ, std::set<TC>>;
	public:
		Symmetry_Filter(const TC& period_in, const Tsec& irsec)
			:period(period_in), irreducible_sector_(irsec) {}
		bool in_irreducible_sector(const TA& Aa, const TAC& Ab) const
		{
			NO_SEC_RETURN_TRUE;
			const TIJ& ap = { Aa, Ab.first };
			if (irreducible_sector_.find(ap) != irreducible_sector_.end())
				if (irreducible_sector_.at(ap).find(Ab.second % this->period) != irreducible_sector_.at(ap).end())
					return true;
			return false;
		}
		bool in_irreducible_sector(const TAC& Aa, const TAC& Ab) const
		{
			NO_SEC_RETURN_TRUE;
			const TC dR = (Ab.second - Aa.second) % this->period;
			const std::pair<TA, TA> ap = { Aa.first, Ab.first };
			if (irreducible_sector_.find(ap) != irreducible_sector_.end())
				if (irreducible_sector_.at(ap).find(dR) != irreducible_sector_.at(ap).end())
					return true;
			return false;
		}
		bool is_I_in_irreducible_sector(const TA& Aa) const
		{
			NO_SEC_RETURN_TRUE;
			for (const auto& apRs : irreducible_sector_)
				if (apRs.first.first == Aa)return true;
			return false;
		}
		bool is_J_in_irreducible_sector(const TA& Ab) const
		{
			NO_SEC_RETURN_TRUE;
			for (const auto& apRs : irreducible_sector_)
				if (apRs.first.second == Ab)return true;
			return false;
		}
		TIJR get_IJR(const TA& I, const TAC& J) const
		{
			return { {I,J.first}, J.second % this->period };
		}
		TIJR get_IJR(const TAC& I, const TAC& J) const
		{
			return { {I.first,J.first}, (J.second - I.second) % this->period };
		}
	private:
		const Tsec& irreducible_sector_;
		const TC& period;
	};

}