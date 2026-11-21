#include <cmath>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <RI/physics/symmetry/Filter_Atom_Symmetry.h>

using Cell = std::array<int, 1>;
using AtomCell = std::pair<int, Cell>;
using Sector = std::map<std::pair<int,int>, std::set<Cell>>;
using Label = RI::Label::ab_ab;

// Call-site tuples from PR #24: all eight positions are exercised via the base interface.
void check(bool actual, bool expected, Label label, const char* position)
{
    if (actual!=expected)
        throw std::runtime_error(RI::Label_Tools::get_name(label)+" callback "+position+" contract mismatch");
}

void check_callbacks(const RI::Filter_Atom<int,AtomCell>& f, Label label,
                     int Aa01, AtomCell Aa2, AtomCell Ab01, AtomCell Ab2,
                     bool left_kept, bool right_kept, bool keep)
{
    switch(label)
    {
        case Label::a0b0_a1b1:
            check(f.filter_GPU_for11(label, Aa01), false, label, "11");
            check(f.filter_GPU_for21(label, Aa01, Ab01), false, label, "21");
            check(f.filter_GPU_for31(label, Aa01, Ab01, Ab2), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), false, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Aa2), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Aa2, Ab2), !keep, label, "32");
            return;
        case Label::a0b1_a1b0:
            check(f.filter_GPU_for11(label, Aa01), false, label, "11");
            check(f.filter_GPU_for21(label, Aa01, Ab01), false, label, "21");
            check(f.filter_GPU_for31(label, Aa01, Ab01, Ab2), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), false, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Aa2), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Aa2, Ab2), !keep, label, "32");
            return;
        case Label::a0b0_a1b2:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), false, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Aa2), !keep, label, "32");
            return;
        case Label::a0b1_a1b2:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), false, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Aa2), !keep, label, "32");
            return;
        case Label::a0b2_a1b0:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), false, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Aa2), !keep, label, "32");
            return;
        case Label::a0b2_a1b1:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), false, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Aa2), !keep, label, "32");
            return;
        case Label::a0b0_a2b1:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a0b1_a2b0:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a1b0_a2b1:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a1b1_a2b0:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab01), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a0b0_a2b2:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab2), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), !keep, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a0b1_a2b2:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab2), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), !keep, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a1b0_a2b2:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab2), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), !keep, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a1b1_a2b2:
            check(f.filter_GPU_for11(label,Aa01), !left_kept, label, "11");
            check(f.filter_GPU_for21(label,Aa01,Aa2), false, label, "21");
            check(f.filter_GPU_for31(label,Aa01,Aa2,Ab2), false, label, "31");
            check(f.filter_GPU_for12(label, Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label, Aa01, Ab01), !keep, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Ab01, Ab2), !keep, label, "32");
            return;
        case Label::a1b2_a2b1:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label,Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label,Aa01,Aa2), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Aa2, Ab01), !keep, label, "32");
            check(f.filter_GPU_for13(label, Aa01), !left_kept, label, "13");
            check(f.filter_GPU_for23(label, Aa01, Ab01), !keep, label, "23");
            return;
        case Label::a0b2_a2b0:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label,Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label,Aa01,Aa2), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Aa2, Ab01), !keep, label, "32");
            check(f.filter_GPU_for13(label, Aa01), !left_kept, label, "13");
            check(f.filter_GPU_for23(label, Aa01, Ab01), !keep, label, "23");
            return;
        case Label::a0b2_a2b1:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label,Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label,Aa01,Aa2), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Aa2, Ab01), !keep, label, "32");
            check(f.filter_GPU_for13(label, Aa01), !left_kept, label, "13");
            check(f.filter_GPU_for23(label, Aa01, Ab01), !keep, label, "23");
            return;
        case Label::a1b2_a2b0:
            check(f.filter_GPU_for11(label,Ab01), !right_kept, label, "11");
            check(f.filter_GPU_for21(label,Ab01,Ab2), false, label, "21");
            check(f.filter_GPU_for31(label,Ab01,Ab2,Aa01), false, label, "31");
            check(f.filter_GPU_for12(label,Aa01), !left_kept, label, "12");
            check(f.filter_GPU_for22(label,Aa01,Aa2), false, label, "22");
            check(f.filter_GPU_for32(label, Aa01, Aa2, Ab01), !keep, label, "32");
            check(f.filter_GPU_for13(label, Aa01), !left_kept, label, "13");
            check(f.filter_GPU_for23(label, Aa01, Ab01), !keep, label, "23");
            return;
        default: throw std::runtime_error("Unknown label");
    }
}

int main() try
{
    const Cell period{{3}};
    const std::vector<Sector> sectors{
        {}, {{{1,0}, {Cell{{1}}}}},
        {{{0,1}, {Cell{{0}}, Cell{{-1}}}}, {{1,1}, {Cell{{1}}}}},
        {{{7,9}, {Cell{{1}}}}}
    };
    std::size_t checked = 0;
    for (const auto& sector : sectors)
    {
        const RI::Filter_Atom_Symmetry<int,Cell,double> filter(period, sector);
        for (auto label : RI::Label::array_ab_ab)
        for (int Aa01=0; Aa01<2; ++Aa01)
        for (int a=0; a<6; ++a)
        for (int b=0; b<6; ++b)
        for (int c=0; c<6; ++c)
        {
            const AtomCell Aa2{a/3,Cell{{a%3}}};
            const AtomCell Ab01{b/3,Cell{{b%3}}};
            const AtomCell Ab2{c/3,Cell{{c%3}}};
            AtomCell left{Aa01,Cell{{0}}}, right;
            // Derive output indices independently from the uncontracted a/b index.
            const std::string name=RI::Label_Tools::get_name(label);
            int ai=3, bi=3;
            for (int i=0;i<3;++i)
            {
                if (name[1]-'0'!=i && name[6]-'0'!=i) ai=i;
                if (name[3]-'0'!=i && name[8]-'0'!=i) bi=i;
            }
            if (ai==2) left=Aa2;
            right=bi==2 ? Ab2 : Ab01;
            int cell=(right.second[0]-left.second[0]+3)%3;
            if (cell>1) cell-=3;
            const auto pair=sector.find({left.first,right.first});
            const bool keep=sector.empty() || (pair!=sector.end() && pair->second.count(Cell{{cell}}));
            bool left_kept=sector.empty(), right_kept=sector.empty();
            for (const auto& entry:sector)
            {
                left_kept = left_kept || entry.first.first==left.first;
                right_kept = right_kept || entry.first.second==right.first;
            }
            check_callbacks(filter,label,Aa01,Aa2,Ab01,Ab2,left_kept,right_kept,keep);
            ++checked;
        }
    }
    std::cout << "PASS " << checked << " label/sector/atom/cell combinations\n";
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
