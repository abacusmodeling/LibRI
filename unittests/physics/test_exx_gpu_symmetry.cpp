// Standalone callback regression: g++ -std=c++11 -Iinclude unittests/physics/test_exx_gpu_symmetry.cpp -o /tmp/test_exx_gpu_symmetry
#include <cmath>
#include "RI/physics/symmetry/Filter_Atom_Symmetry.h"

#include <array>
#include <cassert>
#include <map>
#include <set>
#include <utility>

int main()
{
    using Cell = std::array<int, 1>;
    using AtomCell = std::pair<int, Cell>;
    using Label = RI::Label::ab_ab;
    const Cell zero{{0}};
    const Cell period{{2}};
    const std::map<std::pair<int, int>, std::set<Cell>> sector{
        {{1, 0}, {zero}}
    };
    RI::Filter_Atom_Symmetry<int, Cell, double> filter(period, sector);
    const AtomCell left{1, zero};
    const AtomCell right{0, zero};
    const AtomCell intermediate{2, zero};
    const Cell one{{1}};
    const AtomCell shifted_right{0, one};

    assert(!filter.filter_for1(Label::a0b0_a1b1, 2));
    assert(!filter.filter_for32(Label::a0b0_a1b1, 2, left, right));
    assert(filter.filter_for32(Label::a0b0_a1b1, 2, right, left));
    assert(filter.filter_for32(Label::a0b0_a1b1, 2, left, shifted_right));

    assert(!filter.filter_for1(Label::a0b0_a1b2, 2));
    assert(!filter.filter_for1(Label::a0b0_a1b2, right));
    assert(filter.filter_for1(Label::a0b0_a1b2, left));
    assert(!filter.filter_for32(Label::a0b0_a1b2, 2, left, right));
    assert(filter.filter_for32(Label::a0b0_a1b2, 2, right, left));

    assert(!filter.filter_for1(Label::a0b0_a2b1, 1));
    assert(!filter.filter_for32(Label::a0b0_a2b1, 1, intermediate, right));
    assert(filter.filter_for32(Label::a0b0_a2b1, 1, intermediate, left));

    assert(!filter.filter_for1(Label::a0b0_a2b2, 1));
    assert(!filter.filter_for32(Label::a0b0_a2b2, 1, intermediate, right));
    assert(filter.filter_for32(Label::a0b0_a2b2, 1, intermediate, left));
}
