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
    const Cell one{{1}};
    const std::map<std::pair<int, int>, std::set<Cell>> sector{
        {{1, 0}, {zero}}
    };
    RI::Filter_Atom_Symmetry<int, Cell, double> filter(period, sector);
    const AtomCell left{1, zero};
    const AtomCell right{0, zero};
    const AtomCell intermediate{2, zero};
    const AtomCell shifted_right{0, one};
    std::set<Label> checked;

    // GPU output: (Aa2, Ab2); Aa01 is only an intermediate atom.
    const std::array<Label, 2> output_a2_b2{{Label::a0b0_a1b1, Label::a0b1_a1b0}};
    for (const Label label : output_a2_b2)
    {
        checked.insert(label);
        assert(!filter.filter_for1(label, 2));
        assert(!filter.filter_for32(label, 2, left, right));
        assert(filter.filter_for32(label, 2, right, left));
        assert(filter.filter_for32(label, 2, left, shifted_right));
    }

    // GPU output: (Aa2, Ab01); Ab01 can already be screened on its own.
    const std::array<Label, 4> output_a2_b01{{Label::a0b0_a1b2, Label::a0b1_a1b2,
                                              Label::a0b2_a1b0, Label::a0b2_a1b1}};
    for (const Label label : output_a2_b01)
    {
        checked.insert(label);
        assert(!filter.filter_for1(label, 2));
        assert(!filter.filter_for1(label, right));
        assert(filter.filter_for1(label, left));
        assert(!filter.filter_for32(label, 2, left, right));
        assert(filter.filter_for32(label, 2, right, left));
        assert(filter.filter_for32(label, 2, left, shifted_right));
    }

    // GPU output: (Aa01, Ab2); Ab01 is only an intermediate atom.
    const std::array<Label, 4> output_a01_b2{{Label::a0b0_a2b1, Label::a0b1_a2b0,
                                              Label::a1b0_a2b1, Label::a1b1_a2b0}};
    for (const Label label : output_a01_b2)
    {
        checked.insert(label);
        assert(!filter.filter_for1(label, left.first));
        assert(filter.filter_for1(label, right.first));
        assert(!filter.filter_for32(label, left.first, intermediate, right));
        assert(filter.filter_for32(label, left.first, intermediate, shifted_right));
        assert(filter.filter_for32(label, right.first, intermediate, left));
    }

    // GPU output: (Aa01, Ab01); Ab2 is only an intermediate atom.
    const std::array<Label, 4> output_a01_b01_a2b2{{Label::a0b0_a2b2, Label::a0b1_a2b2,
                                                     Label::a1b0_a2b2, Label::a1b1_a2b2}};
    for (const Label label : output_a01_b01_a2b2)
    {
        checked.insert(label);
        assert(!filter.filter_for1(label, left.first));
        assert(filter.filter_for1(label, right.first));
        assert(!filter.filter_for32(label, left.first, intermediate, right));
        assert(filter.filter_for32(label, left.first, intermediate, shifted_right));
        assert(filter.filter_for32(label, right.first, intermediate, left));
    }

    // GPU output: (Aa01, Ab01); this group screens the pair in filter_for2.
    const std::array<Label, 4> output_a01_b01_a01b2{{Label::a0b2_a2b0, Label::a0b2_a2b1,
                                                      Label::a1b2_a2b0, Label::a1b2_a2b1}};
    for (const Label label : output_a01_b01_a01b2)
    {
        checked.insert(label);
        assert(!filter.filter_for1(label, right));
        assert(filter.filter_for1(label, left));
        assert(!filter.filter_for1(label, left.first));
        assert(!filter.filter_for2(label, left.first, right));
        assert(filter.filter_for2(label, left.first, shifted_right));
        assert(filter.filter_for2(label, right.first, left));
    }

    assert(checked.size() == RI::Label::array_ab_ab.size());
    for (const Label label : RI::Label::array_ab_ab)
    {
        assert(checked.count(label) == 1);
    }
}
