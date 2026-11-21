// ===================
//  Author: maki49
//  date: 2026.11.10
// ===================

#pragma once

#ifdef __GPU_RI

//class Filter_Atom_Symmetry: public Filter_Atom<TA, std::pair<TA, TC>>
//{

    bool filter_GPU_for11(const Label::ab_ab &label, const TA &A1) const override
    {
        switch (Label_Tools::to_Aab_Aab(label))
        {
            case Label::Aab_Aab::a01b01_a01b01: // Aa01 is intermediate.
            case Label::Aab_Aab::a01b01_a01b2:
                return false;
            default: // Aa01 is the left output atom.
                return !symmetry.is_Aa_in_irreducible_sector(A1);
        }
    }

    bool filter_GPU_for11(const Label::ab_ab &, const TAC &A1) const override
    {
        // The first loop with a cell-bearing atom visits output Ab01.
        return !symmetry.is_Ab_in_irreducible_sector(A1.first);
    }

    bool filter_GPU_for21(const Label::ab_ab &label, const TA &A1, const TAC &A2) const override
    {
        // Only the final stage of a01b2_a2b01 has both outputs at level 2.
        if (Label_Tools::to_Aab_Aab(label) == Label::Aab_Aab::a01b2_a2b01)
            return !symmetry.in_irreducible_sector(A1, A2);
        return false;
    }

    bool filter_GPU_for32(const Label::ab_ab &label,
                      const TA &A1, const TAC &A2, const TAC &A3) const override
    {
        switch (Label_Tools::to_Aab_Aab(label))
        {
            case Label::Aab_Aab::a01b01_a01b01: // Aa01, Aa2, Ab2
                return !symmetry.in_irreducible_sector(A2, A3);
            case Label::Aab_Aab::a01b01_a01b2: // Aa01, Ab01, Aa2
                return !symmetry.in_irreducible_sector(A3, A2);
            case Label::Aab_Aab::a01b01_a2b01: // Aa01, Ab01, Ab2
            case Label::Aab_Aab::a01b2_a2b01: // Aa01, Aa2, Ab01
                return !symmetry.in_irreducible_sector(A1, A3);
            case Label::Aab_Aab::a01b01_a2b2: // Aa01, Ab01, Ab2
                return !symmetry.in_irreducible_sector(A1, A2);
            default:
                throw std::invalid_argument("Unknown GPU symmetry label");
        }
    }

//};


#endif