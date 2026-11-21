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

    // The second first-level loop always visits Aa01.
    bool filter_GPU_for12(const Label::ab_ab &label, const TA &A1) const override
    {
        return filter_GPU_for11(label, A1);
    }

    // Only a01b2_a2b01 has a third stage; Aa01 is its left output.
    bool filter_GPU_for13(const Label::ab_ab &, const TA &A1) const override
    {
        return !symmetry.is_Aa_in_irreducible_sector(A1);
    }

    bool filter_GPU_for22(const Label::ab_ab &label, const TA &A1, const TAC &A2) const override
    {
        // Stage 2: Aa01, Ab01 are both outputs only in a01b01_a2b2.
        // In a01b2_a2b01 A2 is Aa2, an unrestricted summation atom.
        if (Label_Tools::to_Aab_Aab(label) == Label::Aab_Aab::a01b01_a2b2)
            return !symmetry.in_irreducible_sector(A1, A2);
        return false;
    }

    bool filter_GPU_for23(const Label::ab_ab &, const TA &A1, const TAC &A2) const override
    {
        // Final two-loop stage of a01b2_a2b01: Aa01, Ab01.
        return !symmetry.in_irreducible_sector(A1, A2);
    }

    // for21 and for31 retain the base no-op: never screen a summation pair.

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