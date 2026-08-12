#############################################################################
## Cryst right-action to the PCP group used by HAP.
#############################################################################

MathPSGClassifierPureTranslationConversion := function(request, matrices)
    local basis, inverseBasis, pcpGroup, pcp, pcpGenerators, imageFunction;
    basis := List(
        request.action.translation_basis,
        row -> List(row, MathPSGClassifierRational)
    );
    inverseBasis := Inverse(basis);
    pcpGroup := AbelianPcpGroup([0, 0, 0]);
    pcp := Pcp(pcpGroup);
    pcpGenerators := GeneratorsOfPcp(pcp);
    imageFunction := function(element)
        local translation, exponents, index;
        translation := element[4]{[1..3]};
        exponents := translation * TransposedMat(inverseBasis);
        return Product(
            [1..3],
            index -> pcpGenerators[index]^exponents[index]
        );
    end;
    return rec(
        image := imageFunction,
        pcp := pcp,
        pcp_group := pcpGroup
    );
end;

MathPSGClassifierCrystConversion := function(matrices)
    local group, isomorphism, pcpGroup;
    group := Group(matrices);
    SetIsAffineCrystGroupOnRight(group, true);
    SetIsSpaceGroup(group, true);
    isomorphism := IsomorphismPcpGroup(group);
    if isomorphism = fail then Error("Cryst-to-PCP conversion failed"); fi;
    pcpGroup := Image(isomorphism);
    return rec(
        image := element -> Image(isomorphism, element),
        pcp := Pcp(pcpGroup),
        pcp_group := pcpGroup
    );
end;
