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

MathPSGClassifierPcpRelatorRowsMod2 := function(pcp)
    local orders, generators, count, rows, earlier, later, reduced, row,
          index;
    orders := RelativeOrdersOfPcp(pcp);
    generators := GeneratorsOfPcp(pcp);
    count := Length(generators);
    rows := [];

    # Power and collection relations are the complete PCP presentation.
    # Only exponent sums modulo two are needed for Hom(G, Z2).
    for later in [1..count] do
        if orders[later] <> 0 then
            reduced := ExponentsByPcp(
                pcp, generators[later]^orders[later]
            );
            row := List([1..count], index -> -reduced[index]);
            row[later] := row[later] + orders[later];
            Add(rows, List(row, value -> value mod 2));
        fi;
    od;
    for later in [1..count] do
        for earlier in [1..later - 1] do
            reduced := ExponentsByPcp(
                pcp, generators[later] * generators[earlier]
            );
            row := List([1..count], index -> -reduced[index]);
            row[earlier] := row[earlier] + 1;
            row[later] := row[later] + 1;
            Add(rows, List(row, value -> value mod 2));
        od;
    od;
    return rows;
end;
