#############################################################################
## Exact Cryst right-action to PCP conversion with replayable word witnesses.
#############################################################################

MathPSGClassifierPcpWord := function(exponents)
    local pieces, index, exponent, factor;
    pieces := [];
    for index in [1..Length(exponents)] do
        exponent := exponents[index];
        if exponent <> 0 then
            factor := Concatenation("p", String(index));
            if exponent <> 1 then
                Append(factor, Concatenation("^", String(exponent)));
            fi;
            Add(pieces, factor);
        fi;
    od;
    if IsEmpty(pieces) then return "1"; fi;
    return JoinStringsWithSeparator(pieces, "*");
end;

MathPSGClassifierAffineWord := function(freeWord)
    local external, result, index;
    external := ExtRepOfObj(freeWord);
    result := [];
    for index in [1, 3..Length(external) - 1] do
        Add(result, [external[index] - 1, external[index + 1]]);
    od;
    return result;
end;

MathPSGClassifierPureTranslationConversion := function(request, matrices)
    local basis, inverseBasis, pcpGroup, pcp, pcpGenerators, images,
          coefficients, matrix, translation, freeGroup, epimorphism,
          preimages, imageFunction, generatorMatrices, index;
    basis := List(
        request.action.translation_basis,
        row -> List(row, MathPSGClassifierRational)
    );
    inverseBasis := Inverse(basis);
    pcpGroup := AbelianPcpGroup([0, 0, 0]);
    pcp := Pcp(pcpGroup);
    pcpGenerators := GeneratorsOfPcp(pcp);
    images := [];
    for matrix in matrices do
        if matrix{[1..3]}{[1..3]} <> IdentityMat(3) then
            Error("pure translation conversion received a point operation");
        fi;
        translation := matrix[4]{[1..3]};
        coefficients := translation * TransposedMat(inverseBasis);
        if not ForAll(coefficients, IsInt) then
            Error("translation generator is outside the declared lattice");
        fi;
        Add(images, Product([1..3], index -> pcpGenerators[index]^coefficients[index]));
    od;
    freeGroup := FreeGroup(Length(matrices));
    epimorphism := GroupHomomorphismByImages(
        freeGroup, pcpGroup, GeneratorsOfGroup(freeGroup), images
    );
    preimages := List(
        pcpGenerators,
        generator -> MathPSGClassifierAffineWord(
            PreImagesRepresentative(epimorphism, generator)
        )
    );
    generatorMatrices := [];
    for index in [1..3] do
        Add(generatorMatrices, IdentityMat(4));
        generatorMatrices[index][4]{[1..3]} := List(
            [1..3], row -> basis[row][index]
        );
    od;
    imageFunction := function(element)
        local translation, exponents;
        if element{[1..3]}{[1..3]} <> IdentityMat(3) then
            Error("point operation is not in the pure translation ambient group");
        fi;
        translation := element[4]{[1..3]};
        exponents := translation * TransposedMat(inverseBasis);
        if not ForAll(exponents, IsInt) then
            Error("affine element is outside the declared translation lattice");
        fi;
        return Product([1..3], index -> pcpGenerators[index]^exponents[index]);
    end;
    return rec(
        image := imageFunction,
        pcp := pcp,
        pcp_group := pcpGroup,
        pcp_generators := pcpGenerators,
        pcp_generator_affines := List(
            generatorMatrices, MathPSGClassifierAffineColumn
        ),
        pcp_generator_preimages := preimages
    );
end;

MathPSGClassifierCrystConversion := function(matrices)
    local group, isomorphism, pcpGroup, pcp, pcpGenerators, images,
          freeGroup, epimorphism, preimages, imageFunction, preimageMatrices;
    group := Group(matrices);
    SetIsAffineCrystGroupOnRight(group, true);
    SetIsSpaceGroup(group, true);
    isomorphism := IsomorphismPcpGroup(group);
    if isomorphism = fail then Error("Cryst-to-PCP isomorphism failed"); fi;
    pcpGroup := Image(isomorphism);
    pcp := Pcp(pcpGroup);
    pcpGenerators := GeneratorsOfPcp(pcp);
    images := List(matrices, matrix -> Image(isomorphism, matrix));
    freeGroup := FreeGroup(Length(matrices));
    epimorphism := GroupHomomorphismByImages(
        freeGroup, pcpGroup, GeneratorsOfGroup(freeGroup), images
    );
    preimages := List(
        pcpGenerators,
        generator -> MathPSGClassifierAffineWord(
            PreImagesRepresentative(epimorphism, generator)
        )
    );
    preimageMatrices := List(
        pcpGenerators,
        generator -> PreImagesRepresentative(isomorphism, generator)
    );
    imageFunction := element -> Image(isomorphism, element);
    return rec(
        image := imageFunction,
        pcp := pcp,
        pcp_group := pcpGroup,
        pcp_generators := pcpGenerators,
        pcp_generator_affines := List(
            preimageMatrices, MathPSGClassifierAffineColumn
        ),
        pcp_generator_preimages := preimages
    );
end;

MathPSGClassifierTransportInclusion := function(inclusion, conversion)
    local matrices, images, order, table, inverses, left, right, position;
    matrices := List(inclusion.literal_elements, MathPSGClassifierAffineRight);
    images := List(
        matrices,
        matrix -> MathPSGClassifierPcpWord(
            ExponentsByPcp(conversion.pcp, conversion.image(matrix))
        )
    );
    order := Length(matrices);
    table := [];
    for left in [1..order] do
        Add(table, []);
        for right in [1..order] do
            position := Position(matrices, matrices[left] * matrices[right]);
            if position = fail then Error("literal stabilizer is not closed"); fi;
            Add(table[left], position - 1);
        od;
    od;
    inverses := [];
    for left in [1..order] do
        position := Position(matrices, matrices[left]^-1);
        if position = fail then Error("literal stabilizer lacks an inverse"); fi;
        Add(inverses, position - 1);
    od;
    return rec(
        inclusion_id := inclusion.inclusion_id,
        inverse_indices := inverses,
        literal_element_digest := inclusion.literal_element_digest,
        literal_elements := inclusion.literal_elements,
        literal_stabilizer_digest := inclusion.literal_stabilizer_digest,
        multiplication_table := table,
        pcp_images := images
    );
end;

MathPSGClassifierRuntimeSmoke := function(request, conversion)
    local ambientResolution, inclusion, images, stabilizer,
          finiteResolution, inclusionMap, chainMap;
    ambientResolution := ResolutionAlmostCrystalGroup(conversion.pcp_group, 1);
    if ambientResolution = fail
       or ambientResolution!.dimension(0) < 1
       or ambientResolution!.dimension(1) < 1 then
        Error("almost-crystallographic resolution smoke failed");
    fi;
    for inclusion in request.inclusions do
        images := List(
            inclusion.literal_elements,
            element -> conversion.image(MathPSGClassifierAffineRight(element))
        );
        stabilizer := Group(images);
        if not IsFinite(stabilizer)
           or Size(stabilizer) <> Length(images) then
            Error("finite stabilizer resolution smoke received a nonliteral subgroup");
        fi;
        finiteResolution := ResolutionFiniteGroup(stabilizer, 1);
        if finiteResolution = fail
           or finiteResolution!.dimension(0) < 1 then
            Error("finite stabilizer resolution smoke failed");
        fi;
        inclusionMap := GroupHomomorphismByFunction(
            stabilizer, conversion.pcp_group, element -> element
        );
        chainMap := EquivariantChainMap(
            finiteResolution, ambientResolution, inclusionMap
        );
        if chainMap = fail or not IsHapEquivariantChainMap(chainMap) then
            Error("equivariant inclusion chain-map smoke failed");
        fi;
    od;
    return true;
end;

MathPSGClassifierBuildCertificate := function(request)
    local matrices, pureTranslation, conversion, affineImages, basis,
          translationMatrices, translationImages, transported, roundtrip,
          index, left, right, core, algorithmBytes;
    matrices := List(
        request.action.affine_generators,
        MathPSGClassifierAffineRight
    );
    pureTranslation := ForAll(
        matrices,
        matrix -> matrix{[1..3]}{[1..3]} = IdentityMat(3)
    );
    if pureTranslation then
        conversion := MathPSGClassifierPureTranslationConversion(request, matrices);
    else
        conversion := MathPSGClassifierCrystConversion(matrices);
    fi;
    affineImages := List(
        matrices,
        matrix -> MathPSGClassifierPcpWord(
            ExponentsByPcp(conversion.pcp, conversion.image(matrix))
        )
    );
    basis := List(
        request.action.translation_basis,
        row -> List(row, MathPSGClassifierRational)
    );
    translationMatrices := [];
    for index in [1..3] do
        Add(translationMatrices, IdentityMat(4));
        translationMatrices[index][4]{[1..3]} := List([1..3], row -> basis[row][index]);
    od;
    translationImages := List(
        translationMatrices,
        matrix -> MathPSGClassifierPcpWord(
            ExponentsByPcp(conversion.pcp, conversion.image(matrix))
        )
    );
    transported := List(
        request.inclusions,
        inclusion -> MathPSGClassifierTransportInclusion(inclusion, conversion)
    );
    if IsBound(GAPInfo.SystemEnvironment.MATHPSG_CLASSIFIER_RUNTIME_SMOKE)
       and GAPInfo.SystemEnvironment.MATHPSG_CLASSIFIER_RUNTIME_SMOKE = "1" then
        if MathPSGClassifierRuntimeSmoke(request, conversion) <> true then
            Error("classifier runtime smoke failed");
        fi;
    fi;
    roundtrip := List(
        [1..Length(matrices)],
        generator -> [[generator - 1, 1]]
    );
    for left in [1..Length(matrices)] do
        for right in [1..Length(matrices)] do
            if left <> right then
                Add(roundtrip, [[left - 1, 1], [right - 1, 1]]);
            fi;
        od;
    od;
    if Length(matrices) >= 2 then Add(roundtrip, [[1, -1], [0, 2]]); fi;
    algorithmBytes := Concatenation(
        "mathpsg-affine-pcp-conversion-v1|",
        StringFile("gap/classifier/lib/protocol.g"), "|",
        StringFile("gap/classifier/lib/affine_pcp.g")
    );
    core := rec(
        affine_generator_images := affineImages,
        catalogue_action_digest := request.action.action_digest,
        conversion_algorithm_digest := Concatenation(
            "sha256:", MathPSGClassifierHexSHA256(algorithmBytes)
        ),
        pcp_normal_form := rec(
            generator_affines := conversion.pcp_generator_affines,
            relative_orders := RelativeOrdersOfPcp(conversion.pcp)
        ),
        pcp_generator_preimages := conversion.pcp_generator_preimages,
        roundtrip_words := roundtrip,
        translation_basis_images := translationImages,
        transported_stabilizers := transported
    );
    core.certificate_digest := MathPSGClassifierDigest(
        "affine-pcp-isomorphism-certificate-v1", core
    );
    return core;
end;
