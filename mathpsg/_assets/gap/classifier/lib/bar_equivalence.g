#############################################################################
## Lean Task-5 data export for the physical solver.
#############################################################################

MathPSGClassifierTask5ResolutionChain := function(
    resolution, degree, word, normalForm
)
    local terms, letter, coefficient, basis, element;
    terms := [];
    for letter in word do
        if Length(letter) = 3 then
            coefficient := letter[1];
            basis := letter[2];
            element := resolution!.elts[letter[3]];
        else
            coefficient := SignInt(letter[1]);
            basis := AbsoluteValue(letter[1]);
            element := resolution!.elts[letter[2]];
        fi;
        Add(terms, rec(
            basis_id := Concatenation(
                "c", String(degree), ":", String(basis - 1)
            ),
            coefficient := coefficient,
            element := normalForm(element)
        ));
    od;
    Sort(terms, function(left, right)
        return left.basis_id < right.basis_id
            or (left.basis_id = right.basis_id and left.element < right.element);
    end);
    return rec(terms := terms);
end;

MathPSGClassifierTask5BarChain := function(word, elementName)
    local terms, letter;
    terms := [];
    for letter in word do
        Add(terms, rec(
            coefficient := letter[1],
            group_tuple := List(
                letter{[3..Length(letter)]}, elementName
            ),
            left_element := elementName(letter[2])
        ));
    od;
    terms := Filtered(terms, term -> not "1" in term.group_tuple);
    Sort(terms, function(left, right)
        if left.left_element <> right.left_element then
            return left.left_element < right.left_element;
        fi;
        return String(left.group_tuple) < String(right.group_tuple);
    end);
    return terms;
end;

# The solver pulls local cochains back only in degrees one and two, and uses
# phi only to coordinate singleton nonidentity grade queries for U(1) Weyl
# shifts.  No homotopies or replay closure are part of this computation.
MathPSGClassifierTask5LocalBarData := function(
    resolution, literalElements, labels, normalForm
)
    local comparison, identityIndex, psi, degree, basis,
          phi, index, element, barWord;
    comparison := BarResolutionEquivalence(resolution);
    identityIndex := Position(
        resolution!.elts, Identity(resolution!.group)
    );
    psi := [[], [], []];
    for degree in [1..2] do
        for basis in [1..resolution!.dimension(degree)] do
            Add(psi[degree + 1], rec(
                basis_id := Concatenation(
                    "c", String(degree), ":", String(basis - 1)
                ),
                image := MathPSGClassifierTask5BarChain(
                    comparison!.psi(
                        degree, [[1, basis, identityIndex]]
                    ),
                    normalForm
                )
            ));
        od;
    od;
    phi := [];
    for index in [2..Length(literalElements)] do
        element := literalElements[index];
        barWord := [[1, Identity(resolution!.group), element]];
        Add(phi, rec(
            group_tuple := [labels[index]],
            image := MathPSGClassifierTask5ResolutionChain(
                resolution, 1, comparison!.phi(1, barWord), normalForm
            )
        ));
    od;
    return rec(phi_on_queries := phi, psi_on_basis := psi);
end;

MathPSGClassifierTask5PrepareLiteralInclusionBatch := function(input)
    local matrices, pureTranslation, request, conversion,
          ambientSpatialResolution, timeGroup, timeResolution, ambient,
          ambientSpatialNormal, normalAmbient;
    matrices := List(
        input.action.affine_generators, MathPSGClassifierAffineRight
    );
    pureTranslation := ForAll(
        matrices,
        matrix -> matrix{[1..3]}{[1..3]} = IdentityMat(3)
    );
    request := rec(action := input.action);
    if pureTranslation then
        conversion := MathPSGClassifierPureTranslationConversion(
            request, matrices
        );
    else
        conversion := MathPSGClassifierCrystConversion(matrices);
    fi;
    ambientSpatialNormal := element ->
        MathPSGClassifierTask5PcpWord(conversion.pcp, element);
    ambientSpatialResolution := MathPSGClassifierTask5AmbientResolution(
        conversion.pcp_group, false
    );
    if input.time_reversal then
        timeGroup := CyclicGroup(IsPermGroup, 2);
        timeResolution := MathPSGClassifierTask5FiniteResolution(timeGroup);
        ambient := ResolutionDirectProduct(
            ambientSpatialResolution, timeResolution
        );
        normalAmbient := element ->
            MathPSGClassifierTask5DirectProductNormalForm(
                ambient, ambientSpatialNormal, "T", element
            );
    else
        timeResolution := fail;
        ambient := ambientSpatialResolution;
        normalAmbient := ambientSpatialNormal;
    fi;
    return rec(
        ambient := ambient,
        conversion := conversion,
        normal_ambient := normalAmbient,
        time_resolution := timeResolution
    );
end;

MathPSGClassifierTask5LiteralInclusionMemberRaw := function(input, context)
    local conversion, literalMatrices, literalSpatialElements,
          literalSpatialGroup, spatialInclusion, timeResolution,
          localSpatialResolution, ambient, localResolution, inclusion,
          labels, literalElements, groupId, gradedLabels,
          localSpatialEmbedding, localTimeEmbedding, localTimeElement,
          localSpatialNormal, normalAmbient, normalLocal, finiteTable;
    conversion := context.conversion;
    ambient := context.ambient;
    timeResolution := context.time_resolution;
    literalMatrices := List(
        input.inclusion.literal_elements, MathPSGClassifierAffineRight
    );
    literalSpatialElements := List(literalMatrices, conversion.image);
    literalSpatialGroup := Group(literalSpatialElements);
    spatialInclusion := GroupHomomorphismByFunction(
        literalSpatialGroup, conversion.pcp_group, element -> element
    );
    labels := ShallowCopy(input.element_labels);
    localSpatialNormal := element ->
        labels[Position(literalSpatialElements, element)];
    localSpatialResolution := MathPSGClassifierTask5FiniteResolution(
        literalSpatialGroup
    );
    if input.time_reversal then
        localResolution := ResolutionDirectProduct(
            localSpatialResolution, timeResolution
        );
        normalAmbient := context.normal_ambient;
        normalLocal := element ->
            MathPSGClassifierTask5DirectProductNormalForm(
                localResolution, localSpatialNormal, "T", element
            );
        inclusion := MathPSGClassifierTask5DirectProductInclusion(
            localResolution, ambient, spatialInclusion
        );
        localSpatialEmbedding := Embedding(localResolution!.group, 1);
        localTimeEmbedding := Embedding(localResolution!.group, 2);
        localTimeElement := ImagesRepresentative(
            localTimeEmbedding,
            GeneratorsOfGroup(Range(localResolution!.secondProjection))[1]
        );
        literalElements := List(
            literalSpatialElements,
            element -> ImagesRepresentative(localSpatialEmbedding, element)
        );
        Append(
            literalElements,
            List(
                ShallowCopy(literalElements),
                element -> element * localTimeElement
            )
        );
        gradedLabels := List(
            literalElements{[
                Length(input.element_labels) + 1..Length(literalElements)
            ]},
            normalLocal
        );
        Append(labels, gradedLabels);
    else
        localResolution := localSpatialResolution;
        inclusion := spatialInclusion;
        normalAmbient := context.normal_ambient;
        normalLocal := localSpatialNormal;
        literalElements := literalSpatialElements;
    fi;
    groupId := input.finite_group_id;
    if input.time_reversal then
        groupId := Concatenation(groupId, "+onsite-T");
    fi;
    finiteTable := rec(
        element_order := labels,
        group_id := groupId,
        identity_index := 0,
        multiplication_table := List(
            literalElements,
            first -> List(
                literalElements,
                second -> Position(literalElements, first * second) - 1
            )
        )
    );
    return rec(
        bar_equivalence := MathPSGClassifierTask5LocalBarData(
            localResolution, literalElements, labels, normalLocal
        ),
        finite_group := finiteTable,
        restriction_maps := MathPSGClassifierTask5InclusionMaps(
            localResolution, ambient, inclusion, normalAmbient
        ),
        source := MathPSGClassifierTask5RawResolution(
            localResolution, normalLocal
        ),
        source_element_images := List(
            literalElements,
            element -> normalAmbient(ImageElm(inclusion, element))
        ),
        target := MathPSGClassifierTask5RawResolution(
            ambient, normalAmbient
        )
    );
end;
