#############################################################################
## On-demand HAP finite-resolution / normalized-bar comparison export.
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
    return rec(degree := degree, terms := terms);
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
    # Pass from HAP's bar model to the normalized quotient: every tuple
    # containing the identity represents a degenerate basis element and is
    # exactly zero there.
    terms := Filtered(terms, term -> not "1" in term.group_tuple);
    Sort(terms, function(left, right)
        if left.left_element <> right.left_element then
            return left.left_element < right.left_element;
        fi;
        return String(left.group_tuple) < String(right.group_tuple);
    end);
    return terms;
end;

MathPSGClassifierTask5BarWord := function(group, elements)
    return [Concatenation([1, Identity(group)], elements)];
end;

MathPSGClassifierTask5CollectResolutionWord := function(word)
    local result, term, found;
    result := [];
    for term in word do
        found := First(result, item ->
            item[2] = term[2] and item[3] = term[3]
        );
        if found = fail then Add(result, ShallowCopy(term));
        else found[1] := found[1] + term[1];
        fi;
    od;
    return Filtered(result, item -> item[1] <> 0);
end;

MathPSGClassifierTask5ApplyResolutionHomotopy := function(
    resolution, degree, word
)
    local result, term, image, letter;
    result := [];
    for term in word do
        image := resolution!.homotopy(degree, [term[2], term[3]]);
        for letter in image do
            Add(result, [
                term[1] * SignInt(letter[1]),
                AbsoluteValue(letter[1]),
                letter[2]
            ]);
        od;
    od;
    return MathPSGClassifierTask5CollectResolutionWord(result);
end;

MathPSGClassifierTask5FiniteHomotopyOnBasis := function(
    resolution, equivalence, degree, basis, rawHomotopy
)
    local identityIndex, input, psi, phiPsi, defect, boundary,
          letter, lifted, term;
    identityIndex := Position(resolution!.elts, Identity(resolution!.group));
    input := [[1, basis, identityIndex]];
    psi := equivalence!.psi(degree, input);
    phiPsi := equivalence!.phi(degree, psi);
    defect := Concatenation(
        [[1, basis, identityIndex]],
        List(phiPsi, item -> [-item[1], item[2], item[3]])
    );
    if degree > 0 then
        boundary := resolution!.boundary(degree, basis);
        for letter in boundary do
            lifted := StructuralCopy(
                rawHomotopy[degree][AbsoluteValue(letter[1])]
            );
            for term in lifted do
                term[1] := -SignInt(letter[1]) * term[1];
                term[3] := Position(
                    resolution!.elts,
                    resolution!.elts[letter[2]] * resolution!.elts[term[3]]
                );
            od;
            Append(defect, lifted);
        od;
    fi;
    return MathPSGClassifierTask5ApplyResolutionHomotopy(
        resolution, degree,
        MathPSGClassifierTask5CollectResolutionWord(defect)
    );
end;

MathPSGClassifierTask5BarEquivalence := function(
    resolution, finiteGroup, phiTuples, homotopyTuples,
    normalForm, elementByName
)
    local equivalence, identity, psi, homotopy, rawHomotopy, degree, basis,
          tuple, elements, barWord, core, result;
    equivalence := BarResolutionEquivalence(resolution);
    identity := Position(resolution!.elts, Identity(resolution!.group));
    psi := [];
    homotopy := [];
    rawHomotopy := [];
    for degree in [0..4] do
        Add(psi, []);
        Add(homotopy, []);
        Add(rawHomotopy, []);
        for basis in [1..resolution!.dimension(degree)] do
            Add(psi[degree + 1], rec(
                basis_id := Concatenation(
                    "c", String(degree), ":", String(basis - 1)
                ),
                image := MathPSGClassifierTask5BarChain(
                    equivalence!.psi(degree, [[1, basis, identity]]), normalForm
                )
            ));
            Add(
                rawHomotopy[degree + 1],
                MathPSGClassifierTask5FiniteHomotopyOnBasis(
                    resolution, equivalence, degree, basis, rawHomotopy
                )
            );
            Add(homotopy[degree + 1], rec(
                basis_id := Concatenation(
                    "c", String(degree), ":", String(basis - 1)
                ),
                image := MathPSGClassifierTask5ResolutionChain(
                    resolution, degree + 1,
                    rawHomotopy[degree + 1][basis], normalForm
                )
            ));
        od;
    od;
    core := rec(
        bar_homotopy_algorithm := "hap-1.70-bar-resolution-equivalence-equiv",
        finite_group := finiteGroup,
        phi_algorithm := "hap-1.70-bar-resolution-equivalence-phi",
        phi_on_queries := [],
        psi_on_basis := psi,
        queried_bar_tuples := homotopyTuples,
        resolution_homotopy_on_basis := homotopy
    );
    for tuple in phiTuples do
        elements := List(tuple, elementByName);
        barWord := MathPSGClassifierTask5BarWord(resolution!.group, elements);
        Add(core.phi_on_queries, rec(
            group_tuple := tuple,
            image := MathPSGClassifierTask5ResolutionChain(
                resolution, Length(tuple),
                equivalence!.phi(Length(tuple), barWord), normalForm
            ),
            bar_homotopy := fail
        ));
        if tuple in homotopyTuples then
            # HAP 1.70's implementation has the opposite sign from the
            # equation printed in its manual: its `equiv` satisfies
            # dH+Hd = psi*phi-id.  Negate the trace so this certificate
            # literally witnesses id-psi*phi = dH+Hd.
            barWord := equivalence!.equiv(Length(tuple), barWord);
            for elements in barWord do
                elements[1] := -elements[1];
            od;
            core.phi_on_queries[Length(core.phi_on_queries)].bar_homotopy :=
                MathPSGClassifierTask5BarChain(
                    barWord, normalForm
                );
        fi;
    od;
    return core;
end;

MathPSGClassifierTask5TargetBarEquivalence := function(
    resolution, inclusionQueries, inclusionBasis, normalForm
)
    local comparison, identity, seeds, degree, basis, word, term, query,
          queries, pending, current, candidate, index, namedQueries,
          elementPairs, element, name, elementByName, core, requiredBasis,
          requiredPending, required, boundary;
    comparison := BarResolutionEquivalence(resolution);
    identity := Identity(resolution!.group);
    requiredBasis := [];
    requiredPending := ShallowCopy(inclusionBasis);
    while not IsEmpty(requiredPending) do
        required := Remove(requiredPending);
        if Position(requiredBasis, required) <> fail then continue; fi;
        Add(requiredBasis, required);
        if required[1] > 0 then
            boundary := resolution!.boundary(required[1], required[2]);
            for term in boundary do
                Add(requiredPending, [
                    required[1] - 1, AbsoluteValue(term[1])
                ]);
            od;
        fi;
    od;
    Sort(requiredBasis, function(left, right)
        return left[1] < right[1]
            or (left[1] = right[1] and left[2] < right[2]);
    end);
    seeds := ShallowCopy(inclusionQueries);
    for required in requiredBasis do
            degree := required[1]; basis := required[2];
            word := comparison!.psi(degree, [[
                1, basis,
                Position(resolution!.elts, identity)
            ]]);
            for term in word do
                query := term{[3..Length(term)]};
                if not identity in query then Add(seeds, query); fi;
            od;
    od;

    # This is the exact finite domain needed to replay phi as a chain map and
    # H as a bar homotopy on every transported source-psi term and every
    # target-psi basis term.  Closing under normalized bar boundaries makes
    # every recursive identity independently replayable in Python.
    queries := [[]];
    pending := ShallowCopy(seeds);
    while not IsEmpty(pending) do
        current := Remove(pending);
        if identity in current or Position(queries, current) <> fail then
            continue;
        fi;
        Add(queries, current);
        if IsEmpty(current) then continue; fi;
        Add(pending, current{[2..Length(current)]});
        Add(pending, current{[1..Length(current) - 1]});
        for index in [1..Length(current) - 1] do
            candidate := Concatenation(
                current{[1..index - 1]},
                [current[index] * current[index + 1]],
                current{[index + 2..Length(current)]}
            );
            if not identity in candidate then Add(pending, candidate); fi;
        od;
    od;

    namedQueries := List(
        queries, tuple -> List(tuple, normalForm)
    );
    Sort(namedQueries, function(left, right)
        if Length(left) <> Length(right) then
            return Length(left) < Length(right);
        fi;
        return String(left) < String(right);
    end);
    elementPairs := [];
    for query in queries do
        for element in query do
            name := normalForm(element);
            if First(elementPairs, pair -> pair[1] = name) = fail then
                Add(elementPairs, [name, element]);
            fi;
        od;
    od;
    elementByName := function(elementName)
        local pair;
        pair := First(elementPairs, item -> item[1] = elementName);
        if pair = fail then Error("target bar query element is absent"); fi;
        return pair[2];
    end;
    core := MathPSGClassifierTask5BarEquivalence(
        resolution, fail, namedQueries, namedQueries,
        normalForm, elementByName
    );
    Unbind(core.finite_group);
    for degree in [0..4] do
        core.psi_on_basis[degree + 1] := List(
            Filtered(requiredBasis, item -> item[1] = degree),
            item -> core.psi_on_basis[degree + 1][item[2]]
        );
        core.resolution_homotopy_on_basis[degree + 1] := List(
            Filtered(requiredBasis, item -> item[1] = degree),
            item -> core.resolution_homotopy_on_basis[degree + 1][item[2]]
        );
    od;
    return core;
end;

MathPSGClassifierTask5ObservedBackend := function()
    local result, manifestPath, lockPath, manifestBytes, lockBytes;
    result := rec(
        backend_lock_digest := fail,
        execution_mode := "diagnostic-local",
        gap_version := GAPInfo.Version,
        packages := [
            rec(name := "Cryst", version := PackageInfo("cryst")[1].Version),
            rec(name := "GAP", version := GAPInfo.Version),
            rec(name := "HAP", version := PackageInfo("hap")[1].Version),
            rec(name := "HAPcryst", version := PackageInfo("hapcryst")[1].Version)
        ],
        runtime_manifest_digest := fail,
        schema_version := 1
    );
    manifestPath := "/opt/mathpsg/classifier-gap/runtime-provenance.json";
    lockPath := "/opt/mathpsg/classifier-gap.lock.json";
    if IsExistingFile(manifestPath) and IsExistingFile(lockPath) then
        manifestBytes := StringFile(manifestPath);
        lockBytes := StringFile(lockPath);
        result.execution_mode := "locked-oci";
        result.backend_lock_digest := Concatenation(
            "sha256:", MathPSGClassifierHexSHA256(lockBytes)
        );
        result.runtime_manifest_digest := Concatenation(
            "sha256:", MathPSGClassifierHexSHA256(manifestBytes)
        );
    fi;
    return result;
end;

MathPSGClassifierTask5ValidateLiteralInclusionInput := function(input, encoded)
    local actionCore, inclusion, inputCore, generator, element,
          literalMatrices, left, right;
    if not IsRecord(input)
       or MathPSGClassifierJson(input) <> encoded
       or not MathPSGClassifierRequireFields(input, [
           "action", "element_labels", "finite_group_id", "inclusion",
           "input_digest", "record_type", "schema_version", "time_reversal"
       ])
       or input.record_type <> "task5-literal-inclusion-export-input"
       or input.schema_version <> 1
       or not input.time_reversal in [true, false]
       or not IsString(input.finite_group_id)
       or IsEmpty(input.finite_group_id)
       or not MathPSGClassifierIsDigest(input.input_digest) then
        Error("invalid Task5 literal-inclusion export envelope");
    fi;
    if not MathPSGClassifierRequireFields(input.action, [
        "action_digest", "affine_generators", "translation_basis"
    ])
       or not IsList(input.action.affine_generators)
       or IsEmpty(input.action.affine_generators)
       or not IsList(input.action.translation_basis)
       or Length(input.action.translation_basis) <> 3
       or not ForAll(
           input.action.translation_basis,
           row -> IsList(row) and Length(row) = 3
       )
       or DeterminantMat(List(
           input.action.translation_basis,
           row -> List(row, MathPSGClassifierRational)
       )) = 0 then
        Error("invalid Task5 affine action");
    fi;
    for generator in input.action.affine_generators do
        MathPSGClassifierAffineRight(generator);
    od;
    actionCore := rec(
        affine_generators := input.action.affine_generators,
        translation_basis := input.action.translation_basis
    );
    if input.action.action_digest <>
       MathPSGClassifierDigest("certified-space-group-action-v1", actionCore) then
        Error("Task5 affine action digest mismatch");
    fi;
    inclusion := input.inclusion;
    if not MathPSGClassifierRequireFields(inclusion, [
        "inclusion_id", "literal_element_digest", "literal_elements",
        "literal_stabilizer_digest"
    ])
       or not IsString(inclusion.inclusion_id)
       or IsEmpty(inclusion.inclusion_id)
       or not MathPSGClassifierIsDigest(inclusion.literal_element_digest)
       or not MathPSGClassifierIsDigest(inclusion.literal_stabilizer_digest)
       or not IsList(inclusion.literal_elements)
       or IsEmpty(inclusion.literal_elements)
       or not IsList(input.element_labels)
       or Length(input.element_labels) <> Length(inclusion.literal_elements)
       or input.element_labels[1] <> "1"
       or Length(Set(input.element_labels)) <> Length(input.element_labels)
       or not ForAll(input.element_labels, label ->
           IsString(label) and not IsEmpty(label) and label <> "T"
       ) then
        Error("invalid Task5 literal inclusion or labels");
    fi;
    for element in inclusion.literal_elements do
        MathPSGClassifierAffineRight(element);
    od;
    if inclusion.literal_element_digest <>
       MathPSGClassifierDigest(
           "literal-stabilizer-authority-v1", inclusion.literal_elements
       ) then
        Error("Task5 literal element digest mismatch");
    fi;
    literalMatrices := List(
        inclusion.literal_elements, MathPSGClassifierAffineRight
    );
    if literalMatrices[1] <> IdentityMat(4)
       or Length(Set(literalMatrices)) <> Length(literalMatrices) then
        Error("Task5 literal inclusion is not canonical identity-first");
    fi;
    for left in literalMatrices do
        if Position(literalMatrices, left^-1) = fail then
            Error("Task5 literal inclusion lacks an inverse");
        fi;
        for right in literalMatrices do
            if Position(literalMatrices, left * right) = fail then
                Error("Task5 literal inclusion is not closed");
            fi;
        od;
    od;
    inputCore := rec(
        action := input.action,
        element_labels := input.element_labels,
        finite_group_id := input.finite_group_id,
        inclusion := input.inclusion,
        time_reversal := input.time_reversal
    );
    if input.input_digest <> MathPSGClassifierDigest(
        "task5-literal-inclusion-export-input-v1", inputCore
    ) then
        Error("Task5 literal-inclusion export input digest mismatch");
    fi;
    return true;
end;

MathPSGClassifierTask5PrepareLiteralInclusionBatch := function(input)
    local matrices, pureTranslation, request, conversion,
          ambientSpatialResolution, timeGroup, timeResolution, ambient,
          ambientSpatialNormal, normalAmbient, ambientConstructionCount,
          buildAmbientResolution;
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
    ambientConstructionCount := 0;
    buildAmbientResolution := function()
        ambientConstructionCount := ambientConstructionCount + 1;
        return MathPSGClassifierTask5AmbientResolution(
            conversion.pcp_group, false
        );
    end;
    ambientSpatialResolution := buildAmbientResolution();
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
        ambient_construction_count := ambientConstructionCount,
        ambient_spatial_resolution := ambientSpatialResolution,
        backend_environment := MathPSGClassifierTask5ObservedBackend(),
        conversion := conversion,
        normal_ambient := normalAmbient,
        time_resolution := timeResolution
    );
end;

MathPSGClassifierTask5LiteralInclusionMemberRaw := function(input, context)
    local conversion, literalMatrices,
          literalSpatialElements, literalSpatialGroup, spatialInclusion,
          timeResolution, localSpatialResolution,
          ambientSpatialResolution, ambient, localResolution, inclusion,
          labels, literalElements, gradedProduct,
          groupId, gradedLabels,
          localSpatialEmbedding, localTimeEmbedding, localTimeElement,
          ambientSpatialNormal, localSpatialNormal, normalAmbient, normalLocal,
          elementName, elementByName, finiteTable, source, target, mapExport,
          comparison, homotopyTuples, phiTuples, nonidentityLabels,
          left, right, witness, degree, basis, word, term, query;
    conversion := context.conversion;
    ambientSpatialResolution := context.ambient_spatial_resolution;
    ambient := context.ambient;
    timeResolution := context.time_resolution;
    literalMatrices := List(
        input.inclusion.literal_elements, MathPSGClassifierAffineRight
    );
    literalSpatialElements := List(literalMatrices, conversion.image);
    literalSpatialGroup := Group(literalSpatialElements);
    if not IsFinite(literalSpatialGroup)
       or Size(literalSpatialGroup) <> Length(literalSpatialElements) then
        Error("Task5 literal inclusion is not the full finite stabilizer");
    fi;
    spatialInclusion := GroupHomomorphismByFunction(
        literalSpatialGroup, conversion.pcp_group, element -> element
    );
    labels := ShallowCopy(input.element_labels);
    ambientSpatialNormal := element ->
        MathPSGClassifierTask5PcpWord(conversion.pcp, element);
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
            List(ShallowCopy(literalElements), element -> element * localTimeElement)
        );
        gradedLabels := List(
            literalElements{[Length(input.element_labels)+1..Length(literalElements)]}, normalLocal
        );
        if Length(Set(Concatenation(labels, gradedLabels))) <> 2 * Length(labels) then Error("Task5 graded element labels collide"); fi;
        Append(labels, gradedLabels);
        gradedProduct := rec(
            spatial_inclusion := spatialInclusion,
            spatial_source_resolution := localSpatialResolution,
            spatial_target_resolution := ambientSpatialResolution
        );
    else
        localResolution := localSpatialResolution;
        inclusion := spatialInclusion;
        normalAmbient := context.normal_ambient;
        normalLocal := localSpatialNormal;
        literalElements := literalSpatialElements;
        gradedProduct := fail;
    fi;
    if literalElements[1] <> Identity(localResolution!.group) then
        Error("Task5 finite-table identity is not canonical");
    fi;
    elementName := element -> labels[Position(literalElements, element)];
    elementByName := name -> literalElements[Position(labels, name)];
    groupId := input.finite_group_id;
    if input.time_reversal then
        groupId := Concatenation(groupId, "+onsite-T");
    fi;
    finiteTable := rec(
        element_order := labels,
        group_id := groupId,
        identity_index := 0,
        inverse_indices := List(literalElements, element ->
            Position(literalElements, element^-1) - 1
        ),
        multiplication_table := List(literalElements, first ->
            List(literalElements, second ->
                Position(literalElements, first * second) - 1
            )
        ),
        table_digest := fail
    );
    if fail in finiteTable.inverse_indices
       or ForAny(finiteTable.multiplication_table, row -> fail in row) then
        Error("Task5 direct-product finite table is incomplete");
    fi;
    source := MathPSGClassifierTask5RawResolution(
        localResolution, normalLocal
    );
    target := MathPSGClassifierTask5RawResolution(ambient, normalAmbient);
    mapExport := MathPSGClassifierTask5InclusionMaps(
        localResolution, ambient, inclusion, normalLocal, normalAmbient,
        gradedProduct
    );
    nonidentityLabels := labels{[2..Length(labels)]};
    homotopyTuples := [[]];
    Append(homotopyTuples, List(nonidentityLabels, item -> [item]));
    Append(homotopyTuples, Cartesian(nonidentityLabels, nonidentityLabels));
    witness := fail;
    for left in [1..Length(literalElements)] do
        for right in [1..Length(literalElements)] do
            if witness = fail
               and literalElements[left] * literalElements[right]
                   <> literalElements[right] * literalElements[left] then
                witness := [labels[right], labels[left], labels[right]];
            fi;
        od;
    od;
    if witness <> fail then Add(homotopyTuples, witness); fi;
    phiTuples := ShallowCopy(homotopyTuples);
    comparison := BarResolutionEquivalence(localResolution);
    for degree in [0..4] do
        for basis in [1..localResolution!.dimension(degree)] do
            word := comparison!.psi(degree, [[
                1, basis,
                Position(
                    localResolution!.elts, Identity(localResolution!.group)
                )
            ]]);
            for term in word do
                query := List(term{[3..Length(term)]}, elementName);
                if not "1" in query and not query in phiTuples then
                    Add(phiTuples, query);
                fi;
            od;
        od;
    od;
    return rec(
        backend_environment := context.backend_environment,
        bar_equivalence := MathPSGClassifierTask5BarEquivalence(
            localResolution, finiteTable, phiTuples, homotopyTuples,
            normalLocal, elementByName
        ),
        bar_comparison_traces := mapExport.bar_comparison_traces,
        chain_map_algorithm := mapExport.chain_map_algorithm,
        diagnostic_backend := mapExport.diagnostic_backend,
        diagnostic_maps := mapExport.diagnostic_maps,
        finite_group := finiteTable,
        lookahead_boundary := source.lookahead_boundary,
        source := source,
        source_element_images := List(
            literalElements,
            element -> normalAmbient(ImageElm(inclusion, element))
        ),
        target := target,
        target_bar_equivalence := mapExport.target_bar_equivalence
    );
end;

MathPSGClassifierTask5LiteralInclusionRaw := function(input, encoded)
    local context;
    MathPSGClassifierTask5ValidateLiteralInclusionInput(input, encoded);
    context := MathPSGClassifierTask5PrepareLiteralInclusionBatch(input);
    return MathPSGClassifierTask5LiteralInclusionMemberRaw(input, context);
end;

MathPSGClassifierTask5ValidateLiteralInclusionBatchInput := function(input, encoded)
    local member, memberInput, core, inclusionIds;
    if not IsRecord(input)
       or MathPSGClassifierJson(input) <> encoded
       or not MathPSGClassifierRequireFields(input, [
           "action", "input_digest", "members", "record_type",
           "schema_version", "time_reversal"
       ])
       or input.record_type <> "task5-literal-inclusion-batch-input"
       or input.schema_version <> 1
       or not input.time_reversal in [true, false]
       or not IsList(input.members)
       or IsEmpty(input.members)
       or not MathPSGClassifierIsDigest(input.input_digest) then
        Error("invalid Task5 literal-inclusion batch envelope");
    fi;
    inclusionIds := [];
    for member in input.members do
        if not IsRecord(member)
           or not MathPSGClassifierRequireFields(member, [
               "element_labels", "finite_group_id", "inclusion", "input_digest"
           ]) then
            Error("invalid Task5 literal-inclusion batch member");
        fi;
        memberInput := rec(
            action := input.action,
            element_labels := member.element_labels,
            finite_group_id := member.finite_group_id,
            inclusion := member.inclusion,
            input_digest := member.input_digest,
            record_type := "task5-literal-inclusion-export-input",
            schema_version := 1,
            time_reversal := input.time_reversal
        );
        MathPSGClassifierTask5ValidateLiteralInclusionInput(
            memberInput, MathPSGClassifierJson(memberInput)
        );
        Add(inclusionIds, member.inclusion.inclusion_id);
    od;
    if inclusionIds <> Set(inclusionIds) then
        Error("Task5 literal-inclusion batch members are not canonical unique");
    fi;
    core := rec(
        action := input.action,
        members := input.members,
        time_reversal := input.time_reversal
    );
    if input.input_digest <> MathPSGClassifierDigest(
        "task5-literal-inclusion-batch-input-v1", core
    ) then
        Error("Task5 literal-inclusion batch digest mismatch");
    fi;
    return true;
end;

MathPSGClassifierTask5LiteralInclusionBatchRaw := function(input, encoded)
    local context, members, member, memberInput, raw;
    MathPSGClassifierTask5ValidateLiteralInclusionBatchInput(input, encoded);
    context := MathPSGClassifierTask5PrepareLiteralInclusionBatch(input);
    members := [];
    for member in input.members do
        memberInput := rec(
            action := input.action,
            element_labels := member.element_labels,
            finite_group_id := member.finite_group_id,
            inclusion := member.inclusion,
            input_digest := member.input_digest,
            record_type := "task5-literal-inclusion-export-input",
            schema_version := 1,
            time_reversal := input.time_reversal
        );
        raw := MathPSGClassifierTask5LiteralInclusionMemberRaw(
            memberInput, context
        );
        Add(members, rec(
            inclusion_id := member.inclusion.inclusion_id,
            member_input_digest := member.input_digest,
            raw_output := raw
        ));
    od;
    return rec(
        action_digest := input.action.action_digest,
        ambient_construction_count := context.ambient_construction_count,
        backend_environment := context.backend_environment,
        batch_input_digest := input.input_digest,
        members := members,
        record_type := "task5-literal-inclusion-batch-output",
        schema_version := 1,
        time_reversal := input.time_reversal
    );
end;

MathPSGClassifierTask5P4mmFixtureRaw := function()
    local rotation, reflection, tx, ty, tz, matrices, conversion,
          literalMatrices, literalImages, literalGroup, labels, elementName,
          elementByName, finiteTable, ambient, localResolution, inclusion,
          normalAmbient, normalLocal, rawResolution, source, target, mapExport,
          lookahead, comparison, phiTuples, homotopyTuples, equivalence,
          degree, basis, word, term, backendEnvironment;
    # Backend metadata is an observation only.  In particular, callers cannot
    # inject a record and have GAP echo it as release evidence.  The Python
    # launcher separately binds these bytes to the process that produced them.
    backendEnvironment := MathPSGClassifierTask5ObservedBackend();
    rotation := [
        [0, 1, 0, 0], [-1, 0, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1]
    ];
    reflection := DiagonalMat([1, -1, 1, 1]);
    tx := IdentityMat(4); tx[4][1] := 1;
    ty := IdentityMat(4); ty[4][2] := 1;
    tz := IdentityMat(4); tz[4][3] := 1;
    matrices := [rotation, reflection, tx, ty, tz];
    conversion := MathPSGClassifierCrystConversion(matrices);
    literalMatrices := [
        IdentityMat(4), rotation, rotation^2, rotation^3,
        reflection, rotation * reflection,
        rotation^2 * reflection, rotation^3 * reflection
    ];
    literalImages := List(literalMatrices, conversion.image);
    literalGroup := Group(literalImages);
    labels := ["1", "r", "r2", "r3", "s", "rs", "r2s", "r3s"];
    elementName := element -> labels[Position(literalImages, element)];
    elementByName := name -> literalImages[Position(labels, name)];
    finiteTable := rec(
        element_order := labels,
        group_id := "p4mm-1a-d4",
        identity_index := 0,
        inverse_indices := List(literalImages, element ->
            Position(literalImages, element^-1) - 1
        ),
        multiplication_table := List(literalImages, left ->
            List(literalImages, right ->
                Position(literalImages, left * right) - 1
            )
        ),
        table_digest := fail
    );
    ambient := MathPSGClassifierTask5AmbientResolution(
        conversion.pcp_group, false
    );
    localResolution := MathPSGClassifierTask5FiniteResolution(literalGroup);
    inclusion := GroupHomomorphismByFunction(
        literalGroup, conversion.pcp_group, element -> element
    );
    normalAmbient := element -> MathPSGClassifierTask5PcpWord(
        conversion.pcp, element
    );
    normalLocal := elementName;
    rawResolution := function(resolution, normalForm)
        return rec(
            basis := List([0..4], degree ->
                MathPSGClassifierTask5Basis(
                    degree, resolution!.dimension(degree)
                )
            ),
            boundaries := List([1..4], degree ->
                MathPSGClassifierTask5BoundaryMatrix(
                    resolution, degree, normalForm
                )
            ),
            degree_five_basis := MathPSGClassifierTask5Basis(
                5, resolution!.dimension(5)
            ),
            lookahead_boundary := MathPSGClassifierTask5BoundaryMatrix(
                resolution, 5, normalForm
            )
        );
    end;
    source := rawResolution(localResolution, normalLocal);
    target := rawResolution(ambient, normalAmbient);
    mapExport := MathPSGClassifierTask5InclusionMaps(
        localResolution, ambient, inclusion, normalLocal, normalAmbient
    );
    lookahead := MathPSGClassifierTask5BoundaryMatrix(
        localResolution, 5, normalLocal
    );
    comparison := BarResolutionEquivalence(localResolution);
    homotopyTuples := [[]];
    Append(homotopyTuples, List(labels{[2..8]}, item -> [item]));
    Append(homotopyTuples, Cartesian(labels{[2..8]}, labels{[2..8]}));
    Add(homotopyTuples, ["s", "r", "s"]);
    phiTuples := ShallowCopy(homotopyTuples);
    for degree in [0..4] do
        for basis in [1..localResolution!.dimension(degree)] do
            word := comparison!.psi(degree, [[
                1, basis,
                Position(localResolution!.elts, Identity(literalGroup))
            ]]);
            for term in word do
                word := List(term{[3..Length(term)]}, elementName);
                if not "1" in word and not word in phiTuples then
                    Add(phiTuples, word);
                fi;
            od;
        od;
    od;
    equivalence := MathPSGClassifierTask5BarEquivalence(
        localResolution, finiteTable,
        phiTuples, homotopyTuples, normalLocal, elementByName
    );
    return rec(
        backend_environment := backendEnvironment,
        bar_equivalence := equivalence,
        bar_comparison_traces := mapExport.bar_comparison_traces,
        finite_group := finiteTable,
        lookahead_boundary := lookahead,
        chain_map_algorithm := mapExport.chain_map_algorithm,
        diagnostic_backend := mapExport.diagnostic_backend,
        diagnostic_maps := mapExport.diagnostic_maps,
        source := source,
        source_element_images := List(
            literalImages,
            element -> MathPSGClassifierTask5PcpWord(
                conversion.pcp, element
            )
        ),
        target_bar_equivalence := mapExport.target_bar_equivalence,
        target := target
    );
end;
