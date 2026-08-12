#############################################################################
## Literal finite-stabilizer inclusion maps through exported degree four.
#############################################################################

MathPSGClassifierTask5MapMatrix := function(
    mapping, sourceResolution, targetResolution, degree, targetNormalForm
)
    local entries, identity, column, letter, coefficient, row, element;
    entries := [];
    identity := Position(
        sourceResolution!.elts, Identity(sourceResolution!.group)
    );
    for column in [1..sourceResolution!.dimension(degree)] do
        for letter in mapping([[1, column, identity]], degree) do
            if Length(letter) = 3 then
                coefficient := letter[1]; row := letter[2] - 1;
                element := targetNormalForm(targetResolution!.elts[letter[3]]);
            else
                coefficient := SignInt(letter[1]); row := AbsoluteValue(letter[1]) - 1;
                element := targetNormalForm(targetResolution!.elts[letter[2]]);
            fi;
            MathPSGClassifierTask5AddTerm(
                entries, row, column - 1, element, coefficient
            );
        od;
    od;
    return rec(
        column_count := sourceResolution!.dimension(degree),
        entries := MathPSGClassifierTask5CanonicalizeEntries(entries),
        row_count := targetResolution!.dimension(degree)
    );
end;

MathPSGClassifierTask5ProductMapMatrix := function(
    spatialChainMap, spatialSourceResolution, spatialTargetResolution,
    sourceResolution, targetResolution, degree, targetNormalForm
)
    local entries, sourceIdentity, targetSpatialEmbedding, column, vector,
          spatialDegree, timeDegree, spatialBasis, timeBasis, letter,
          coefficient, spatialRow, spatialElement, targetRow, targetElement;
    entries := [];
    sourceIdentity := Position(
        spatialSourceResolution!.elts,
        Identity(spatialSourceResolution!.group)
    );
    targetSpatialEmbedding := Embedding(targetResolution!.group, 1);
    for column in [1..sourceResolution!.dimension(degree)] do
        vector := sourceResolution!.Int2Vector(degree, column);
        spatialDegree := vector[1];
        timeDegree := vector[2];
        spatialBasis := vector[3];
        timeBasis := vector[4];
        for letter in spatialChainMap!.mapping(
            [[1, spatialBasis, sourceIdentity]], spatialDegree
        ) do
            if Length(letter) = 3 then
                coefficient := letter[1];
                spatialRow := letter[2];
                spatialElement := spatialTargetResolution!.elts[letter[3]];
            else
                coefficient := SignInt(letter[1]);
                spatialRow := AbsoluteValue(letter[1]);
                spatialElement := spatialTargetResolution!.elts[letter[2]];
            fi;
            targetRow := targetResolution!.Vector2Int(
                spatialDegree, timeDegree, spatialRow, timeBasis
            );
            targetElement := ImagesRepresentative(
                targetSpatialEmbedding, spatialElement
            );
            MathPSGClassifierTask5AddTerm(
                entries, targetRow - 1, column - 1,
                targetNormalForm(targetElement), coefficient
            );
        od;
    od;
    return rec(
        column_count := sourceResolution!.dimension(degree),
        entries := MathPSGClassifierTask5CanonicalizeEntries(entries),
        row_count := targetResolution!.dimension(degree)
    );
end;

MathPSGClassifierTask5BarComparisonTraces := function(
    sourceResolution, targetResolution, inclusionHom,
    sourceNormalForm, targetNormalForm
)
    local sourceComparison, targetComparison, identity, traces, degree,
          basis, sourceBar, targetBar, term, mapped, targetOutput,
          targetQueryElements, targetRequiredBasis, query, required;
    sourceComparison := BarResolutionEquivalence(sourceResolution);
    targetComparison := BarResolutionEquivalence(targetResolution);
    identity := Position(
        sourceResolution!.elts, Identity(sourceResolution!.group)
    );
    traces := [];
    targetQueryElements := [];
    targetRequiredBasis := [];
    for degree in [0..4] do
        for basis in [1..sourceResolution!.dimension(degree)] do
            sourceBar := sourceComparison!.psi(
                degree, [[1, basis, identity]]
            );
            targetBar := [];
            for term in sourceBar do
                mapped := [term[1], ImageElm(inclusionHom, term[2])];
                Append(
                    mapped,
                    List(
                        term{[3..Length(term)]},
                        element -> ImageElm(inclusionHom, element)
                    )
                );
                Add(targetBar, mapped);
            od;
            targetOutput := targetComparison!.phi(degree, targetBar);
            for term in targetOutput do
                required := [degree, AbsoluteValue(term[2])];
                if Position(targetRequiredBasis, required) = fail then
                    Add(targetRequiredBasis, required);
                fi;
            od;
            for term in targetBar do
                query := term{[3..Length(term)]};
                if not Identity(targetResolution!.group) in query
                   and Position(targetQueryElements, query) = fail then
                    Add(targetQueryElements, query);
                fi;
            od;
            Add(traces, rec(
                degree := degree,
                source_basis_id := Concatenation(
                    "c", String(degree), ":", String(basis - 1)
                ),
                source_psi := rec(
                    degree := degree,
                    terms := MathPSGClassifierTask5BarChain(
                        sourceBar, sourceNormalForm
                    )
                ),
                target_phi_input := rec(
                    degree := degree,
                    terms := MathPSGClassifierTask5BarChain(
                        targetBar, targetNormalForm
                    )
                ),
                target_phi_output := MathPSGClassifierTask5ResolutionChain(
                    targetResolution, degree, targetOutput, targetNormalForm
                )
            ));
        od;
    od;
    return rec(
        target_query_elements := targetQueryElements,
        target_required_basis := targetRequiredBasis,
        traces := traces
    );
end;

MathPSGClassifierTask5InclusionMaps := function(arg)
    local sourceResolution, targetResolution, inclusionHom,
          sourceNormalForm, targetNormalForm, gradedProduct, chainMap,
          comparison, diagnostic, diagnosticBackend, targetEquivalence;
    if not Length(arg) in [5, 6] then
        Error("Task5 inclusion-map builder expects five or six arguments");
    fi;
    sourceResolution := arg[1];
    targetResolution := arg[2];
    inclusionHom := arg[3];
    sourceNormalForm := arg[4];
    targetNormalForm := arg[5];
    gradedProduct := fail;
    if Length(arg) = 6 then gradedProduct := arg[6]; fi;
    # HAP 1.70's direct EquivariantChainMap is retained as an explicit
    # diagnostic.  For finite -> almost-crystallographic inclusions it can
    # violate dF=Fd already in degree one.  The authoritative map is the
    # standard comparison phi_target o inclusion_bar o psi_source; Python
    # independently replays every displayed square and rejects the direct
    # map with digest-bound failure/residue observations; the frozen p4mm
    # outcome remains compatibility data, not generic vocabulary.
    if gradedProduct = fail then
        chainMap := EquivariantChainMap(
            sourceResolution, targetResolution, inclusionHom
        );
        diagnosticBackend := "HAP-1.70-EquivariantChainMap";
    else
        chainMap := EquivariantChainMap(
            gradedProduct.spatial_source_resolution,
            gradedProduct.spatial_target_resolution,
            gradedProduct.spatial_inclusion
        );
        diagnosticBackend :=
            "HAP-1.70-EquivariantChainMap-Tensor-Identity-C2";
    fi;
    if chainMap = fail then Error("HAP inclusion chain map failed"); fi;
    comparison := MathPSGClassifierTask5BarComparisonTraces(
        sourceResolution, targetResolution, inclusionHom,
        sourceNormalForm, targetNormalForm
    );
    targetEquivalence := MathPSGClassifierTask5TargetBarEquivalence(
        targetResolution, comparison.target_query_elements,
        comparison.target_required_basis, targetNormalForm
    );
    if gradedProduct = fail then
        diagnostic := List(
            [0..4],
            degree -> MathPSGClassifierTask5MapMatrix(
                chainMap!.mapping, sourceResolution, targetResolution,
                degree, targetNormalForm
            )
        );
    else
        diagnostic := List(
            [0..4],
            degree -> MathPSGClassifierTask5ProductMapMatrix(
                chainMap,
                gradedProduct.spatial_source_resolution,
                gradedProduct.spatial_target_resolution,
                sourceResolution, targetResolution, degree,
                targetNormalForm
            )
        );
    fi;
    return rec(
        bar_comparison_traces := comparison.traces,
        chain_map_algorithm :=
            "hap-1.70-bar-phi-target-inclusion-psi-source",
        diagnostic_backend := diagnosticBackend,
        diagnostic_maps := diagnostic,
        target_bar_equivalence := targetEquivalence
    );
end;

MathPSGClassifierTask5DirectProductInclusion := function(
    sourceResolution, targetResolution, spatialInclusion
)
    local sourceGroup, targetGroup, targetSpatialEmbedding,
          targetTimeEmbedding, targetTimeGenerator;
    sourceGroup := sourceResolution!.group;
    targetGroup := targetResolution!.group;
    targetSpatialEmbedding := Embedding(targetGroup, 1);
    targetTimeEmbedding := Embedding(targetGroup, 2);
    targetTimeGenerator := GeneratorsOfGroup(
        Range(targetResolution!.secondProjection)
    )[1];
    return GroupHomomorphismByFunction(
        sourceGroup, targetGroup,
        function(element)
            local spatial, time, image;
            spatial := ImagesRepresentative(
                sourceResolution!.firstProjection, element
            );
            time := ImagesRepresentative(
                sourceResolution!.secondProjection, element
            );
            image := ImagesRepresentative(
                targetSpatialEmbedding,
                ImageElm(spatialInclusion, spatial)
            );
            if time <> Identity(Range(sourceResolution!.secondProjection)) then
                image := image * ImagesRepresentative(
                    targetTimeEmbedding, targetTimeGenerator
                );
            fi;
            return image;
        end
    );
end;
