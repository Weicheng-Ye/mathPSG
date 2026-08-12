#############################################################################
## Literal finite-stabilizer restrictions used by the physical solver.
#############################################################################

# Build the authoritative resolution map
#
#     phi_target o inclusion_bar o psi_source
#
# directly in one degree.  HAP's EquivariantChainMap is not used here: for
# finite-to-almost-crystallographic inclusions it can fail to be a chain map.
MathPSGClassifierTask5RestrictionMatrix := function(
    sourceResolution, targetResolution, inclusionHom,
    sourceComparison, targetComparison, degree, targetNormalForm
)
    local sourceIdentity, entries, column, sourceBar, targetBar, term,
          mapped, targetOutput, letter, coefficient, row, element;
    sourceIdentity := Position(
        sourceResolution!.elts, Identity(sourceResolution!.group)
    );
    entries := [];
    for column in [1..sourceResolution!.dimension(degree)] do
        sourceBar := sourceComparison!.psi(
            degree, [[1, column, sourceIdentity]]
        );
        targetBar := [];
        for term in sourceBar do
            mapped := [term[1], ImageElm(inclusionHom, term[2])];
            Append(
                mapped,
                List(
                    term{[3..Length(term)]},
                    item -> ImageElm(inclusionHom, item)
                )
            );
            Add(targetBar, mapped);
        od;
        targetOutput := targetComparison!.phi(degree, targetBar);
        for letter in targetOutput do
            if Length(letter) = 3 then
                coefficient := letter[1];
                row := letter[2] - 1;
                element := targetNormalForm(targetResolution!.elts[letter[3]]);
            else
                coefficient := SignInt(letter[1]);
                row := AbsoluteValue(letter[1]) - 1;
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

MathPSGClassifierTask5InclusionMaps := function(
    sourceResolution, targetResolution, inclusionHom, targetNormalForm
)
    local sourceComparison, targetComparison;
    sourceComparison := BarResolutionEquivalence(sourceResolution);
    targetComparison := BarResolutionEquivalence(targetResolution);
    return List(
        [1..2],
        degree -> MathPSGClassifierTask5RestrictionMatrix(
            sourceResolution, targetResolution, inclusionHom,
            sourceComparison, targetComparison,
            degree, targetNormalForm
        )
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
