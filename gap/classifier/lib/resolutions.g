#############################################################################
## Full sparse low-degree resolution export.  HAP is an untrusted producer:
## the Python verifier replays every product and every d_{n-1} d_n identity.
#############################################################################

MathPSGClassifierTask5PcpWord := function(pcp, element)
    local exponents, pieces, index, exponent, factor;
    exponents := ExponentsByPcp(pcp, element);
    pieces := [];
    for index in [1..Length(exponents)] do
        exponent := exponents[index];
        if exponent <> 0 then
            factor := Concatenation("p", String(index));
            if exponent <> 1 then
                factor := Concatenation(factor, "^", String(exponent));
            fi;
            Add(pieces, factor);
        fi;
    od;
    if IsEmpty(pieces) then return "1"; fi;
    return JoinStringsWithSeparator(pieces, "*");
end;

MathPSGClassifierTask5Basis := function(degree, rank)
    return List(
        [0..rank - 1],
        index -> Concatenation("c", String(degree), ":", String(index))
    );
end;

MathPSGClassifierTask5AddTerm := function(entries, row, column, element, coefficient)
    local entry, term;
    entry := First(entries, item -> item.row = row and item.column = column);
    if entry = fail then
        entry := rec(column := column, row := row, terms := []);
        Add(entries, entry);
    fi;
    term := First(entry.terms, item -> item[2] = element);
    if term = fail then
        Add(entry.terms, [coefficient, element]);
    else
        term[1] := term[1] + coefficient;
    fi;
end;

MathPSGClassifierTask5CanonicalizeEntries := function(entries)
    local entry;
    for entry in entries do
        entry.terms := Filtered(entry.terms, term -> term[1] <> 0);
        Sort(entry.terms, function(left, right) return left[2] < right[2]; end);
    od;
    entries := Filtered(entries, entry -> not IsEmpty(entry.terms));
    Sort(entries, function(left, right)
        return left.row < right.row
            or (left.row = right.row and left.column < right.column);
    end);
    return entries;
end;

MathPSGClassifierTask5BoundaryMatrix := function(resolution, degree, normalForm)
    local entries, column, letter, coefficient, row, element;
    entries := [];
    for column in [1..resolution!.dimension(degree)] do
        for letter in resolution!.boundary(degree, column) do
            coefficient := SignInt(letter[1]);
            row := AbsoluteValue(letter[1]) - 1;
            element := normalForm(resolution!.elts[letter[2]]);
            MathPSGClassifierTask5AddTerm(
                entries, row, column - 1, element, coefficient
            );
        od;
    od;
    return rec(
        column_count := resolution!.dimension(degree),
        entries := MathPSGClassifierTask5CanonicalizeEntries(entries),
        row_count := resolution!.dimension(degree - 1)
    );
end;

MathPSGClassifierTask5ResolutionCore := function(
    resolution, groupId, affineCertificate, finiteGroup, construction,
    backendBinding, catalogueRecordDigest
)
    local basis, boundaries, degree, degreeFiveBasis, lookahead;
    basis := [];
    for degree in [0..4] do
        Add(basis, MathPSGClassifierTask5Basis(
            degree, resolution!.dimension(degree)
        ));
    od;
    boundaries := [];
    for degree in [1..4] do
        Add(boundaries, MathPSGClassifierTask5BoundaryMatrix(
            resolution, degree, construction.normal_form
        ));
    od;
    degreeFiveBasis := MathPSGClassifierTask5Basis(
        5, resolution!.dimension(5)
    );
    lookahead := MathPSGClassifierTask5BoundaryMatrix(
        resolution, 5, construction.normal_form
    );
    return rec(
        affine_pcp_certificate := affineCertificate,
        affine_pcp_certificate_digest := affineCertificate.certificate_digest,
        backend_environment_id := backendBinding.backend_environment_id,
        backend_lock_digest := backendBinding.backend_lock_digest,
        basis := basis,
        boundaries := boundaries,
        catalogue_record_digest := catalogueRecordDigest,
        construction := construction.name,
        degree_five_basis := degreeFiveBasis,
        finite_group := finiteGroup,
        group_id := groupId,
        lookahead_boundary := lookahead,
        max_degree := 4,
        runtime_provenance_digest := backendBinding.runtime_provenance_digest
    );
end;

MathPSGClassifierTask5ResolutionCertificate := function(
    resolution, groupId, affineCertificate, finiteGroup, construction,
    backendBinding, catalogueRecordDigest
)
    local core, result;
    core := MathPSGClassifierTask5ResolutionCore(
        resolution, groupId, affineCertificate, finiteGroup, construction,
        backendBinding, catalogueRecordDigest
    );
    result := ShallowCopy(core);
    result.record_type := "free-resolution-certificate";
    result.resolution_id := MathPSGClassifierDigest(
        "task5-free-resolution-certificate-v1", core
    );
    result.schema_version := 1;
    return result;
end;

MathPSGClassifierTask5AmbientResolution := function(pcpGroup, withTime)
    local spatial, time;
    # One lookahead degree is required because EquivariantChainMap computes
    # its degree-four value using the target contraction into degree five.
    spatial := ResolutionAlmostCrystalGroup(pcpGroup, 5);
    if not withTime then return spatial; fi;
    time := ResolutionFiniteGroup(CyclicGroup(IsPermGroup, 2), 5);
    return ResolutionDirectProduct(spatial, time);
end;

MathPSGClassifierTask5FiniteResolution := function(group)
    # Degree five is a construction-only lookahead for the degree-four
    # contracting homotopies; certificate matrices are truncated at four.
    return ResolutionFiniteGroup(group, 5);
end;

MathPSGClassifierTask5RawResolution := function(resolution, normalForm)
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

MathPSGClassifierTask5DirectProductNormalForm := function(
    resolution, spatialNormalForm, timeName, element
)
    local spatial, time, spatialName;
    spatial := ImagesRepresentative(resolution!.firstProjection, element);
    time := ImagesRepresentative(resolution!.secondProjection, element);
    spatialName := spatialNormalForm(spatial);
    if time = Identity(Range(resolution!.secondProjection)) then
        return spatialName;
    fi;
    if spatialName = "1" then return timeName; fi;
    return Concatenation(spatialName, "+", timeName);
end;

MathPSGClassifierTask5LiveD4Probe := function()
    local group, resolution, equivalence, generators, identity, word, image;
    group := DihedralGroup(IsPermGroup, 8);
    resolution := ResolutionFiniteGroup(group, 4);
    equivalence := BarResolutionEquivalence(resolution);
    generators := GeneratorsOfGroup(group);
    if generators[1] * generators[2] = generators[2] * generators[1] then
        Error("D4 probe unexpectedly commutes");
    fi;
    identity := Identity(group);
    word := [[1, identity, generators[1], generators[2], generators[1]]];
    image := equivalence!.equiv(3, word);
    if image = fail then Error("length-three bar homotopy probe failed"); fi;
    return Concatenation(
        "task5-live-d4:",
        JoinStringsWithSeparator(
            List([0..4], degree -> String(resolution!.dimension(degree))), ","
        ),
        ":noncommuting:length3"
    );
end;
