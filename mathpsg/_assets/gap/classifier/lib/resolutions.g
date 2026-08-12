#############################################################################
## Low-degree resolution data used by the physical solver.
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

MathPSGClassifierTask5AmbientResolution := function(pcpGroup, withTime)
    local spatial, time;
    spatial := ResolutionAlmostCrystalGroup(pcpGroup, 3);
    if not withTime then return spatial; fi;
    time := ResolutionFiniteGroup(CyclicGroup(IsPermGroup, 2), 3);
    return ResolutionDirectProduct(spatial, time);
end;

MathPSGClassifierTask5FiniteResolution := function(group)
    return ResolutionFiniteGroup(group, 3);
end;

MathPSGClassifierTask5RawResolution := function(resolution, normalForm)
    return rec(
        basis := List([0..3], degree ->
            MathPSGClassifierTask5Basis(
                degree, resolution!.dimension(degree)
            )
        ),
        boundaries := List([1..3], degree ->
            MathPSGClassifierTask5BoundaryMatrix(
                resolution, degree, normalForm
            )
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
