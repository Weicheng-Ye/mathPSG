#############################################################################
##
##  Exact affine and JSON helpers for the MathPSG Cryst export protocol.
##
#############################################################################

MathPSGJsonEscape := function(value)
    local out, character, code;
    out := "";
    for character in value do
        code := INT_CHAR(character);
        if character = '"' then
            Append(out, "\\\"");
        elif character = '\\' then
            Append(out, "\\\\");
        elif character = '\n' then
            Append(out, "\\n");
        elif character = '\r' then
            Append(out, "\\r");
        elif character = '\t' then
            Append(out, "\\t");
        elif code < 32 then
            Error("unsupported JSON control character");
        else
            Add(out, character);
        fi;
    od;
    return out;
end;

MathPSGJsonString := value -> Concatenation("\"", MathPSGJsonEscape(value), "\"");

MathPSGJson := function(value)
    local names, pieces, name;
    if value = true then
        return "true";
    elif value = false then
        return "false";
    elif value = fail then
        return "null";
    elif IsInt(value) then
        return String(value);
    elif IsString(value) and Length(value) > 0 then
        return MathPSGJsonString(value);
    elif IsRecord(value) then
        names := SortedList(RecNames(value));
        pieces := [];
        for name in names do
            Add(pieces, Concatenation(MathPSGJsonString(name), ":", MathPSGJson(value.(name))));
        od;
        return Concatenation("{", JoinStringsWithSeparator(pieces, ","), "}");
    elif IsList(value) then
        pieces := List(value, MathPSGJson);
        return Concatenation("[", JoinStringsWithSeparator(pieces, ","), "]");
    fi;
    Error("unsupported value in MathPSG JSON encoder");
end;

MathPSGSettingString := function(setting)
    if IsChar(setting) then
        return [setting];
    fi;
    return String(setting);
end;

MathPSGRationalString := function(value)
    return Concatenation(
        "q(", String(NumeratorRat(value)), ",", String(DenominatorRat(value)), ")"
    );
end;

MathPSGRationalVector := vector -> List(vector, MathPSGRationalString);

MathPSGRationalMatrix := matrix -> List(matrix, MathPSGRationalVector);

MathPSGAffineFromRight := function(element)
    local dimension, linear;
    dimension := Length(element) - 1;
    linear := element{[1..dimension]}{[1..dimension]};
    return rec(
        matrix := MathPSGRationalMatrix(TransposedMat(linear)),
        translation := MathPSGRationalVector(element[dimension + 1]{[1..dimension]})
    );
end;

MathPSGColumnBasis := function(rowBasis, ambientDimension)
    return List(
        [1..ambientDimension],
        coordinate -> List(rowBasis, vector -> vector[coordinate])
    );
end;

MathPSGParameterNames := function(dimension)
    return List([1..dimension], index -> Concatenation("lambda", String(index)));
end;

MathPSGHexSHA256 := function(value)
    local digest;
    digest := LowercaseString(HexSHA256(value));
    if Length(digest) > 64 then
        Error("SHA-256 hexadecimal encoding exceeds 64 digits");
    fi;
    return Concatenation(
        String(ListWithIdenticalEntries(64 - Length(digest), '0')),
        digest
    );
end;

MathPSGBranch := function(translation, rowBasis)
    local dimension, core;
    dimension := Length(rowBasis);
    core := rec(
        basis := MathPSGRationalMatrix(MathPSGColumnBasis(rowBasis, Length(translation))),
        offset := MathPSGRationalVector(translation),
        parameter_dimension := dimension,
        parameter_names := MathPSGParameterNames(dimension)
    );
    core.branch_digest := Concatenation("sha256:", MathPSGHexSHA256(MathPSGJson(core)));
    return core;
end;

MathPSGPositionRecord := position -> rec(
    basis := WyckoffBasis(position),
    spaceGroup := WyckoffSpaceGroup(position),
    translation := WyckoffTranslation(position)
);

MathPSGSamePositionFamily := function(left, right)
    return WyckoffTranslation(left) = WyckoffTranslation(right)
       and WyckoffBasis(left) = WyckoffBasis(right);
end;

MathPSGPointOperationSubgroup := function(group)
    local dimension, identity, generators;
    dimension := DimensionOfMatrixGroup(group) - 1;
    identity := IdentityMat(dimension);
    generators := Filtered(
        GeneratorsOfGroup(group),
        element -> element{[1..dimension]}{[1..dimension]} <> identity
    );
    return AffineCrystGroupOnRight(generators, One(group));
end;

MathPSGCorrectBranchTransport := function(group, sourceRecord, targetRecord, representative)
    local dimension, linear, imageTranslation, correction, correctionElement,
          transport;
    dimension := DimensionOfMatrixGroup(group) - 1;
    linear := representative{[1..dimension]}{[1..dimension]};
    imageTranslation := sourceRecord.translation * linear
                      + representative[dimension + 1]{[1..dimension]};
    correction := targetRecord.translation - imageTranslation;
    if VectorModL(correction, TranslationBasis(group)) <> 0 * correction then
        Error("Cryst branch reduction is not a translation-lattice correction");
    fi;

    correctionElement := IdentityMat(dimension + 1);
    correctionElement[dimension + 1]{[1..dimension]} := correction;
    transport := representative * correctionElement;
    linear := transport{[1..dimension]}{[1..dimension]};
    if sourceRecord.translation * linear
         + transport[dimension + 1]{[1..dimension]}
         <> targetRecord.translation then
        Error("corrected transport does not map the reference offset exactly");
    fi;
    if IsEmpty(sourceRecord.basis) then
        if not IsEmpty(targetRecord.basis) then
            Error("corrected transport changes the parameter dimension");
        fi;
    elif sourceRecord.basis * linear <> targetRecord.basis then
        Error("corrected transport does not induce the stored branch parametrization");
    fi;
    if not transport in group then
        Error("corrected branch transport is not an ambient group element");
    fi;
    return rec(lattice_correction := correction, transport := transport);
end;

MathPSGExactBranchTransport := function(group, pointOperationSubgroup, reference, target)
    local sourceRecord, targetRecord, representative;
    sourceRecord := MathPSGPositionRecord(reference);
    targetRecord := MathPSGPositionRecord(target);
    representative := RepresentativeAction(
        pointOperationSubgroup,
        sourceRecord,
        targetRecord,
        ImageAffineSubspaceLatticePointwise
    );
    if representative = fail then
        Error("Cryst did not supply a transport between Wyckoff orbit branches");
    fi;
    return MathPSGCorrectBranchTransport(
        group,
        sourceRecord,
        targetRecord,
        representative
    ).transport;
end;

MathPSGVerifyFamilyFixation := function(position, elements)
    local translation, basis, dimension, element, linear;
    translation := WyckoffTranslation(position);
    basis := WyckoffBasis(position);
    dimension := Length(translation);
    for element in elements do
        linear := element{[1..dimension]}{[1..dimension]};
        if translation * linear + element[dimension + 1]{[1..dimension]}
             <> translation then
            return false;
        fi;
        if not IsEmpty(basis) and basis * linear <> basis then
            return false;
        fi;
    od;
    return true;
end;

MathPSGArgumentValue := function(name)
    local args, positions;
    args := GAPInfo.SystemCommandLine;
    positions := Positions(args, name);
    if Length(positions) <> 1 or positions[1] = Length(args) then
        return fail;
    fi;
    return args[positions[1] + 1];
end;

MathPSGValidateOptionPairs := function(allowedNames)
    local args, separators, tail, names;
    args := GAPInfo.SystemCommandLine;
    separators := Positions(args, "--");
    if Length(separators) <> 1 then
        return false;
    fi;
    tail := args{[separators[1] + 1..Length(args)]};
    if IsEmpty(tail) or Length(tail) mod 2 <> 0 then
        return false;
    fi;
    names := tail{[1,3..Length(tail) - 1]};
    return Length(Set(names)) = Length(names)
       and ForAll(names, name -> name in allowedNames);
end;

MathPSGWriteEncodedFile := function(path, encoded)
    local result;
    result := CALL_WITH_CATCH(FileString, [path, encoded]);
    return result[1] = true
       and result[2] <> fail
       and result[2] = Length(encoded);
end;

MathPSGErrorEncoding := function(code, message)
    local payload;
    payload := rec(
        error := rec(code := code, message := message),
        protocol_version := 1,
        record_type := "catalogue-export-error",
        status := "error"
    );
    return Concatenation(MathPSGJson(payload), "\n");
end;

MathPSGWriteError := function(path, code, message)
    local encoded;
    encoded := MathPSGErrorEncoding(code, message);
    if path <> fail and MathPSGWriteEncodedFile(path, encoded) then
        return true;
    fi;
    SizeScreen([4096, 24]);
    Print(encoded);
    return false;
end;

MathPSGPreflightOutputDirectory := function(path)
    local attempt, probe;
    if not IsDirectoryPath(path) then
        return false;
    fi;
    for attempt in [1..128] do
        probe := Concatenation(
            path,
            "/.mathpsg-catalogue-write-probe-",
            String(attempt)
        );
        if not IsExistingFile(probe) then
            if not MathPSGWriteEncodedFile(probe, "x") then
                return false;
            fi;
            if RemoveFile(probe) <> true then
                return false;
            fi;
            return true;
        fi;
    od;
    return false;
end;

MathPSGDirectEnvironment := function()
    local versions, packageNames, name, package;
    versions := rec(
        alnuth := "3.2.1",
        autpgrp := "1.11.1",
        cryst := "4.1.30",
        gap := "4.15.1",
        polenta := "1.3.11",
        polycyclic := "2.17",
        radiroot := "2.9"
    );
    if GAPInfo.Version <> versions.gap then
        Error("GAP version differs from environments/catalogue-gap.lock.json");
    fi;
    packageNames := Filtered(RecNames(versions), name -> name <> "gap");
    for name in packageNames do
        package := PackageInfo(name);
        if IsEmpty(package) or package[1].Version <> versions.(name) then
            Error(Concatenation(
                "GAP package ", name,
                " differs from environments/catalogue-gap.lock.json"
            ));
        fi;
    od;
    return rec(
        certification_status := "uncertified-direct",
        load_policy := "exact-version-only-needed",
        versions := versions
    );
end;

MathPSGEnvironmentEvidence := function()
    local names, loadedPackages, name, loaded, available;
    names := SortedList(RecNames(GAPInfo.PackagesLoaded));
    loadedPackages := [];
    for name in names do
        loaded := GAPInfo.PackagesLoaded.(name);
        available := PackageInfo(name);
        Add(loadedPackages, rec(
            available_installation_paths := List(
                available,
                package -> package.InstallationPath
            ),
            installation_path := loaded[1],
            name := name,
            version := loaded[2]
        ));
    od;
    return rec(
        gap_version := GAPInfo.Version,
        load_request := rec(
            exact_version := "4.1.30",
            only_needed := true,
            package := "cryst"
        ),
        loaded_packages := loadedPackages,
        options := rec(
            autoload_disabled := GAPInfo.CommandLineOptions.A,
            bare := GAPInfo.CommandLineOptions.bare,
            nointeract := GAPInfo.CommandLineOptions.nointeract,
            norepl := GAPInfo.CommandLineOptions.norepl,
            user_root_disabled := GAPInfo.CommandLineOptions.r,
            workspace_restore_disabled := GAPInfo.CommandLineOptions.R
        ),
        root_paths := GAPInfo.RootPaths
    );
end;

MathPSGExportOne := function(number)
    local request, data, group, positions, package, candidates, position,
          branch, stabilizer, orbitPositions, pointOperationSubgroup, branches,
          transports, target, transport, stabilizerElements;
    request := rec(dim := 3, nr := number);
    data := SpaceGroupDataIT(request);
    group := SpaceGroupOnRightIT(3, number, request.setting);
    positions := WyckoffPositions(group);
    package := PackageInfo("cryst")[1];
    pointOperationSubgroup := MathPSGPointOperationSubgroup(group);
    candidates := [];
    for position in positions do
        branch := MathPSGBranch(
            WyckoffTranslation(position),
            WyckoffBasis(position)
        );
        orbitPositions := Concatenation(
            [position],
            Filtered(
                WyckoffOrbit(position),
                target -> not MathPSGSamePositionFamily(position, target)
            )
        );
        branches := List(
            orbitPositions,
            target -> MathPSGBranch(WyckoffTranslation(target), WyckoffBasis(target))
        );
        transports := [];
        for target in orbitPositions do
            transport := MathPSGExactBranchTransport(
                group,
                pointOperationSubgroup,
                position,
                target
            );
            Add(transports, rec(
                ambient_element := MathPSGAffineFromRight(transport),
                exact_transport_verified := true,
                parameter_action := rec(
                    matrix := MathPSGRationalMatrix(IdentityMat(branch.parameter_dimension)),
                    translation := MathPSGRationalVector(0 * [1..branch.parameter_dimension])
                ),
                parameter_dimension := branch.parameter_dimension,
                target_branch_digest := MathPSGBranch(
                    WyckoffTranslation(target),
                    WyckoffBasis(target)
                ).branch_digest
            ));
        od;
        stabilizer := WyckoffStabilizer(position);
        stabilizerElements := SortedList(Elements(stabilizer));
        if not MathPSGVerifyFamilyFixation(position, stabilizerElements) then
            Error("embedded stabilizer does not fix the reference family exactly");
        fi;
        Add(candidates, rec(
            orbit := rec(
                branch_transports := transports,
                branches := branches,
                parameter_dimension := branch.parameter_dimension,
                parameter_names := branch.parameter_names,
                primitive_orbit_size := Length(orbitPositions),
                reference_branch_digest := branch.branch_digest
            ),
            stabilizer := rec(
                embedded_elements := List(stabilizerElements, MathPSGAffineFromRight),
                fixation_verified := true,
                order := Size(stabilizer),
                reference_branch_digest := branch.branch_digest
            )
        ));
    od;
    Sort(candidates, function(left, right)
        return MathPSGJson(left) < MathPSGJson(right);
    end);
    return rec(
        candidates := candidates,
        coordinate_convention := rec(
            affine_action := "x_column -> matrix*x_column + translation",
            composition_law := "C(g*h)=C(h)*C(g)",
            rational_encoding := "q(n,d)",
            source_action := "Cryst right-row homogeneous matrices",
            translation_policy := "full-unreduced"
        ),
        environment := MathPSGDirectEnvironment(),
        protocol_version := 1,
        record_type := "catalogue-gap-export",
        source := rec(cryst := package.Version, gap := GAPInfo.Version),
        space_group := rec(
            international_number := number,
            setting := MathPSGSettingString(request.setting)
        ),
        space_group_action := rec(
            source_generators := List(GeneratorsOfGroup(group), MathPSGAffineFromRight),
            source_right_homogeneous_matrices := List(
                GeneratorsOfGroup(group),
                MathPSGRationalMatrix
            ),
            translation_basis := MathPSGRationalMatrix(TransposedMat(TranslationBasis(group)))
        ),
        status := "success"
    );
end;
