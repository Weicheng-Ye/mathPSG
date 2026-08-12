#############################################################################
## Minimal Cryst-to-JSON helpers used by classify.
#############################################################################

MathPSGJsonEscape := function(value)
    local out, character;
    out := "";
    for character in value do
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
            Add(pieces, Concatenation(
                MathPSGJsonString(name), ":", MathPSGJson(value.(name))
            ));
        od;
        return Concatenation("{", JoinStringsWithSeparator(pieces, ","), "}");
    elif IsList(value) then
        return Concatenation(
            "[", JoinStringsWithSeparator(List(value, MathPSGJson), ","), "]"
        );
    fi;
    Error("unsupported value in JSON output");
end;

MathPSGRationalString := value -> Concatenation(
    "q(", String(NumeratorRat(value)), ",", String(DenominatorRat(value)), ")"
);

MathPSGRationalVector := vector -> List(vector, MathPSGRationalString);
MathPSGRationalMatrix := matrix -> List(matrix, MathPSGRationalVector);

MathPSGColumnBasis := function(rowBasis, ambientDimension)
    return List(
        [1..ambientDimension],
        coordinate -> List(rowBasis, vector -> vector[coordinate])
    );
end;

MathPSGAffineFromRight := function(element)
    local dimension, linear;
    dimension := Length(element) - 1;
    linear := element{[1..dimension]}{[1..dimension]};
    return rec(
        matrix := MathPSGRationalMatrix(TransposedMat(linear)),
        translation := MathPSGRationalVector(
            element[dimension + 1]{[1..dimension]}
        )
    );
end;

MathPSGArgumentValue := function(name)
    local arguments, positions;
    arguments := GAPInfo.SystemCommandLine;
    positions := Positions(arguments, name);
    if Length(positions) <> 1 or positions[1] = Length(arguments) then
        return fail;
    fi;
    return arguments[positions[1] + 1];
end;

MathPSGSettingString := function(setting)
    if IsChar(setting) then
        return [setting];
    fi;
    return String(setting);
end;

MathPSGWriteEncodedFile := function(path, encoded)
    local result;
    result := CALL_WITH_CATCH(FileString, [path, encoded]);
    return result[1] = true and result[2] <> fail;
end;

MathPSGWriteError := function(path, message)
    local encoded;
    encoded := Concatenation(MathPSGJson(rec(error := message)), "\n");
    if path <> fail and MathPSGWriteEncodedFile(path, encoded) then
        return;
    fi;
    Print(encoded);
end;

MathPSGPositionGeometry := function(position)
    return rec(
        basis := MathPSGRationalMatrix(
            MathPSGColumnBasis(WyckoffBasis(position), 3)
        ),
        offset := MathPSGRationalVector(WyckoffTranslation(position))
    );
end;

MathPSGSamePositionFamily := function(left, right)
    return WyckoffTranslation(left) = WyckoffTranslation(right)
       and WyckoffBasis(left) = WyckoffBasis(right);
end;

MathPSGExportOne := function(number)
    local request, data, group, positions, candidates, position,
          orbitPositions, stabilizerElements, generators, embedded;
    request := rec(dim := 3, nr := number);
    data := SpaceGroupDataIT(request);
    group := SpaceGroupOnRightIT(3, number, request.setting);
    positions := WyckoffPositions(group);
    candidates := [];
    for position in positions do
        orbitPositions := Concatenation(
            [position],
            Filtered(
                WyckoffOrbit(position),
                target -> not MathPSGSamePositionFamily(position, target)
            )
        );
        stabilizerElements := SortedList(Elements(WyckoffStabilizer(position)));
        embedded := List(stabilizerElements, MathPSGAffineFromRight);
        Sort(embedded, function(left, right)
            return MathPSGJson(left) < MathPSGJson(right);
        end);
        Add(candidates, rec(
            orbit := rec(
                branches := List(orbitPositions, MathPSGPositionGeometry),
                primitive_orbit_size := Length(orbitPositions)
            ),
            stabilizer := rec(
                embedded_elements := embedded,
                order := Length(embedded)
            )
        ));
    od;
    generators := List(GeneratorsOfGroup(group), MathPSGAffineFromRight);
    Sort(generators, function(left, right)
        return MathPSGJson(left) < MathPSGJson(right);
    end);
    return rec(
        candidates := candidates,
        space_group := rec(
            setting := MathPSGSettingString(request.setting)
        ),
        space_group_action := rec(
            source_generators := generators,
            translation_basis := MathPSGRationalMatrix(
                TransposedMat(TranslationBasis(group))
            )
        )
    );
end;
