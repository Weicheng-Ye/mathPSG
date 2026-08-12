#############################################################################
## JSON and exact-affine helpers used by the physical calculation.
#############################################################################

MathPSGClassifierJsonEscape := function(value)
    local out, character;
    out := "";
    for character in value do
        if character = '"' then Append(out, "\\\"");
        elif character = '\\' then Append(out, "\\\\");
        elif character = '\n' then Append(out, "\\n");
        elif character = '\r' then Append(out, "\\r");
        elif character = '\t' then Append(out, "\\t");
        else Add(out, character);
        fi;
    od;
    return out;
end;

MathPSGClassifierJsonString := value -> Concatenation(
    "\"", MathPSGClassifierJsonEscape(value), "\""
);

MathPSGClassifierJson := function(value)
    local names, pieces, name;
    if value = true then return "true";
    elif value = false then return "false";
    elif value = fail then return "null";
    elif IsInt(value) then return String(value);
    elif IsString(value) and Length(value) > 0 then
        return MathPSGClassifierJsonString(value);
    elif IsRecord(value) then
        names := SortedList(RecNames(value));
        pieces := [];
        for name in names do
            Add(pieces, Concatenation(
                MathPSGClassifierJsonString(name), ":",
                MathPSGClassifierJson(value.(name))
            ));
        od;
        return Concatenation("{", JoinStringsWithSeparator(pieces, ","), "}");
    elif IsList(value) then
        return Concatenation(
            "[", JoinStringsWithSeparator(List(value, MathPSGClassifierJson), ","), "]"
        );
    fi;
    Error("unsupported JSON value");
end;

MathPSGClassifierRational := function(value)
    local pieces;
    pieces := SplitString(value{[3..Length(value) - 1]}, ",");
    return Int(pieces[1]) / Int(pieces[2]);
end;

MathPSGClassifierAffineRight := function(value)
    local matrix, translation, right, row;
    matrix := List(value.matrix, row -> List(row, MathPSGClassifierRational));
    translation := List(value.translation, MathPSGClassifierRational);
    right := List(TransposedMat(matrix), row -> Concatenation(row, [0]));
    Add(right, Concatenation(translation, [1]));
    return right;
end;
