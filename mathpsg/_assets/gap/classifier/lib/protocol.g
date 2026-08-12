#############################################################################
## Strict canonical JSON and exact-affine helpers for classifier protocol v1.
#############################################################################

MathPSGClassifierJsonEscape := function(value)
    local out, character, code;
    out := "";
    for character in value do
        code := INT_CHAR(character);
        if character = '"' then Append(out, "\\\"");
        elif character = '\\' then Append(out, "\\\\");
        elif character = '\n' then Append(out, "\\n");
        elif character = '\r' then Append(out, "\\r");
        elif character = '\t' then Append(out, "\\t");
        elif code < 32 then Error("unsupported JSON control character");
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
    Error("unsupported classifier JSON value");
end;

MathPSGClassifierHexSHA256 := function(value)
    local digest;
    digest := LowercaseString(HexSHA256(value));
    if Length(digest) > 64 then Error("SHA-256 encoding exceeds 64 digits"); fi;
    return Concatenation(
        String(ListWithIdenticalEntries(64 - Length(digest), '0')), digest
    );
end;

MathPSGClassifierDigest := function(domain, value)
    return Concatenation(
        "sha256:",
        MathPSGClassifierHexSHA256(Concatenation(
            "mathpsg-gap-classifier-v1|", domain, "|",
            MathPSGClassifierJson(value)
        ))
    );
end;

MathPSGClassifierArgumentValue := function(name)
    local args, separators, tail, positions;
    args := GAPInfo.SystemCommandLine;
    separators := Positions(args, "--");
    if Length(separators) <> 1 then return fail; fi;
    tail := args{[separators[1] + 1..Length(args)]};
    positions := Positions(tail, name);
    if Length(positions) <> 1 or positions[1] = Length(tail) then return fail; fi;
    return tail[positions[1] + 1];
end;

MathPSGClassifierValidateArguments := function()
    local args, separators, tail, names;
    args := GAPInfo.SystemCommandLine;
    separators := Positions(args, "--");
    if Length(separators) <> 1 then return false; fi;
    tail := args{[separators[1] + 1..Length(args)]};
    if Length(tail) <> 4 then return false; fi;
    names := tail{[1, 3]};
    return Set(names) = Set(["--request", "--response"]);
end;

MathPSGClassifierWrite := function(path, value)
    local encoded, result;
    encoded := MathPSGClassifierJson(value);
    result := CALL_WITH_CATCH(FileString, [path, encoded]);
    return result[1] = true and result[2] = Length(encoded);
end;

MathPSGClassifierFailureResponse := function(requestDigest, code, stage, message)
    return rec(
        affine_pcp_certificate := fail,
        environment := fail,
        failures := [rec(code := code, context := rec(), message := message, stage := stage)],
        problem := fail,
        protocol_version := 1,
        record_type := "gap-classifier-response",
        request_digest := requestDigest,
        status := "error"
    );
end;

MathPSGClassifierIsDigest := function(value)
    local hexadecimal;
    if not IsString(value) or Length(value) <> 71
       or value{[1..7]} <> "sha256:" then return false; fi;
    hexadecimal := "0123456789abcdef";
    return ForAll(value{[8..71]}, character -> character in hexadecimal);
end;

MathPSGClassifierRequireFields := function(value, names)
    return IsRecord(value) and SortedList(RecNames(value)) = SortedList(names);
end;

MathPSGClassifierRational := function(value)
    local inner, pieces, numerator, denominator, result;
    if not IsString(value) or Length(value) < 6
       or value{[1..2]} <> "q(" or value[Length(value)] <> ')' then
        Error("invalid exact rational spelling");
    fi;
    inner := value{[3..Length(value) - 1]};
    pieces := SplitString(inner, ",");
    if Length(pieces) <> 2 or IsEmpty(pieces[1]) or IsEmpty(pieces[2]) then
        Error("invalid exact rational spelling");
    fi;
    numerator := Int(pieces[1]);
    denominator := Int(pieces[2]);
    if numerator = fail or denominator = fail or denominator <= 0 then
        Error("invalid exact rational spelling");
    fi;
    result := numerator / denominator;
    if Concatenation(
        "q(", String(NumeratorRat(result)), ",", String(DenominatorRat(result)), ")"
    ) <> value then Error("noncanonical exact rational spelling"); fi;
    return result;
end;

MathPSGClassifierRationalString := value -> Concatenation(
    "q(", String(NumeratorRat(value)), ",", String(DenominatorRat(value)), ")"
);

MathPSGClassifierAffineRight := function(value)
    local matrix, translation, right, row;
    if not MathPSGClassifierRequireFields(value, ["matrix", "translation"])
       or not IsList(value.matrix) or Length(value.matrix) <> 3
       or not ForAll(value.matrix, item -> IsList(item) and Length(item) = 3)
       or not IsList(value.translation) or Length(value.translation) <> 3 then
        Error("invalid affine transformation shape");
    fi;
    matrix := List(value.matrix, row -> List(row, MathPSGClassifierRational));
    translation := List(value.translation, MathPSGClassifierRational);
    if DeterminantMat(matrix) = 0 then Error("singular affine linear part"); fi;
    right := List(TransposedMat(matrix), row -> Concatenation(row, [0]));
    Add(right, Concatenation(translation, [1]));
    return right;
end;

MathPSGClassifierAffineColumn := function(value)
    local linear, translation;
    linear := TransposedMat(value{[1..3]}{[1..3]});
    translation := value[4]{[1..3]};
    return rec(
        matrix := List(linear, row -> List(row, MathPSGClassifierRationalString)),
        translation := List(translation, MathPSGClassifierRationalString)
    );
end;

MathPSGClassifierParseRequest := function(path)
    local encoded, parsed, result, request, actionCore, requestCore, generator,
          inclusion, element;
    encoded := StringFile(path);
    if encoded = fail then Error("request could not be read"); fi;
    parsed := CALL_WITH_CATCH(JsonStringToGap, [encoded]);
    if parsed[1] <> true or not IsRecord(parsed[2]) then
        Error("request is not strict JSON");
    fi;
    request := parsed[2];
    if MathPSGClassifierJson(request) <> encoded then
        Error("request bytes are not canonical JSON");
    fi;
    if not MathPSGClassifierRequireFields(request, [
        "action", "inclusions", "max_degree", "protocol_version", "record_type",
        "request_digest", "time_reversal"
    ]) or request.protocol_version <> 1
       or request.record_type <> "gap-classifier-request"
       or request.max_degree <> 4
       or not request.time_reversal in [true, false]
       or not MathPSGClassifierIsDigest(request.request_digest) then
        Error("invalid classifier request envelope");
    fi;
    if not MathPSGClassifierRequireFields(request.action, [
        "action_digest", "affine_generators", "translation_basis"
    ]) or not IsList(request.action.affine_generators)
       or IsEmpty(request.action.affine_generators)
       or not IsList(request.action.translation_basis) then
        Error("invalid classifier action envelope");
    fi;
    for generator in request.action.affine_generators do
        MathPSGClassifierAffineRight(generator);
    od;
    if Length(request.action.translation_basis) <> 3
       or not ForAll(request.action.translation_basis, row -> IsList(row) and Length(row) = 3)
       or DeterminantMat(List(
           request.action.translation_basis,
           row -> List(row, MathPSGClassifierRational)
       )) = 0 then Error("invalid translation basis"); fi;
    actionCore := rec(
        affine_generators := request.action.affine_generators,
        translation_basis := request.action.translation_basis
    );
    if request.action.action_digest <>
       MathPSGClassifierDigest("certified-space-group-action-v1", actionCore) then
        Error("action digest mismatch");
    fi;
    if not IsList(request.inclusions) or IsEmpty(request.inclusions) then
        Error("literal inclusions are absent");
    fi;
    for inclusion in request.inclusions do
        if not MathPSGClassifierRequireFields(inclusion, [
            "inclusion_id", "literal_element_digest", "literal_elements",
            "literal_stabilizer_digest"
        ]) or not IsString(inclusion.inclusion_id)
           or not MathPSGClassifierIsDigest(inclusion.literal_element_digest)
           or not MathPSGClassifierIsDigest(inclusion.literal_stabilizer_digest)
           or not IsList(inclusion.literal_elements) or IsEmpty(inclusion.literal_elements) then
            Error("invalid literal inclusion");
        fi;
        for element in inclusion.literal_elements do
            MathPSGClassifierAffineRight(element);
        od;
        if inclusion.literal_element_digest <>
           MathPSGClassifierDigest(
               "literal-stabilizer-authority-v1", inclusion.literal_elements
           ) then
            Error("literal element digest mismatch");
        fi;
    od;
    requestCore := rec(
        action := request.action,
        inclusions := request.inclusions,
        max_degree := request.max_degree,
        time_reversal := request.time_reversal
    );
    if request.request_digest <>
       MathPSGClassifierDigest("gap-classifier-request-v1", requestCore) then
        Error("request digest mismatch");
    fi;
    return request;
end;
