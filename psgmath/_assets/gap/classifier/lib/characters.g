#############################################################################
## Exact mod-two presentation reduction with replayable row/column witnesses.
#############################################################################

MathPSGClassifierTask5IdentityGF2 := function(size)
    return List([1..size], row -> List([1..size], column ->
        (function() if row = column then return 1; else return 0; fi; end)()
    ));
end;

MathPSGClassifierTask5CharacterReduction := function(matrix, columns)
    local work, rows, rowChange, columnChange, rank, pivot, i, j, tmp;
    work := List(matrix, row -> List(row, value -> value mod 2));
    rows := Length(work);
    rowChange := MathPSGClassifierTask5IdentityGF2(rows);
    columnChange := MathPSGClassifierTask5IdentityGF2(columns);
    rank := 0;
    while rank < Minimum(rows, columns) do
        pivot := fail;
        for i in [rank + 1..rows] do
            for j in [rank + 1..columns] do
                if work[i][j] = 1 then pivot := [i, j]; break; fi;
            od;
            if pivot <> fail then break; fi;
        od;
        if pivot = fail then break; fi;
        rank := rank + 1;
        tmp := work[rank]; work[rank] := work[pivot[1]]; work[pivot[1]] := tmp;
        tmp := rowChange[rank]; rowChange[rank] := rowChange[pivot[1]];
        rowChange[pivot[1]] := tmp;
        if pivot[2] <> rank then
            for i in [1..rows] do
                tmp := work[i][rank]; work[i][rank] := work[i][pivot[2]];
                work[i][pivot[2]] := tmp;
            od;
            for i in [1..columns] do
                tmp := columnChange[i][rank];
                columnChange[i][rank] := columnChange[i][pivot[2]];
                columnChange[i][pivot[2]] := tmp;
            od;
        fi;
        for i in [1..rows] do
            if i <> rank and work[i][rank] = 1 then
                work[i] := List([1..columns], j -> (work[i][j] + work[rank][j]) mod 2);
                rowChange[i] := List([1..rows], j ->
                    (rowChange[i][j] + rowChange[rank][j]) mod 2
                );
            fi;
        od;
        for j in [1..columns] do
            if j <> rank and work[rank][j] = 1 then
                for i in [1..rows] do work[i][j] := (work[i][j] + work[i][rank]) mod 2; od;
                for i in [1..columns] do
                    columnChange[i][j] := (columnChange[i][j] + columnChange[i][rank]) mod 2;
                od;
            fi;
        od;
    od;
    return rec(
        column_change := columnChange,
        normal_form := work,
        rank := rank,
        row_change := rowChange
    );
end;

MathPSGClassifierTask5CharacterCertificateCore := function(
    groupId, resolutionId, presentationKind, presentationDigest,
    generatorOrder, relatorWords, relatorMatrix
)
    local reduction;
    reduction := MathPSGClassifierTask5CharacterReduction(
        relatorMatrix, Length(generatorOrder)
    );
    return rec(
        column_change := reduction.column_change,
        generator_order := generatorOrder,
        group_id := groupId,
        normal_form := reduction.normal_form,
        presentation_digest := presentationDigest,
        presentation_kind := presentationKind,
        relator_matrix_mod2 := relatorMatrix,
        relator_words := relatorWords,
        resolution_id := resolutionId,
        row_change := reduction.row_change
    );
end;
