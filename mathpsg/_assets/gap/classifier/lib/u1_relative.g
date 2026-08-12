#############################################################################
## Diagnostic coefficient twisting of the full exported group-ring matrices.
#############################################################################

MathPSGClassifierTask5TwistMatrix := function(matrix, character)
    local dense, entry, term;
    dense := List([1..matrix.row_count], row ->
        ListWithIdenticalEntries(matrix.column_count, 0)
    );
    for entry in matrix.entries do
        for term in entry.terms do
            dense[entry.row + 1][entry.column + 1] :=
                dense[entry.row + 1][entry.column + 1]
                + term[1] * character(term[2]);
        od;
    od;
    return rec(
        column_count := matrix.column_count,
        rows := dense,
        row_count := matrix.row_count
    );
end;

MathPSGClassifierTask5TwistedDiagnostics := function(boundaries, character)
    return List(boundaries, matrix ->
        MathPSGClassifierTask5TwistMatrix(matrix, character)
    );
end;
