if LoadPackage("cryst", false : OnlyNeeded) <> true then
    Print("{\"error\":\"Cryst is required\",\"status\":\"error\"}\n");
    QUIT_GAP(2);
fi;

Read("gap/catalogue/lib/normalize_affine.g");

MathPSGNumberText := MathPSGArgumentValue("--international-number");
MathPSGOutputPath := MathPSGArgumentValue("--json-output");
if MathPSGOutputPath = fail or MathPSGNumberText = fail
   or not ForAll(MathPSGNumberText, IsDigitChar) then
    MathPSGWriteError(MathPSGOutputPath, "invalid catalogue arguments");
    QUIT_GAP(2);
fi;

MathPSGNumber := Int(MathPSGNumberText);
if MathPSGNumber < 1 or MathPSGNumber > 230 then
    MathPSGWriteError(MathPSGOutputPath, "space-group number must be in 1..230");
    QUIT_GAP(2);
fi;

MathPSGResult := CALL_WITH_CATCH(MathPSGExportOne, [MathPSGNumber]);
if MathPSGResult[1] <> true
   or not MathPSGWriteEncodedFile(
       MathPSGOutputPath,
       Concatenation(MathPSGJson(MathPSGResult[2]), "\n")
   ) then
    MathPSGWriteError(MathPSGOutputPath, "catalogue computation failed");
    QUIT_GAP(2);
fi;
QUIT_GAP(0);
