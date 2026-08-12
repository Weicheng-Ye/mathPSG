MathPSGCrystLoadInfo := rec();;
MathPSGCrystLoadResult := LoadPackage(
    "cryst",
    "=4.1.30",
    false : OnlyNeeded, LoadInfo := MathPSGCrystLoadInfo
);
Read("gap/catalogue/lib/normalize_affine.g");

MathPSGNumberText := MathPSGArgumentValue("--international-number");
MathPSGOutputPath := MathPSGArgumentValue("--json-output");
MathPSGEvidencePath := MathPSGArgumentValue("--environment-evidence-output");

if not MathPSGValidateOptionPairs([
    "--environment-evidence-output",
    "--international-number",
    "--json-output"
]) then
    MathPSGWriteError(MathPSGOutputPath, "invalid-arguments", "unknown, duplicate, or incomplete exporter option");
    QUIT_GAP(2);
fi;

if MathPSGOutputPath = fail then
    MathPSGWriteError(fail, "invalid-arguments", "--json-output must occur exactly once");
    QUIT_GAP(2);
fi;

if MathPSGNumberText = fail or not ForAll(MathPSGNumberText, IsDigitChar) then
    MathPSGWriteError(MathPSGOutputPath, "invalid-arguments", "--international-number must be an integer from 1 through 230");
    QUIT_GAP(2);
fi;

MathPSGNumber := Int(MathPSGNumberText);
if MathPSGNumber < 1 or MathPSGNumber > 230 then
    MathPSGWriteError(MathPSGOutputPath, "invalid-space-group", "International number must be from 1 through 230");
    QUIT_GAP(2);
fi;

if MathPSGCrystLoadResult <> true then
    MathPSGWriteError(MathPSGOutputPath, "missing-cryst", "Cryst 4.1.30 is required");
    QUIT_GAP(2);
fi;

MathPSGResult := CALL_WITH_CATCH(MathPSGExportOne, [MathPSGNumber]);
if MathPSGResult[1] <> true then
    MathPSGWriteError(MathPSGOutputPath, "catalogue-export-failed", "Cryst export failed");
    QUIT_GAP(2);
fi;

if MathPSGEvidencePath <> fail and not MathPSGWriteEncodedFile(
    MathPSGEvidencePath,
    Concatenation(MathPSGJson(MathPSGEnvironmentEvidence()), "\n")
) then
    MathPSGWriteError(
        fail,
        "environment-evidence-write-failed",
        "GAP environment evidence could not be written completely"
    );
    QUIT_GAP(2);
fi;

if not MathPSGWriteEncodedFile(
    MathPSGOutputPath,
    Concatenation(MathPSGJson(MathPSGResult[2]), "\n")
) then
    MathPSGWriteError(
        fail,
        "output-write-failed",
        "catalogue output could not be written completely"
    );
    QUIT_GAP(2);
fi;
QUIT_GAP(0);
