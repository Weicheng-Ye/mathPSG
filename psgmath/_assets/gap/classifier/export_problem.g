MathPSGClassifierJsonLoad := LoadPackage(
    "json", "=2.2.3", false : OnlyNeeded
);
Read("gap/classifier/lib/protocol.g");

MathPSGClassifierRequestPath := MathPSGClassifierArgumentValue("--request");
MathPSGClassifierResponsePath := MathPSGClassifierArgumentValue("--response");

if not MathPSGClassifierValidateArguments()
   or MathPSGClassifierRequestPath = fail
   or MathPSGClassifierResponsePath = fail then
    if MathPSGClassifierResponsePath <> fail then
        MathPSGClassifierWrite(
            MathPSGClassifierResponsePath,
            MathPSGClassifierFailureResponse(
                Concatenation("sha256:", String(ListWithIdenticalEntries(64, '0'))),
                "invalid_request", "protocol", "classifier arguments are invalid"
            )
        );
    fi;
    QUIT_GAP(2);
fi;

MathPSGClassifierCrystLoad := LoadPackage(
    "cryst", "=4.1.30", false : OnlyNeeded
);
MathPSGClassifierHAPLoad := LoadPackage(
    "hap", "=1.70", false : OnlyNeeded
);
MathPSGClassifierHAPcrystLoad := LoadPackage(
    "hapcryst", "=0.1.15", false : OnlyNeeded
);

if GAPInfo.Version <> "4.15.1"
   or MathPSGClassifierJsonLoad <> true
   or MathPSGClassifierCrystLoad <> true
   or MathPSGClassifierHAPLoad <> true
   or MathPSGClassifierHAPcrystLoad <> true then
    MathPSGClassifierWrite(
        MathPSGClassifierResponsePath,
        MathPSGClassifierFailureResponse(
            Concatenation("sha256:", String(ListWithIdenticalEntries(64, '0'))),
            "backend_failed", "environment", "required local GAP packages are unavailable"
        )
    );
    QUIT_GAP(2);
fi;

Read("gap/classifier/lib/affine_pcp.g");

MathPSGClassifierEnvironment := function()
    local core, result;
    core := rec(
        execution_mode := "diagnostic_local",
        packages := [
            rec(
                name := "Cryst", version := "4.1.30"
            ),
            rec(
                name := "GAP", version := "4.15.1"
            ),
            rec(
                name := "HAP", version := "1.70"
            ),
            rec(
                name := "HAPcryst", version := "0.1.15"
            )
        ],
        release_certified := false,
        runtime_provenance_digest := fail
    );
    result := ShallowCopy(core);
    result.environment_id := MathPSGClassifierDigest("classifier-environment-v1", core);
    return result;
end;

MathPSGClassifierRun := function()
    local request, certificate;
    request := MathPSGClassifierParseRequest(MathPSGClassifierRequestPath);
    certificate := MathPSGClassifierBuildCertificate(request);
    return rec(
        affine_pcp_certificate := certificate,
        environment := MathPSGClassifierEnvironment(),
        failures := [],
        problem := fail,
        protocol_version := 1,
        record_type := "gap-classifier-response",
        request_digest := request.request_digest,
        status := "conversion_only"
    );
end;

MathPSGClassifierResult := CALL_WITH_CATCH(MathPSGClassifierRun, []);
if MathPSGClassifierResult[1] <> true then
    MathPSGClassifierWrite(
        MathPSGClassifierResponsePath,
        MathPSGClassifierFailureResponse(
            Concatenation("sha256:", String(ListWithIdenticalEntries(64, '0'))),
            "backend_failed", "certificate", "classifier construction failed"
        )
    );
    QUIT_GAP(2);
fi;

if not MathPSGClassifierWrite(
    MathPSGClassifierResponsePath, MathPSGClassifierResult[2]
) then
    QUIT_GAP(2);
fi;
QUIT_GAP(0);
