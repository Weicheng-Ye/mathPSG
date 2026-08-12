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

MathPSGClassifierRequiredLock :=
    "c92c0cef1c72a061a642ccdbb297adafd52ffad2779a84755d9e626363edb25d";
MathPSGClassifierRequiredOCI :=
    "sha256:726b772a1aae0cfa22fd3cdba89bb424c65eed01744a265a0f55078649a2b95d";

MathPSGClassifierRuntimeManifestPath :=
    "/opt/mathpsg/classifier-gap/runtime-provenance.json";
MathPSGClassifierLockPath := "/opt/mathpsg/classifier-gap.lock.json";
MathPSGClassifierRuntimeManifest := fail;
MathPSGClassifierRuntimeManifestBytes := fail;
MathPSGClassifierLockedRuntime := false;
if IsExistingFile(MathPSGClassifierRuntimeManifestPath)
   and IsExistingFile(MathPSGClassifierLockPath)
   and MathPSGClassifierHexSHA256(StringFile(MathPSGClassifierLockPath))
       = MathPSGClassifierRequiredLock then
    MathPSGClassifierRuntimeManifestBytes :=
        StringFile(MathPSGClassifierRuntimeManifestPath);
    MathPSGClassifierRuntimeManifestResult := CALL_WITH_CATCH(
        JsonStringToGap, [MathPSGClassifierRuntimeManifestBytes]
    );
    if MathPSGClassifierRuntimeManifestResult[1] = true then
        MathPSGClassifierRuntimeManifest :=
            MathPSGClassifierRuntimeManifestResult[2];
        MathPSGClassifierLockedRuntime :=
            IsRecord(MathPSGClassifierRuntimeManifest)
            and IsBound(MathPSGClassifierRuntimeManifest.schema_version)
            and MathPSGClassifierRuntimeManifest.schema_version = 1
            and IsBound(MathPSGClassifierRuntimeManifest.lock_digest)
            and MathPSGClassifierRuntimeManifest.lock_digest
                = Concatenation("sha256:", MathPSGClassifierRequiredLock)
            and IsBound(MathPSGClassifierRuntimeManifest.base_image)
            and IsBound(MathPSGClassifierRuntimeManifest.base_image.index_digest)
            and MathPSGClassifierRuntimeManifest.base_image.index_digest
                = MathPSGClassifierRequiredOCI
            and IsBound(MathPSGClassifierRuntimeManifest.external_runtime)
            and IsBound(
                MathPSGClassifierRuntimeManifest.external_runtime.nq_executable
            )
            and MathPSGClassifierRuntimeManifest.external_runtime
                .nq_executable.provider = "pinned_oci_image"
            and IsBound(MathPSGClassifierRuntimeManifest.external_runtime.polymake)
            and MathPSGClassifierRuntimeManifest.external_runtime.polymake.provider
                = "excluded_by_authenticated_api_closure";
    fi;
fi;
MathPSGClassifierDiagnosticRuntime :=
    IsBound(GAPInfo.SystemEnvironment.MATHPSG_CLASSIFIER_DIAGNOSTIC)
    and GAPInfo.SystemEnvironment.MATHPSG_CLASSIFIER_DIAGNOSTIC = "1";

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
   or MathPSGClassifierHAPcrystLoad <> true
   or not (MathPSGClassifierLockedRuntime or MathPSGClassifierDiagnosticRuntime) then
    MathPSGClassifierWrite(
        MathPSGClassifierResponsePath,
        MathPSGClassifierFailureResponse(
            Concatenation("sha256:", String(ListWithIdenticalEntries(64, '0'))),
            "backend_failed", "environment", "classifier environment is not locked"
        )
    );
    QUIT_GAP(2);
fi;

Read("gap/classifier/lib/affine_pcp.g");

MathPSGClassifierEnvironment := function()
    local core, result;
    core := rec(
        execution_mode := "diagnostic_local",
        lock_digest := Concatenation("sha256:", MathPSGClassifierRequiredLock),
        oci_image_digest := MathPSGClassifierRequiredOCI,
        packages := [
            rec(
                archive_sha256 := "sha256:90aae4bf7eabdb94bceebef0d984c8d6ea9e9c60d8268913498526565b693a7f",
                license_sha256 := "sha256:e9c68e5cf6425d8749ca7112dcd96049a25bfdf055c39ddf800456dc12353c01",
                name := "Cryst", version := "4.1.30"
            ),
            rec(
                archive_sha256 := "sha256:2a81d008e1638f638a035b1cd981ca39436bdabbef8c29b15b24fceb2af678e4",
                license_sha256 := "sha256:8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643",
                name := "GAP", version := "4.15.1"
            ),
            rec(
                archive_sha256 := "sha256:300e776141be73f807a2fbdfc0ce45d871c8d4a765dc2ca3b49ba38db9d51861",
                license_sha256 := "sha256:edaef632cbb643e4e7a221717a6c441a4c1a7c918e6e4d56debc3d8739b233f6",
                name := "HAP", version := "1.70"
            ),
            rec(
                archive_sha256 := "sha256:dda392457ecc9fcffd7d86b3633da455e9fe65118d7bcf4039cc5d4d05edfc94",
                license_sha256 := "sha256:ab15fd526bd8dd18a9e77ebc139656bf4d33e97fc7238cd11bf60e2b9b8666c6",
                name := "HAPcryst", version := "0.1.15"
            )
        ],
        release_certified := false,
        runtime_provenance_digest := fail
    );
    if MathPSGClassifierLockedRuntime then
        core.execution_mode := "locked_image";
        core.runtime_provenance_digest := Concatenation(
            "sha256:",
            MathPSGClassifierHexSHA256(MathPSGClassifierRuntimeManifestBytes)
        );
    fi;
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
