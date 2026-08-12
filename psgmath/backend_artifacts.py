"""Canonical, capability-free recipes for production backend artifact replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from threading import RLock
from typing import Any, Generic, Protocol, TypeVar, Union
import weakref


_PROTOCOL = b"mathpsg-backend-artifact-v1|"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*\Z")
_MAX_RECIPE_BYTES = 64 * 1024
_ARTIFACT_KINDS = frozenset(
    {
        "ambient-resolution",
        "inclusion",
        "local-skeleton",
        "relative-layer",
    }
)
_MODES = frozenset({"Z2", "U1"})

ResultScalar = Union[str, int, bool, None]
ResultSummary = tuple[tuple[str, ResultScalar], ...]
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


_REPLAY_IDENTITY_LOCK = RLock()
_REPLAY_IDENTITIES: dict[str, list[weakref.ReferenceType[object]]] = {}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_digest(value: object, path: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected sha256 digest")
    return value


def _require_identifier(value: object, path: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected canonical identifier")
    return value


def _normalise_digests(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected a finite digest sequence")
    result = tuple(
        _require_digest(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise ValueError(f"{path}: expected at least one signed bundle")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{path}: expected unique canonical bundle order")
    return result


def _normalise_identifiers(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected a finite identifier sequence")
    return tuple(
        _require_identifier(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _normalise_bits(value: object, path: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}: expected GF(2) bits or null")
    result = tuple(value)
    if any(type(bit) is not int or bit not in (0, 1) for bit in result):
        raise ValueError(f"{path}: expected GF(2) bits")
    return result


def _normalise_result_summary(value: object, path: str) -> ResultSummary:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = tuple(value)
    else:
        raise TypeError(f"{path}: expected a result-summary mapping")
    normalised: list[tuple[str, ResultScalar]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise TypeError(f"{path}[{index}]: expected key/value pair")
        key = _require_identifier(pair[0], f"{path}[{index}].key")
        scalar = pair[1]
        if type(scalar) is int:
            if scalar < 0:
                raise ValueError(f"{path}.{key}: expected nonnegative summary integer")
        elif type(scalar) not in (str, bool) and scalar is not None:
            raise TypeError(
                f"{path}.{key}: summary values must be canonical JSON scalars"
            )
        if type(scalar) is str and not scalar:
            raise ValueError(f"{path}.{key}: expected nonempty summary text")
        normalised.append((key, scalar))
    result = tuple(sorted(normalised))
    if not result or len({key for key, _ in result}) != len(result):
        raise ValueError(f"{path}: expected a nonempty unique summary")
    summary = dict(result)
    _require_digest(summary.get("result_digest"), f"{path}.result_digest")
    return result


def _summary_mapping(value: ResultSummary) -> dict[str, ResultScalar]:
    return {key: scalar for key, scalar in value}


def _recipe_core(
    *,
    algorithm_id: str,
    artifact_kind: str,
    bundle_ids: tuple[str, ...],
    mode: str,
    request_digest: str,
    result_summary: ResultSummary,
    rho_bits: tuple[int, ...] | None,
    skeleton_ids: tuple[str, ...],
    store_manifest_digest: str,
    trust_root_digest: str,
) -> dict[str, Any]:
    return {
        "algorithm_id": algorithm_id,
        "artifact_kind": artifact_kind,
        "bundle_ids": list(bundle_ids),
        "mode": mode,
        "request_digest": request_digest,
        "result_summary": _summary_mapping(result_summary),
        "rho_bits": None if rho_bits is None else list(rho_bits),
        "skeleton_ids": list(skeleton_ids),
        "store_manifest_digest": store_manifest_digest,
        "trust_root_digest": trust_root_digest,
    }


def _recipe_digest(**core: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _PROTOCOL + b"artifact-recipe-v1|" + _canonical_json(core)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecipe:
    algorithm_id: str
    artifact_kind: str
    bundle_ids: tuple[str, ...]
    mode: str
    recipe_digest: str
    request_digest: str
    result_summary: ResultSummary
    rho_bits: tuple[int, ...] | None
    skeleton_ids: tuple[str, ...]
    store_manifest_digest: str
    trust_root_digest: str

    def __post_init__(self) -> None:
        algorithm = _require_identifier(self.algorithm_id, "$recipe.algorithm_id")
        if type(self.artifact_kind) is not str or self.artifact_kind not in _ARTIFACT_KINDS:
            raise ValueError("$recipe.artifact_kind: unsupported artifact kind")
        if type(self.mode) is not str or self.mode not in _MODES:
            raise ValueError("$recipe.mode: expected Z2 or U1")
        bundles = _normalise_digests(self.bundle_ids, "$recipe.bundle_ids")
        request = _require_digest(self.request_digest, "$recipe.request_digest")
        store = _require_digest(
            self.store_manifest_digest, "$recipe.store_manifest_digest"
        )
        root = _require_digest(self.trust_root_digest, "$recipe.trust_root_digest")
        rho = _normalise_bits(self.rho_bits, "$recipe.rho_bits")
        skeletons = _normalise_identifiers(
            self.skeleton_ids, "$recipe.skeleton_ids"
        )
        summary = _normalise_result_summary(
            self.result_summary, "$recipe.result_summary"
        )
        digest = _require_digest(self.recipe_digest, "$recipe.recipe_digest")
        object.__setattr__(self, "bundle_ids", bundles)
        object.__setattr__(self, "rho_bits", rho)
        object.__setattr__(self, "skeleton_ids", skeletons)
        object.__setattr__(self, "result_summary", summary)
        expected = _recipe_digest(
            **_recipe_core(
                algorithm_id=algorithm,
                artifact_kind=self.artifact_kind,
                bundle_ids=bundles,
                mode=self.mode,
                request_digest=request,
                result_summary=summary,
                rho_bits=rho,
                skeleton_ids=skeletons,
                store_manifest_digest=store,
                trust_root_digest=root,
            )
        )
        if digest != expected:
            raise ValueError("$recipe.recipe_digest: does not bind the recipe")


def make_artifact_recipe(
    *,
    artifact_kind: str,
    request_digest: str,
    store_manifest_digest: str,
    trust_root_digest: str,
    bundle_ids: Sequence[str],
    algorithm_id: str,
    mode: str,
    rho_bits: Sequence[int] | None,
    skeleton_ids: Sequence[str],
    result_summary: Mapping[str, ResultScalar] | Sequence[tuple[str, ResultScalar]],
) -> ArtifactRecipe:
    bundles = _normalise_digests(bundle_ids, "$recipe.bundle_ids")
    rho = _normalise_bits(rho_bits, "$recipe.rho_bits")
    skeletons = _normalise_identifiers(skeleton_ids, "$recipe.skeleton_ids")
    summary = _normalise_result_summary(result_summary, "$recipe.result_summary")
    core = _recipe_core(
        algorithm_id=_require_identifier(algorithm_id, "$recipe.algorithm_id"),
        artifact_kind=artifact_kind,
        bundle_ids=bundles,
        mode=mode,
        request_digest=_require_digest(request_digest, "$recipe.request_digest"),
        result_summary=summary,
        rho_bits=rho,
        skeleton_ids=skeletons,
        store_manifest_digest=_require_digest(
            store_manifest_digest, "$recipe.store_manifest_digest"
        ),
        trust_root_digest=_require_digest(
            trust_root_digest, "$recipe.trust_root_digest"
        ),
    )
    return ArtifactRecipe(
        algorithm_id=core["algorithm_id"],
        artifact_kind=core["artifact_kind"],
        bundle_ids=bundles,
        mode=core["mode"],
        recipe_digest=_recipe_digest(**core),
        request_digest=core["request_digest"],
        result_summary=summary,
        rho_bits=rho,
        skeleton_ids=skeletons,
        store_manifest_digest=core["store_manifest_digest"],
        trust_root_digest=core["trust_root_digest"],
    )


def artifact_recipe_mapping(recipe: ArtifactRecipe) -> dict[str, Any]:
    if type(recipe) is not ArtifactRecipe:
        raise TypeError("expected an exact ArtifactRecipe")
    # Reconstructing the dataclass replays normalization and the content hash,
    # including after hostile ``object.__setattr__`` mutation.
    checked = ArtifactRecipe(
        recipe.algorithm_id,
        recipe.artifact_kind,
        recipe.bundle_ids,
        recipe.mode,
        recipe.recipe_digest,
        recipe.request_digest,
        recipe.result_summary,
        recipe.rho_bits,
        recipe.skeleton_ids,
        recipe.store_manifest_digest,
        recipe.trust_root_digest,
    )
    return {
        **_recipe_core(
            algorithm_id=checked.algorithm_id,
            artifact_kind=checked.artifact_kind,
            bundle_ids=checked.bundle_ids,
            mode=checked.mode,
            request_digest=checked.request_digest,
            result_summary=checked.result_summary,
            rho_bits=checked.rho_bits,
            skeleton_ids=checked.skeleton_ids,
            store_manifest_digest=checked.store_manifest_digest,
            trust_root_digest=checked.trust_root_digest,
        ),
        "recipe_digest": checked.recipe_digest,
        "record_type": "mathpsg-artifact-recipe",
        "schema_version": 1,
    }


def dumps_artifact_recipe(recipe: ArtifactRecipe) -> bytes:
    return _canonical_json(artifact_recipe_mapping(recipe))


def _strict_json(data: bytes) -> Any:
    if type(data) is not bytes:
        raise TypeError("artifact recipe must be exact bytes")
    if len(data) > _MAX_RECIPE_BYTES:
        raise ValueError("artifact recipe exceeds the size bound")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"artifact recipe has duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=no_duplicates,
            parse_float=lambda token: (_ for _ in ()).throw(
                ValueError("artifact recipe floating-point JSON is forbidden")
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("artifact recipe non-finite JSON is forbidden")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact recipe is invalid strict JSON") from error
    if _canonical_json(value) != data:
        raise ValueError("artifact recipe bytes are not canonical JSON")
    return value


def loads_artifact_recipe(data: bytes) -> ArtifactRecipe:
    value = _strict_json(data)
    if not isinstance(value, Mapping):
        raise TypeError("artifact recipe root must be an object")
    expected = {
        "algorithm_id",
        "artifact_kind",
        "bundle_ids",
        "mode",
        "recipe_digest",
        "record_type",
        "request_digest",
        "result_summary",
        "rho_bits",
        "schema_version",
        "skeleton_ids",
        "store_manifest_digest",
        "trust_root_digest",
    }
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"artifact recipe missing field {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"artifact recipe has unexpected field {sorted(extra)[0]}")
    if (
        type(value["record_type"]) is not str
        or value["record_type"] != "mathpsg-artifact-recipe"
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("artifact recipe has an unsupported schema")
    if not isinstance(value["bundle_ids"], list):
        raise TypeError("$recipe.bundle_ids: expected array")
    if value["rho_bits"] is not None and not isinstance(value["rho_bits"], list):
        raise TypeError("$recipe.rho_bits: expected array or null")
    if not isinstance(value["skeleton_ids"], list):
        raise TypeError("$recipe.skeleton_ids: expected array")
    if not isinstance(value["result_summary"], Mapping):
        raise TypeError("$recipe.result_summary: expected object")
    return ArtifactRecipe(
        algorithm_id=value["algorithm_id"],
        artifact_kind=value["artifact_kind"],
        bundle_ids=tuple(value["bundle_ids"]),
        mode=value["mode"],
        recipe_digest=value["recipe_digest"],
        request_digest=value["request_digest"],
        result_summary=tuple(value["result_summary"].items()),
        rho_bits=None if value["rho_bits"] is None else tuple(value["rho_bits"]),
        skeleton_ids=tuple(value["skeleton_ids"]),
        store_manifest_digest=value["store_manifest_digest"],
        trust_root_digest=value["trust_root_digest"],
    )


@dataclass(frozen=True, slots=True)
class ArtifactReplay(Generic[T]):
    """One typed resolver result; its resolver cannot attest its own summary."""

    value: T


class ArtifactRecipeResolver(Protocol[T_co]):
    def __call__(self, recipe: ArtifactRecipe, /) -> ArtifactReplay[T_co]: ...


class ArtifactResultSummarizer(Protocol[T_contra]):
    """Caller-trusted derivation of a canonical summary from a typed value."""

    def __call__(
        self, value: T_contra, /
    ) -> Mapping[str, ResultScalar] | Sequence[tuple[str, ResultScalar]]: ...


def _claim_fresh_replay(recipe_digest: str, value: object) -> None:
    """Reject reuse of any still-live typed value for the same trusted recipe."""

    def discard(reference: weakref.ReferenceType[object]) -> None:
        with _REPLAY_IDENTITY_LOCK:
            current = _REPLAY_IDENTITIES.get(recipe_digest)
            if current is None:
                return
            live = [
                item
                for item in current
                if item is not reference and item() is not None
            ]
            if live:
                _REPLAY_IDENTITIES[recipe_digest] = live
            else:
                _REPLAY_IDENTITIES.pop(recipe_digest, None)

    try:
        issued = weakref.ref(value, discard)
    except TypeError as error:
        raise TypeError(
            "artifact replay values must support weak references for bounded freshness"
        ) from error
    with _REPLAY_IDENTITY_LOCK:
        live: list[weakref.ReferenceType[object]] = []
        for reference in _REPLAY_IDENTITIES.get(recipe_digest, ()):
            previous = reference()
            if previous is None:
                continue
            if previous is value:
                raise ValueError(
                    "artifact resolver reused a stale value instead of fresh replay"
                )
            live.append(reference)
        live.append(issued)
        _REPLAY_IDENTITIES[recipe_digest] = live


def resolve_artifact_recipe(
    data: bytes,
    *,
    expected_recipe_digest: str,
    resolver: ArtifactRecipeResolver[T],
    result_summarizer: ArtifactResultSummarizer[T],
) -> T:
    """Resolve one trusted recipe and independently summarize a fresh value."""

    recipe = loads_artifact_recipe(data)
    trusted_digest = _require_digest(
        expected_recipe_digest, "$expected_recipe_digest"
    )
    if recipe.recipe_digest != trusted_digest:
        raise ValueError(
            "artifact recipe digest differs from the caller-trusted binding"
        )
    if not callable(resolver):
        raise TypeError("artifact recipe resolver must be callable")
    if not callable(result_summarizer):
        raise TypeError("artifact result summarizer must be callable")
    trusted_recipe = loads_artifact_recipe(dumps_artifact_recipe(recipe))
    replay = resolver(recipe)
    if type(replay) is not ArtifactReplay:
        raise TypeError("artifact recipe resolver must return ArtifactReplay")
    try:
        replayed_recipe = loads_artifact_recipe(dumps_artifact_recipe(recipe))
    except (TypeError, ValueError) as error:
        raise ValueError("artifact resolver mutated the parsed recipe") from error
    if replayed_recipe != trusted_recipe:
        raise ValueError("artifact resolver mutated the parsed recipe")
    summary = _normalise_result_summary(
        result_summarizer(replay.value), "$artifact_replay.result_summary"
    )
    if summary != trusted_recipe.result_summary:
        raise ValueError("artifact replay result summary differs from the recipe")
    _claim_fresh_replay(trusted_digest, replay.value)
    return replay.value


__all__ = [
    "ArtifactRecipe",
    "ArtifactRecipeResolver",
    "ArtifactReplay",
    "ArtifactResultSummarizer",
    "artifact_recipe_mapping",
    "dumps_artifact_recipe",
    "loads_artifact_recipe",
    "make_artifact_recipe",
    "resolve_artifact_recipe",
]
