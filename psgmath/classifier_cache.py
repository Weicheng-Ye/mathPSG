"""Exact dependency-keyed, atomic caches for certified classifier artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Literal


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROTOCOL = b"mathpsg-classifier-cache-v1|"
_CACHE_SCHEMA_VERSION = 1


class CacheCorruptError(ValueError):
    """Raised when a cache value or dependency witness cannot be trusted."""


_DEPENDENCY_SCHEMAS: dict[str, frozenset[str]] = {
    "ambient-resolution": frozenset(
        {
            "affine_pcp_conversion",
            "catalogue_normalization",
            "external_artifacts",
            "gap_environment",
            "group_action",
            "artifact_plan",
            "time_reversal",
        }
    ),
    "z2-local-skeleton": frozenset(
        {
            "local_library",
            "artifact_plan",
            "external_artifacts",
            "restricted_grade",
            "stabilizer_normalization",
            "stabilizer_table",
            "target_model",
        }
    ),
    "u1-local-skeleton": frozenset(
        {
            "derived_q",
            "external_artifacts",
            "local_library",
            "artifact_plan",
            "restricted_grade",
            "restricted_rho",
            "stabilizer_normalization",
            "stabilizer_table",
            "target_model",
        }
    ),
    "inclusion": frozenset(
        {
            "affine_pcp_transport",
            "ambient_resolution",
            "catalogue_manifest",
            "catalogue_normalization",
            "catalogue_record",
            "external_artifacts",
            "group_action",
            "literal_stabilizer",
            "artifact_plan",
        }
    ),
    "relative-layer": frozenset(
        {
            "affine_pcp_transport",
            "ambient_resolution",
            "catalogue_manifest",
            "catalogue_normalization",
            "catalogue_record_set",
            "external_artifacts",
            "group_action",
            "igg",
            "inclusion_set",
            "local_library",
            "artifact_plan",
            "rho",
            "symbolic_orbit_tuple",
            "target_model",
        }
    ),
    "routing-verification": frozenset(
        {
            "affine_transport",
            "candidate_record_set",
            "catalogue_manifest",
            "comparison_algorithm",
            "geometry_normalization",
            "group_action",
            "group_setting",
            "request",
            "routing_schema",
            "structural_result",
            "verifier_library",
        }
    ),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: str, value: object) -> str:
    return "sha256:" + hashlib.sha256(
        _PROTOCOL + domain.encode("ascii") + b"|" + _canonical_json(value)
    ).hexdigest()


def _raw_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_digest(value: object, path: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{path}: expected sha256 digest")
    return value


def _strict_json(data: bytes, *, context: str) -> Any:
    if type(data) is not bytes:
        raise CacheCorruptError(f"{context}: expected bytes")

    def pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CacheCorruptError(f"{context}: duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs_without_duplicates,
            parse_float=lambda token: (_ for _ in ()).throw(
                CacheCorruptError(f"{context}: floating-point JSON is forbidden")
            ),
            parse_constant=lambda token: (_ for _ in ()).throw(
                CacheCorruptError(f"{context}: non-finite JSON is forbidden")
            ),
        )
    except CacheCorruptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CacheCorruptError(f"{context}: invalid strict JSON") from error
    try:
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as error:
        raise CacheCorruptError(f"{context}: unsupported JSON value") from error
    if canonical != data:
        raise CacheCorruptError(f"{context}: bytes are not canonical JSON")
    return value


def _key_mapping(key: "CacheKey") -> dict[str, object]:
    return {
        "algorithm_digest": key.algorithm_digest,
        "artifact_kind": key.artifact_kind,
        "dependency_digests": [list(item) for item in key.dependency_digests],
        "key_digest": key.digest,
        "schema_version": key.schema_version,
    }


@dataclass(frozen=True, slots=True)
class CacheKey:
    artifact_kind: str
    schema_version: int
    algorithm_digest: str
    dependency_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        expected = _DEPENDENCY_SCHEMAS.get(self.artifact_kind)
        if expected is None:
            raise ValueError("$CacheKey.artifact_kind: unsupported artifact kind")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("$CacheKey.schema_version: expected version 1")
        _require_digest(self.algorithm_digest, "$CacheKey.algorithm_digest")
        dependencies = tuple(tuple(item) for item in self.dependency_digests)
        if any(
            len(item) != 2 or type(item[0]) is not str or type(item[1]) is not str
            for item in dependencies
        ):
            raise TypeError("$CacheKey.dependency_digests: expected name/digest pairs")
        if dependencies != tuple(sorted(dependencies)):
            raise ValueError("$CacheKey.dependency_digests: expected canonical sorted order")
        names = tuple(name for name, _ in dependencies)
        if len(set(names)) != len(names):
            raise ValueError("$CacheKey.dependency_digests: duplicate dependency name")
        actual = set(names)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            raise ValueError(
                "$CacheKey.dependency_digests: missing dependency " + missing[0]
            )
        if unexpected:
            raise ValueError(
                "$CacheKey.dependency_digests: unexpected dependency " + unexpected[0]
            )
        for name, digest in dependencies:
            if not name or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
                raise ValueError(f"$CacheKey.dependency_digests.{name}: invalid name")
            _require_digest(digest, f"$CacheKey.dependency_digests.{name}")
        object.__setattr__(self, "dependency_digests", dependencies)

    @property
    def digest(self) -> str:
        return _digest(
            "cache-key",
            {
                "algorithm_digest": self.algorithm_digest,
                "artifact_kind": self.artifact_kind,
                "dependency_digests": [list(item) for item in self.dependency_digests],
                "schema_version": self.schema_version,
            },
        )


def _bits(value: Sequence[int], path: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{path}: expected a bit sequence")
    result = tuple(value)
    if not result or any(type(bit) is not int or bit not in (0, 1) for bit in result):
        raise ValueError(f"{path}: expected a nonempty exact bit sequence")
    return result


def _bit_digest(domain: str, value: Sequence[int]) -> str:
    return _digest(domain, list(value))


def make_local_skeleton_cache_key(
    igg: Literal["Z2", "U1"],
    *,
    algorithm_digest: str,
    target_model_digest: str,
    stabilizer_table_digest: str,
    stabilizer_normalization_digest: str,
    local_library_digest: str,
    plan_digest: str,
    external_artifact_provenance_digest: str,
    restricted_grade: Sequence[int],
    restricted_rho: Sequence[int] | None = None,
    derived_q: Sequence[int] | None = None,
) -> CacheKey:
    """Build the exact local-library key in canonical local table order."""

    if igg not in ("Z2", "U1"):
        raise ValueError("igg must be Z2 or U1")
    grade = _bits(restricted_grade, "restricted_grade")
    dependencies = {
        "local_library": _require_digest(local_library_digest, "local_library_digest"),
        "artifact_plan": _require_digest(plan_digest, "plan_digest"),
        "external_artifacts": _require_digest(
            external_artifact_provenance_digest,
            "external_artifact_provenance_digest",
        ),
        "restricted_grade": _bit_digest("normalized-restricted-grade", grade),
        "stabilizer_normalization": _require_digest(
            stabilizer_normalization_digest, "stabilizer_normalization_digest"
        ),
        "stabilizer_table": _require_digest(
            stabilizer_table_digest, "stabilizer_table_digest"
        ),
        "target_model": _require_digest(target_model_digest, "target_model_digest"),
    }
    if igg == "Z2":
        if derived_q is not None:
            raise ValueError("Z2 local skeletons do not have a q assignment")
        # rho has no mathematical role in the Z2 target and is deliberately
        # excluded even when a generic caller happens to carry it.
        return CacheKey(
            "z2-local-skeleton",
            1,
            _require_digest(algorithm_digest, "algorithm_digest"),
            tuple(sorted(dependencies.items())),
        )

    if restricted_rho is None:
        raise ValueError("U1 local skeleton key requires restricted rho")
    rho = _bits(restricted_rho, "restricted_rho")
    if len(rho) != len(grade):
        raise ValueError("restricted grade and rho lengths differ")
    expected_q = tuple(left ^ right for left, right in zip(grade, rho, strict=True))
    if derived_q is not None and _bits(derived_q, "derived_q") != expected_q:
        raise CacheCorruptError("cache_corrupt: stale q differs from a + rho")
    dependencies["restricted_rho"] = _bit_digest("normalized-restricted-rho", rho)
    dependencies["derived_q"] = _bit_digest("normalized-derived-q", expected_q)
    return CacheKey(
        "u1-local-skeleton",
        1,
        _require_digest(algorithm_digest, "algorithm_digest"),
        tuple(sorted(dependencies.items())),
    )


@dataclass(frozen=True, slots=True)
class ClassifierCache:
    root: Path
    _path_identities: tuple[tuple[Path, int, int], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _artifact_identities: tuple[tuple[str, int, int], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("$ClassifierCache.root: expected pathlib.Path")
        self._validate_root(ValueError)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("$ClassifierCache.root: existing path is not a directory")
        object.__setattr__(
            self,
            "_path_identities",
            self._capture_path_identities(ValueError),
        )
        object.__setattr__(
            self,
            "_artifact_identities",
            self._capture_artifact_identities(ValueError),
        )

    def _path_chain(self) -> tuple[Path, ...]:
        return (self.root, *self.root.parents)

    @staticmethod
    def _directory_identity(
        path: Path,
        error_type: type[Exception],
    ) -> tuple[int, int] | None:
        try:
            status = os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise error_type(f"cache path identity is unavailable: {path}") from error
        if stat.S_ISLNK(status.st_mode):
            raise error_type(f"cache path is a symbolic link: {path}")
        if not stat.S_ISDIR(status.st_mode):
            raise error_type(f"cache path is not a directory: {path}")
        return (status.st_dev, status.st_ino)

    def _capture_path_identities(
        self,
        error_type: type[Exception],
    ) -> tuple[tuple[Path, int, int], ...]:
        identities: list[tuple[Path, int, int]] = []
        for candidate in self._path_chain():
            identity = self._directory_identity(candidate, error_type)
            if identity is not None:
                identities.append((candidate, *identity))
        return tuple(identities)

    def _revalidate_path_identities(self) -> None:
        for candidate, expected_device, expected_inode in self._path_identities:
            identity = self._directory_identity(candidate, CacheCorruptError)
            if identity is None:
                raise CacheCorruptError(
                    f"cache path identity disappeared after construction: {candidate}"
                )
            if identity != (expected_device, expected_inode):
                raise CacheCorruptError(
                    f"cache path identity was substituted after construction: {candidate}"
                )

    def _capture_artifact_identities(
        self,
        error_type: type[Exception],
    ) -> tuple[tuple[str, int, int], ...]:
        identities: list[tuple[str, int, int]] = []
        for artifact_kind in sorted(_DEPENDENCY_SCHEMAS):
            identity = self._directory_identity(
                self.root / artifact_kind,
                error_type,
            )
            if identity is not None:
                identities.append((artifact_kind, *identity))
        return tuple(identities)

    def _ensure_bound_root(self) -> None:
        self._validate_root(CacheCorruptError)
        self._revalidate_path_identities()
        bound_paths = {candidate for candidate, _, _ in self._path_identities}
        root_identity = self._directory_identity(self.root, CacheCorruptError)
        if root_identity is not None:
            if self.root not in bound_paths:
                raise CacheCorruptError(
                    "cache root appeared after construction without a trusted identity"
                )
            return

        unexpected = tuple(
            candidate
            for candidate in self._path_chain()
            if candidate not in bound_paths
            and self._directory_identity(candidate, CacheCorruptError) is not None
        )
        if unexpected:
            raise CacheCorruptError(
                "cache path appeared after construction without a trusted identity: "
                f"{unexpected[0]}"
            )
        try:
            self.root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise CacheCorruptError(
                "cache root was concurrently substituted during creation"
            ) from error
        except OSError as error:
            raise CacheCorruptError("cache root could not be created safely") from error
        self._validate_root(CacheCorruptError)
        self._revalidate_path_identities()
        identities = self._capture_path_identities(CacheCorruptError)
        if not identities or identities[0][0] != self.root:
            raise CacheCorruptError("cache root identity could not be bound")
        object.__setattr__(self, "_path_identities", identities)

    def _ensure_bound_artifact_directory(self, key: CacheKey) -> None:
        self._ensure_bound_root()
        artifact_directory = self.root / key.artifact_kind
        bound = {
            artifact_kind: (device, inode)
            for artifact_kind, device, inode in self._artifact_identities
        }
        identity = self._directory_identity(
            artifact_directory,
            CacheCorruptError,
        )
        expected = bound.get(key.artifact_kind)
        if expected is not None:
            if identity is None:
                raise CacheCorruptError(
                    "cache artifact directory identity disappeared after binding"
                )
            if identity != expected:
                raise CacheCorruptError(
                    "cache artifact directory identity was substituted after binding"
                )
            return
        if identity is not None:
            raise CacheCorruptError(
                "cache artifact directory appeared without a trusted identity"
            )
        try:
            artifact_directory.mkdir(exist_ok=False)
        except FileExistsError as error:
            raise CacheCorruptError(
                "cache artifact directory was concurrently substituted during creation"
            ) from error
        except OSError as error:
            raise CacheCorruptError(
                "cache artifact directory could not be created safely"
            ) from error
        self._ensure_bound_root()
        identity = self._directory_identity(
            artifact_directory,
            CacheCorruptError,
        )
        if identity is None:
            raise CacheCorruptError(
                "cache artifact directory identity could not be bound"
            )
        object.__setattr__(
            self,
            "_artifact_identities",
            tuple(
                sorted(
                    (*self._artifact_identities, (key.artifact_kind, *identity)),
                    key=lambda item: item[0],
                )
            ),
        )

    def _validate_root(self, error_type: type[Exception]) -> None:
        if not self.root.is_absolute():
            raise error_type("cache root must be an absolute path without traversal")
        if ".." in self.root.parts:
            raise error_type("cache root parent traversal is forbidden")
        for candidate in (self.root, *self.root.parents):
            if candidate.is_symlink():
                raise error_type(
                    f"cache root has a symbolic-link ancestor: {candidate}"
                )

    def entry_path(self, key: CacheKey) -> Path:
        if type(key) is not CacheKey:
            raise TypeError("cache key must be CacheKey")
        return self.root / key.artifact_kind / f"{key.digest.removeprefix('sha256:')}.json"

    def _assert_no_symlink_substitution(self, key: CacheKey, path: Path) -> None:
        artifact_directory = self.root / key.artifact_kind
        if path.is_symlink():
            raise CacheCorruptError("cache entry is a symbolic link")
        self._ensure_bound_artifact_directory(key)
        if self.root.exists() and not self.root.is_dir():
            raise CacheCorruptError("cache root is not a directory")
        if artifact_directory.is_symlink():
            raise CacheCorruptError("cache artifact directory is a symbolic link")
        if artifact_directory.exists() and not artifact_directory.is_dir():
            raise CacheCorruptError("cache artifact path is not a directory")

    def _decode_entry(self, key: CacheKey, data: bytes) -> bytes:
        value = _strict_json(data, context="cache entry")
        if not isinstance(value, Mapping):
            raise CacheCorruptError("cache entry: expected object")
        expected_fields = {
            "artifact_digest",
            "cache_key",
            "payload",
            "schema_version",
        }
        if set(value) != expected_fields:
            raise CacheCorruptError("cache entry: schema fields differ")
        if value["schema_version"] != _CACHE_SCHEMA_VERSION:
            raise CacheCorruptError("cache entry: schema version differs")
        if value["cache_key"] != _key_mapping(key):
            raise CacheCorruptError("cache entry: dependency key differs")
        payload = _canonical_json(value["payload"])
        if value["artifact_digest"] != _raw_digest(payload):
            raise CacheCorruptError("cache entry: artifact digest differs")
        # Reparse the returned artifact boundary as strictly as a fresh build.
        _strict_json(payload, context="cache payload")
        return payload

    def _entry_bytes(self, key: CacheKey, payload: bytes) -> bytes:
        parsed = _strict_json(payload, context="cache builder payload")
        return _canonical_json(
            {
                "artifact_digest": _raw_digest(payload),
                "cache_key": _key_mapping(key),
                "payload": parsed,
                "schema_version": _CACHE_SCHEMA_VERSION,
            }
        )

    def get_or_build(self, key: CacheKey, builder: Callable[[], bytes]) -> bytes:
        if type(key) is not CacheKey:
            raise TypeError("cache key must be CacheKey")
        if not callable(builder):
            raise TypeError("cache builder must be callable")
        path = self.entry_path(key)
        self._assert_no_symlink_substitution(key, path)
        if path.exists():
            try:
                return self._decode_entry(key, path.read_bytes())
            except CacheCorruptError:
                raise
            except OSError as error:
                raise CacheCorruptError("cache entry: unable to read") from error

        payload = builder()
        if type(payload) is not bytes:
            raise TypeError("cache builder must return bytes")
        encoded = self._entry_bytes(key, payload)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        self._assert_no_symlink_substitution(key, path)
        temporary_path: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
            )
            temporary_path = Path(raw_path)
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            # Verify the exact bytes which will cross the atomic boundary.
            self._decode_entry(key, temporary_path.read_bytes())
            self._assert_no_symlink_substitution(key, path)
            os.replace(temporary_path, path)
            temporary_path = None
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Directory fsync is not available on every supported host;
                # atomic replace still prevents a partial file from becoming a hit.
                pass
        except CacheCorruptError:
            raise
        except OSError as error:
            raise CacheCorruptError("cache entry: atomic write failed") from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
        return self._decode_entry(key, path.read_bytes())


__all__ = [
    "CacheCorruptError",
    "CacheKey",
    "ClassifierCache",
    "make_local_skeleton_cache_key",
]
