"""Cross-machine read-only audits preserve typed state and implementation hashes."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _audit_identity import verify_configuration_identity, verify_implementation_sources
from supervised_nmf._identity import configuration_identity, implementation_sources


def identity_transform(X, **kwargs):
    return X


def identities():
    estimator = FunctionTransformer(identity_transform,
        kw_args={"path": "/inputs/user-data", "sha256": "f" * 64})
    candidates = [{"validate": False}, {"validate": True}]
    return (configuration_identity({"estimator": estimator, "candidates": candidates}),
            implementation_sources(estimator, candidates))


def relocated(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            filename = value["path"].replace("\\", "/").rsplit("/", 1)[-1]
            return {"path": "/home/remote/env/package/" + filename, "sha256": value["sha256"]}
        return {key: relocated(item) for key, item in value.items()}
    return [relocated(item) for item in value] if isinstance(value, list) else value


def first_with_key(value, key):
    if isinstance(value, dict):
        if key in value:
            return value
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return None
    for item in children:
        found = first_with_key(item, key)
        if found is not None:
            return found
    return None


def test_read_only_identity_accepts_source_relocation_without_changing_user_paths():
    configuration, sources = identities()
    remote = relocated(configuration)
    remote_sources = {name.partition("@")[0] + "@/home/remote/" + name.rsplit("/", 1)[-1]: digest
                      for name, digest in sources.items()}
    assert remote != configuration
    assert verify_configuration_identity(remote, configuration)
    assert verify_implementation_sources(remote_sources, sources)
    # The ordinary package identity is still different; this helper never
    # rewrites the source-bound identity used for resuming cached computation.
    assert remote_sources != sources


@pytest.mark.parametrize("change", ["source_hash", "source_type", "code_hash", "user_path", "typed_bool", "relative_path"])
def test_relocation_does_not_accept_changed_implementation_or_typed_parameters(change):
    configuration, _ = identities()
    altered = relocated(configuration)
    if change == "source_hash":
        first_with_key(altered, "sha256")["sha256"] = "0" * 64
    elif change == "source_type":
        first_with_key(altered, "type")["type"] = "different.Implementation"
    elif change == "code_hash":
        first_with_key(altered, "code_sha256")["code_sha256"] = "0" * 64
    elif change == "relative_path":
        first_with_key(altered, "sha256")["path"] = "relative/implementation.py"
    else:
        def replace(value):
            if isinstance(value, dict):
                if change == "user_path" and value == {"kind": "str", "value": "/inputs/user-data"}:
                    value["value"] = "/inputs/different-data"
                elif change == "typed_bool" and value.get("kind") == "bool":
                    value["value"] = int(value["value"])
                for item in value.values():
                    replace(item)
            elif isinstance(value, list):
                for item in value:
                    replace(item)
        replace(altered)
    with pytest.raises(ValueError, match="Read-only audit structural identity"):
        verify_configuration_identity(altered, configuration)


@pytest.mark.parametrize("change", ["missing", "extra", "hash", "qualified_name", "relative_path"])
def test_relocated_source_inventory_rejects_missing_extra_or_changed_entries(change):
    _, sources = identities()
    altered = deepcopy(sources)
    key = next(iter(altered))
    digest = altered[key]
    if change == "missing":
        altered.pop(key)
    elif change == "extra":
        altered["extra.Implementation@/home/remote/extra.py"] = digest
    elif change == "hash":
        altered[key] = "0" * 64
    else:
        altered.pop(key)
        name, _, path = key.partition("@")
        altered[("different.Implementation@" + path) if change == "qualified_name"
                else name + "@relative/implementation.py"] = digest
    with pytest.raises(ValueError, match="Read-only audit implementation source"):
        verify_implementation_sources(altered, sources)
