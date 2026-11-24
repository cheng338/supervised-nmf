"""Comparison readers must account for draws skipped by saved models and splits."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import ExtraTreeClassifier
from sklearn.utils import check_random_state

from supervised_nmf import evaluation
from supervised_nmf._identity import configuration_identity, implementation_sources


class Interrupted(BaseException):
    pass


class CallbackTree(ExtraTreeClassifier):
    def __init__(self, draw, random_state=0):
        self.draw = draw
        super().__init__(random_state=random_state, min_samples_leaf=2, max_features=2)

    def fit(self, X, y):
        self.random_state = self.draw()
        return super().fit(X, y)


class HeldOutDrawTree(ExtraTreeClassifier):
    def __init__(self, draw):
        self.draw = draw
        super().__init__(random_state=337)

    def predict(self, X):
        if len(X) == 59:
            self.draw()
        return super().predict(X)


class MetadataTree(ExtraTreeClassifier):
    def __init__(self, payload):
        self.payload = payload
        super().__init__(random_state=337)

    def predict(self, X):
        if len(X) == 59:
            self.payload.dtype.metadata["stream"].bit_generator.seed_seq.pool[0] += 1
        return super().predict(X)


def seed_pool(generator):
    engine = generator.bit_generator
    return (engine.seed_seq if hasattr(engine, "seed_seq") else engine._seed_seq).pool


class FieldTitleTree(ExtraTreeClassifier):
    def __init__(self, payload):
        self.payload = payload
        super().__init__(random_state=337)

    def predict(self, X):
        if len(X) == 59:
            dtype = self.payload.dtype
            if "inner" in dtype.names:
                dtype = dtype.fields["inner"][0]
                if dtype.subdtype:
                    dtype = dtype.subdtype[0]
            title = dtype.fields["word"][2]
            if isinstance(title, tuple):
                title = title[1]
            if isinstance(title, np.random.Generator):
                seed_pool(title)[0] += 1
        return super().predict(X)


class MissingValueRNGTree(ExtraTreeClassifier):
    def __init__(self, payload):
        self.payload = payload
        super().__init__(random_state=337, min_samples_leaf=2, max_features=2)

    def fit(self, X, y):
        self.random_state = int(self.payload.dtype.na_object.integers(2**30))
        return super().fit(X, y)


class StringParameterTree(ExtraTreeClassifier):
    def __init__(self, words, generator):
        self.words = words
        self.generator = generator
        super().__init__(random_state=337, min_samples_leaf=2, max_features=2)

    def fit(self, X, y):
        word = self.words.astype(np.dtypes.StringDType(na_object=None))[0]
        self.random_state = int(self.generator.integers(2**29)) + (0 if word is None else ord(word[0]))
        return super().fit(X, y)


class PoolParameterTree(ExtraTreeClassifier):
    def __init__(self, words):
        self.words = words
        super().__init__(random_state=337, min_samples_leaf=2, max_features=2)

    def fit(self, X, y):
        self.random_state = int(self.words[0])
        return super().fit(X, y)


class MutatingSplitRNG(np.random.RandomState):
    def shuffle(self, values):
        self.words[0] += 1
        return super().shuffle(values)


class ChildProbability(ClassifierMixin, BaseEstimator):
    fits = 0

    def __init__(self, generator):
        self.generator = generator

    def fit(self, X, y):
        type(self).fits += 1
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.probability_ = float(self.generator.spawn(1)[0].random())
        return self

    def predict(self, X):
        return np.full(len(X), int(self.probability_ >= .5))

    def predict_proba(self, X):
        return np.tile([1 - self.probability_, self.probability_], (len(X), 1))


def callback_tree(rng):
    def draw():
        if isinstance(rng, np.random.Generator):
            return int(rng.integers(2**30))
        return rng.randint(2**30)

    return CallbackTree(draw), {}


def tree(seed):
    return (ExtraTreeClassifier(random_state=seed, min_samples_leaf=2, max_features=2),
            {"max_depth": [2, 4]})


@pytest.fixture(autouse=True)
def restore_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def compare():
    rng = np.random.RandomState(7421)
    X = rng.normal(size=(149, 7))
    y = (X[:, 0] * X[:, 1] - .6 * X[:, 2] + rng.normal(size=149) * .4 > 0).astype(int)

    def run(folder, models, split=73):
        np.random.seed(44)
        return evaluation.run_comparison(models, X[:90], y[:90], X[90:], y[90:],
                                         output_dir=folder, seed=split)

    return run


def scientific(folder, name):
    cv = json.loads((folder / f"{name}-cv.json").read_bytes())
    record = json.loads((folder / f"{name}-evaluation.json").read_bytes())
    assert record["evaluation"]["status"] == "valid" and not record["is_fallback"]
    return (cv["folds"], [[f["score"] for f in c["folds"]] for c in cv["candidates"]],
            cv["selected_candidate_index"], cv["selected_parameters"],
            record["evaluation"]["predictions"])


def archive(folder):
    return {p.name: p.read_bytes() for p in folder.glob("*.json")}


def assert_unverified_fit_generator_refused(compare, folder, models):
    before = archive(folder)
    with patch.object(evaluation, "calibration_search") as calibrate, patch.object(evaluation, "write_json") as write:
        with pytest.raises(ValueError, match="Cannot verify the fit generator in this saved comparison"):
            compare(folder, models)
        calibrate.assert_not_called()
        write.assert_not_called()
    assert archive(folder) == before


def interrupt_after_fold(monkeypatch, name):
    write = evaluation.write_json

    def publish(path, report):
        write(path, report)
        if Path(path).name == f"{name}-cv.json" and len(report["candidates"][0]["folds"]) == 1:
            raise Interrupted

    monkeypatch.setattr(evaluation, "write_json", publish)


@pytest.mark.parametrize("stage", ["evaluation", "partial_cv"])
def test_previously_archived_first_rng_user_cannot_follow_another(tmp_path, compare, monkeypatch, stage):
    folder, joint, single = (tmp_path / tag for tag in ("saved", "joint", "single"))
    fixed, implicit = tree(337), tree(None)
    # For partial CV, sklearn clones the explicit singleton for each tree.
    # The saved fold leaves the global RNG untouched, so calibration alone
    # would accept this prefix at the wrong position in the comparison.
    second = implicit if stage == "evaluation" else tree(check_random_state(None))
    compare(joint, {"f": fixed, "a": implicit, "b": second})
    compare(single, {"f": fixed, "b": second})
    assert scientific(joint, "b") != scientific(single, "b")
    compare(folder, {"f": fixed, "a": implicit})
    if stage == "partial_cv" and np.__version__.split(".", 1)[0] != "2":
        assert_unverified_fit_generator_refused(compare, folder, {"f": fixed, "b": second})
        assert_unverified_fit_generator_refused(compare, folder, {"f": fixed, "a": implicit, "b": second})
        return
    if stage == "evaluation":
        compare(folder, {"f": fixed, "b": second})
        assert scientific(folder, "b") == scientific(single, "b")
    else:
        with monkeypatch.context() as interrupted:
            interrupt_after_fold(interrupted, "b")
            with pytest.raises(Interrupted):
                compare(folder, {"f": fixed, "b": second})
        report = json.loads((folder / "b-cv.json").read_bytes())
        assert len(report["candidates"][0]["folds"]) == 1
        assert not report.get("global_rng_prefix_changed")
    before = archive(folder)
    with pytest.raises(ValueError, match="global RNG"):
        compare(folder, {"f": fixed, "a": implicit, "b": second})
    assert archive(folder) == before


def test_archives_published_after_this_caller_creates_plan(tmp_path, compare, monkeypatch):
    folds = evaluation._comparison_folds
    fixed, implicit = tree(337), tree(None)
    published = {}

    def archive_before_first_model(*args, **kwargs):
        result = folds(*args, **kwargs)
        if result[0] is not None:
            # Another process can publish models between fold commitment and
            # the caller's first model read, without changing its process RNG.
            state = np.random.get_state()
            try:
                compare(tmp_path, {"f": fixed, "a": implicit})
                compare(tmp_path, {"f": fixed, "b": implicit})
            finally:
                np.random.set_state(state)
            published.update(archive(tmp_path))
        return result

    monkeypatch.setattr(evaluation, "_comparison_folds", archive_before_first_model)
    with pytest.raises(ValueError, match="earlier archived model"):
        compare(tmp_path, {"f": fixed, "a": implicit, "b": implicit})
    assert published and archive(tmp_path) == published


@pytest.mark.parametrize("writer_shared", [False, True], ids=["to-singleton", "to-independent"])
def test_archived_callback_cannot_switch_global_ownership(tmp_path, compare, writer_shared):
    def models(shared):
        rng = check_random_state(None) if shared else np.random.RandomState(44)

        def draw():
            check_random_state(None).randint(2**30)
            return rng.randint(2**30)

        return {"f": tree(337), "b": (CallbackTree(draw, random_state=None), {})}

    writer, reader = tmp_path / "writer", tmp_path / "reader"
    compare(writer, models(writer_shared))
    compare(reader, models(not writer_shared))
    assert scientific(writer, "b") != scientific(reader, "b")
    before = archive(writer)
    with pytest.raises(ValueError, match="incompatible|fit generator"):
        compare(writer, models(not writer_shared))
    assert archive(writer) == before


@pytest.mark.parametrize("engine_wrapper", [False, True], ids=["singleton", "engine"])
@pytest.mark.parametrize("ambient", [False, True], ids=["independent", "ambient"])
def test_append_after_model_with_captured_rng(tmp_path, compare, engine_wrapper, ambient):
    def models():
        rng = check_random_state(None) if ambient else np.random.RandomState(44)
        if engine_wrapper:
            rng = np.random.Generator(rng._bit_generator)
        return {"a": callback_tree(rng), "b": tree(None)}

    joint, folder = tmp_path / "joint", tmp_path / "saved"
    compare(joint, models())
    compare(folder, {"a": models()["a"]})
    assert scientific(joint, "a") == scientific(folder, "a")
    if ambient:
        before = archive(folder)
        with pytest.raises(ValueError, match="global RNG|fit generator"):
            compare(folder, models())
        assert archive(folder) == before
    elif int(np.__version__.split(".")[0]) == 2:
        compare(folder, models())
        assert scientific(folder, "b") == scientific(joint, "b")
    else:
        with pytest.raises(ValueError, match="fit generator"):
            compare(folder, models())


@pytest.mark.parametrize(("stage", "later"), [("append", "callback"), ("partial_cv", "tree"),
                                              ("evaluation", "engine"), ("append", "independent")])
def test_private_stream_across_models(tmp_path, compare, monkeypatch, stage, later):
    def models():
        rng = np.random.RandomState(51)
        callback, grid = callback_tree(rng)
        # A held-out prediction can advance the stream after clean CV folds.
        first = (HeldOutDrawTree(callback.draw), grid) if stage == "evaluation" else (callback, grid)
        if later == "engine":
            second = callback_tree(np.random.Generator(rng._bit_generator))
        elif later == "callback":
            second = callback_tree(rng)
        else:
            second = tree(np.random.RandomState(51) if later == "independent" else rng)
        return {"f": tree(337), "a": first, "b": second}

    folder, joint, single = (tmp_path / tag for tag in ("saved", "joint", "single"))
    compare(joint, models())
    first = models()
    compare(folder, {"f": first["f"], "a": first["a"]})
    fresh = models()
    without_a = {"f": fresh["f"], "b": fresh["b"]}
    compare(single, without_a)
    if later != "independent":
        assert scientific(joint, "b") != scientific(single, "b")
    if np.__version__.split(".", 1)[0] != "2":
        if stage in ("partial_cv", "evaluation"):
            fresh = models()
            assert_unverified_fit_generator_refused(compare, folder, {"f": fresh["f"], "b": fresh["b"]})
        assert_unverified_fit_generator_refused(compare, folder, models())
        return
    if later == "independent":
        compare(folder, models())
        assert scientific(folder, "b") == scientific(joint, "b") == scientific(single, "b")
        return
    if stage == "partial_cv":
        with monkeypatch.context() as interrupted:
            interrupt_after_fold(interrupted, "b")
            with pytest.raises(Interrupted):
                compare(folder, without_a)
    elif stage == "evaluation":
        fresh = models()
        compare(folder, {"f": fresh["f"], "b": fresh["b"]})
    before = archive(folder)
    with pytest.raises(ValueError, match="earlier archived callback"):
        compare(folder, models())
    assert archive(folder) == before


@pytest.mark.parametrize("engine_wrapper", [False, True], ids=["splitter", "engine"])
@pytest.mark.parametrize("shared", [False, True], ids=["independent", "shared"])
def test_append_after_restoring_private_splitter(tmp_path, compare, engine_wrapper, shared):
    splitter = np.random.RandomState(73)

    def models():
        splitter.seed(73)
        rng = splitter if shared else np.random.RandomState(73)
        fitted = callback_tree(np.random.Generator(rng._bit_generator)) if engine_wrapper else tree(rng)
        return {"a": tree(337), "b": fitted}

    joint, folder = tmp_path / "joint", tmp_path / "saved"
    compare(joint, models(), split=splitter)
    compare(folder, {"a": models()["a"]}, split=splitter)
    if shared or int(np.__version__.split(".")[0]) != 2:
        before = archive(folder)
        with pytest.raises(ValueError, match="shared split RNG|fit generator"):
            compare(folder, models(), split=splitter)
        assert archive(folder) == before
    else:
        compare(folder, models(), split=splitter)
        assert scientific(folder, "b") == scientific(joint, "b")


@pytest.mark.parametrize("writer_shared", [False, True], ids=["to-shared", "to-independent"])
@pytest.mark.parametrize("stage", ["evaluation", "partial_cv"])
def test_saved_model_cannot_switch_split_ownership(tmp_path, compare, monkeypatch, writer_shared, stage):
    splitter = np.random.RandomState(73)

    def models(shared):
        splitter.seed(73)
        return {"a": tree(337), "b": tree(splitter if shared else np.random.RandomState(73))}

    if stage == "evaluation":
        compare(tmp_path, models(writer_shared), split=splitter)
    else:
        with monkeypatch.context() as interrupted:
            interrupt_after_fold(interrupted, "b")
            with pytest.raises(Interrupted):
                compare(tmp_path, models(writer_shared), split=splitter)
    before = archive(tmp_path)
    with pytest.raises(ValueError, match="shared split RNG|incompatible|fit generator"):
        compare(tmp_path, models(not writer_shared), split=splitter)
    assert archive(tmp_path) == before


def mutable_entropy_models(kind, *, during_fit=False):
    entropy = [17, 31]
    if kind == "array":
        entropy = np.array(entropy, dtype=np.uint32)
    generator = np.random.default_rng(np.random.SeedSequence(entropy))

    def bump():
        entropy[0] += 1
        return 337

    first = CallbackTree(bump) if during_fit else HeldOutDrawTree(bump)
    return {"a": (first, {}), "b": (ChildProbability(generator), {})}


@pytest.mark.parametrize("kind", ["array", "list"])
def test_new_child_with_mutable_entropy_after_archived_callback(tmp_path, compare, monkeypatch, kind):
    joint, single, saved = (tmp_path / name for name in ("joint", "single", "saved"))
    compare(joint, mutable_entropy_models(kind))
    compare(single, {"b": mutable_entropy_models(kind)["b"]})
    joint_cv, *_, joint_preds = scientific(joint, "b")
    single_cv, *_, single_preds = scientific(single, "b")
    assert joint_cv == single_cv
    assert joint_preds["positive_class_probability"] != single_preds["positive_class_probability"]
    compare(saved, {"a": mutable_entropy_models(kind)["a"]})
    before = archive(saved)
    monkeypatch.setattr(ChildProbability, "fits", 0)
    for names in (("a", "b"), ("b",)):
        group = mutable_entropy_models(kind)
        with pytest.raises(ValueError, match="fit generator"):
            compare(saved, {name: group[name] for name in names})
        assert ChildProbability.fits == 0 and archive(saved) == before

    def qualified_models():
        group = mutable_entropy_models(kind)
        group["b"] = ChildProbability(np.random.default_rng(31)), {}
        return group

    if np.__version__.split(".")[0] == "2":
        reference = tmp_path / "qualified"
        compare(reference, qualified_models())
        compare(saved, qualified_models())
        assert scientific(saved, "b") == scientific(reference, "b")


@pytest.mark.parametrize("partial", [False, True], ids=["evaluation", "partial-cv"])
def test_unverified_generator_after_concurrent_publication(tmp_path, compare, monkeypatch, partial):
    calibrate = evaluation.calibration_search
    folds = evaluation._comparison_folds
    folder = tmp_path / "saved"
    published = []

    def publish_a():
        # Each caller has its own configuration, as in separate processes.
        group = mutable_entropy_models("array", during_fit=partial)
        if partial:
            with monkeypatch.context() as interrupted:
                interrupt_after_fold(interrupted, "a")
                with pytest.raises(Interrupted):
                    compare(folder, {"a": group["a"]})
        else:
            compare(folder, {"a": group["a"]})
        published.append(True)

    if partial:
        def publish_after_cv_path_check(*args, **kwargs):
            with monkeypatch.context() as writer:
                writer.setattr(evaluation, "calibration_search", calibrate)
                publish_a()
            return calibrate(*args, **kwargs)

        monkeypatch.setattr(evaluation, "calibration_search", publish_after_cv_path_check)
    else:
        def publish_after_plan(*args, **kwargs):
            result = folds(*args, **kwargs)
            if result[0] is not None:
                publish_a()
            return result

        monkeypatch.setattr(evaluation, "_comparison_folds", publish_after_plan)
    monkeypatch.setattr(ChildProbability, "fits", 0)
    with pytest.raises(ValueError, match="fit generator"):
        compare(folder, mutable_entropy_models("array", during_fit=partial))
    assert published == [True] and ChildProbability.fits == 0
    assert (folder / "a-evaluation.json").is_file()
    assert not list(folder.glob("b-*.json"))


@pytest.mark.skipif(np.__version__.split(".")[0] != "2", reason="requires qualified NumPy 2 seed pools")
def test_resuming_a_partial_model_skips_a_pool_mutation(tmp_path, compare, monkeypatch):
    def models():
        generator = np.random.default_rng(31)
        pool = generator.bit_generator.seed_seq.pool

        def bump():
            pool[0] += 1
            return 337

        def draw():
            return int(generator.bit_generator.seed_seq.generate_state(1)[0])

        return {"a": (CallbackTree(bump), {}), "b": (CallbackTree(draw), {})}

    with monkeypatch.context() as interrupted:
        interrupt_after_fold(interrupted, "a")
        with pytest.raises(Interrupted):
            compare(tmp_path, models())
    original = json.loads((tmp_path / "a-cv.json").read_bytes())["candidates"][0]["folds"]
    assert len(original) == 1
    with pytest.raises(ValueError, match="earlier archived callback"):
        compare(tmp_path, models())
    cv = json.loads((tmp_path / "a-cv.json").read_bytes())
    assert cv["candidates"][0]["folds"][:1] == original
    assert (tmp_path / "a-evaluation.json").is_file()
    assert not list(tmp_path.glob("b-*.json"))


def pool_backed_models(backing, opaque_first, step=1):
    generator = np.random.default_rng(31)
    pool = generator.bit_generator.seed_seq.pool
    if backing == "retained-estimator":
        source = ExtraTreeClassifier(random_state=337)
        source.pool_ = pool
    elif backing == "structured-scalar":
        source = pool.view([("word", pool.dtype)])[0]
    else:
        source = pool[:0]
        if backing in ("strided", "wrapped-strided"):
            source = np.lib.stride_tricks.as_strided(source)
        if backing == "wrapped-memoryview":
            source = np.asarray(memoryview(source))
        if backing.startswith("wrapped-"):
            source = source.view()
        assert not np.may_share_memory(source, pool)

    def storage():
        value = source
        if isinstance(value, ExtraTreeClassifier):
            return value.pool_
        while True:
            if isinstance(value, memoryview):
                value = value.obj
            elif getattr(value, "base", None) is not None:
                value = value.base
            else:
                return value

    assert storage() is pool

    def bump_view():
        storage()[0] += step

    def bump_rng():
        generator.bit_generator.seed_seq.pool[0] += step

    def draw_rng():
        return int(generator.bit_generator.seed_seq.generate_state(1)[0])

    def draw_view():
        return int(storage()[0])

    return {"a": (HeldOutDrawTree(bump_view if opaque_first else bump_rng), {}),
            "b": (CallbackTree(draw_rng if opaque_first else draw_view), {})}


@pytest.mark.skipif(np.__version__.split(".")[0] != "2", reason="requires qualified NumPy 2 seed pools")
@pytest.mark.parametrize("opaque_first", [False, True], ids=["opaque-after-rng", "rng-after-opaque"])
@pytest.mark.parametrize("backing", ["strided", "wrapped-strided", "wrapped-memoryview", "direct",
                                     "structured-scalar", "retained-estimator"])
def test_captured_state_can_reach_another_models_pool(tmp_path, compare, opaque_first, backing):
    def models():
        return pool_backed_models(backing, opaque_first)

    joint, saved, single = (tmp_path / t for t in ("joint", "saved", "single"))
    compare(joint, models())
    compare(single, {"b": models()["b"]})
    assert scientific(joint, "b") != scientific(single, "b")
    compare(saved, {"a": models()["a"]})
    before = archive(saved)
    with pytest.raises(ValueError, match="captured state|RNG or buffer"):
        compare(saved, models())
    assert archive(saved) == before
    fixed = tree(337)
    compare(saved, {"a": models()["a"], "c": fixed})
    intact = tmp_path / "fixed"
    compare(intact, {"a": models()["a"], "c": fixed})
    assert scientific(saved, "c") == scientific(intact, "c")


@pytest.mark.skipif(np.__version__.split(".")[0] != "2", reason="requires qualified NumPy 2 seed pools")
@pytest.mark.parametrize("stage", ["evaluation", "partial_cv"])
def test_archive_from_unverified_prefix_cannot_be_read_alone_or_reordered(tmp_path, compare, monkeypatch, stage):
    saved, first, second = (tmp_path / t for t in ("saved", "first", "second"))

    def models(name, step):
        group = pool_backed_models("wrapped-strided", False, step)
        return {name: group["a"], "b": group["b"]}

    compare(first, models("a", 1))
    compare(second, models("c", 2))
    assert scientific(first, "b") != scientific(second, "b")
    if stage == "partial_cv":
        with monkeypatch.context() as interrupted:
            interrupt_after_fold(interrupted, "b")
            with pytest.raises(Interrupted):
                compare(saved, models("a", 1))
    else:
        compare(saved, models("a", 1))
    before = archive(saved)
    with pytest.raises(ValueError, match="incompatible"):
        compare(saved, {"b": models("a", 1)["b"]})
    assert archive(saved) == before
    with pytest.raises(ValueError, match="captured state"):
        compare(saved, models("c", 2))
    assert all((saved / name).read_bytes() == content for name, content in before.items())
    assert scientific(second, "c") == scientific(saved, "c")


@pytest.mark.skipif(np.__version__.split(".")[0] != "2", reason="requires qualified NumPy 2 seed pools")
@pytest.mark.parametrize("first", ["dtype-metadata", "bare-pool", "hidden-rng"])
@pytest.mark.parametrize("stage", ["evaluation", "partial_cv"])
def test_opaque_pool_reader_requires_its_original_prefix(tmp_path, compare, monkeypatch, first, stage):
    def models():
        generator = np.random.default_rng(31)
        pool = generator.bit_generator.seed_seq.pool
        source = np.lib.stride_tricks.as_strided(pool[:0]).view()

        def draw():
            owner = source
            while owner.base is not None:
                owner = owner.base
            return int(owner[0])

        if first == "dtype-metadata":
            # deepcopy of the array preserves the Generator inside its dtype.
            payload = np.array([1], dtype=np.dtype("uint32", metadata={"stream": generator}))
            writer = MetadataTree(payload)
        elif first == "bare-pool":
            def bump():
                pool[0] += 1

            writer = HeldOutDrawTree(bump)
        else:
            holder = ExtraTreeClassifier(random_state=337)
            holder.stream_ = generator

            def bump():
                holder.stream_.bit_generator.seed_seq.pool[0] += 1

            writer = HeldOutDrawTree(bump)
        return {"a": (writer, {}), "b": (CallbackTree(draw), {})}

    joint, single = tmp_path / "joint", tmp_path / "single"
    compare(joint, models())
    compare(single, {"b": models()["b"]})
    paired, standalone = scientific(joint, "b"), scientific(single, "b")
    assert paired[0] == standalone[0] and paired[1] != standalone[1]
    assert paired[-1]["predicted_labels"] != standalone[-1]["predicted_labels"]
    assert paired[-1]["positive_class_probability"] != standalone[-1]["positive_class_probability"]
    saved = joint
    if stage == "partial_cv":
        saved = tmp_path / "partial"
        with monkeypatch.context() as interrupted:
            interrupt_after_fold(interrupted, "b")
            with pytest.raises(Interrupted):
                compare(saved, models())
    before = archive(saved)
    with pytest.raises(ValueError, match="incompatible"):
        compare(saved, {"b": models()["b"]})
    assert archive(saved) == before


@pytest.mark.parametrize("stage", ["evaluation", "partial_cv"])
def test_opaque_capture_can_read_mutable_splitter(tmp_path, compare, monkeypatch, stage):
    splitter = np.random.RandomState(73)
    holder = ExtraTreeClassifier(random_state=337)
    holder.stream_ = splitter

    def draw():
        return holder.stream_.get_state()[2]

    models = {"b": (CallbackTree(draw), {})}
    joint, independent = tmp_path / "joint", tmp_path / "independent"
    compare(joint, models, split=splitter)
    splitter.seed(73)
    compare(independent, models, split=73)
    paired, standalone = scientific(joint, "b"), scientific(independent, "b")
    assert paired[0] == standalone[0] and paired[-1] != standalone[-1]
    saved = joint
    if stage == "partial_cv":
        saved = tmp_path / "partial"
        splitter.seed(73)
        with monkeypatch.context() as interrupted:
            interrupt_after_fold(interrupted, "b")
            with pytest.raises(Interrupted):
                compare(saved, models, split=splitter)
    before = archive(saved)
    splitter.seed(73)
    with pytest.raises(ValueError, match="captured state"):
        compare(saved, models, split=splitter)
    assert archive(saved) == before


def test_appending_after_custom_split_rng_would_skip_its_mutation(tmp_path, compare):
    splitter = MutatingSplitRNG(73)

    def models():
        splitter.seed(73)
        splitter.words = [337]
        return {"a": tree(337), "b": (PoolParameterTree(splitter.words), {})}

    joint, single, saved = (tmp_path / t for t in ("joint", "without-mutation", "saved"))
    compare(joint, models(), split=splitter)
    assert splitter.words[0] > 337
    compare(single, {"b": models()["b"]}, split=73)
    paired, standalone = scientific(joint, "b"), scientific(single, "b")
    assert paired[0] == standalone[0] and paired[-1] != standalone[-1]
    compare(saved, {"a": models()["a"]}, split=splitter)
    before = archive(saved)
    with pytest.raises(ValueError, match="custom split RNG"):
        compare(saved, {"b": models()["b"]}, split=splitter)
    assert splitter.words == [337] and archive(saved) == before


@pytest.mark.parametrize("explicit", [False, True], ids=["generated", "explicit"])
def test_calibration_replays_or_bypasses_custom_split_rng(tmp_path, explicit):
    X = np.random.RandomState(121).normal(size=(60, 4))
    y = np.tile([0, 1], 30)
    folds = evaluation.build_cv_folds(X, y, seed=73) if explicit else None
    splitter = MutatingSplitRNG(73)
    splitter.words = [337]

    def run():
        splitter.seed(73)
        splitter.words[0] = 337
        result = evaluation.calibration_search(
            PoolParameterTree(splitter.words), {}, X, y, seed=splitter, folds=folds,
            checkpoint_path=tmp_path / "calibration.json", refit=False)
        assert result.report["state"] == "cv_completed"
        assert (splitter.words[0] == 337) is explicit
        return result.report["candidates"][0]["folds"]

    original = run()
    assert run() == original


@pytest.mark.skipif(np.__version__.split(".")[0] != "2", reason="requires seed_seq public access")
@pytest.mark.parametrize("buffer", ["pool", "entropy", "structured-scalar", "grid-pool", "visible-scalar"])
def test_writer_can_change_an_ordinary_parameter(tmp_path, compare, buffer):
    def models():
        holder = ExtraTreeClassifier(random_state=337)
        generator = np.random.default_rng(31 if buffer == "visible-scalar" else [17, 31])
        holder.stream_ = generator
        attribute = "entropy" if buffer == "entropy" else "pool"
        words = getattr(holder.stream_.bit_generator.seed_seq, attribute)
        if buffer in ("structured-scalar", "visible-scalar"):
            words = words.view([("word", words.dtype)])[0]

        def bump_hidden():
            getattr(holder.stream_.bit_generator.seed_seq, attribute)[0] += 1

        def bump_visible():
            generator.bit_generator.seed_seq.pool[0] += 1

        reader = ((PoolParameterTree((337,)), {"words": [words]}) if buffer == "grid-pool"
                  else (PoolParameterTree(words), {}))
        return {"a": (HeldOutDrawTree(bump_visible if buffer == "visible-scalar" else bump_hidden), {}), "b": reader}

    joint, single, saved = (tmp_path / t for t in ("joint", "single", "saved"))
    compare(joint, models())
    compare(single, {"b": models()["b"]})
    paired, standalone = scientific(joint, "b"), scientific(single, "b")
    assert paired[0] == standalone[0] and paired[-1] != standalone[-1]
    compare(saved, {"a": models()["a"]})
    before = archive(saved)
    with pytest.raises(ValueError, match="captured state|RNG or buffer"):
        compare(saved, models())
    assert archive(saved) == before


@pytest.mark.parametrize("location", ["constructor", "grid"])
def test_numpy_scalar_after_unverified_model(tmp_path, compare, monkeypatch, location):
    # The callback captures a view. Its backing array is independent of the
    # second model's immutable scalar, even when recovery skips this callback.
    view = np.arange(4)[1:]

    def read_view():
        return int(view[0])

    first = CallbackTree(read_view), {}
    if location == "constructor":
        second = ExtraTreeClassifier(random_state=337, max_depth=np.int64(2)), {}
    else:
        second = ExtraTreeClassifier(random_state=337), {"max_features": [np.float64(0.5)]}
    models = {"a": first, "b": second}
    single, joint, saved = (tmp_path / name for name in ("single", "joint", "saved"))
    compare(single, {"b": second})
    expected = scientific(single, "b")
    compare(joint, models)
    assert scientific(joint, "b") == expected

    compare(saved, {"a": first})
    archived_first = (saved / "a-evaluation.json").read_bytes()
    with monkeypatch.context() as interrupted:
        interrupt_after_fold(interrupted, "b")
        with pytest.raises(Interrupted):
            compare(saved, models)
    compare(saved, models)
    assert scientific(saved, "b") == expected
    assert (saved / "a-evaluation.json").read_bytes() == archived_first
    with patch.object(evaluation, "calibration_search") as calibrate:
        compare(saved, models)
        calibrate.assert_not_called()
    assert scientific(saved, "b") == expected


def field_title_models(layout, title_kind):
    generator = np.random.default_rng(31)
    title = {"generator": generator, "tuple": ("stream", generator), "string": "measurement"}[title_kind]
    dtype = np.dtype([((title, "word"), "u4")])
    if layout == "nested":
        dtype = np.dtype([("inner", dtype)])
    elif layout == "subarray":
        dtype = np.dtype([("inner", dtype, (2,))])
    payload = np.zeros(1, dtype=dtype)
    return {"a": (FieldTitleTree(payload), {}), "b": (PoolParameterTree(seed_pool(generator)), {})}


@pytest.mark.parametrize("layout", ["direct", "nested", "subarray"])
@pytest.mark.parametrize("title_kind", ["generator", "tuple"])
def test_field_title_can_retain_a_later_models_seed_pool(tmp_path, compare, layout, title_kind):
    joint, single, saved = (tmp_path / t for t in ("joint", "single", "saved"))
    models = field_title_models(layout, title_kind)
    fixed = tree(337)
    compare(joint, {**models, "fixed": fixed})
    compare(single, {"b": field_title_models(layout, title_kind)["b"]})
    paired, standalone = scientific(joint, "b"), scientific(single, "b")
    assert paired[0] == standalone[0] and paired[1] != standalone[1]
    assert paired[-1]["predicted_labels"] != standalone[-1]["predicted_labels"]
    assert paired[-1]["positive_class_probability"] != standalone[-1]["positive_class_probability"]
    compare(saved, {"a": field_title_models(layout, title_kind)["a"]})
    before = archive(saved)
    with (patch.object(PoolParameterTree, "fit", autospec=True, side_effect=PoolParameterTree.fit) as fit,
          patch.object(evaluation, "write_json", wraps=evaluation.write_json) as write):
        with pytest.raises(ValueError, match="fit generator|captured state|RNG or buffer"):
            compare(saved, field_title_models(layout, title_kind))
        fit.assert_not_called()
        write.assert_not_called()
    assert archive(saved) == before

    compare(saved, {"fixed": fixed})
    assert scientific(saved, "fixed") == scientific(joint, "fixed")
    assert (saved / "a-evaluation.json").read_bytes() == before["a-evaluation.json"]


@pytest.mark.parametrize("layout", ["direct", "nested", "subarray"])
def test_string_field_title_allows_comparison_restart(tmp_path, compare, layout):
    joint, saved = tmp_path / "joint", tmp_path / "saved"
    compare(joint, field_title_models(layout, "string"))
    compare(saved, {"a": field_title_models(layout, "string")["a"]})
    before = archive(saved)
    compare(saved, field_title_models(layout, "string"))
    assert scientific(saved, "b") == scientific(joint, "b")
    for name in ("a-cv.json", "a-evaluation.json"):
        assert (saved / name).read_bytes() == before[name]


@pytest.mark.parametrize(("attachment", "unverified"), [
    ("field-name-subclass", True), ("void-subclass", True), ("record", False), ("integer-title", False),
    ("generator-sentinel", True), ("string-sentinel", False), ("absent-sentinel", False)])
def test_dtype_reference_qualification(attachment, unverified):
    generator = np.random.default_rng(31)
    if attachment == "field-name-subclass":
        class FieldName(str):
            def get_params(self, deep=False):
                return {}

        name = FieldName("word")
        name.stream = generator
        dtype = np.dtype([(name, "u4")])
        assert dtype.names[0] is name
    elif attachment in ("void-subclass", "record"):
        class ScalarWithStream(np.void):
            stream = generator

        scalar = ScalarWithStream if unverified else np.record
        dtype = np.dtype((scalar, np.dtype([("word", "u4")])))
        assert dtype.type is scalar
    elif attachment == "integer-title":
        dtype = np.dtype([((9, "word"), "u4")])
    else:
        if not hasattr(np.dtypes, "StringDType"):
            pytest.skip("requires NumPy variable-width strings")
        options = {"generator-sentinel": {"na_object": generator},
                   "string-sentinel": {"na_object": "NA"}, "absent-sentinel": {}}
        dtype = np.dtypes.StringDType(**options[attachment])
    limits = []
    configuration_identity(dtype, _split_limits=limits)
    assert ("unverified captured state" in limits) is unverified
    assert ("unverified fit RNG" in limits) is unverified


@pytest.mark.skipif(not hasattr(np.dtypes, "StringDType"), reason="requires NumPy variable-width strings")
def test_dtype_reference_as_only_calibration_rng_prevents_partial_reuse(tmp_path, monkeypatch):
    X = np.random.RandomState(121).normal(size=(60, 4))
    y = np.tile([0, 1], 30)
    generator = np.random.default_rng(31)
    payload = np.array(["ok"], dtype=np.dtypes.StringDType(na_object=generator))
    model = MissingValueRNGTree(payload)

    def run(path):
        # Keep the sentinel's identity and restore the exact entry RNG state.
        generator.bit_generator.state = np.random.PCG64(31).state
        return evaluation.calibration_search(model, {}, X, y, seed=73, checkpoint_path=path, refit=False)

    complete = run(tmp_path / "reference.json")
    path = tmp_path / "b-cv.json"
    with monkeypatch.context() as interrupted:
        interrupt_after_fold(interrupted, "b")
        with pytest.raises(Interrupted):
            run(path)
    before = path.read_bytes()
    folds = json.loads(before)["candidates"][0]["folds"]
    assert len(folds) == 1 and folds[0]["score"] == complete.report["candidates"][0]["folds"][0]["score"]
    with (patch.object(MissingValueRNGTree, "fit", autospec=True, side_effect=MissingValueRNGTree.fit) as fit,
          patch.object(evaluation, "write_json", wraps=evaluation.write_json) as write):
        with pytest.raises(ValueError, match="verifiable clone behavior"):
            run(path)
        fit.assert_not_called()
        write.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.skipif(not hasattr(np.dtypes, "StringDType"), reason="requires NumPy variable-width strings")
@pytest.mark.parametrize(("writer", "reader"), [("missing", "literal"), ("literal", "missing"),
                                                ("literal", "other-literal"), ("missing", "missing"),
                                                ("literal", "literal")])
def test_string_parameter_values_bind_calibration_folds(tmp_path, monkeypatch, writer, reader):
    X = np.random.RandomState(121).normal(size=(60, 4))
    y = np.tile([0, 1], 30)
    generator = np.random.default_rng(31)
    sentinel = "x" * 256
    dtype = np.dtypes.StringDType(na_object=sentinel)
    ordinary = np.dtypes.StringDType()
    words = {"missing": np.array([sentinel], dtype=dtype),
             "literal": np.array([sentinel], dtype=ordinary).astype(dtype),
             "other-literal": np.array(["y" * 256], dtype=ordinary).astype(dtype)}
    assert words["missing"].tolist() == words["literal"].tolist()

    def run(which, path):
        return evaluation.calibration_search(StringParameterTree(words[which], generator), {}, X, y, seed=73,
                                             checkpoint_path=path, refit=False)

    reference = run(reader, tmp_path / "reference.json")
    path = tmp_path / "b-cv.json"
    with monkeypatch.context() as interrupted:
        interrupt_after_fold(interrupted, "b")
        with pytest.raises(Interrupted):
            run(writer, path)
    before = path.read_bytes()
    saved = json.loads(before)["candidates"][0]["folds"]
    expected = [fold["score"] for fold in reference.report["candidates"][0]["folds"]]
    assert len(saved) == 1
    if writer == reader:
        resumed = run(reader, path).report["candidates"][0]["folds"]
        assert resumed[:1] == saved and [fold["score"] for fold in resumed] == expected
        return
    assert saved[0]["score"] != expected[0]
    with pytest.raises(ValueError, match="incompatible"):
        run(reader, path)
    assert path.read_bytes() == before
    assert configuration_identity(words[writer]) != configuration_identity(words[reader])


@pytest.mark.skipif(not hasattr(np.dtypes, "StringDType"), reason="requires NumPy variable-width strings")
@pytest.mark.parametrize("coerce", [False, True])
def test_string_parameter_observes_strided_empty_literals_and_missing_values(coerce):
    plain = np.dtypes.StringDType(coerce=coerce)
    dtype = np.dtypes.StringDType(na_object="", coerce=coerce)
    words = np.array([["", "skip", "a\0b"], ["end", "skip", ""]], dtype=plain).astype(dtype)
    words[1, 2] = ""
    view = words[::-1, ::2]
    observed = []
    implementation_sources(view, _observation=observed)
    expected = [configuration_identity(item) for item in ("end", None, "", "a\0b")]
    assert observed[0]["content"] == configuration_identity(view)["content"] == expected
    assert configuration_identity(view[:0])["content"] == []


@pytest.mark.skipif(np.__version__.split(".")[0] != "2", reason="requires qualified NumPy 2 seed pools")
@pytest.mark.parametrize("fixed_predecessor", [False, True])
def test_standalone_deterministic_view_retains_archive_reuse(tmp_path, compare, monkeypatch, fixed_predecessor):
    def model():
        return {"b": pool_backed_models("wrapped-strided", False)["b"]}

    models = {"fixed": tree(337)} if fixed_predecessor else {}
    compare(tmp_path, {**models, **model()})
    before = archive(tmp_path)

    def unexpected_calibration(*args, **kwargs):
        pytest.fail("Loading the complete evaluation should skip fitting.")

    monkeypatch.setattr(evaluation, "calibration_search", unexpected_calibration)
    compare(tmp_path, model())
    assert json.loads((tmp_path / "comparison.json").read_bytes())["models"] == {
        "b": json.loads(before["b-evaluation.json"])}
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items()
               if name != "comparison.json")


@pytest.mark.skipif(np.__version__.split(".")[0] != "2", reason="requires qualified NumPy 2 seed pools")
def test_unverified_prefix_checks_concurrent_cv_publication_under_lock(tmp_path, compare, monkeypatch):
    writer, reader = tmp_path / "writer", tmp_path / "reader"
    with monkeypatch.context() as interrupted:
        interrupt_after_fold(interrupted, "b")
        with pytest.raises(Interrupted):
            compare(writer, pool_backed_models("wrapped-strided", False))
    published = (writer / "b-cv.json").read_bytes()
    calibrate = evaluation.calibration_search

    def publish_before_lock(*args, **kwargs):
        path = Path(kwargs["checkpoint_path"])
        if path.name == "b-cv.json":
            assert not path.exists()
            path.write_bytes(published)
        return calibrate(*args, **kwargs)

    monkeypatch.setattr(evaluation, "calibration_search", publish_before_lock)
    with pytest.raises(ValueError, match="unverifiable comparison RNG prefix"):
        compare(reader, pool_backed_models("wrapped-strided", False))
    assert (reader / "b-cv.json").read_bytes() == published
    assert not (reader / "b-evaluation.json").exists()
