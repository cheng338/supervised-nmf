"""Empty CSV record handling without relaxing malformed attribute validation."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from supervised_nmf.datasets import _attribute_labels, load_celeba


HEADER = "image_id,Smiling,Wearing_Necktie"
ROWS = ("a.png,1,-1", "b.png,0,1", "c.png,-1,1")
LABELS = {"a.png": 1, "b.png": 0, "c.png": 0}


class AttributeCSVRecordTests(unittest.TestCase):
    """Parser-only checks do not import or depend on optional Pillow."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = 0

    def _fixture(self, content, suffix=".csv"):
        self.index += 1
        path = self.root / f"attributes-{self.index}{suffix}"
        path.write_bytes(content.encode("utf-8"))
        return path

    def _parse(self, content, suffix=".csv", attribute="Smiling"):
        path = self._fixture(content, suffix)
        before = path.read_bytes()
        try:
            return _attribute_labels(path, attribute)
        finally:
            self.assertEqual(path.read_bytes(), before)

    def test_empty_records_preserve_labels_for_all_line_endings(self):
        for ending in ("\n", "\r\n", "\r"):
            with self.subTest(ending=repr(ending)):
                clean = ending.join((HEADER, *ROWS)) + ending
                # Empty data records directly after header, between data and at EOF.
                spaced = ending.join((HEADER, "", ROWS[0], "", ROWS[1], ROWS[2], "")) + ending
                self.assertEqual(self._parse(clean), LABELS)
                self.assertEqual(self._parse(spaced), LABELS)

    def test_header_with_only_empty_data_still_rejects_empty_file(self):
        for ending in ("\n", "\r\n", "\r"):
            for empty_records in (0, 3):
                with self.subTest(ending=repr(ending), empty_records=empty_records):
                    with self.assertRaisesRegex(ValueError, "^Attribute file is empty$"):
                        self._parse(HEADER + ending * (empty_records + 1))

    def test_nonempty_empty_looking_records_are_not_discarded(self):
        prefix = HEADER + "\n\n" + ROWS[0] + "\n\n"
        for record, message in (
            (" \t ", "Malformed attribute row 4"),
            (",,", "Invalid image identifier ''"),
            (" , , ", "Invalid image identifier ''"),
            ('""', "Malformed attribute row 4"),
            ('" \t "', "Malformed attribute row 4"),
            ('"","",""', "Invalid image identifier ''"),
        ):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, message):
                self._parse(prefix + record + "\n")

    def test_malformed_fields_and_duplicates_after_empty_records_still_fail(self):
        prefix = HEADER + "\n\n" + ROWS[0] + "\n\n"
        for record, message in (
            (",1,0", "Invalid image identifier ''"),
            ("b.png,,1", "Invalid binary label ''"),
            ("b.png,2,1", "Invalid binary label '2'"),
            ("b.png,1", "Malformed attribute row 4"),
            ("b.png,1,0,0", "Malformed attribute row 4"),
            (ROWS[0], "Duplicate image identifier 'a.png'"),
        ):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, message):
                self._parse(prefix + record + "\n")

    def test_row_ordinal_counts_empty_logical_records_not_physical_lines(self):
        content = 'image_id,Smiling,Note\n\na.png,1,"alpha\n\nomega"\n\nb.png,-1\n'
        path = self._fixture(content)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.reader(stream))
        self.assertEqual(records, [["image_id", "Smiling", "Note"], [],
                                  ["a.png", "1", "alpha\n\nomega"], [], ["b.png", "-1"]])
        with self.assertRaisesRegex(ValueError, "Malformed attribute row 4"):
            _attribute_labels(path, "Smiling")

    def test_quoted_multiline_fields_keep_existing_semantics(self):
        for content, expected in (
            (HEADER + '\n\na.png," \n\n1 ",-1\n\n', {"a.png": 1}),
            ('image_id,Smiling,Note\n\na.png,1,"alpha\n\nomega"\n\n', {"a.png": 1}),
            (HEADER + '\n\n"a\n\nb.png",1,-1\n\n', {"a\n\nb.png": 1}),
        ):
            with self.subTest(content=content):
                self.assertEqual(self._parse(content), expected)

    def test_bom_suffix_and_first_line_csv_detection_are_unchanged(self):
        content = HEADER + "\n\n" + "\n".join(ROWS) + "\n\n"
        for prefix, suffix in (("\ufeff", ".csv"), ("", ".CSV"), ("\ufeff", ".data")):
            with self.subTest(prefix=repr(prefix), suffix=suffix):
                self.assertEqual(self._parse(prefix + content, suffix), LABELS)

    def test_headers_remain_strict_and_are_not_skipped(self):
        for content, suffix, message in (
            ("", ".csv", "Attribute CSV must start"),
            ("\n" + HEADER + "\n" + ROWS[0], ".csv", "Attribute CSV must start"),
            (" \t \n" + HEADER + "\n" + ROWS[0], ".csv", "Attribute CSV must start"),
            ("\n" + HEADER + "\n" + ROWS[0], ".data", "Official attribute text must begin"),
            ("id,Smiling\n\na.png,1\n", ".csv", "Attribute CSV must start"),
            ("image_id,Smiling,Smiling\n\n" + ROWS[0], ".csv", "Duplicate attribute names"),
            ("image_id,Wearing_Necktie\n\na.png,1\n", ".csv", "Attribute 'Smiling' is missing"),
        ):
            with self.subTest(content=content, suffix=suffix), self.assertRaisesRegex(ValueError, message):
                self._parse(content, suffix)

    def test_official_text_blank_lines_and_count_check_are_unchanged(self):
        body = "Smiling Wearing_Necktie\n\na.png 1 -1\n \t \nb.png 0 1\nc.png -1 1\n\n"
        self.assertEqual(self._parse("3\n" + body, ".txt"), LABELS)
        for count in (2, 4):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "Attribute row count does not match"):
                self._parse(f"{count}\n" + body, ".txt")


@unittest.skipUnless(importlib.util.find_spec("PIL"), "optional Pillow dependency unavailable")
class AttributeCSVLoaderTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.ids = ["a.png", "b.png", "c.png", "d.png", "v.png", "w.png"]
        self.rows = [f"{identifier},{label}" for identifier, label in
                     zip(self.ids, (1, -1, 0, 1, -1, 1))]
        for identifier, intensity in zip(self.ids[:4], (32, 64, 128, 224)):
            Image.new("RGB", (2, 3), (intensity,) * 3).save(self.images / identifier)
        self.attrs = self.root / "attributes.csv"
        self.parts = self.root / "partitions.txt"
        self.parts.write_text("\n".join(f"{name} {0 if index < 4 else 1}"
                                       for index, name in enumerate(self.ids)) + "\n", encoding="utf-8")

    def test_empty_records_preserve_full_and_seeded_subsample_results(self):
        clean = "image_id,Smiling\n" + "\n".join(reversed(self.rows)) + "\n"
        spaced = "image_id,Smiling\n\n" + "\n\n".join(reversed(self.rows)) + "\n\n"
        for maximum in (None, 2):
            with self.subTest(max_samples=maximum):
                options = {"max_samples": maximum, "random_state": 17, "image_size": (3, 2)}
                self.attrs.write_bytes(clean.encode("utf-8"))
                first = load_celeba(self.images, self.attrs, self.parts, **options)
                self.attrs.write_bytes(spaced.encode("utf-8"))
                second = load_celeba(self.images, self.attrs, self.parts, **options)
                self.assertEqual(set(first), set(second))
                for field in ("data", "target", "image_ids"):
                    self.assertEqual(first[field].dtype, second[field].dtype)
                    np.testing.assert_array_equal(first[field], second[field])
                self.assertEqual(first.metadata, second.metadata)
                self.assertEqual(len(second.target), 4 if maximum is None else maximum)
                self.assertEqual(self.attrs.read_bytes(), spaced.encode("utf-8"))

    def test_malformed_records_after_empty_records_fail_before_image_opening(self):
        from PIL import Image

        for record, message in ((" \t ", "Malformed attribute row 4"),
                                (",1", "Invalid image identifier ''"),
                                ("b.png,", "Invalid binary label ''")):
            with self.subTest(record=record):
                self.attrs.write_text("image_id,Smiling\n\na.png,1\n\n" + record + "\n", encoding="utf-8")
                with patch.object(Image, "open", side_effect=AssertionError("Image access before attribute validation")) as opened:
                    with self.assertRaisesRegex(ValueError, message):
                        load_celeba(self.images, self.attrs, self.parts)
                    opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
