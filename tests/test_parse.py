"""Tests for scripts/parse.py changelog parser."""

from pathlib import Path

from scripts.parse import parse_changelog


def _write_changelog(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "CHANGELOG.md"
    p.write_text(content)
    return p


def test_basic_parsing(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- Added new feature
- Fixed a bug
""")
    entries = parse_changelog(path, {"1.0.0": "2025-01-01"})
    assert len(entries) == 2
    assert entries[0]["version"] == "1.0.0"
    assert entries[0]["text"] == "Added new feature"
    assert entries[0]["entry_index"] == 0
    assert entries[1]["entry_index"] == 1


def test_multiple_versions(tmp_path):
    path = _write_changelog(tmp_path, """\
## 2.0.0
- Added X

## 1.0.0
- Fixed Y
""")
    entries = parse_changelog(path, {})
    assert len(entries) == 2
    assert entries[0]["version"] == "2.0.0"
    assert entries[1]["version"] == "1.0.0"


def test_version_dates(tmp_path):
    path = _write_changelog(tmp_path, """\
## 2.0.0
- Entry A

## 1.0.0
- Entry B
""")
    entries = parse_changelog(path, {"2.0.0": "2025-06-01"})
    assert entries[0]["date"] == "2025-06-01"
    assert entries[1]["date"] is None


def test_prefix_detection(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- Added a widget
- Fixed the crash
- Improved performance
- Changed the default
- Removed old code
- Some other thing
""")
    entries = parse_changelog(path, {})
    assert entries[0]["prefix"] == "Added"
    assert entries[1]["prefix"] == "Fixed"
    assert entries[2]["prefix"] == "Improved"
    assert entries[3]["prefix"] == "Changed"
    assert entries[4]["prefix"] == "Removed"
    assert entries[5]["prefix"] is None


def test_prefix_normalization_case(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- added lowercase prefix
""")
    entries = parse_changelog(path, {})
    assert entries[0]["prefix"] == "Added"


def test_breaking_change_prefix_normalized(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- Breaking change: removed old API
""")
    entries = parse_changelog(path, {})
    assert entries[0]["prefix"] == "Breaking"
    assert entries[0]["is_breaking"] is True


def test_vscode_detection(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- [VSCode] Fixed sidebar issue
- Fixed regular issue
""")
    entries = parse_changelog(path, {})
    assert entries[0]["is_vscode"] is True
    assert entries[0]["prefix"] == "Fixed"
    assert entries[1]["is_vscode"] is False


def test_bold_prefix_stripping(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- **Security:** Fixed silent sandbox disable
""")
    entries = parse_changelog(path, {})
    assert entries[0]["prefix"] == "Fixed"


def test_backtick_stripping(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- `some_flag` Added support for new flag
""")
    entries = parse_changelog(path, {})
    assert entries[0]["prefix"] == "Added"


def test_breaking_change_in_text(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- Removed old API (breaking change)
""")
    entries = parse_changelog(path, {})
    assert entries[0]["is_breaking"] is True


def test_non_breaking_entry(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- Added a feature
""")
    entries = parse_changelog(path, {})
    assert entries[0]["is_breaking"] is False


def test_lines_before_first_version_skipped(tmp_path):
    path = _write_changelog(tmp_path, """\
# Changelog

Some preamble text.

- This should be ignored

## 1.0.0
- Real entry
""")
    entries = parse_changelog(path, {})
    assert len(entries) == 1
    assert entries[0]["text"] == "Real entry"


def test_non_entry_lines_skipped(tmp_path):
    path = _write_changelog(tmp_path, """\
## 1.0.0
- Entry one
Some continuation text
### Subsection
- Entry two
""")
    entries = parse_changelog(path, {})
    assert len(entries) == 2


def test_empty_changelog(tmp_path):
    path = _write_changelog(tmp_path, "")
    entries = parse_changelog(path, {})
    assert entries == []


def test_entry_index_resets_per_version(tmp_path):
    path = _write_changelog(tmp_path, """\
## 2.0.0
- First in v2
- Second in v2

## 1.0.0
- First in v1
""")
    entries = parse_changelog(path, {})
    assert entries[0]["entry_index"] == 0
    assert entries[1]["entry_index"] == 1
    assert entries[2]["entry_index"] == 0
