from pathlib import Path

from image_archive.indexing.discovery import discover_image_files


def test_discover_image_files_finds_supported_assets(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "photo.jpg").write_bytes(b"not-really-a-photo")
    (tmp_path / "nested" / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "cover.png").write_bytes(b"png")

    files = discover_image_files([tmp_path])

    assert [item.name for item in files] == ["cover.png", "photo.jpg"]


def test_discover_image_files_skips_video_frame_like_folders(tmp_path: Path) -> None:
    (tmp_path / "Photos Library.photoslibrary" / "resources" / "derivatives").mkdir(parents=True)
    (tmp_path / "Photos Library.photoslibrary" / "resources" / "derivatives" / "frame001.jpg").write_bytes(b"x")
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "clip_frame_002.png").write_bytes(b"y")
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "real_photo.jpg").write_bytes(b"z")

    files = discover_image_files(
        [tmp_path],
        excluded_dir_names={"photo library.photoslibrary", ".photoslibrary", "derivatives", "frames"},
        excluded_path_fragments=[".photoslibrary/resources/derivatives"],
    )

    assert [item.name for item in files] == ["real_photo.jpg"]
