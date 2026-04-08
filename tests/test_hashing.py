from image_archive.indexing.fingerprint import file_sha256, stat_signature


def test_hashing_and_signatures_change_with_content(tmp_path) -> None:
    path = tmp_path / "sample.jpg"
    path.write_bytes(b"alpha")

    first_hash = file_sha256(path)
    first_signature = stat_signature(path)

    path.write_bytes(b"beta")

    assert file_sha256(path) != first_hash
    assert stat_signature(path) != first_signature
