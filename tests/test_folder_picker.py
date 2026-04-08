from image_archive.services.folder_picker import _build_rows, _toggle_selected


def test_folder_picker_rows_show_selectable_folders(tmp_path) -> None:
    current = tmp_path / "library"
    child = current / "child"
    child.mkdir(parents=True)
    selected = [current.resolve()]

    rows = _build_rows(current.resolve(), selected)

    current_row = rows[0]
    child_row = next(row for row in rows if row.path == child.resolve())
    assert current_row.selectable is True
    assert child_row.selectable is True
    assert current_row.path in selected


def test_folder_picker_toggle_selected_adds_and_removes_path(tmp_path) -> None:
    selected = []
    folder = tmp_path / "folder"
    folder.mkdir()

    _toggle_selected(selected, folder)
    assert selected == [folder.resolve()]

    _toggle_selected(selected, folder)
    assert selected == []
