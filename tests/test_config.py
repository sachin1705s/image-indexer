from image_archive.config import AppConfig, ConfigManager, default_config_path, default_data_dir


def test_default_config_uses_user_app_directory() -> None:
    manager = ConfigManager()
    config = AppConfig()

    assert manager.path == default_config_path()
    assert config.app_data_dir == str(default_data_dir())
    assert config.sqlite_path == str(default_data_dir() / "archive.db")
