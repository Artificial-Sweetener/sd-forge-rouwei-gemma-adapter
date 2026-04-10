"""Test installer-managed bootstrap and launcher generation behavior."""

from __future__ import annotations

from pathlib import Path

import install


def test_extract_base_package_handles_comparison_requirements() -> None:
    """Extract the base package name from one comparison requirement string."""

    assert install.extract_base_package("safetensors>=0.5.2") == "safetensors"


def test_install_requirements_skips_empty_lines(tmp_path: Path) -> None:
    """Skip blank and comment-only lines while reading requirements."""

    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("\n# comment only\n", encoding="utf-8")
    install.install_requirements(requirements_file)


def test_should_upgrade_package_when_missing() -> None:
    """Require install when the package is missing."""

    assert install.should_upgrade_package(None, "4.53.1") is True


def test_should_upgrade_package_when_older() -> None:
    """Require install when the package is below the required version."""

    assert install.should_upgrade_package("4.48.1", "4.53.1") is True


def test_should_upgrade_package_when_new_enough() -> None:
    """Skip install when the package already satisfies the minimum version."""

    assert install.should_upgrade_package("4.53.1", "4.53.1") is False
    assert install.should_upgrade_package("4.54.0", "4.53.1") is False


def test_ensure_required_transformers_version_upgrades_when_missing(
    monkeypatch,
) -> None:
    """Upgrade host transformers when it is not installed."""

    pip_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(install, "get_installed_version", lambda _: None)
    monkeypatch.setattr(
        install.launch,
        "run_pip",
        lambda command, description: pip_calls.append((command, description)),
    )

    changed = install.ensure_required_transformers_version()

    assert changed is True
    assert pip_calls == [
        (
            'install -U "transformers==4.53.1"',
            "sd-forge-rouwei-gemma-adapter runtime: upgrading transformers from None to 4.53.1",
        )
    ]


def test_ensure_required_transformers_version_skips_when_current(
    monkeypatch,
) -> None:
    """Skip host upgrade when transformers already meets the requirement."""

    pip_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(install, "get_installed_version", lambda _: "4.53.1")
    monkeypatch.setattr(
        install.launch,
        "run_pip",
        lambda command, description: pip_calls.append((command, description)),
    )

    changed = install.ensure_required_transformers_version()

    assert changed is False
    assert pip_calls == []


def test_resolve_reforge_root_from_extension_repo(tmp_path: Path) -> None:
    """Derive the host ReForge root from the extension repo path."""

    host_root = tmp_path / "stable-diffusion-webui-reForge"
    extension_root = host_root / "extensions" / "sd-forge-rouwei-gemma-adapter"
    extension_root.mkdir(parents=True)
    (host_root / "webui.bat").write_text("@echo off\r\n", encoding="utf-8")

    resolved = install.resolve_reforge_root(extension_root)

    assert resolved == host_root


def test_extract_batch_variable_returns_empty_when_missing() -> None:
    """Return an empty string when the requested batch variable is absent."""

    assert install.extract_batch_variable("@echo off\r\n", "COMMANDLINE_ARGS") == ""


def test_extract_batch_variable_reads_first_assignment() -> None:
    """Extract one variable assignment from a normal user launcher."""

    contents = (
        "@echo off\r\n"
        "set PYTHON=\r\n"
        "set COMMANDLINE_ARGS= --listen --api\r\n"
        "set COMMANDLINE_ARGS= --ignore-this\r\n"
    )

    assert (
        install.extract_batch_variable(contents, "COMMANDLINE_ARGS") == "--listen --api"
    )


def test_read_user_launcher_settings_reads_supported_variables(tmp_path: Path) -> None:
    """Read the launcher variables that the managed launcher needs."""

    user_launcher = tmp_path / "webui-user.bat"
    user_launcher.write_text(
        "@echo off\r\n"
        "set PYTHON=C:\\Python310\\python.exe\r\n"
        "set GIT=C:\\Program Files\\Git\\bin\\git.exe\r\n"
        "set VENV_DIR=C:\\venvs\\reforge\r\n"
        "set COMMANDLINE_ARGS= --listen --api\r\n",
        encoding="utf-8",
    )

    settings = install.read_user_launcher_settings(user_launcher)

    assert settings == {
        "PYTHON": "C:\\Python310\\python.exe",
        "GIT": "C:\\Program Files\\Git\\bin\\git.exe",
        "VENV_DIR": "C:\\venvs\\reforge",
        "COMMANDLINE_ARGS": "--listen --api",
    }


def test_merge_commandline_args_appends_skip_install_once() -> None:
    """Append `--skip-install` when it is missing."""

    assert (
        install.merge_commandline_args("--listen --api")
        == "--listen --api --skip-install"
    )


def test_merge_commandline_args_preserves_existing_skip_install() -> None:
    """Keep a single `--skip-install` token when already present."""

    assert install.merge_commandline_args("--listen --skip-install --api") == (
        "--listen --skip-install --api"
    )


def test_merge_commandline_args_uses_skip_install_for_empty_input() -> None:
    """Produce the required launcher flag when no user args are defined."""

    assert install.merge_commandline_args("") == "--skip-install"


def test_render_managed_launcher_contains_required_runtime_logic() -> None:
    """Render a launcher that repairs transformers and preserves user settings."""

    launcher = install.render_managed_launcher("4.53.1")

    assert "webui-user-gemma-adapter.bat" not in launcher
    assert "Managed by sd-forge-rouwei-gemma-adapter" in launcher
    assert "USER_LAUNCHER=%REFORGE_ROOT%webui-user.bat" in launcher
    assert "COMMANDLINE_ARGS" in launcher
    assert "--skip-install" in launcher
    assert 'pip install "transformers==4.53.1"' in launcher
    assert 'call "%WEBUI_BAT%" %*' in launcher


def test_write_managed_launcher_writes_expected_file(tmp_path: Path) -> None:
    """Write the managed launcher into the host ReForge root."""

    host_root = tmp_path / "stable-diffusion-webui-reForge"
    host_root.mkdir()

    launcher_path = install.write_managed_launcher(host_root)

    assert launcher_path == host_root / "webui-user-gemma-adapter.bat"
    assert launcher_path.is_file()
    assert "transformers==4.53.1" in launcher_path.read_text(encoding="utf-8")


def test_report_setup_result_mentions_generated_launcher(capsys) -> None:
    """Tell the user to launch with the generated managed launcher."""

    install.report_setup_result(
        Path("C:\\reforge\\webui-user-gemma-adapter.bat"), False
    )

    output = capsys.readouterr().out
    assert "webui-user-gemma-adapter.bat" in output
    assert "Launch ReForge with that file." in output
