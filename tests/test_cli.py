"""Tests for the CLI entry point."""

import os
from click.testing import CliRunner

from gptempest.main import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert 'TEMPEST' in result.output


def test_cli_generate_config(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ['--generate_config'])
        assert result.exit_code == 0
        assert os.path.exists('default_tempest_config.yaml')
