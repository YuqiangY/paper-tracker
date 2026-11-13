"""Tests for deploy failure/timeout handling and post-mortem verification.

Covers the false-alarm case: deploy.sh is killed by the outer timeout even
though the commit already reached origin (e.g. SSH session hangs after a
successful push). The pipeline must report ok in that case, not failure.
"""

import subprocess

import pytest

from main import _is_site_deployed, _run_deploy


def _git(cwd, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _commit_site(work, content: str) -> None:
    (work / "site" / "index.html").write_text(content)
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "update site")


@pytest.fixture()
def work_repo(tmp_path):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True
    )
    subprocess.run(["git", "init", "-b", "main", str(work)], check=True, capture_output=True)
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    _git(work, "remote", "add", "origin", str(origin))
    (work / "site").mkdir()
    (work / "site" / "index.html").write_text("v1")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "init")
    _git(work, "push", "-u", "origin", "main")
    return work


def test_run_deploy_ok(work_repo):
    # Arrange & Act & Assert
    assert _run_deploy("true", cwd=str(work_repo), timeout=30) == "ok"


def test_reports_failure_when_commit_never_reached_origin(work_repo):
    # Arrange: commit exists locally but was NOT pushed
    _commit_site(work_repo, "v2")

    # Act
    result = _run_deploy("false", cwd=str(work_repo), timeout=30)

    # Assert
    assert result.startswith("failed")


def test_timeout_but_already_pushed_reports_ok(work_repo):
    # Arrange: deploy.sh committed AND pushed, then hung and was killed
    _commit_site(work_repo, "v2")
    _git(work_repo, "push", "origin", "main")

    # Act
    result = _run_deploy("sleep 30", cwd=str(work_repo), timeout=1)

    # Assert: not a failure — the site is provably deployed
    assert not result.startswith("failed")
    assert "ok" in result


def test_timeout_with_uncommitted_site_reports_failed(work_repo):
    # Arrange: deploy.sh was killed before it could commit site/
    (work_repo / "site" / "index.html").write_text("v2")

    # Act
    result = _run_deploy("sleep 30", cwd=str(work_repo), timeout=1)

    # Assert
    assert result.startswith("failed")


def test_timeout_with_commit_not_pushed_reports_failed(work_repo):
    # Arrange: deploy.sh committed but was killed before the push landed
    _commit_site(work_repo, "v2")

    # Act
    result = _run_deploy("sleep 30", cwd=str(work_repo), timeout=1)

    # Assert
    assert result.startswith("failed")


def test_is_site_deployed_false_when_site_dirty(work_repo):
    # Arrange
    (work_repo / "site" / "index.html").write_text("v2")

    # Act & Assert
    assert _is_site_deployed(str(work_repo)) is False


def test_is_site_deployed_true_when_synced(work_repo):
    # Arrange: clean tree, HEAD == origin/main (fixture state)

    # Act & Assert
    assert _is_site_deployed(str(work_repo)) is True
