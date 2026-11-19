from autoforge.core.agent import _reuse_hint


class Registry:
    def __init__(self, *names):
        self.names = set(names)

    def get(self, name):
        return object() if name in self.names else None


def test_existing_read_tools_are_suggested():
    hint = _reuse_hint("print(open('a.txt').read())", Registry("read_local_file"))
    assert "read_local_file" in hint
    assert "read_file_lines" not in hint


def test_writes_and_database_work_are_not_steered():
    r = Registry("read_local_file")
    assert not _reuse_hint("open('a', 'w').write('x')", r)
    assert not _reuse_hint("open('a', mode='a')", r)
    assert not _reuse_hint("conn.execute('select 1')", r)


def test_bad_code_missing_registry_and_calculation():
    r = Registry("read_local_file")
    assert not _reuse_hint("if :", r)
    assert not _reuse_hint("print(2+2)", r)
    assert not _reuse_hint("open('a').read()", Registry())


def test_subprocess_suggestion_is_filtered():
    assert "run_shell_command" in _reuse_hint(
        "import subprocess; subprocess.run(['git', 'status'])",
        Registry("run_shell_command"))
