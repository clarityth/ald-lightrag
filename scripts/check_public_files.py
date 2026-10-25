"""Check tracked publication files without printing sensitive matches."""
import re
import subprocess
from pathlib import Path

files = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
violations = []
patterns = ['/' + r'Users/[^/\s]+/', r'sk-proj-[A-Za-z0-9_-]{20,}', r'gh[pousr]_[A-Za-z0-9]{20,}']
for name in files:
    path = Path(name)
    if path.name == '.DS_Store' or path.name.startswith('.env') and path.name != '.env.example':
        violations.append(name)
    if any(part in {'internal', '_workspace', 'cache', 'results', '__pycache__'} for part in path.parts):
        violations.append(name)
    if path.is_file():
        text = path.read_text(errors='replace')
        if any(re.search(pattern, text) for pattern in patterns):
            violations.append(name)
assert not violations, 'Publication check failed: ' + ', '.join(sorted(set(violations)))
print(f'Publication check passed: {len(files)} tracked files')
