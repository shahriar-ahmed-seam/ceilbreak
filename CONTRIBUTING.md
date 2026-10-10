# Contributing to CeilBreak

Thank you for your interest in contributing to CeilBreak. We maintain strict production-grade software engineering standards.

---

## 1. Development Environment Setup

Clone the repository and install all dependencies inside an isolated virtual environment:

```bash
git clone https://github.com/shahriar-ahmed-seam/ceilbreak.git
cd ceilbreak

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 2. Engineering Standards and Code Quality

All pull requests must satisfy the following validation gates prior to submission:

1. **Test Suite Execution**:
   ```bash
   pytest -v
   ```
   100% test pass rate is mandatory.

2. **Linter and Code Formatting**:
   ```bash
   ruff check .
   ```
   Zero linter errors or warnings allowed.

3. **Documentation Hygiene**:
   - Zero emojis in all documentation files and code comments.
   - All architecture diagrams must be authored as structured Mermaid (`mermaid`) flowcharts.
   - Maintain typing annotations across all public functions and classes.

---

## 3. GitHub Issue and PR Workflow

1. File a formal GitHub issue with problem statement and technical requirements.
2. Branch off `main` using standard prefixes: `feat/...`, `fix/...`, `docs/...`.
3. Submit a pull request referencing the corresponding issue.
4. Ensure linear history and passing test verifications.
