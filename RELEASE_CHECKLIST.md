# Public release checklist

Before publishing GraphSkillAA:

1. Keep only source code, configurations, prompts, fixed split metadata, tests,
   documentation, and the framework figure.
2. Do not commit `.env`, API keys, raw benchmark payloads, model weights, run
   outputs, logs, caches, or virtual environments.
3. Run the offline release checks:

   ```bash
   PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
   PYTHONPYCACHEPREFIX=/tmp/graphskillaa_pyc python -m compileall -q graphopt scripts
   python scripts/verify_release.py
   ```

4. Review `git status --short` and inspect every staged change.
5. Add the project license chosen by the copyright holders before declaring the
   repository open source. No license is selected automatically by this package.

The anonymous review package has additional identity checks and must not include
the public repository's `.git` directory or URL.
