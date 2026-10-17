#!/bin/bash
set -euo pipefail

python3 - <<'PY'
import json
import sys
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

auth_file: Final = Path.home() / ".pi/agent/auth.json"

try:
    auth: Final = json.loads(auth_file.read_text())
    entry: Final = auth.get("deepseek") if isinstance(auth, dict) else None
    key: Final = entry.get("key") if isinstance(entry, dict) else None
    if not isinstance(key, str) or not key.strip():
        sys.exit(f"list-deepseek-models: Missing DeepSeek key in {auth_file}")

    request: Final = Request(
        "https://api.deepseek.com/models",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        json.dump(json.load(response), sys.stdout, indent=2)
    sys.stdout.write("\n")
except HTTPError as error:
    sys.exit(f"list-deepseek-models: API returned HTTP {error.code}")
except (OSError, ValueError, URLError) as error:
    sys.exit(f"list-deepseek-models: {error}")
PY
