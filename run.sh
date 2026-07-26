#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

litellm_data_dir="${LITELLM_DIR:-${HOME}/.litellm}"
chatgpt_token_dir="${HOME}/.config/litellm/chatgpt"
github_copilot_token_dir="${HOME}/.config/litellm/github_copilot"
config_file="${litellm_data_dir}/config.yaml"

mkdir -p "${chatgpt_token_dir}" "${github_copilot_token_dir}"

if grep -q "chatgpt/" "${config_file}" 2>/dev/null; then
    if [ ! -f "${chatgpt_token_dir}/auth.json" ] || ! grep -q '"access_token"' "${chatgpt_token_dir}/auth.json" 2>/dev/null; then
        echo "WARNING: Config has ChatGPT models but no token found. Run ./login-chatgpt.sh first."
    fi
fi

if grep -q "github_copilot/" "${config_file}" 2>/dev/null; then
    if [ ! -f "${github_copilot_token_dir}/api-key.json" ] || ! grep -q '"token"' "${github_copilot_token_dir}/api-key.json" 2>/dev/null; then
        echo "WARNING: Config has GitHub Copilot models but no token found. Run ./login-github-copilot.sh first."
    fi
fi

uv sync --extra proxy --extra extra_proxy 2>&1
exec uv run litellm \
    --config "${config_file}" \
    --port 4000 \
    --host 0.0.0.0
