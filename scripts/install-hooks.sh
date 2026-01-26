#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BioInsight AI - Git Hooks Installer
# ═══════════════════════════════════════════════════════════════
# Run this script to install git hooks for secret detection
#
# Usage: ./scripts/install-hooks.sh
# ═══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "🔧 Installing BioInsight AI git hooks..."

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Copy pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'HOOK_CONTENT'
#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# BioInsight AI - Pre-commit Hook (Secret Detection)
# ═══════════════════════════════════════════════════════════════

echo "🔍 Checking for secrets in staged files..."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BLOCKED_FILES=('.env' '.env.local' '.env.production' '.env.development' 'credentials.json' 'service-account.json' 'vertex-app-key.json')

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
    echo -e "${GREEN}✓ No files staged${NC}"
    exit 0
fi

FOUND_SECRETS=0

for blocked in "${BLOCKED_FILES[@]}"; do
    for file in $STAGED_FILES; do
        if [[ "$file" == "$blocked" ]] || [[ "$file" == *"/$blocked" ]]; then
            echo -e "${RED}❌ BLOCKED: '$file'${NC}"
            FOUND_SECRETS=1
        fi
    done
done

for file in $STAGED_FILES; do
    if [[ "$file" =~ \.(png|jpg|jpeg|gif|ico|pdf|zip|tar|gz|pyc|so|dylib)$ ]]; then
        continue
    fi
    [ ! -f "$file" ] && continue

    CONTENT=$(git show ":$file" 2>/dev/null) || continue

    # OpenAI
    if echo "$CONTENT" | grep -qE 'sk-proj-[A-Za-z0-9_-]{20,}'; then
        echo -e "${RED}❌ OpenAI key in: $file${NC}"; FOUND_SECRETS=1
    fi
    if echo "$CONTENT" | grep -qE 'sk-[A-Za-z0-9]{48,}'; then
        echo -e "${RED}❌ OpenAI key in: $file${NC}"; FOUND_SECRETS=1
    fi

    # Anthropic
    if echo "$CONTENT" | grep -qE 'sk-ant-[A-Za-z0-9_-]{20,}'; then
        echo -e "${RED}❌ Anthropic key in: $file${NC}"; FOUND_SECRETS=1
    fi

    # Google
    if echo "$CONTENT" | grep -qE 'AIzaSy[A-Za-z0-9_-]{33}'; then
        echo -e "${RED}❌ Google key in: $file${NC}"; FOUND_SECRETS=1
    fi

    # AWS
    if echo "$CONTENT" | grep -qE 'AKIA[0-9A-Z]{16}'; then
        echo -e "${RED}❌ AWS key in: $file${NC}"; FOUND_SECRETS=1
    fi

    # Groq
    if echo "$CONTENT" | grep -qE 'gsk_[A-Za-z0-9]{40,}'; then
        echo -e "${RED}❌ Groq key in: $file${NC}"; FOUND_SECRETS=1
    fi

    # Perplexity
    if echo "$CONTENT" | grep -qE 'pplx-[A-Za-z0-9]{40,}'; then
        echo -e "${RED}❌ Perplexity key in: $file${NC}"; FOUND_SECRETS=1
    fi

    # Private keys
    if echo "$CONTENT" | grep -q 'BEGIN.*PRIVATE KEY'; then
        echo -e "${RED}❌ Private key in: $file${NC}"; FOUND_SECRETS=1
    fi
done

if [ $FOUND_SECRETS -eq 1 ]; then
    echo -e "${RED}COMMIT BLOCKED: Secrets detected!${NC}"
    echo "Use 'git commit --no-verify' to bypass (not recommended)"
    exit 1
fi

echo -e "${GREEN}✓ No secrets detected${NC}"
exit 0
HOOK_CONTENT

# Make hook executable
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will now:"
echo "  • Block commits containing .env files"
echo "  • Detect API keys and secrets in code"
echo "  • Prevent accidental credential leaks"
echo ""
echo "To bypass in emergencies: git commit --no-verify"
