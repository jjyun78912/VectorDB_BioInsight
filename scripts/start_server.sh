#!/bin/bash
# BioInsight Server Startup Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting BioInsight Servers...${NC}"

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if FastAPI dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing FastAPI dependencies...${NC}"
    pip install fastapi uvicorn python-multipart
fi

# Start FastAPI backend
echo -e "${GREEN}Starting FastAPI backend on port 8000...${NC}"
cd "$PROJECT_ROOT"
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Check if frontend dependencies are installed
if [ ! -d "$PROJECT_ROOT/frontend/react_app/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd "$PROJECT_ROOT/frontend/react_app"
    npm install
fi

# Start React frontend
echo -e "${GREEN}Starting React frontend on port 3000...${NC}"
cd "$PROJECT_ROOT/frontend/react_app"
npm run dev &
FRONTEND_PID=$!

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}BioInsight is running!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "Backend API: ${YELLOW}http://localhost:8000${NC}"
echo -e "API Docs:    ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "Frontend:    ${YELLOW}http://localhost:3000${NC}"
echo ""
echo -e "Press Ctrl+C to stop all servers"

# Trap Ctrl+C and kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait for both processes
wait
