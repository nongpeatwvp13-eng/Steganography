#!/bin/bash

echo "🧪 LSB Steganography - Local Testing Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}➜ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run test
run_test() {
    local test_name=$1
    local test_command=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    print_step "Testing: $test_name"
    
    if eval $test_command > /dev/null 2>&1; then
        print_success "$test_name - PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_error "$test_name - FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

echo "=== Phase 1: Environment Check ==="
echo ""

# Test 1: Check Node.js
print_step "Checking Node.js installation..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION is installed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "Node.js is not installed"
    print_warning "Please install Node.js 18+ from https://nodejs.org/"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 2: Check Python
print_step "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "$PYTHON_VERSION is installed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "Python3 is not installed"
    print_warning "Please install Python 3.8+ from https://python.org/"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 3: Check npm
print_step "Checking npm installation..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm $NPM_VERSION is installed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "npm is not installed"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "=== Phase 2: Project Structure Check ==="
echo ""

# Test 4: Check if we're in the right directory
run_test "package.json exists" "[ -f 'package.json' ]"
run_test "vite.config.js exists" "[ -f 'vite.config.js' ]"
run_test "Frontend directory exists" "[ -d 'Frontend' ]"
run_test "Backend directory exists" "[ -d 'Backend' ]"
run_test "Backend/app.py exists" "[ -f 'Backend/app.py' ]"

echo ""
echo "=== Phase 3: Dependencies Check ==="
echo ""

# Test 5: Check Node.js dependencies
print_step "Checking if node_modules exists..."
if [ -d "node_modules" ]; then
    print_success "node_modules directory exists"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_warning "node_modules not found. Installing dependencies..."
    npm install
    if [ $? -eq 0 ]; then
        print_success "npm packages installed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "Failed to install npm packages"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 6: Check Python dependencies
print_step "Checking Python dependencies..."
python3 -c "import flask" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Flask is installed"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_warning "Flask not found. Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Python packages installed successfully"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        print_error "Failed to install Python packages"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "=== Phase 4: Port Availability Check ==="
echo ""

# Test 7: Check if port 3000 is available
print_step "Checking if port 3000 is available..."
lsof -ti:3000 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_warning "Port 3000 is in use"
    print_step "Attempting to free port 3000..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    sleep 1
    lsof -ti:3000 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_error "Failed to free port 3000"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    else
        print_success "Port 3000 is now available"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    fi
else
    print_success "Port 3000 is available"
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 8: Check if port 5000 is available
print_step "Checking if port 5000 is available..."
lsof -ti:5000 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_warning "Port 5000 is in use"
    print_step "Attempting to free port 5000..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    sleep 1
    lsof -ti:5000 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_error "Failed to free port 5000"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    else
        print_success "Port 5000 is now available"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    fi
else
    print_success "Port 5000 is available"
    PASSED_TESTS=$((PASSED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "=== Phase 5: Configuration Files Check ==="
echo ""

# Test 9: Verify Vite config
print_step "Verifying Vite configuration..."
if grep -q "root: 'Frontend'" vite.config.js; then
    print_success "Vite config root is correct"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "Vite config root is incorrect"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 10: Check Frontend files
run_test "Frontend/index.html exists" "[ -f 'Frontend/index.html' ]"
run_test "Frontend/main.js exists" "[ -f 'Frontend/main.js' ]"
run_test "Frontend/api.js exists" "[ -f 'Frontend/api.js' ]"
run_test "Frontend/ui.js exists" "[ -f 'Frontend/ui.js' ]"

echo ""
echo "=== Phase 6: Backend Validation ==="
echo ""

# Test 11: Check Python syntax
print_step "Checking Python syntax in Backend/app.py..."
python3 -m py_compile Backend/app.py 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Backend/app.py syntax is valid"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "Backend/app.py has syntax errors"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 12: Check Backend imports
print_step "Checking Backend imports..."
python3 -c "
import sys
sys.path.insert(0, 'Backend')
try:
    from Functions.Stego import encode_message, decode_message
    from Analyze.image_analyzer import comprehensive_analysis
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "Backend imports are valid"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_error "Backend imports failed"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "=============================================="
echo "=== Test Summary ==="
echo "=============================================="
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

# Calculate percentage
if [ $TOTAL_TESTS -gt 0 ]; then
    PERCENTAGE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Success Rate: $PERCENTAGE%"
    echo ""
fi

# Final recommendation
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}=============================================="
    echo "✓ ALL TESTS PASSED!"
    echo "=============================================="
    echo ""
    echo "Your project is ready to run locally!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./start.sh"
    echo "  2. Open browser: http://localhost:3000"
    echo "  3. Test the application features"
    echo ""
    echo "After successful local testing:"
    echo "  - Commit and push to GitHub"
    echo "  - Create a Codespace"
    echo "  - Run ./start.sh in Codespace"
    echo -e "${NC}"
elif [ $FAILED_TESTS -le 3 ]; then
    echo -e "${YELLOW}=============================================="
    echo "⚠ MINOR ISSUES DETECTED"
    echo "=============================================="
    echo ""
    echo "Some tests failed, but you can still try running:"
    echo "  ./fix.sh"
    echo "Then run this test again:"
    echo "  ./test-local.sh"
    echo -e "${NC}"
else
    echo -e "${RED}=============================================="
    echo "✗ CRITICAL ISSUES DETECTED"
    echo "=============================================="
    echo ""
    echo "Please fix the issues above before running."
    echo "Try running:"
    echo "  ./fix.sh"
    echo ""
    echo "Or check the DEPLOYMENT_GUIDE.md for help."
    echo -e "${NC}"
fi

exit $FAILED_TESTS
