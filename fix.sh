#!/bin/bash

echo "🔧 LSB Steganography - Quick Fix Script"
echo "======================================="
echo ""

# Function to print colored output
print_status() {
    echo -e "\033[1;34m➜\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m✓\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m✗\033[0m $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found!"
    print_status "Copying config files from Config/ directory..."
    
    if [ -f "Config/package.json" ]; then
        cp Config/package.json .
        cp Config/vite.config.js .
        cp Config/tailwind.config.js . 2>/dev/null || true
        cp Config/postcss.config.js . 2>/dev/null || true
        print_success "Config files copied!"
    else
        print_error "Config directory not found. Please check your project structure."
        exit 1
    fi
fi

# Fix 1: Ensure correct file structure
print_status "Checking project structure..."
if [ -d "Frontend" ] && [ -d "Backend" ]; then
    print_success "Project structure is correct"
else
    print_error "Missing Frontend or Backend directory!"
    exit 1
fi

# Fix 2: Install Node.js dependencies
print_status "Checking Node.js dependencies..."
if [ ! -d "node_modules" ]; then
    print_status "Installing npm packages..."
    npm install
    if [ $? -eq 0 ]; then
        print_success "Node.js dependencies installed"
    else
        print_error "Failed to install Node.js dependencies"
        print_status "Try running: npm install --legacy-peer-deps"
    fi
else
    print_success "Node.js dependencies already installed"
fi

# Fix 3: Check Python dependencies
print_status "Checking Python dependencies..."
python3 -c "import flask, flask_cors, PIL, numpy, cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    print_status "Installing Python packages..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Python dependencies installed"
    else
        print_error "Failed to install Python dependencies"
        print_status "Try running: pip install -r requirements.txt --user"
    fi
else
    print_success "Python dependencies already installed"
fi

# Fix 4: Kill processes on ports 3000 and 5000
print_status "Checking for processes on ports 3000 and 5000..."
lsof -ti:3000 >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status "Killing process on port 3000..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    print_success "Port 3000 is now free"
fi

lsof -ti:5000 >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status "Killing process on port 5000..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    print_success "Port 5000 is now free"
fi

# Fix 5: Ensure Backend/__pycache__ is clean
print_status "Cleaning Python cache..."
find Backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find Backend -type f -name "*.pyc" -delete 2>/dev/null
print_success "Python cache cleaned"

# Fix 6: Verify Vite config
print_status "Verifying Vite configuration..."
if grep -q "root: 'Frontend'" vite.config.js; then
    print_success "Vite config is correct"
else
    print_error "Vite config may have issues. Check vite.config.js"
fi

echo ""
echo "======================================="
print_success "All fixes applied!"
echo ""
print_status "You can now run the application with:"
echo "  ./start.sh"
echo "  OR"
echo "  npm run start:dev"
echo ""
print_status "For manual start:"
echo "  Terminal 1: cd Backend && python3 app.py"
echo "  Terminal 2: npm run dev"
echo ""
