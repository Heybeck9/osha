# 🤖 WADE Autonomous Development Environment - Setup Manual

## 📋 **Complete Installation Guide**

This manual provides step-by-step instructions to set up WADE (Autonomous Development Environment) on your local system with full functionality including chat-driven development, embedded VS Code, terminal integration, and execution mode switching.

---

## 🖥️ **System Requirements**

### **Minimum Requirements:**
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, Arch Linux, Kali Linux)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4 cores (8 cores recommended)
- **Network**: Internet connection for initial setup

### **Recommended Requirements:**
- **OS**: Ubuntu 22.04 LTS or Arch Linux
- **RAM**: 16GB+ 
- **Storage**: 20GB+ SSD
- **CPU**: 8+ cores
- **GPU**: Optional (for faster LLM inference)

---

## 🔧 **Prerequisites Installation**

### **Step 1: Update System**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# Arch Linux
sudo pacman -Syu

# Install essential tools
sudo apt install -y curl wget git build-essential python3 python3-pip python3-venv nodejs npm
```

### **Step 2: Install Docker (Required for VS Code Server)**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Logout and login again for group changes to take effect
```

### **Step 3: Install Ollama (Local LLM Server)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify installation
ollama --version
```

---

## 🚀 **WADE Installation**

### **Step 1: Clone WADE Repository**
```bash
# Create WADE directory
mkdir -p ~/wade_workspace
cd ~/wade_workspace

# Clone or download WADE files
git clone https://github.com/your-repo/wade.git .
# OR manually create the files from the provided code
```

### **Step 2: Create WADE Directory Structure**
```bash
# Create the complete directory structure
mkdir -p ~/wade_workspace/{wade_env,payloads/{sandboxed,live,archive},memory/{evolution,chains},logs,tools}

# Set permissions
chmod +x ~/wade_workspace/executioner_native
```

### **Step 3: Install Python Dependencies**
```bash
cd ~/wade_workspace/wade_env

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install fastapi uvicorn websockets python-multipart jinja2 aiofiles
pip install requests beautifulsoup4 lxml selenium
pip install networkx numpy pandas
pip install python-dotenv pyyaml
pip install psutil

# Install development tools
pip install pytest black flake8 mypy
```

### **Step 4: Install VS Code Server**
```bash
# Install code-server (VS Code in browser)
curl -fsSL https://code-server.dev/install.sh | sh

# Create VS Code config directory
mkdir -p ~/.config/code-server

# Create VS Code config
cat > ~/.config/code-server/config.yaml << EOF
bind-addr: 0.0.0.0:12001
auth: none
password: 
cert: false
EOF
```

### **Step 5: Download and Configure LLM Models**
```bash
# Download Phind CodeLlama (recommended)
ollama pull phind-codellama:latest

# Download backup models
ollama pull deepseek-coder:6.7b
ollama pull wizardlm:7b

# Verify models are installed
ollama list
```

---

## ⚙️ **Configuration**

### **Step 1: Create WADE Configuration File**
```bash
cat > ~/wade_workspace/wade_config.yaml << EOF
# WADE Configuration
server:
  host: "0.0.0.0"
  port: 12000
  debug: false

models:
  primary: "phind-codellama:latest"
  fallback: "deepseek-coder:6.7b"
  creative: "wizardlm:7b"

execution:
  default_mode: "simulation"
  workspace_path: "~/wade_workspace/wade_env"
  max_concurrent_tasks: 5

security:
  enable_live_mode: true
  require_confirmation: true
  log_all_operations: true

tools:
  vscode_port: 12001
  terminal_enabled: true
  browser_preview: true
EOF
```

### **Step 2: Create Environment Variables**
```bash
# Create .env file
cat > ~/wade_workspace/.env << EOF
WADE_WORKSPACE=~/wade_workspace
WADE_ENV=~/wade_workspace/wade_env
OLLAMA_HOST=http://localhost:11434
VSCODE_PORT=12001
WADE_PORT=12000
PYTHONPATH=~/wade_workspace
EOF

# Source environment
echo "source ~/wade_workspace/.env" >> ~/.bashrc
source ~/.bashrc
```

### **Step 3: Create Systemd Services (Optional)**
```bash
# Create WADE service
sudo tee /etc/systemd/system/wade.service << EOF
[Unit]
Description=WADE Autonomous Development Environment
After=network.target ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/wade_workspace
Environment=PATH=$HOME/wade_workspace/wade_env/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$HOME/wade_workspace/wade_env/venv/bin/python $HOME/wade_workspace/wade_native_interface.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create VS Code service
sudo tee /etc/systemd/system/wade-vscode.service << EOF
[Unit]
Description=WADE VS Code Server
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=/usr/bin/code-server --bind-addr 0.0.0.0:12001 --auth none $HOME/wade_workspace/wade_env
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable services
sudo systemctl daemon-reload
sudo systemctl enable wade wade-vscode
```

---

## 🎯 **First Launch**

### **Step 1: Start Core Services**
```bash
# Start Ollama (if not already running)
sudo systemctl start ollama

# Verify Ollama is working
curl http://localhost:11434/api/tags

# Start VS Code Server
code-server --bind-addr 0.0.0.0:12001 --auth none ~/wade_workspace/wade_env &

# Verify VS Code is accessible
curl http://localhost:12001
```

### **Step 2: Launch WADE**
```bash
cd ~/wade_workspace/wade_env
source venv/bin/activate

# Start WADE Native Interface
python3 ../wade_native_interface.py
```

### **Step 3: Access WADE Interface**
Open your web browser and navigate to:
- **WADE Interface**: `http://localhost:12000`
- **VS Code Editor**: `http://localhost:12001`

---

## 🧪 **Testing Installation**

### **Test 1: Basic Functionality**
1. Open WADE interface at `http://localhost:12000`
2. Verify the chat panel loads
3. Check that VS Code iframe loads properly
4. Confirm terminal panel is responsive

### **Test 2: Execution Mode Toggle**
1. Click the execution mode toggle (🧪 SIMULATION ⇄ 🔥 LIVE)
2. Verify confirmation dialog appears for live mode
3. Check that mode indicator updates correctly

### **Test 3: Chat Functionality**
1. Type a simple request: "Create a hello world Python script"
2. Verify AI responds and creates a task
3. Check that task execution logs appear in terminal

### **Test 4: VS Code Integration**
1. Verify VS Code loads in the embedded iframe
2. Create a new file in VS Code
3. Confirm file appears in the workspace

---

## 🔧 **Advanced Configuration**

### **Custom Model Configuration**
```bash
# Add custom model to Ollama
ollama create custom-model -f Modelfile

# Update WADE config to use custom model
vim ~/wade_workspace/wade_config.yaml
```

### **Network Configuration**
```bash
# Configure firewall (if needed)
sudo ufw allow 12000
sudo ufw allow 12001
sudo ufw allow 11434

# For remote access, update bind addresses
# Edit wade_native_interface.py and change host to "0.0.0.0"
```

### **Performance Tuning**
```bash
# Increase Ollama memory limit
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_NUM_PARALLEL=4

# Optimize Python for performance
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Issue 1: WADE Interface Won't Load**
```bash
# Check if port is in use
sudo netstat -tlnp | grep 12000

# Kill conflicting processes
sudo fuser -k 12000/tcp

# Restart WADE
cd ~/wade_workspace/wade_env && source venv/bin/activate && python3 ../wade_native_interface.py
```

#### **Issue 2: VS Code Server Not Loading**
```bash
# Check VS Code server status
ps aux | grep code-server

# Restart VS Code server
pkill code-server
code-server --bind-addr 0.0.0.0:12001 --auth none ~/wade_workspace/wade_env &
```

#### **Issue 3: Ollama Connection Failed**
```bash
# Check Ollama service
sudo systemctl status ollama

# Restart Ollama
sudo systemctl restart ollama

# Test connection
curl http://localhost:11434/api/tags
```

#### **Issue 4: Models Not Responding**
```bash
# Check available models
ollama list

# Re-download model if corrupted
ollama rm phind-codellama:latest
ollama pull phind-codellama:latest

# Test model directly
ollama run phind-codellama:latest "Hello, how are you?"
```

### **Log Locations**
- **WADE Logs**: `~/wade_workspace/logs/wade.log`
- **VS Code Logs**: `~/.local/share/code-server/logs/`
- **Ollama Logs**: `sudo journalctl -u ollama`

---

## 🔒 **Security Considerations**

### **Production Deployment**
```bash
# Enable authentication for VS Code
cat > ~/.config/code-server/config.yaml << EOF
bind-addr: 0.0.0.0:12001
auth: password
password: your_secure_password_here
cert: false
EOF

# Use HTTPS (recommended)
# Generate SSL certificates and update config
```

### **Firewall Configuration**
```bash
# Restrict access to localhost only (recommended for local use)
sudo ufw deny 12000
sudo ufw deny 12001
sudo ufw allow from 127.0.0.1 to any port 12000
sudo ufw allow from 127.0.0.1 to any port 12001
```

---

## 📚 **Usage Examples**

### **Example 1: Create a Web Scraper**
1. Open WADE interface
2. Type: "Create a Python web scraper that extracts headlines from news websites"
3. Watch as WADE creates the scraper, tests it, and shows results

### **Example 2: Security Assessment**
1. Switch to SIMULATION mode
2. Type: "Perform a security assessment of a web application"
3. WADE will create a comprehensive security testing pipeline

### **Example 3: API Development**
1. Type: "Build a REST API with FastAPI for user management"
2. WADE will generate the API, create tests, and provide documentation

---

## 🆘 **Support and Updates**

### **Getting Help**
- Check logs in `~/wade_workspace/logs/`
- Review configuration in `~/wade_workspace/wade_config.yaml`
- Test individual components (Ollama, VS Code, Python environment)

### **Updates**
```bash
# Update WADE
cd ~/wade_workspace
git pull origin main

# Update dependencies
cd wade_env && source venv/bin/activate
pip install --upgrade -r requirements.txt

# Update models
ollama pull phind-codellama:latest
```

### **Backup Configuration**
```bash
# Create backup
tar -czf wade_backup_$(date +%Y%m%d).tar.gz ~/wade_workspace/

# Restore from backup
tar -xzf wade_backup_YYYYMMDD.tar.gz -C ~/
```

---

## ✅ **Installation Complete!**

You now have a fully functional WADE Autonomous Development Environment with:
- ✅ Chat-driven development interface
- ✅ Embedded VS Code editor
- ✅ Real terminal integration
- ✅ Execution mode switching (Simulation ⇄ Live)
- ✅ Local LLM integration (Phind CodeLlama)
- ✅ Task chain execution
- ✅ Self-evolution capabilities

**Access your WADE environment at**: `http://localhost:12000`

**Happy autonomous coding!** 🚀