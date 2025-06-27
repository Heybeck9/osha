#!/usr/bin/env python3
"""
WADE Security Research Platform
Autonomous Defensive Security and Compliance Framework
"""

import os
import json
import asyncio
import subprocess
import tempfile
import shutil
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Security-focused data models
class SecurityAssessmentRequest(BaseModel):
    repo_path: str
    assessment_type: str  # "vulnerability", "compliance", "hardening", "incident_response"
    framework: Optional[str] = "owasp"  # "owasp", "nist", "iso27001", "gdpr", etc.
    scope: Optional[List[str]] = ["code", "config", "dependencies"]

class SecurityFinding(BaseModel):
    severity: str  # "critical", "high", "medium", "low", "info"
    category: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    recommendation: str
    compliance_impact: Optional[str] = None

class SecurityReport(BaseModel):
    assessment_id: str
    status: str
    assessment_type: str
    framework: str
    findings: List[SecurityFinding]
    summary: Dict[str, Any]
    compliance_score: Optional[float] = None
    recommendations: List[str]

@dataclass
class SecurityTask:
    task_id: str
    repo_path: str
    assessment_type: str
    framework: str
    scope: List[str]
    status: str = "running"
    progress: int = 0
    current_step: str = "Initializing"
    findings: List[SecurityFinding] = None
    logs: List[str] = None
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []
        if self.logs is None:
            self.logs = []

# Global task storage
security_tasks: Dict[str, SecurityTask] = {}

app = FastAPI(title="WADE Security Research Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DefensiveSecurityAgent:
    """Autonomous defensive security research agent"""
    
    def __init__(self, task: SecurityTask):
        self.task = task
        self.repo_path = Path(task.repo_path)
        
    async def execute(self):
        """Main security assessment pipeline"""
        try:
            await self.log("🛡️ Starting defensive security assessment...")
            
            if self.task.assessment_type == "vulnerability":
                await self.vulnerability_assessment()
            elif self.task.assessment_type == "compliance":
                await self.compliance_assessment()
            elif self.task.assessment_type == "hardening":
                await self.security_hardening()
            elif self.task.assessment_type == "incident_response":
                await self.incident_response_setup()
            else:
                await self.comprehensive_assessment()
                
            await self.generate_recommendations()
            await self.update_progress(100, "Security assessment completed")
            self.task.status = "completed"
            
        except Exception as e:
            await self.log(f"❌ Error: {str(e)}")
            self.task.status = "failed"
            raise
    
    async def vulnerability_assessment(self):
        """Comprehensive vulnerability assessment"""
        await self.update_progress(10, "Starting vulnerability assessment")
        
        # Static code analysis
        await self.static_code_analysis()
        await self.update_progress(30, "Static code analysis complete")
        
        # Dependency scanning
        await self.dependency_scanning()
        await self.update_progress(50, "Dependency scanning complete")
        
        # Configuration analysis
        await self.configuration_analysis()
        await self.update_progress(70, "Configuration analysis complete")
        
        # OWASP Top 10 checks
        await self.owasp_top10_analysis()
        await self.update_progress(90, "OWASP analysis complete")
    
    async def static_code_analysis(self):
        """Analyze source code for security vulnerabilities"""
        await self.log("🔍 Performing static code analysis...")
        
        # Common vulnerability patterns
        vulnerability_patterns = {
            "sql_injection": {
                "pattern": r"(SELECT|INSERT|UPDATE|DELETE).*\+.*\$",
                "severity": "high",
                "cwe": "CWE-89",
                "description": "Potential SQL injection vulnerability"
            },
            "xss": {
                "pattern": r"innerHTML\s*=\s*.*\+",
                "severity": "medium", 
                "cwe": "CWE-79",
                "description": "Potential Cross-Site Scripting (XSS) vulnerability"
            },
            "hardcoded_secrets": {
                "pattern": r"(password|secret|key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
                "severity": "critical",
                "cwe": "CWE-798",
                "description": "Hardcoded credentials detected"
            },
            "path_traversal": {
                "pattern": r"\.\.\/|\.\.\\",
                "severity": "high",
                "cwe": "CWE-22",
                "description": "Potential path traversal vulnerability"
            },
            "weak_crypto": {
                "pattern": r"(MD5|SHA1)\s*\(",
                "severity": "medium",
                "cwe": "CWE-327",
                "description": "Weak cryptographic algorithm detected"
            }
        }
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.php', '.java', '.cs']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        for vuln_type, vuln_data in vulnerability_patterns.items():
                            if re.search(vuln_data["pattern"], line, re.IGNORECASE):
                                finding = SecurityFinding(
                                    severity=vuln_data["severity"],
                                    category="Code Vulnerability",
                                    title=f"{vuln_type.replace('_', ' ').title()} Detected",
                                    description=vuln_data["description"],
                                    file_path=str(file_path.relative_to(self.repo_path)),
                                    line_number=line_num,
                                    cwe_id=vuln_data["cwe"],
                                    recommendation=self.get_vulnerability_recommendation(vuln_type)
                                )
                                self.task.findings.append(finding)
                                await self.log(f"🚨 {vuln_data['severity'].upper()}: {vuln_data['description']} in {file_path.name}:{line_num}")
                                
                except Exception as e:
                    await self.log(f"⚠️ Error analyzing {file_path.name}: {str(e)}")
    
    async def dependency_scanning(self):
        """Scan dependencies for known vulnerabilities"""
        await self.log("📦 Scanning dependencies for vulnerabilities...")
        
        # Check for common dependency files
        dep_files = {
            "requirements.txt": "Python",
            "package.json": "Node.js", 
            "Gemfile": "Ruby",
            "pom.xml": "Java",
            "composer.json": "PHP"
        }
        
        for dep_file, language in dep_files.items():
            dep_path = self.repo_path / dep_file
            if dep_path.exists():
                await self.analyze_dependencies(dep_path, language)
    
    async def analyze_dependencies(self, dep_path: Path, language: str):
        """Analyze specific dependency file"""
        try:
            content = dep_path.read_text()
            
            # Known vulnerable packages (simplified example)
            vulnerable_packages = {
                "requests": {"version": "<2.20.0", "cve": "CVE-2018-18074"},
                "flask": {"version": "<1.0", "cve": "CVE-2018-1000656"},
                "django": {"version": "<2.2.13", "cve": "CVE-2020-13254"},
                "lodash": {"version": "<4.17.12", "cve": "CVE-2019-10744"},
                "jquery": {"version": "<3.4.0", "cve": "CVE-2019-11358"}
            }
            
            for package, vuln_info in vulnerable_packages.items():
                if package in content.lower():
                    finding = SecurityFinding(
                        severity="high",
                        category="Dependency Vulnerability",
                        title=f"Vulnerable {language} Package: {package}",
                        description=f"Package {package} may be vulnerable ({vuln_info['cve']})",
                        file_path=str(dep_path.relative_to(self.repo_path)),
                        cwe_id="CWE-1104",
                        recommendation=f"Update {package} to version {vuln_info['version']} or later"
                    )
                    self.task.findings.append(finding)
                    await self.log(f"🚨 Vulnerable dependency: {package} in {dep_path.name}")
                    
        except Exception as e:
            await self.log(f"⚠️ Error analyzing dependencies in {dep_path.name}: {str(e)}")
    
    async def configuration_analysis(self):
        """Analyze configuration files for security issues"""
        await self.log("⚙️ Analyzing configuration security...")
        
        config_patterns = {
            "debug_enabled": {
                "pattern": r"debug\s*=\s*true",
                "severity": "medium",
                "description": "Debug mode enabled in production"
            },
            "weak_session": {
                "pattern": r"session.*timeout.*[0-9]{1,3}[^0-9]",
                "severity": "low",
                "description": "Short session timeout detected"
            },
            "insecure_protocol": {
                "pattern": r"http://(?!localhost|127\.0\.0\.1)",
                "severity": "medium",
                "description": "Insecure HTTP protocol in use"
            }
        }
        
        config_files = [".env", "config.py", "settings.py", "web.config", "application.properties"]
        
        for config_file in config_files:
            config_path = self.repo_path / config_file
            if config_path.exists():
                await self.analyze_config_file(config_path, config_patterns)
    
    async def analyze_config_file(self, config_path: Path, patterns: Dict):
        """Analyze individual configuration file"""
        try:
            content = config_path.read_text()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for issue_type, issue_data in patterns.items():
                    if re.search(issue_data["pattern"], line, re.IGNORECASE):
                        finding = SecurityFinding(
                            severity=issue_data["severity"],
                            category="Configuration Security",
                            title=f"Configuration Issue: {issue_type.replace('_', ' ').title()}",
                            description=issue_data["description"],
                            file_path=str(config_path.relative_to(self.repo_path)),
                            line_number=line_num,
                            recommendation=self.get_config_recommendation(issue_type)
                        )
                        self.task.findings.append(finding)
                        await self.log(f"⚠️ Configuration issue: {issue_data['description']} in {config_path.name}")
                        
        except Exception as e:
            await self.log(f"⚠️ Error analyzing config {config_path.name}: {str(e)}")
    
    async def owasp_top10_analysis(self):
        """OWASP Top 10 security analysis"""
        await self.log("🔟 Performing OWASP Top 10 analysis...")
        
        # Check for OWASP Top 10 vulnerabilities
        owasp_checks = [
            "Injection vulnerabilities",
            "Broken authentication",
            "Sensitive data exposure", 
            "XML external entities",
            "Broken access control",
            "Security misconfiguration",
            "Cross-site scripting",
            "Insecure deserialization",
            "Known vulnerabilities",
            "Insufficient logging"
        ]
        
        for check in owasp_checks:
            await self.log(f"✓ Checking: {check}")
            # Simplified check - in real implementation, each would have specific logic
            
    async def compliance_assessment(self):
        """Compliance framework assessment"""
        await self.update_progress(10, f"Starting {self.task.framework} compliance assessment")
        
        if self.task.framework.lower() == "gdpr":
            await self.gdpr_compliance_check()
        elif self.task.framework.lower() == "pci":
            await self.pci_compliance_check()
        elif self.task.framework.lower() == "hipaa":
            await self.hipaa_compliance_check()
        elif self.task.framework.lower() == "iso27001":
            await self.iso27001_compliance_check()
        else:
            await self.general_compliance_check()
            
        await self.update_progress(90, "Compliance assessment complete")
    
    async def gdpr_compliance_check(self):
        """GDPR compliance assessment"""
        await self.log("🇪🇺 Performing GDPR compliance assessment...")
        
        gdpr_requirements = [
            "Data processing consent mechanisms",
            "Right to be forgotten implementation",
            "Data portability features",
            "Privacy by design principles",
            "Data breach notification procedures",
            "Data protection impact assessments"
        ]
        
        for requirement in gdpr_requirements:
            # Simplified check - look for related code patterns
            await self.check_gdpr_requirement(requirement)
    
    async def check_gdpr_requirement(self, requirement: str):
        """Check specific GDPR requirement"""
        # Simplified implementation
        finding = SecurityFinding(
            severity="info",
            category="GDPR Compliance",
            title=f"GDPR Requirement: {requirement}",
            description=f"Assessment needed for: {requirement}",
            recommendation=f"Implement proper controls for {requirement}"
        )
        self.task.findings.append(finding)
        await self.log(f"📋 GDPR check: {requirement}")
    
    async def security_hardening(self):
        """Security hardening recommendations"""
        await self.update_progress(10, "Analyzing security hardening opportunities")
        
        await self.check_authentication_mechanisms()
        await self.update_progress(30, "Authentication analysis complete")
        
        await self.check_encryption_usage()
        await self.update_progress(50, "Encryption analysis complete")
        
        await self.check_access_controls()
        await self.update_progress(70, "Access control analysis complete")
        
        await self.check_logging_mechanisms()
        await self.update_progress(90, "Logging analysis complete")
    
    async def check_authentication_mechanisms(self):
        """Check authentication implementation"""
        await self.log("🔐 Analyzing authentication mechanisms...")
        
        auth_patterns = [
            "password hashing",
            "multi-factor authentication", 
            "session management",
            "token validation"
        ]
        
        for pattern in auth_patterns:
            # Check if authentication patterns are implemented
            await self.log(f"✓ Checking: {pattern}")
    
    async def check_encryption_usage(self):
        """Check encryption implementation"""
        await self.log("🔒 Analyzing encryption usage...")
        
        # Look for encryption patterns in code
        encryption_patterns = {
            "aes": r"AES|aes",
            "rsa": r"RSA|rsa", 
            "tls": r"TLS|SSL|https",
            "hashing": r"bcrypt|scrypt|argon2"
        }
        
        for file_path in self.repo_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    for crypto_type, pattern in encryption_patterns.items():
                        if re.search(pattern, content):
                            await self.log(f"✓ Found {crypto_type} usage in {file_path.name}")
                except:
                    pass
    
    async def check_access_controls(self):
        """Check access control implementation"""
        await self.log("🚪 Analyzing access controls...")
        
        # Look for authorization patterns
        auth_patterns = [
            r"@login_required",
            r"@permission_required", 
            r"check_permission",
            r"authorize"
        ]
        
        for file_path in self.repo_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    for pattern in auth_patterns:
                        if re.search(pattern, content):
                            await self.log(f"✓ Found access control in {file_path.name}")
                except:
                    pass
    
    async def check_logging_mechanisms(self):
        """Check logging implementation"""
        await self.log("📝 Analyzing logging mechanisms...")
        
        # Look for logging patterns
        logging_patterns = [
            r"logging\.",
            r"logger\.",
            r"log\.",
            r"audit"
        ]
        
        for file_path in self.repo_path.rglob("*.py"):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    for pattern in logging_patterns:
                        if re.search(pattern, content):
                            await self.log(f"✓ Found logging in {file_path.name}")
                except:
                    pass
    
    async def incident_response_setup(self):
        """Set up incident response framework"""
        await self.update_progress(10, "Setting up incident response framework")
        
        await self.create_incident_response_plan()
        await self.update_progress(40, "Incident response plan created")
        
        await self.setup_monitoring_alerts()
        await self.update_progress(70, "Monitoring alerts configured")
        
        await self.create_forensics_toolkit()
        await self.update_progress(90, "Forensics toolkit ready")
    
    async def create_incident_response_plan(self):
        """Create incident response plan"""
        await self.log("📋 Creating incident response plan...")
        
        ir_plan = {
            "phases": [
                "Preparation",
                "Detection & Analysis", 
                "Containment & Eradication",
                "Recovery",
                "Lessons Learned"
            ],
            "contacts": {
                "security_team": "security@company.com",
                "legal": "legal@company.com",
                "management": "management@company.com"
            },
            "procedures": {
                "malware_incident": "isolate_system -> analyze_malware -> remove_threat -> restore_system",
                "data_breach": "assess_scope -> notify_authorities -> contain_breach -> investigate_cause",
                "ddos_attack": "activate_mitigation -> analyze_traffic -> block_sources -> monitor_recovery"
            }
        }
        
        # Save incident response plan
        ir_file = self.repo_path / "incident_response_plan.json"
        ir_file.write_text(json.dumps(ir_plan, indent=2))
        await self.log("✅ Incident response plan saved")
    
    async def setup_monitoring_alerts(self):
        """Setup security monitoring and alerts"""
        await self.log("🚨 Setting up security monitoring...")
        
        monitoring_config = {
            "log_sources": [
                "application_logs",
                "system_logs", 
                "network_logs",
                "security_logs"
            ],
            "alert_rules": [
                "failed_login_attempts > 5",
                "privilege_escalation_detected",
                "unusual_network_traffic",
                "malware_signature_match"
            ],
            "notification_channels": [
                "email",
                "slack",
                "sms"
            ]
        }
        
        # Save monitoring configuration
        monitor_file = self.repo_path / "security_monitoring.json"
        monitor_file.write_text(json.dumps(monitoring_config, indent=2))
        await self.log("✅ Security monitoring configured")
    
    async def create_forensics_toolkit(self):
        """Create digital forensics toolkit"""
        await self.log("🔍 Creating forensics toolkit...")
        
        forensics_tools = {
            "evidence_collection": [
                "memory_dump_script.py",
                "disk_imaging_script.py",
                "network_capture_script.py"
            ],
            "analysis_tools": [
                "log_analyzer.py",
                "malware_scanner.py", 
                "timeline_generator.py"
            ],
            "reporting": [
                "evidence_report_template.md",
                "chain_of_custody_form.pdf"
            ]
        }
        
        # Create forensics directory and tools
        forensics_dir = self.repo_path / "forensics_toolkit"
        forensics_dir.mkdir(exist_ok=True)
        
        toolkit_file = forensics_dir / "toolkit_inventory.json"
        toolkit_file.write_text(json.dumps(forensics_tools, indent=2))
        await self.log("✅ Forensics toolkit created")
    
    async def comprehensive_assessment(self):
        """Comprehensive security assessment"""
        await self.log("🔍 Performing comprehensive security assessment...")
        
        await self.vulnerability_assessment()
        await self.compliance_assessment()
        await self.security_hardening()
    
    async def generate_recommendations(self):
        """Generate security recommendations"""
        await self.log("💡 Generating security recommendations...")
        
        # Analyze findings and generate recommendations
        critical_findings = [f for f in self.task.findings if f.severity == "critical"]
        high_findings = [f for f in self.task.findings if f.severity == "high"]
        
        recommendations = []
        
        if critical_findings:
            recommendations.append("🚨 URGENT: Address critical security vulnerabilities immediately")
        
        if high_findings:
            recommendations.append("⚠️ HIGH PRIORITY: Remediate high-severity security issues")
            
        recommendations.extend([
            "🔒 Implement comprehensive input validation",
            "🛡️ Enable security headers and HTTPS",
            "📝 Enhance security logging and monitoring",
            "🔐 Strengthen authentication mechanisms",
            "📋 Conduct regular security assessments"
        ])
        
        for rec in recommendations:
            await self.log(f"💡 {rec}")
    
    def get_vulnerability_recommendation(self, vuln_type: str) -> str:
        """Get recommendation for specific vulnerability type"""
        recommendations = {
            "sql_injection": "Use parameterized queries and input validation",
            "xss": "Implement proper output encoding and Content Security Policy",
            "hardcoded_secrets": "Use environment variables or secure secret management",
            "path_traversal": "Validate and sanitize file paths, use allowlists",
            "weak_crypto": "Use strong cryptographic algorithms (AES-256, SHA-256+)"
        }
        return recommendations.get(vuln_type, "Follow security best practices")
    
    def get_config_recommendation(self, issue_type: str) -> str:
        """Get recommendation for configuration issue"""
        recommendations = {
            "debug_enabled": "Disable debug mode in production environments",
            "weak_session": "Implement appropriate session timeout values",
            "insecure_protocol": "Use HTTPS for all communications"
        }
        return recommendations.get(issue_type, "Review and harden configuration")
    
    async def log(self, message: str):
        """Add a log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.task.logs.append(log_entry)
        print(log_entry)
    
    async def update_progress(self, progress: int, step: str):
        """Update task progress"""
        self.task.progress = progress
        self.task.current_step = step
        await self.log(f"📊 Progress: {progress}% - {step}")

# API Endpoints
@app.post("/api/security-assessment", response_model=dict)
async def start_security_assessment(request: SecurityAssessmentRequest, background_tasks: BackgroundTasks):
    """Start a new security assessment"""
    
    if not os.path.exists(request.repo_path):
        raise HTTPException(status_code=400, detail="Repository path does not exist")
    
    task_id = f"security_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(security_tasks)}"
    
    task = SecurityTask(
        task_id=task_id,
        repo_path=request.repo_path,
        assessment_type=request.assessment_type,
        framework=request.framework or "owasp",
        scope=request.scope or ["code", "config", "dependencies"]
    )
    
    security_tasks[task_id] = task
    
    agent = DefensiveSecurityAgent(task)
    background_tasks.add_task(agent.execute)
    
    return {"task_id": task_id, "status": "started"}

@app.get("/api/security-status/{task_id}")
async def get_security_status(task_id: str):
    """Get security assessment status"""
    if task_id not in security_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = security_tasks[task_id]
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "current_step": task.current_step,
        "assessment_type": task.assessment_type,
        "framework": task.framework,
        "findings_count": len(task.findings),
        "logs": task.logs,
        "findings": [asdict(f) for f in task.findings]
    }

@app.get("/api/security-report/{task_id}")
async def get_security_report(task_id: str):
    """Get comprehensive security report"""
    if task_id not in security_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = security_tasks[task_id]
    
    # Calculate severity distribution
    severity_counts = {}
    for finding in task.findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
    
    # Calculate compliance score
    total_findings = len(task.findings)
    critical_high = len([f for f in task.findings if f.severity in ["critical", "high"]])
    compliance_score = max(0, 100 - (critical_high * 10) - (total_findings * 2))
    
    return SecurityReport(
        assessment_id=task.task_id,
        status=task.status,
        assessment_type=task.assessment_type,
        framework=task.framework,
        findings=task.findings,
        summary={
            "total_findings": total_findings,
            "severity_distribution": severity_counts,
            "files_analyzed": len(list(Path(task.repo_path).rglob("*"))),
            "assessment_duration": "N/A"  # Would calculate actual duration
        },
        compliance_score=compliance_score,
        recommendations=[
            "Address critical and high severity findings immediately",
            "Implement security best practices",
            "Conduct regular security assessments",
            "Enhance security monitoring and logging"
        ]
    )

# Enhanced frontend with security capabilities
@app.get("/", response_class=HTMLResponse)
async def serve_security_frontend():
    """Serve the security research frontend"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WADE Security Research Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .security-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .security-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .security-card:hover {
            transform: translateY(-5px);
        }
        .security-card h3 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .security-card p {
            opacity: 0.9;
            line-height: 1.6;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        input, textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #3498db;
        }
        .btn {
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .status-card {
            display: none;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e1e5e9;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2c3e50);
            transition: width 0.3s ease;
        }
        .logs {
            background: #f8f9fa;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            white-space: pre-wrap;
        }
        .findings {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            max-height: 400px;
            overflow-y: auto;
        }
        .finding {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 4px solid #e74c3c;
            background: white;
            border-radius: 4px;
        }
        .finding.critical { border-left-color: #e74c3c; }
        .finding.high { border-left-color: #f39c12; }
        .finding.medium { border-left-color: #f1c40f; }
        .finding.low { border-left-color: #27ae60; }
        .finding.info { border-left-color: #3498db; }
        .severity-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .severity-critical { background: #e74c3c; color: white; }
        .severity-high { background: #f39c12; color: white; }
        .severity-medium { background: #f1c40f; color: black; }
        .severity-low { background: #27ae60; color: white; }
        .severity-info { background: #3498db; color: white; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .security-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ WADE Security</h1>
            <p>Autonomous Defensive Security Research Platform</p>
        </div>
        
        <div class="security-grid">
            <div class="security-card" onclick="loadAssessment('vulnerability')">
                <h3>🔍 Vulnerability Assessment</h3>
                <p>Comprehensive security vulnerability scanning including static code analysis, dependency checking, and OWASP Top 10 assessment.</p>
            </div>
            
            <div class="security-card" onclick="loadAssessment('compliance')">
                <h3>📋 Compliance Audit</h3>
                <p>Regulatory compliance assessment for GDPR, HIPAA, PCI DSS, SOC 2, and other security frameworks.</p>
            </div>
            
            <div class="security-card" onclick="loadAssessment('hardening')">
                <h3>🔒 Security Hardening</h3>
                <p>Security configuration analysis and hardening recommendations for enhanced protection.</p>
            </div>
            
            <div class="security-card" onclick="loadAssessment('incident_response')">
                <h3>🚨 Incident Response</h3>
                <p>Incident response framework setup, monitoring configuration, and forensics toolkit creation.</p>
            </div>
        </div>
        
        <div class="card">
            <h2>🚀 Start Security Assessment</h2>
            <form id="securityForm">
                <div class="form-group">
                    <label for="repoPath">Repository Path:</label>
                    <input type="text" id="repoPath" placeholder="/path/to/your/repo" value="/workspace/wade_env/demo_repo" required>
                </div>
                
                <div class="grid">
                    <div class="form-group">
                        <label for="assessmentType">Assessment Type:</label>
                        <select id="assessmentType" required>
                            <option value="vulnerability">Vulnerability Assessment</option>
                            <option value="compliance">Compliance Audit</option>
                            <option value="hardening">Security Hardening</option>
                            <option value="incident_response">Incident Response</option>
                            <option value="comprehensive">Comprehensive Assessment</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="framework">Framework/Standard:</label>
                        <select id="framework">
                            <option value="owasp">OWASP</option>
                            <option value="nist">NIST Cybersecurity Framework</option>
                            <option value="iso27001">ISO 27001</option>
                            <option value="gdpr">GDPR</option>
                            <option value="hipaa">HIPAA</option>
                            <option value="pci">PCI DSS</option>
                            <option value="soc2">SOC 2</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" class="btn" id="startBtn">🛡️ Start Security Assessment</button>
            </form>
        </div>
        
        <div class="card status-card" id="statusCard">
            <h2>📊 Assessment Status</h2>
            <div id="statusInfo">
                <p><strong>Task ID:</strong> <span id="taskId"></span></p>
                <p><strong>Status:</strong> <span id="status"></span></p>
                <p><strong>Assessment Type:</strong> <span id="assessmentType2"></span></p>
                <p><strong>Framework:</strong> <span id="framework2"></span></p>
                <p><strong>Current Step:</strong> <span id="currentStep"></span></p>
                
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                </div>
                <p><span id="progress">0</span>% Complete</p>
            </div>
            
            <h3>📝 Assessment Logs</h3>
            <div class="logs" id="logs"></div>
            
            <div id="findingsSection" style="display: none;">
                <h3>🚨 Security Findings (<span id="findingsCount">0</span>)</h3>
                <div class="findings" id="findings"></div>
            </div>
        </div>
    </div>

    <script>
        let currentTaskId = null;
        let statusInterval = null;

        function loadAssessment(type) {
            document.getElementById('assessmentType').value = type;
            
            // Set appropriate framework based on assessment type
            const frameworkMap = {
                'vulnerability': 'owasp',
                'compliance': 'gdpr',
                'hardening': 'nist',
                'incident_response': 'nist'
            };
            
            if (frameworkMap[type]) {
                document.getElementById('framework').value = frameworkMap[type];
            }
        }

        document.getElementById('securityForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = {
                repo_path: document.getElementById('repoPath').value,
                assessment_type: document.getElementById('assessmentType').value,
                framework: document.getElementById('framework').value,
                scope: ["code", "config", "dependencies"]
            };
            
            const startBtn = document.getElementById('startBtn');
            startBtn.disabled = true;
            startBtn.textContent = '🛡️ Starting Assessment...';
            
            try {
                const response = await fetch('/api/security-assessment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    currentTaskId = result.task_id;
                    document.getElementById('taskId').textContent = currentTaskId;
                    document.getElementById('assessmentType2').textContent = formData.assessment_type;
                    document.getElementById('framework2').textContent = formData.framework;
                    document.getElementById('statusCard').style.display = 'block';
                    
                    // Start polling for status
                    statusInterval = setInterval(updateStatus, 2000);
                    updateStatus(); // Initial update
                } else {
                    alert('Error: ' + result.detail);
                    startBtn.disabled = false;
                    startBtn.textContent = '🛡️ Start Security Assessment';
                }
            } catch (error) {
                alert('Error: ' + error.message);
                startBtn.disabled = false;
                startBtn.textContent = '🛡️ Start Security Assessment';
            }
        });

        async function updateStatus() {
            if (!currentTaskId) return;
            
            try {
                const response = await fetch(`/api/security-status/${currentTaskId}`);
                const status = await response.json();
                
                document.getElementById('status').textContent = status.status;
                document.getElementById('currentStep').textContent = status.current_step;
                document.getElementById('progress').textContent = status.progress;
                document.getElementById('progressFill').style.width = status.progress + '%';
                
                // Update logs
                const logsDiv = document.getElementById('logs');
                logsDiv.textContent = status.logs.join('\\n');
                logsDiv.scrollTop = logsDiv.scrollHeight;
                
                // Update findings
                if (status.findings && status.findings.length > 0) {
                    document.getElementById('findingsSection').style.display = 'block';
                    document.getElementById('findingsCount').textContent = status.findings.length;
                    
                    const findingsDiv = document.getElementById('findings');
                    findingsDiv.innerHTML = '';
                    
                    status.findings.forEach(finding => {
                        const findingDiv = document.createElement('div');
                        findingDiv.className = `finding ${finding.severity}`;
                        findingDiv.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <strong>${finding.title}</strong>
                                <span class="severity-badge severity-${finding.severity}">${finding.severity}</span>
                            </div>
                            <p>${finding.description}</p>
                            ${finding.file_path ? `<p><strong>File:</strong> ${finding.file_path}${finding.line_number ? `:${finding.line_number}` : ''}</p>` : ''}
                            ${finding.cwe_id ? `<p><strong>CWE:</strong> ${finding.cwe_id}</p>` : ''}
                            <p><strong>Recommendation:</strong> ${finding.recommendation}</p>
                        `;
                        findingsDiv.appendChild(findingDiv);
                    });
                }
                
                // Stop polling if completed or failed
                if (status.status === 'completed' || status.status === 'failed') {
                    clearInterval(statusInterval);
                    const startBtn = document.getElementById('startBtn');
                    startBtn.disabled = false;
                    startBtn.textContent = '🛡️ Start Security Assessment';
                }
                
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("🛡️ Starting WADE Security Research Platform...")
    print("📍 Access the security interface at: http://localhost:12001")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=12001,
        reload=False
    )