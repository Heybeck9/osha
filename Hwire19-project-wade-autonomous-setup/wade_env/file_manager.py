#!/usr/bin/env python3
"""
WADE File Manager - Complete file operations with upload/download
Handles workspace files, project management, and file operations
"""

import os
import shutil
import hashlib
import mimetypes
import zipfile
import tarfile
import json
import logging
from typing import Dict, List, Optional, Any, BinaryIO
from pathlib import Path
from datetime import datetime
import aiofiles
import aiofiles.os
from dataclasses import dataclass, asdict

@dataclass
class FileInfo:
    name: str
    path: str
    size: int
    type: str  # file, directory
    mime_type: Optional[str]
    created: str
    modified: str
    permissions: str
    hash_sha256: Optional[str] = None
    is_hidden: bool = False
    is_executable: bool = False

@dataclass
class UploadResult:
    success: bool
    file_path: str
    file_size: int
    message: str
    hash_sha256: Optional[str] = None

class FileManager:
    def __init__(self, workspace_path: str = "/workspace/wade_env"):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard directories
        self.create_standard_directories()
        
        # File type restrictions
        self.allowed_extensions = {
            '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml', '.md', '.txt',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
            '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.php',
            '.sql', '.xml', '.csv', '.log', '.conf', '.cfg', '.ini',
            '.dockerfile', '.gitignore', '.env', '.toml', '.lock'
        }
        
        self.dangerous_extensions = {
            '.exe', '.msi', '.dmg', '.pkg', '.deb', '.rpm', '.app',
            '.scr', '.pif', '.com', '.bat', '.cmd', '.vbs', '.jar'
        }
        
        # Size limits
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_total_size = 1024 * 1024 * 1024  # 1GB
    
    def create_standard_directories(self):
        """Create standard workspace directories"""
        standard_dirs = [
            "src",
            "tests", 
            "docs",
            "scripts",
            "config",
            "data",
            "logs",
            "temp",
            "uploads",
            "downloads",
            "projects",
            "payloads",
            "memory"
        ]
        
        for dir_name in standard_dirs:
            (self.workspace_path / dir_name).mkdir(exist_ok=True)
    
    async def list_files(self, directory: str = "", show_hidden: bool = False) -> List[FileInfo]:
        """List files in directory"""
        try:
            target_dir = self.workspace_path / directory if directory else self.workspace_path
            
            if not target_dir.exists() or not target_dir.is_dir():
                return []
            
            files = []
            
            for item in target_dir.iterdir():
                # Skip hidden files unless requested
                if item.name.startswith('.') and not show_hidden:
                    continue
                
                try:
                    stat = item.stat()
                    
                    file_info = FileInfo(
                        name=item.name,
                        path=str(item.relative_to(self.workspace_path)),
                        size=stat.st_size if item.is_file() else 0,
                        type="directory" if item.is_dir() else "file",
                        mime_type=mimetypes.guess_type(str(item))[0] if item.is_file() else None,
                        created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        permissions=oct(stat.st_mode)[-3:],
                        is_hidden=item.name.startswith('.'),
                        is_executable=os.access(item, os.X_OK)
                    )
                    
                    # Calculate hash for small files
                    if item.is_file() and stat.st_size < 10 * 1024 * 1024:  # 10MB
                        file_info.hash_sha256 = await self._calculate_file_hash(item)
                    
                    files.append(file_info)
                    
                except Exception as e:
                    logging.error(f"Error processing file {item}: {e}")
                    continue
            
            # Sort: directories first, then files, alphabetically
            files.sort(key=lambda x: (x.type == "file", x.name.lower()))
            
            return files
            
        except Exception as e:
            logging.error(f"Error listing files in {directory}: {e}")
            return []
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logging.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    async def read_file(self, file_path: str) -> Optional[str]:
        """Read text file content"""
        try:
            full_path = self.workspace_path / file_path
            
            if not full_path.exists() or not full_path.is_file():
                return None
            
            # Check if file is text
            mime_type, _ = mimetypes.guess_type(str(full_path))
            if mime_type and not mime_type.startswith('text/'):
                return None
            
            async with aiofiles.open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                return await f.read()
                
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None
    
    async def write_file(self, file_path: str, content: str) -> bool:
        """Write content to file"""
        try:
            full_path = self.workspace_path / file_path
            
            # Ensure directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Security check
            if not self._is_safe_path(full_path):
                logging.error(f"Unsafe file path: {file_path}")
                return False
            
            async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            logging.info(f"File written: {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error writing file {file_path}: {e}")
            return False
    
    async def upload_file(self, file_data: bytes, filename: str, target_directory: str = "uploads") -> UploadResult:
        """Upload file to workspace"""
        try:
            # Validate filename
            if not self._is_valid_filename(filename):
                return UploadResult(
                    success=False,
                    file_path="",
                    file_size=0,
                    message=f"Invalid filename: {filename}"
                )
            
            # Check file size
            if len(file_data) > self.max_file_size:
                return UploadResult(
                    success=False,
                    file_path="",
                    file_size=len(file_data),
                    message=f"File too large: {len(file_data)} bytes (max: {self.max_file_size})"
                )
            
            # Check file extension
            file_ext = Path(filename).suffix.lower()
            if file_ext in self.dangerous_extensions:
                return UploadResult(
                    success=False,
                    file_path="",
                    file_size=len(file_data),
                    message=f"Dangerous file type: {file_ext}"
                )
            
            # Create target directory
            target_dir = self.workspace_path / target_directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename if file exists
            target_path = target_dir / filename
            counter = 1
            while target_path.exists():
                name_part = Path(filename).stem
                ext_part = Path(filename).suffix
                target_path = target_dir / f"{name_part}_{counter}{ext_part}"
                counter += 1
            
            # Write file
            async with aiofiles.open(target_path, 'wb') as f:
                await f.write(file_data)
            
            # Calculate hash
            file_hash = hashlib.sha256(file_data).hexdigest()
            
            relative_path = str(target_path.relative_to(self.workspace_path))
            
            logging.info(f"File uploaded: {relative_path} ({len(file_data)} bytes)")
            
            return UploadResult(
                success=True,
                file_path=relative_path,
                file_size=len(file_data),
                message="File uploaded successfully",
                hash_sha256=file_hash
            )
            
        except Exception as e:
            logging.error(f"Error uploading file {filename}: {e}")
            return UploadResult(
                success=False,
                file_path="",
                file_size=0,
                message=f"Upload failed: {str(e)}"
            )
    
    async def download_file(self, file_path: str) -> Optional[bytes]:
        """Download file from workspace"""
        try:
            full_path = self.workspace_path / file_path
            
            if not full_path.exists() or not full_path.is_file():
                return None
            
            # Security check
            if not self._is_safe_path(full_path):
                logging.error(f"Unsafe file path for download: {file_path}")
                return None
            
            async with aiofiles.open(full_path, 'rb') as f:
                return await f.read()
                
        except Exception as e:
            logging.error(f"Error downloading file {file_path}: {e}")
            return None
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file or directory"""
        try:
            full_path = self.workspace_path / file_path
            
            if not full_path.exists():
                return False
            
            # Security check
            if not self._is_safe_path(full_path):
                logging.error(f"Unsafe file path for deletion: {file_path}")
                return False
            
            if full_path.is_file():
                await aiofiles.os.remove(full_path)
            elif full_path.is_dir():
                shutil.rmtree(full_path)
            
            logging.info(f"Deleted: {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")
            return False
    
    async def create_directory(self, directory_path: str) -> bool:
        """Create directory"""
        try:
            full_path = self.workspace_path / directory_path
            
            # Security check
            if not self._is_safe_path(full_path):
                logging.error(f"Unsafe directory path: {directory_path}")
                return False
            
            full_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory created: {directory_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating directory {directory_path}: {e}")
            return False
    
    async def move_file(self, source_path: str, target_path: str) -> bool:
        """Move/rename file or directory"""
        try:
            source_full = self.workspace_path / source_path
            target_full = self.workspace_path / target_path
            
            if not source_full.exists():
                return False
            
            # Security checks
            if not self._is_safe_path(source_full) or not self._is_safe_path(target_full):
                logging.error(f"Unsafe paths for move: {source_path} -> {target_path}")
                return False
            
            # Ensure target directory exists
            target_full.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source_full), str(target_full))
            logging.info(f"Moved: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error moving {source_path} to {target_path}: {e}")
            return False
    
    async def copy_file(self, source_path: str, target_path: str) -> bool:
        """Copy file or directory"""
        try:
            source_full = self.workspace_path / source_path
            target_full = self.workspace_path / target_path
            
            if not source_full.exists():
                return False
            
            # Security checks
            if not self._is_safe_path(source_full) or not self._is_safe_path(target_full):
                logging.error(f"Unsafe paths for copy: {source_path} -> {target_path}")
                return False
            
            # Ensure target directory exists
            target_full.parent.mkdir(parents=True, exist_ok=True)
            
            if source_full.is_file():
                shutil.copy2(str(source_full), str(target_full))
            elif source_full.is_dir():
                shutil.copytree(str(source_full), str(target_full))
            
            logging.info(f"Copied: {source_path} -> {target_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error copying {source_path} to {target_path}: {e}")
            return False
    
    async def create_archive(self, files: List[str], archive_name: str, archive_type: str = "zip") -> Optional[str]:
        """Create archive from files"""
        try:
            archive_path = self.workspace_path / "downloads" / archive_name
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            if archive_type == "zip":
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files:
                        full_path = self.workspace_path / file_path
                        if full_path.exists():
                            zipf.write(full_path, file_path)
            
            elif archive_type == "tar.gz":
                with tarfile.open(archive_path, 'w:gz') as tarf:
                    for file_path in files:
                        full_path = self.workspace_path / file_path
                        if full_path.exists():
                            tarf.add(full_path, file_path)
            
            else:
                return None
            
            relative_path = str(archive_path.relative_to(self.workspace_path))
            logging.info(f"Archive created: {relative_path}")
            return relative_path
            
        except Exception as e:
            logging.error(f"Error creating archive: {e}")
            return None
    
    async def extract_archive(self, archive_path: str, target_directory: str = "temp") -> bool:
        """Extract archive"""
        try:
            full_archive_path = self.workspace_path / archive_path
            target_dir = self.workspace_path / target_directory
            
            if not full_archive_path.exists():
                return False
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(full_archive_path, 'r') as zipf:
                    zipf.extractall(target_dir)
            
            elif archive_path.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(full_archive_path, 'r:gz') as tarf:
                    tarf.extractall(target_dir)
            
            elif archive_path.endswith('.tar'):
                with tarfile.open(full_archive_path, 'r') as tarf:
                    tarf.extractall(target_dir)
            
            else:
                return False
            
            logging.info(f"Archive extracted: {archive_path} -> {target_directory}")
            return True
            
        except Exception as e:
            logging.error(f"Error extracting archive {archive_path}: {e}")
            return False
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe (within workspace)"""
        try:
            # Resolve path and check if it's within workspace
            resolved_path = path.resolve()
            workspace_resolved = self.workspace_path.resolve()
            
            return str(resolved_path).startswith(str(workspace_resolved))
            
        except Exception:
            return False
    
    def _is_valid_filename(self, filename: str) -> bool:
        """Check if filename is valid"""
        if not filename or filename in ['.', '..']:
            return False
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        if any(char in filename for char in invalid_chars):
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        return True
    
    async def get_workspace_stats(self) -> Dict[str, Any]:
        """Get workspace statistics"""
        try:
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for root, dirs, files in os.walk(self.workspace_path):
                dir_count += len(dirs)
                for file in files:
                    file_path = Path(root) / file
                    try:
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except:
                        pass
            
            return {
                "total_size": total_size,
                "file_count": file_count,
                "directory_count": dir_count,
                "workspace_path": str(self.workspace_path),
                "max_file_size": self.max_file_size,
                "max_total_size": self.max_total_size,
                "usage_percentage": (total_size / self.max_total_size) * 100
            }
            
        except Exception as e:
            logging.error(f"Error getting workspace stats: {e}")
            return {}
    
    async def search_files(self, query: str, file_type: Optional[str] = None) -> List[FileInfo]:
        """Search files by name or content"""
        try:
            results = []
            
            for root, dirs, files in os.walk(self.workspace_path):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.workspace_path)
                    
                    # Skip hidden files
                    if file.startswith('.'):
                        continue
                    
                    # Filter by file type
                    if file_type and not file.endswith(file_type):
                        continue
                    
                    # Search in filename
                    if query.lower() in file.lower():
                        try:
                            stat = file_path.stat()
                            file_info = FileInfo(
                                name=file,
                                path=str(relative_path),
                                size=stat.st_size,
                                type="file",
                                mime_type=mimetypes.guess_type(str(file_path))[0],
                                created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                permissions=oct(stat.st_mode)[-3:],
                                is_hidden=file.startswith('.'),
                                is_executable=os.access(file_path, os.X_OK)
                            )
                            results.append(file_info)
                        except:
                            pass
            
            return results[:100]  # Limit results
            
        except Exception as e:
            logging.error(f"Error searching files: {e}")
            return []

# Global file manager instance
file_manager = FileManager()

async def main():
    """Test file manager"""
    logging.basicConfig(level=logging.INFO)
    
    # Test file operations
    await file_manager.write_file("test.txt", "Hello, WADE!")
    content = await file_manager.read_file("test.txt")
    print(f"File content: {content}")
    
    # Test file listing
    files = await file_manager.list_files()
    print(f"Files in workspace: {len(files)}")
    
    # Test workspace stats
    stats = await file_manager.get_workspace_stats()
    print(f"Workspace stats: {stats}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())