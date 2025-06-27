#!/usr/bin/env python3
"""
WADE Project Sync - Export and synchronize project data
Handles project export, backup, and cloud synchronization
"""

import os
import json
import logging
import asyncio
import zipfile
import tarfile
import tempfile
import shutil
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO, Union
import io

# Import WADE components
try:
    from settings_manager import settings_manager
    from file_manager import file_manager
except ImportError:
    # For standalone testing
    from wade_env.settings_manager import settings_manager
    from wade_env.file_manager import file_manager

# Optional cloud provider imports
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

try:
    import dropbox
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("project_sync")

class ProjectSync:
    """
    Project synchronization and export manager
    Handles project export, backup, and cloud synchronization
    """
    
    def __init__(self):
        """Initialize the project sync manager"""
        self.settings = self._load_sync_settings()
        self.workspace_path = self.settings.get("workspace_path", "/workspace")
        self.backup_path = self.settings.get("backup_path", "/workspace/backups")
        self.export_format = self.settings.get("export_format", "zip")
        self.cloud_providers = self._initialize_cloud_providers()
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_path, exist_ok=True)
    
    def _load_sync_settings(self) -> Dict[str, Any]:
        """Load sync settings from settings manager"""
        try:
            sync_settings = settings_manager.get_settings_dict().get("project_sync", {})
            if not sync_settings:
                # Initialize with defaults if not present
                sync_settings = {
                    "workspace_path": "/workspace",
                    "backup_path": "/workspace/backups",
                    "export_format": "zip",
                    "auto_backup_enabled": True,
                    "auto_backup_interval_hours": 24,
                    "cloud_sync_enabled": False,
                    "cloud_providers": {
                        "google_drive": {
                            "enabled": False,
                            "credentials_file": "",
                            "bucket_name": ""
                        },
                        "dropbox": {
                            "enabled": False,
                            "access_token": "",
                            "folder_path": "/WADE_Backups"
                        }
                    }
                }
                settings_manager.update_settings("project_sync", sync_settings)
            return sync_settings
        except Exception as e:
            logger.error(f"Error loading sync settings: {e}")
            return {
                "workspace_path": "/workspace",
                "backup_path": "/workspace/backups",
                "export_format": "zip",
                "auto_backup_enabled": True,
                "auto_backup_interval_hours": 24,
                "cloud_sync_enabled": False,
                "cloud_providers": {}
            }
    
    def _initialize_cloud_providers(self) -> Dict[str, Any]:
        """Initialize cloud provider clients based on settings"""
        providers = {}
        
        # Initialize Google Drive client if enabled and available
        gd_settings = self.settings.get("cloud_providers", {}).get("google_drive", {})
        if GOOGLE_DRIVE_AVAILABLE and gd_settings.get("enabled", False):
            try:
                credentials_file = gd_settings.get("credentials_file", "")
                bucket_name = gd_settings.get("bucket_name", "")
                
                if os.path.exists(credentials_file):
                    credentials = service_account.Credentials.from_service_account_file(credentials_file)
                    client = storage.Client(credentials=credentials)
                    bucket = client.bucket(bucket_name)
                    
                    providers["google_drive"] = {
                        "client": client,
                        "bucket": bucket,
                        "enabled": True
                    }
                    logger.info("Google Drive client initialized successfully")
                else:
                    logger.warning(f"Google Drive credentials file not found: {credentials_file}")
                    providers["google_drive"] = {"enabled": False}
            except Exception as e:
                logger.error(f"Error initializing Google Drive client: {e}")
                providers["google_drive"] = {"enabled": False}
        else:
            providers["google_drive"] = {"enabled": False}
        
        # Initialize Dropbox client if enabled and available
        db_settings = self.settings.get("cloud_providers", {}).get("dropbox", {})
        if DROPBOX_AVAILABLE and db_settings.get("enabled", False):
            try:
                access_token = db_settings.get("access_token", "")
                folder_path = db_settings.get("folder_path", "/WADE_Backups")
                
                if access_token:
                    client = dropbox.Dropbox(access_token)
                    # Test connection
                    client.users_get_current_account()
                    
                    providers["dropbox"] = {
                        "client": client,
                        "folder_path": folder_path,
                        "enabled": True
                    }
                    logger.info("Dropbox client initialized successfully")
                else:
                    logger.warning("Dropbox access token not provided")
                    providers["dropbox"] = {"enabled": False}
            except Exception as e:
                logger.error(f"Error initializing Dropbox client: {e}")
                providers["dropbox"] = {"enabled": False}
        else:
            providers["dropbox"] = {"enabled": False}
        
        return providers
    
    async def export_project(self, include_chat_history: bool = True, 
                           include_memory: bool = True,
                           include_settings: bool = True) -> Dict[str, Any]:
        """
        Export the current project as a zip or tar archive
        Returns metadata about the export including the file path
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"wade_project_export_{timestamp}"
        
        if self.export_format == "zip":
            export_path = os.path.join(self.backup_path, f"{export_filename}.zip")
            await self._create_zip_archive(export_path, include_chat_history, include_memory, include_settings)
        else:  # tar.gz
            export_path = os.path.join(self.backup_path, f"{export_filename}.tar.gz")
            await self._create_tar_archive(export_path, include_chat_history, include_memory, include_settings)
        
        # Calculate file size and hash
        file_size = os.path.getsize(export_path)
        file_hash = self._calculate_file_hash(export_path)
        
        metadata = {
            "filename": os.path.basename(export_path),
            "path": export_path,
            "format": self.export_format,
            "timestamp": timestamp,
            "size_bytes": file_size,
            "size_human": self._human_readable_size(file_size),
            "sha256_hash": file_hash,
            "includes": {
                "chat_history": include_chat_history,
                "memory": include_memory,
                "settings": include_settings
            }
        }
        
        logger.info(f"Project exported successfully to {export_path}")
        return metadata
    
    async def _create_zip_archive(self, export_path: str, include_chat_history: bool,
                                include_memory: bool, include_settings: bool) -> None:
        """Create a zip archive of the project"""
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add project files
            for root, _, files in os.walk(self.workspace_path):
                # Skip backup directory and hidden directories
                if (os.path.abspath(root).startswith(os.path.abspath(self.backup_path)) or
                    os.path.basename(root).startswith('.')):
                    continue
                
                for file in files:
                    # Skip hidden files and temporary files
                    if file.startswith('.') or file.endswith('.tmp'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.workspace_path)
                    
                    try:
                        zipf.write(file_path, arcname)
                    except Exception as e:
                        logger.error(f"Error adding file to zip: {file_path} - {e}")
            
            # Add chat history if requested
            if include_chat_history:
                chat_history = await self._get_chat_history()
                zipf.writestr("wade_export/chat_history.json", json.dumps(chat_history, indent=2))
            
            # Add memory data if requested
            if include_memory:
                memory_data = await self._get_memory_data()
                zipf.writestr("wade_export/memory_data.json", json.dumps(memory_data, indent=2))
            
            # Add settings if requested
            if include_settings:
                settings_data = settings_manager.get_settings_dict()
                zipf.writestr("wade_export/settings.json", json.dumps(settings_data, indent=2))
            
            # Add export metadata
            metadata = {
                "export_date": datetime.datetime.now().isoformat(),
                "wade_version": "3.0.0",
                "includes": {
                    "chat_history": include_chat_history,
                    "memory": include_memory,
                    "settings": include_settings
                }
            }
            zipf.writestr("wade_export/metadata.json", json.dumps(metadata, indent=2))
    
    async def _create_tar_archive(self, export_path: str, include_chat_history: bool,
                                include_memory: bool, include_settings: bool) -> None:
        """Create a tar.gz archive of the project"""
        with tarfile.open(export_path, 'w:gz') as tarf:
            # Add project files
            for root, _, files in os.walk(self.workspace_path):
                # Skip backup directory and hidden directories
                if (os.path.abspath(root).startswith(os.path.abspath(self.backup_path)) or
                    os.path.basename(root).startswith('.')):
                    continue
                
                for file in files:
                    # Skip hidden files and temporary files
                    if file.startswith('.') or file.endswith('.tmp'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    arcname = os.path.join("wade_export", os.path.relpath(file_path, self.workspace_path))
                    
                    try:
                        tarf.add(file_path, arcname=arcname)
                    except Exception as e:
                        logger.error(f"Error adding file to tar: {file_path} - {e}")
            
            # Add chat history if requested
            if include_chat_history:
                chat_history = await self._get_chat_history()
                chat_json = json.dumps(chat_history, indent=2).encode('utf-8')
                
                chat_info = tarfile.TarInfo("wade_export/chat_history.json")
                chat_info.size = len(chat_json)
                chat_info.mtime = int(datetime.datetime.now().timestamp())
                
                tarf.addfile(chat_info, io.BytesIO(chat_json))
            
            # Add memory data if requested
            if include_memory:
                memory_data = await self._get_memory_data()
                memory_json = json.dumps(memory_data, indent=2).encode('utf-8')
                
                memory_info = tarfile.TarInfo("wade_export/memory_data.json")
                memory_info.size = len(memory_json)
                memory_info.mtime = int(datetime.datetime.now().timestamp())
                
                tarf.addfile(memory_info, io.BytesIO(memory_json))
            
            # Add settings if requested
            if include_settings:
                settings_data = settings_manager.get_settings_dict()
                settings_json = json.dumps(settings_data, indent=2).encode('utf-8')
                
                settings_info = tarfile.TarInfo("wade_export/settings.json")
                settings_info.size = len(settings_json)
                settings_info.mtime = int(datetime.datetime.now().timestamp())
                
                tarf.addfile(settings_info, io.BytesIO(settings_json))
            
            # Add export metadata
            metadata = {
                "export_date": datetime.datetime.now().isoformat(),
                "wade_version": "3.0.0",
                "includes": {
                    "chat_history": include_chat_history,
                    "memory": include_memory,
                    "settings": include_settings
                }
            }
            metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
            
            metadata_info = tarfile.TarInfo("wade_export/metadata.json")
            metadata_info.size = len(metadata_json)
            metadata_info.mtime = int(datetime.datetime.now().timestamp())
            
            tarf.addfile(metadata_info, io.BytesIO(metadata_json))
    
    async def _get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the current chat history"""
        try:
            # This would normally come from a chat history manager
            # For now, we'll use a mock implementation
            chat_history_path = os.path.join(self.workspace_path, "wade_env", "chat_history.json")
            if os.path.exists(chat_history_path):
                with open(chat_history_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    async def _get_memory_data(self) -> Dict[str, Any]:
        """Get the current memory data"""
        try:
            # This would normally come from a memory manager
            # For now, we'll use a mock implementation
            memory_path = os.path.join(self.workspace_path, "wade_env", "memory_data.json")
            if os.path.exists(memory_path):
                with open(memory_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error getting memory data: {e}")
            return {}
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human-readable size"""
        if size_bytes == 0:
            return "0B"
        size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.log(size_bytes, 1024))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    async def sync_to_cloud(self, export_path: str, provider: str = "all") -> Dict[str, Any]:
        """
        Sync a project export to cloud storage
        Returns metadata about the sync operation
        """
        if not self.settings.get("cloud_sync_enabled", False):
            return {"success": False, "message": "Cloud sync is disabled in settings"}
        
        results = {}
        
        # Check if the export file exists
        if not os.path.exists(export_path):
            return {"success": False, "message": f"Export file not found: {export_path}"}
        
        # Sync to Google Drive if requested
        if (provider == "all" or provider == "google_drive") and self.cloud_providers["google_drive"]["enabled"]:
            gd_result = await self._sync_to_google_drive(export_path)
            results["google_drive"] = gd_result
        
        # Sync to Dropbox if requested
        if (provider == "all" or provider == "dropbox") and self.cloud_providers["dropbox"]["enabled"]:
            db_result = await self._sync_to_dropbox(export_path)
            results["dropbox"] = db_result
        
        # Overall success if at least one provider succeeded
        success = any(results.get(p, {}).get("success", False) for p in results)
        
        return {
            "success": success,
            "timestamp": datetime.datetime.now().isoformat(),
            "file": os.path.basename(export_path),
            "providers": results
        }
    
    async def _sync_to_google_drive(self, export_path: str) -> Dict[str, Any]:
        """Sync a file to Google Drive"""
        try:
            if not GOOGLE_DRIVE_AVAILABLE:
                return {"success": False, "message": "Google Drive integration not available"}
            
            if not self.cloud_providers["google_drive"]["enabled"]:
                return {"success": False, "message": "Google Drive integration not enabled"}
            
            bucket = self.cloud_providers["google_drive"]["bucket"]
            filename = os.path.basename(export_path)
            blob = bucket.blob(f"wade_backups/{filename}")
            
            # Upload the file
            blob.upload_from_filename(export_path)
            
            return {
                "success": True,
                "message": "File uploaded to Google Drive successfully",
                "url": blob.public_url if blob.public else None,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error syncing to Google Drive: {e}")
            return {"success": False, "message": f"Error syncing to Google Drive: {str(e)}"}
    
    async def _sync_to_dropbox(self, export_path: str) -> Dict[str, Any]:
        """Sync a file to Dropbox"""
        try:
            if not DROPBOX_AVAILABLE:
                return {"success": False, "message": "Dropbox integration not available"}
            
            if not self.cloud_providers["dropbox"]["enabled"]:
                return {"success": False, "message": "Dropbox integration not enabled"}
            
            client = self.cloud_providers["dropbox"]["client"]
            folder_path = self.cloud_providers["dropbox"]["folder_path"]
            filename = os.path.basename(export_path)
            dropbox_path = f"{folder_path}/{filename}"
            
            # Upload the file
            with open(export_path, "rb") as f:
                client.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            
            # Create a shared link
            shared_link = client.sharing_create_shared_link_with_settings(dropbox_path)
            
            return {
                "success": True,
                "message": "File uploaded to Dropbox successfully",
                "url": shared_link.url if hasattr(shared_link, 'url') else None,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error syncing to Dropbox: {e}")
            return {"success": False, "message": f"Error syncing to Dropbox: {str(e)}"}
    
    async def import_project(self, import_path: str, extract_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Import a project from a zip or tar archive
        Returns metadata about the import operation
        """
        if not os.path.exists(import_path):
            return {"success": False, "message": f"Import file not found: {import_path}"}
        
        # Determine the extraction path
        if extract_to is None:
            extract_to = os.path.join(self.workspace_path, "imported_project")
        
        # Create extraction directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        
        # Extract the archive based on its format
        if import_path.endswith('.zip'):
            success, message = await self._extract_zip(import_path, extract_to)
        elif import_path.endswith('.tar.gz') or import_path.endswith('.tgz'):
            success, message = await self._extract_tar(import_path, extract_to)
        else:
            return {"success": False, "message": "Unsupported archive format"}
        
        if not success:
            return {"success": False, "message": message}
        
        # Load metadata if available
        metadata_path = os.path.join(extract_to, "wade_export", "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        # Import chat history if available
        chat_history_path = os.path.join(extract_to, "wade_export", "chat_history.json")
        if os.path.exists(chat_history_path):
            await self._import_chat_history(chat_history_path)
        
        # Import settings if available
        settings_path = os.path.join(extract_to, "wade_export", "settings.json")
        if os.path.exists(settings_path):
            await self._import_settings(settings_path)
        
        return {
            "success": True,
            "message": "Project imported successfully",
            "extract_path": extract_to,
            "metadata": metadata,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    async def _extract_zip(self, zip_path: str, extract_to: str) -> tuple[bool, str]:
        """Extract a zip archive"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_to)
            return True, "Zip archive extracted successfully"
        except Exception as e:
            logger.error(f"Error extracting zip archive: {e}")
            return False, f"Error extracting zip archive: {str(e)}"
    
    async def _extract_tar(self, tar_path: str, extract_to: str) -> tuple[bool, str]:
        """Extract a tar.gz archive"""
        try:
            with tarfile.open(tar_path, 'r:gz') as tarf:
                # Check for any suspicious paths (security check)
                for member in tarf.getmembers():
                    if member.name.startswith('/') or '..' in member.name:
                        return False, "Archive contains suspicious paths"
                
                tarf.extractall(extract_to)
            return True, "Tar archive extracted successfully"
        except Exception as e:
            logger.error(f"Error extracting tar archive: {e}")
            return False, f"Error extracting tar archive: {str(e)}"
    
    async def _import_chat_history(self, chat_history_path: str) -> bool:
        """Import chat history from a JSON file"""
        try:
            with open(chat_history_path, 'r') as f:
                chat_history = json.load(f)
            
            # This would normally update the chat history in the application
            # For now, we'll just log it
            logger.info(f"Imported chat history with {len(chat_history)} messages")
            return True
        except Exception as e:
            logger.error(f"Error importing chat history: {e}")
            return False
    
    async def _import_settings(self, settings_path: str) -> bool:
        """Import settings from a JSON file"""
        try:
            with open(settings_path, 'r') as f:
                settings_data = json.load(f)
            
            # Update settings
            for section, section_data in settings_data.items():
                settings_manager.update_settings(section, section_data)
            
            logger.info(f"Imported settings with {len(settings_data)} sections")
            return True
        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            return False
    
    async def list_exports(self) -> List[Dict[str, Any]]:
        """List all available project exports"""
        exports = []
        
        if not os.path.exists(self.backup_path):
            return exports
        
        for filename in os.listdir(self.backup_path):
            if filename.startswith("wade_project_export_") and (filename.endswith(".zip") or filename.endswith(".tar.gz")):
                file_path = os.path.join(self.backup_path, filename)
                file_size = os.path.getsize(file_path)
                
                # Parse timestamp from filename
                timestamp_str = filename.replace("wade_project_export_", "").split(".")[0]
                try:
                    timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").isoformat()
                except ValueError:
                    timestamp = None
                
                exports.append({
                    "filename": filename,
                    "path": file_path,
                    "format": "zip" if filename.endswith(".zip") else "tar.gz",
                    "timestamp": timestamp,
                    "size_bytes": file_size,
                    "size_human": self._human_readable_size(file_size)
                })
        
        # Sort by timestamp (newest first)
        exports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return exports
    
    async def delete_export(self, export_path: str) -> bool:
        """Delete a project export file"""
        if not os.path.exists(export_path):
            return False
        
        try:
            os.remove(export_path)
            return True
        except Exception as e:
            logger.error(f"Error deleting export: {e}")
            return False
    
    async def schedule_auto_backup(self) -> None:
        """Schedule automatic backups based on settings"""
        if not self.settings.get("auto_backup_enabled", True):
            logger.info("Automatic backups are disabled")
            return
        
        interval_hours = self.settings.get("auto_backup_interval_hours", 24)
        logger.info(f"Scheduling automatic backups every {interval_hours} hours")
        
        while True:
            # Wait for the specified interval
            await asyncio.sleep(interval_hours * 3600)
            
            # Create a backup
            try:
                logger.info("Creating automatic backup")
                await self.export_project()
            except Exception as e:
                logger.error(f"Error creating automatic backup: {e}")

# Create singleton instance
project_sync = ProjectSync()

# For testing
if __name__ == "__main__":
    import math  # Required for _human_readable_size
    
    async def test_project_sync():
        # Test export
        export_result = await project_sync.export_project()
        print("Export result:", export_result)
        
        # Test listing exports
        exports = await project_sync.list_exports()
        print(f"Found {len(exports)} exports")
        for export in exports:
            print(f"- {export['filename']} ({export['size_human']})")
    
    asyncio.run(test_project_sync())