#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Based on CometML's comet_upload.py script
"""
Comet ML Offline Experiment Uploader
Upload offline experiment archives to Comet ML backend with enhanced features.
"""

import argparse
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is required. Install it with: pip install tqdm")
    sys.exit(1)

try:
    from comet_ml.exceptions import InvalidAPIKey, OfflineExperimentUploadFailed
    from comet_ml.offline import upload_single_offline_experiment
    from comet_ml.config import get_config
    from comet_ml.api import get_api_key
except ImportError:
    print("Error: comet_ml is required. Install it with: pip install comet_ml")
    sys.exit(1)

# Create a custom logger for this utility
logger = logging.getLogger('comet_uploader')


class UploadStats:
    """Thread-safe class to track upload statistics."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.success_count = 0
        self.fail_count = 0
        self.processed_files = []
        self.failed_files = []
    
    def add_success(self, filename: str):
        with self._lock:
            self.success_count += 1
            self.processed_files.append(filename)
    
    def add_failure(self, filename: str):
        with self._lock:
            self.fail_count += 1
            self.failed_files.append(filename)
    
    def get_counts(self):
        with self._lock:
            return self.success_count, self.fail_count


def get_parser_arguments(parser):
    """Add command line arguments to the parser."""
    parser.add_argument(
        "archives", 
        nargs = "+", 
        help = "the offline experiment archives to upload"
    )
    parser.add_argument(
        "--force-upload",
        help = "force upload offline experiments that were already uploaded",
        action = "store_true",
        default = False,
    )
    parser.add_argument(
        "--workspace",
        dest = "override_workspace",
        help = "upload all offline experiment archives to that workspace instead of the original workspace",
        action = "store",
    )
    parser.add_argument(
        "--project-name",
        dest = "override_project_name",
        help = "upload all offline experiment archives to that project instead of the original project",
        action = "store",
    )
    parser.add_argument(
        "--mark",
        help = "rename successfully uploaded files to {filename}.uploaded",
        action = "store_true",
        default = False,
    )
    parser.add_argument(
        "--n-threads",
        type = int,
        default = 1,
        help = "number of threads to use for parallel uploads (default: 1)",
    )


def rename_uploaded_file(filepath: str) -> bool:
    """Rename a file to mark it as uploaded. Returns True if successful."""
    try:
        original_path = Path(filepath)
        new_path = original_path.with_name(f"{original_path.name}.uploaded")
        original_path.rename(new_path)
        return True
    except Exception as e:
        logger.error(f"Failed to rename {filepath}: {str(e)}")
        return False


def upload_single_file(
    filename: str,
    api_key: str,
    force_upload: bool,
    override_workspace: Optional[str],
    override_project_name: Optional[str],
    mark_uploaded: bool,
    stats: UploadStats,
    pbar: tqdm,
) -> None:
    """Upload a single file and update progress."""
    try:
        success = upload_single_offline_experiment(
            offline_archive_path=filename,
            api_key=api_key,
            force_upload=force_upload,
            override_workspace=override_workspace,
            override_project_name=override_project_name,
        )
        
        if success:
            # Mark file as uploaded if requested
            rename_success = True
            if mark_uploaded:
                rename_success = rename_uploaded_file(filename)
            
            if rename_success:
                stats.add_success(filename)
                logger.debug(f"Successfully uploaded: {filename}")
            else:
                stats.add_failure(filename)
                logger.error(f"Upload succeeded but failed to rename: {filename}")
        else:
            stats.add_failure(filename)
            logger.error(f"Upload failed: {filename}")
            
    except InvalidAPIKey:
        stats.add_failure(filename)
        logger.error(f"Invalid API key - failed to upload: {filename}")
        raise  # Re-raise to stop other threads
    except Exception as e:
        stats.add_failure(filename)
        logger.error(f"Unexpected error uploading {filename}: {str(e)}")
    
    finally:
        # Update progress bar
        success_count, fail_count = stats.get_counts()
        pbar.set_description(f"Success: {success_count}, Failed: {fail_count}")
        pbar.update(1)


def upload_files(args) -> None:
    """Main upload function with threading support."""
    # Validate files exist
    valid_archives = []
    for archive in args.archives:
        if not os.path.exists(archive):
            logger.error(f"File not found: {archive}")
            continue
        valid_archives.append(archive)
    
    if not valid_archives:
        logger.error("No valid archive files found")
        sys.exit(1)
    
    # Get API configuration
    try:
        config = get_config()
        api_key = get_api_key(None, config)
    except Exception as e:
        logger.error(f"Failed to get API configuration: {str(e)}")
        sys.exit(1)
    
    # Initialize progress tracking
    stats = UploadStats()
    total_files = len(valid_archives)
    
    # Create progress bar
    pbar = tqdm(
        total=total_files,
        desc="Success: 0, Failed: 0",
        unit="file"
    )
    
    try:
        # Multi-threaded execution
        with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
            # Submit all upload tasks
            futures = []
            for filename in valid_archives:
                future = executor.submit(
                    upload_single_file,
                    filename = filename,
                    api_key = api_key,
                    force_upload = args.force_upload,
                    override_workspace = args.override_workspace,
                    override_project_name = args.override_project_name,
                    mark_uploaded = args.mark,
                    stats = stats,
                    pbar = pbar,
                )
                futures.append(future)
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will re-raise any exceptions
                except InvalidAPIKey:
                    # Stop all other uploads if API key is invalid
                    executor.shutdown(wait=False)
                    logger.error("Invalid API key - stopping all uploads")
                    sys.exit(1)
                except Exception:
                    # Individual upload failures are already logged
                    pass
    
    finally:
        pbar.close()
    
    # Final summary
    success_count, fail_count = stats.get_counts()
    logger.info(f"Upload completed: {success_count} successful, {fail_count} failed")
    
    if success_count > 0:
        logger.info(f"Successfully uploaded {success_count} experiments")
    
    if fail_count > 0:
        logger.error(f"Failed to upload {fail_count} experiments")
        if stats.failed_files:
            logger.error(f"Failed files: {', '.join(stats.failed_files)}")
        sys.exit(1)


def main(args=None):
    """Main entry point."""
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    get_parser_arguments(parser)
    parsed_args = parser.parse_args(args)
    
    # Validate n_threads argument
    if parsed_args.n_threads < 1:
        parser.error("--n-threads must be at least 1")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    upload_files(parsed_args)


if __name__ == '__main__':
    main() 