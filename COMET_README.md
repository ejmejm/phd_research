# Comet ML Offline Experiment Uploader

A CLI utility for uploading offline Comet ML experiment archives to the cloud with enhanced features including multi-threading, progress tracking, and file marking.

## Features

- **Multi-threaded uploads**: Use `--n-threads` to upload multiple files concurrently
- **Progress tracking**: Real-time progress bar showing success/failure counts
- **File marking**: Automatically rename uploaded files with `--mark` flag
- **Enhanced logging**: Custom logger with detailed error reporting
- **Robust error handling**: Continues processing other files even if some fail

## Installation

### Requirements

```bash
pip install comet_ml tqdm
```

### Setup

1. Make the script executable:
```bash
chmod +x comet_uploader.py
```

2. Set up your Comet ML API key (if not already configured):
```bash
export COMET_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

Upload a single experiment archive:
```bash
python comet_uploader.py experiment.zip
```

Upload multiple experiment archives:
```bash
python comet_uploader.py exp1.zip exp2.zip exp3.zip
```

### Advanced Options

**Multi-threaded upload:**
```bash
python comet_uploader.py --n-threads 4 *.zip
```

**Mark uploaded files (rename to .uploaded):**
```bash
python comet_uploader.py --mark experiment.zip
# Creates: experiment.zip.uploaded after successful upload
```

**Force upload (re-upload already uploaded experiments):**
```bash
python comet_uploader.py --force-upload experiment.zip
```

**Override workspace and project:**
```bash
python comet_uploader.py --workspace my-workspace --project-name my-project *.zip
```

**Combine all options:**
```bash
python comet_uploader.py --n-threads 8 --mark --force-upload --workspace production *.zip
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `archives` | Experiment archive files to upload (required) | - |
| `--force-upload` | Force upload experiments that were already uploaded | False |
| `--workspace` | Override workspace for all experiments | None |
| `--project-name` | Override project name for all experiments | None |
| `--mark` | Rename uploaded files to `{filename}.uploaded` | False |
| `--n-threads` | Number of concurrent upload threads | 1 |

## Output

The utility provides:
- Real-time progress bar with success/failure counts
- Detailed logging of each upload attempt
- Final summary of upload results
- List of failed files (if any)

### Example Output

```
Success: 3, Failed: 1: 80%|████████  | 4/5 [00:45<00:11,  2.1s/file]
2024-01-15 10:30:45 - comet_uploader - INFO - Successfully uploaded: exp1.zip
2024-01-15 10:30:47 - comet_uploader - ERROR - Upload failed: corrupted_exp.zip
2024-01-15 10:30:50 - comet_uploader - INFO - Successfully uploaded: exp2.zip
2024-01-15 10:30:52 - comet_uploader - INFO - Successfully uploaded: exp3.zip
2024-01-15 10:30:54 - comet_uploader - INFO - Successfully uploaded: exp4.zip
2024-01-15 10:30:54 - comet_uploader - INFO - Upload completed: 4 successful, 1 failed
```

## Error Handling

- **Invalid API Key**: Stops all uploads immediately
- **Missing files**: Skips non-existent files and continues
- **Upload failures**: Logs errors and continues with remaining files
- **Rename failures**: Counts as failed upload even if upload succeeded

## Threading Considerations

- Use `--n-threads` for faster uploads when you have multiple files
- Recommended thread count: 2-8 depending on your network and system
- Each thread handles one file at a time
- Progress bar and statistics are thread-safe

## Differences from Original

This utility improves upon the original `comet_upload.py` with:

1. **Removed deprecated `--force-reupload`** (use `--force-upload` instead)
2. **Custom logger name** (`comet_uploader` instead of `comet_ml`)
3. **File marking capability** with `--mark` flag
4. **Multi-threading support** with `--n-threads`
5. **Progress bar** with real-time success/failure tracking
6. **Enhanced error handling** and logging
7. **Google-style Python formatting**

## License

Based on the original Comet ML source code (MIT License).
