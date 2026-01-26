#!/usr/bin/env python3
"""
BioInsight AI - GCP Cloud Storage Sync Utility

Cloud Run의 임시 파일 시스템을 보완하기 위해
Cloud Storage와 로컬 디렉토리를 동기화합니다.

사용법:
    # 앱 시작 시 모델/데이터 다운로드
    python storage_sync.py download

    # 결과 업로드
    python storage_sync.py upload --path /app/output/job_123
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GCS 버킷 설정
BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'bioinsight-data')
PROJECT_ID = os.getenv('GCP_PROJECT_ID', '')

# 동기화할 디렉토리 매핑
SYNC_DIRS = {
    'models': {
        'local': '/app/models',
        'remote': 'models/',
        'direction': 'download',  # 앱 시작 시 다운로드
    },
    'chroma_db': {
        'local': '/app/chroma_db',
        'remote': 'chroma_db/',
        'direction': 'download',
    },
    'output': {
        'local': '/app/output',
        'remote': 'output/',
        'direction': 'upload',  # 결과는 업로드
    },
}


def get_storage_client():
    """GCS 클라이언트 초기화"""
    try:
        from google.cloud import storage
        return storage.Client(project=PROJECT_ID)
    except Exception as e:
        logger.error(f"GCS 클라이언트 초기화 실패: {e}")
        return None


def download_directory(bucket_name: str, remote_prefix: str, local_dir: str):
    """GCS에서 디렉토리 다운로드"""
    client = get_storage_client()
    if not client:
        return False

    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=remote_prefix)

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for blob in blobs:
        if blob.name.endswith('/'):
            continue

        relative_path = blob.name[len(remote_prefix):]
        local_file = local_path / relative_path
        local_file.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_file))
        count += 1

    logger.info(f"다운로드 완료: {count}개 파일 → {local_dir}")
    return True


def upload_directory(bucket_name: str, local_dir: str, remote_prefix: str):
    """로컬 디렉토리를 GCS에 업로드"""
    client = get_storage_client()
    if not client:
        return False

    bucket = client.bucket(bucket_name)
    local_path = Path(local_dir)

    if not local_path.exists():
        logger.warning(f"로컬 디렉토리 없음: {local_dir}")
        return False

    count = 0
    for local_file in local_path.rglob('*'):
        if local_file.is_dir():
            continue

        relative_path = local_file.relative_to(local_path)
        blob_name = f"{remote_prefix}{relative_path}"

        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_file))
        count += 1

    logger.info(f"업로드 완료: {count}개 파일 → gs://{bucket_name}/{remote_prefix}")
    return True


def upload_file(bucket_name: str, local_file: str, remote_path: str):
    """단일 파일 업로드"""
    client = get_storage_client()
    if not client:
        return False

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_file)

    logger.info(f"업로드: {local_file} → gs://{bucket_name}/{remote_path}")
    return True


def download_file(bucket_name: str, remote_path: str, local_file: str):
    """단일 파일 다운로드"""
    client = get_storage_client()
    if not client:
        return False

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)

    Path(local_file).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_file)

    logger.info(f"다운로드: gs://{bucket_name}/{remote_path} → {local_file}")
    return True


def sync_on_startup():
    """앱 시작 시 필요한 데이터 동기화"""
    logger.info("=== 앱 시작 시 데이터 동기화 ===")

    for name, config in SYNC_DIRS.items():
        if config['direction'] == 'download':
            logger.info(f"동기화 중: {name}")
            download_directory(
                BUCKET_NAME,
                config['remote'],
                config['local']
            )


def sync_job_result(job_id: str):
    """분석 결과를 GCS에 업로드"""
    local_dir = f"/app/output/{job_id}"
    remote_prefix = f"output/{job_id}/"

    return upload_directory(BUCKET_NAME, local_dir, remote_prefix)


def get_result_url(job_id: str, filename: str = "report.html") -> str:
    """결과 파일의 공개 URL 반환"""
    return f"https://storage.googleapis.com/{BUCKET_NAME}/output/{job_id}/{filename}"


def main():
    parser = argparse.ArgumentParser(description='GCP Storage 동기화 유틸리티')
    parser.add_argument('action', choices=['download', 'upload', 'startup', 'sync-job'],
                       help='수행할 작업')
    parser.add_argument('--path', type=str, help='동기화할 경로')
    parser.add_argument('--job-id', type=str, help='Job ID (sync-job용)')

    args = parser.parse_args()

    if args.action == 'startup':
        sync_on_startup()

    elif args.action == 'download':
        if args.path:
            download_directory(BUCKET_NAME, args.path, f"/app/{args.path}")
        else:
            sync_on_startup()

    elif args.action == 'upload':
        if args.path:
            remote = args.path.replace('/app/', '')
            upload_directory(BUCKET_NAME, args.path, remote)
        else:
            logger.error("--path 필요")

    elif args.action == 'sync-job':
        if args.job_id:
            sync_job_result(args.job_id)
        else:
            logger.error("--job-id 필요")


if __name__ == '__main__':
    main()
