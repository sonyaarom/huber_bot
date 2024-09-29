from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import boto3
import json
import shutil
import re
import hashlib
import tempfile
from unpack_funcs import unzip_file
from download_funcs import download_file

# Import your utility functions here
# Ensure these are accessible to Airflow

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sitemap_processing',
    default_args=default_args,
    description='A DAG to process sitemaps',
    schedule_interval=timedelta(days=1),
)

def download_sitemap(**kwargs):
    url = 'https://www.wiwi.hu-berlin.de/sitemap.xml.gz'
    content = unzip_file(download_file(url))
    kwargs['ti'].xcom_push(key='sitemap_content', value=content)

def parse_and_filter_sitemap(**kwargs):
    ti = kwargs['ti']
    content = ti.xcom_pull(key='sitemap_content', task_ids='download_sitemap')
    pattern = r'<url>\s*<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>'
    
    exclude_extensions = ['.jpg', '.pdf', '.jpeg', '.png']
    exclude_patterns = ['view']
    include_patterns = ['/en/']
    allowed_base_url = 'https://www.wiwi.hu-berlin.de'

    sitemap_entries = parse_sitemap(content, pattern)
    filtered_entries = filter_sitemap_entries(sitemap_entries, exclude_extensions, exclude_patterns, include_patterns)
    safe_entries, unsafe_entries = security_check_urls(filtered_entries, allowed_base_url)
    
    data_dict = create_matches_dict(safe_entries)
    ti.xcom_push(key='processed_data', value=data_dict)
    ti.xcom_push(key='stats', value={
        'total': len(sitemap_entries),
        'filtered': len(filtered_entries),
        'safe': len(safe_entries),
        'unsafe': len(unsafe_entries)
    })

def create_json_file(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(key='processed_data', task_ids='parse_and_filter_sitemap')
    temp_file_path, filename = create_temp_json_file(data_dict, custom_filename='sitemap_data')
    ti.xcom_push(key='temp_file_path', value=temp_file_path)
    ti.xcom_push(key='filename', value=filename)

def store_locally(**kwargs):
    ti = kwargs['ti']
    temp_file_path = ti.xcom_pull(key='temp_file_path', task_ids='create_json_file')
    filename = ti.xcom_pull(key='filename', task_ids='create_json_file')
    local_directory = '../assets/json_files'
    os.makedirs(local_directory, exist_ok=True)
    local_file_path = os.path.join(local_directory, filename)
    shutil.copy2(temp_file_path, local_file_path)
    print(f"File stored locally at: {local_file_path}")

def upload_to_s3(**kwargs):
    ti = kwargs['ti']
    temp_file_path = ti.xcom_pull(key='temp_file_path', task_ids='create_json_file')
    filename = ti.xcom_pull(key='filename', task_ids='create_json_file')
    bucket_name = 'hu-chatbot-schema'
    s3_key_prefix = 'sitemap_data'
    s3_key = f"{s3_key_prefix}/{filename}"
    
    with open(temp_file_path, 'rb') as file:
        s3_client = boto3.client('s3')
        s3_client.upload_fileobj(file, bucket_name, s3_key)
    
    head_response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
    version_id = head_response.get('VersionId', 'null')
    
    print(f"File uploaded to S3 bucket: {bucket_name}")
    print(f"S3 key: {s3_key}")
    print(f"S3 version ID: {version_id}")

def cleanup(**kwargs):
    ti = kwargs['ti']
    temp_file_path = ti.xcom_pull(key='temp_file_path', task_ids='create_json_file')
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)
    print("Temporary file cleaned up")

# Define tasks
t1 = PythonOperator(
    task_id='download_sitemap',
    python_callable=download_sitemap,
    provide_context=True,
    dag=dag,
)

t2 = PythonOperator(
    task_id='parse_and_filter_sitemap',
    python_callable=parse_and_filter_sitemap,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='create_json_file',
    python_callable=create_json_file,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='store_locally',
    python_callable=store_locally,
    provide_context=True,
    dag=dag,
)

t5 = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_to_s3,
    provide_context=True,
    dag=dag,
)

t6 = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
t1 >> t2 >> t3 >> [t4, t5] >> t6