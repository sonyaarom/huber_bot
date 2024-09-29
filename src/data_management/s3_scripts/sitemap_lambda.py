import json
import boto3
from botocore.exceptions import ClientError
import urllib3
import os
from datetime import datetime

# Set up logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS client
s3 = boto3.client('s3')

# Configuration
BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
CHAT_ID = os.environ['TELEGRAM_CHAT_ID']
TRIGGER_BUCKET = 'hu-chatbot-schema'

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    if not BOT_TOKEN or not CHAT_ID:
        logger.error("Telegram bot token or chat ID is missing from environment variables")
        return

    if 'Records' in event and event['Records'][0].get('eventSource') == 'aws:s3':
        process_s3_event(event)
    else:
        logger.info("Received non-S3 event. Ignoring.")

def process_s3_event(event):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        logger.info(f"Processing file: {key} in bucket: {bucket}")
        
        new_content = get_file_content(bucket, key)
        if new_content is None:
            return
        
        versions = get_object_versions(bucket, key)
        if len(versions) > 1:
            previous_version = versions[1]['VersionId']
            old_content = get_file_content(bucket, key, previous_version)
            
            if old_content is None:
                return
            
            removed, delta, added = get_changes(old_content, new_content)
            if removed or delta or added:
                trigger_file_key = write_trigger_file(bucket, key, removed, delta, added)
                message = format_change_message(key, removed, delta, added, trigger_file_key)
                logger.info(message)
                send_telegram_message(message)
            else:
                message = f"No changes detected in file {key}"
                logger.info(message)
                send_telegram_message(message)
        else:
            trigger_file_key = write_trigger_file(bucket, key, {}, {}, new_content)
            message = f"New file uploaded: {key}\nTrigger file created: {trigger_file_key}"
            logger.info(message)
            send_telegram_message(message)
    
    except Exception as e:
        error_message = f"Error processing S3 event: {str(e)}"
        logger.error(error_message, exc_info=True)
        send_telegram_message(error_message)

def get_file_content(bucket, key, version_id=None):
    try:
        kwargs = {'Bucket': bucket, 'Key': key}
        if version_id:
            kwargs['VersionId'] = version_id
        
        response = s3.get_object(**kwargs)
        content = json.loads(response['Body'].read().decode('utf-8'))
        return content
    
    except ClientError as e:
        logger.error(f"Error accessing S3: {str(e)}")
        send_telegram_message(f"Error accessing file {key}: {str(e)}")
        return None

def get_object_versions(bucket, key):
    try:
        response = s3.list_object_versions(Bucket=bucket, Prefix=key)
        return response.get('Versions', [])
    except ClientError as e:
        logger.error(f"Error getting object versions: {str(e)}")
        send_telegram_message(f"Error getting versions for file {key}: {str(e)}")
        return []

def get_changes(old_content, new_content):
    removed = {}
    delta = {}
    added = {}
    
    for key, value in old_content.items():
        if key not in new_content:
            removed[key] = value
        elif new_content[key] != value:
            delta[key] = new_content[key]
    
    for key, value in new_content.items():
        if key not in old_content:
            added[key] = value
    
    return removed, delta, added

def write_trigger_file(source_bucket, source_key, removed, delta, added):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    trigger_key = f"triggers/{timestamp}_{source_bucket}_{source_key.replace('/', '_')}.json"
    trigger_content = json.dumps({
        'source_bucket': source_bucket,
        'source_key': source_key,
        'timestamp': timestamp,
        'removed': removed,
        'delta': delta,
        'added': added
    })
    try:
        s3.put_object(Bucket=TRIGGER_BUCKET, Key=trigger_key, Body=trigger_content)
        logger.info(f"Trigger file written to S3: {trigger_key}")
        return trigger_key
    except ClientError as e:
        logger.error(f"Error writing trigger file to S3: {str(e)}")
        send_telegram_message(f"Error creating trigger for file {source_key}: {str(e)}")
        return None

def format_change_message(key, removed, delta, added, trigger_file_key):
    message = f"Changes detected in file {key}\n"
    message += f"Removed entries: {len(removed)}\n"
    message += f"Modified entries: {len(delta)}\n"
    message += f"Added entries: {len(added)}\n"
    message += f"Trigger file created: {trigger_file_key}"
    return message

def send_telegram_message(message):
    http = urllib3.PoolManager()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = http.request("POST", url, fields=payload)
        if response.status != 200:
            logger.error(f"Failed to send Telegram message. Status: {response.status}, Response: {response.data.decode('utf-8')}")
        else:
            logger.info("Telegram message sent successfully")
    except Exception as e:
        logger.error(f"Error sending Telegram message: {str(e)}", exc_info=True)

# Uncomment to test Telegram messaging
# send_telegram_message("Test message from Lambda function")