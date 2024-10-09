import re
import hashlib
from datetime import datetime
import tempfile
import json
from unpack_funcs import unzip_file
from download_funcs import download_file

def parse_sitemap(content, pattern):
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    matches = re.findall(pattern, content, re.DOTALL)
    return [{'url': url.strip(), 'lastmod': lastmod.strip()} for url, lastmod in matches]


def filter_sitemap_entries(entries, exclude_extensions=None, exclude_patterns=None, include_patterns=None):
    filtered = entries
    if exclude_extensions:
        filtered = [entry for entry in filtered 
                    if not any(entry['url'].lower().endswith(ext.lower()) for ext in exclude_extensions)]
    if exclude_patterns:
        filtered = [entry for entry in filtered 
                    if not any(pattern in entry['url'] for pattern in exclude_patterns)]
    if include_patterns:
        filtered = [entry for entry in filtered 
                    if any(pattern in entry['url'] for pattern in include_patterns)]
    return filtered


def security_check_urls(entries, allowed_base_url):
    safe = [entry for entry in entries if entry['url'].startswith(allowed_base_url)]
    unsafe = [entry for entry in entries if not entry['url'].startswith(allowed_base_url)]
    return safe, unsafe

def convert_to_date(datetime_string, format="%Y-%m-%dT%H:%M:%S%z"):
    return datetime.strptime(datetime_string, format).strftime("%Y-%m-%d")

def create_matches_dict(entries):
    return {
        hashlib.md5(entry['url'].encode('utf-8')).hexdigest(): {
            'url': entry['url'],
            'last_updated': convert_to_date(entry['lastmod'])
        }
        for entry in entries
    }

def create_temp_json_file(data_dict, custom_filename=None):
    if custom_filename:
        filename = custom_filename if custom_filename.endswith('.json') else f"{custom_filename}.json"
    else:
        filename = f"sitemap_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
        json.dump(data_dict, tmp_file, indent=2)
        temp_file_path = tmp_file.name

    return temp_file_path, filename


def process_sitemap(url, exclude_extensions, exclude_patterns, include_patterns, allowed_base_url):
    content = unzip_file(download_file(url))
    pattern = r'<url>\s*<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>'
    
    sitemap_entries = parse_sitemap(content, pattern)
    filtered_entries = filter_sitemap_entries(sitemap_entries, exclude_extensions, exclude_patterns, include_patterns)
    safe_entries, unsafe_entries = security_check_urls(filtered_entries, allowed_base_url)
    
    data_dict = create_matches_dict(safe_entries)
    return data_dict, len(sitemap_entries), len(filtered_entries), len(safe_entries), len(unsafe_entries)
