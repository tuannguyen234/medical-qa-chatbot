import os
from datetime import datetime
from loguru import logger
import sys
import time
import functools
from pydantic import BaseModel, HttpUrl
from fastapi import Header, HTTPException
from typing import List, Dict
import re
import httpx
from dotenv import load_dotenv
from collections import defaultdict
load_dotenv()
import asyncio
async def sort_urls_by_post_index(urls):
    def extract_index(url):
        match = re.search(r'-posts-(\d+)\.png', url)
        return int(match.group(1)) if match else float('inf')

    return sorted(urls, key=extract_index)

def create_logger(module_name: str = None):
    """Create a logger with improved formatting for better readability"""
    
    # Base folder for logs
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)

    # Module-specific folder
    if module_name:
        log_folder = os.path.join(log_folder, module_name)
        os.makedirs(log_folder, exist_ok=True)
    
    debug_log = os.path.join(log_folder, "debug.log")
    application_log = os.path.join(log_folder, "application.log")
    error_log = os.path.join(log_folder, "error.log")
    # Remove default handler
    logger.remove()
    
    log_format_file = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    log_format_console = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Add custom handler with improved format (for dev)
    if os.getenv("ENV") != "production":
        logger.add(
            sys.stderr,
            format=log_format_console,
            level="INFO"
        )
    
    # Add file handler for persistent logs
    logger.add(
        debug_log, 
        rotation="10 MB",
        retention="1 week",
        format=log_format_file,
        level="DEBUG",
        backtrace=True,
        diagnose=True
    )
    logger.add(
        application_log,
        # filter=lambda record: record["level"].name not in ("ERROR", "CRITICAL"), # Ko nên lọc để giữ mạch thời gian
        rotation="50 MB",
        compression="zip",
        retention=None,
        format=log_format_file,
        enqueue=True,
        level="INFO"  # Chỉ ghi log từ INFO trở lên
    )
    logger.add(
        error_log,
        rotation="50 MB",
        compression="zip",
        retention=None,
        format=log_format_file,
        enqueue=True,
        level="ERROR"
    )
    
    return logger

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"⏱️ Hàm {func.__name__} thực hiện trong {end - start:.4f} giây")
        return result
    return wrapper

def async_timing_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        logger.info(f"⏱️ Hàm async {func.__name__} thực thi trong {end - start:.4f} giây")
        return result
    return wrapper


async def group_urls_by_section(urls: List[str]) -> Dict[str, List[str]]:
    """
    Nhóm các URL theo phần tên như 'details', 'life_events', 'work_and_education', v.v.
    Gộp 'about' và 'contact_and_basic_info' vào nhóm 'basic_info'.
    Đồng thời tạo thêm nhóm phụ 'about_and_contact_info' nếu có contact info.

    Args:
        urls (List[str]): Danh sách các URL hình ảnh Facebook.

    Returns:
        Dict[str, List[str]]: Dictionary với key là tên nhóm và value là danh sách URL tương ứng.
    """
    grouped = defaultdict(list)
    pattern = re.compile(r'about(?:_)?([a-z_]+)?-\d+\.png')

    # Các nhóm mặc định cần có
    expected_groups = ['basic_info', 'work_and_education', 'life_events', 'details', 'about_and_contact_info']

    for url in urls:
        match = pattern.search(url)
        if match:
            raw_group = match.group(1)
            if raw_group is None:
                group = 'basic_info'
                grouped[group].append(url)
            elif raw_group == 'contact_and_basic_info':
                grouped['basic_info'].append(url)
                grouped['about_and_contact_info'].append(url)  # Nhóm riêng biệt
            elif raw_group == 'work_and_education':
                grouped['work_and_education'].append(url)
            elif raw_group == 'life_events':
                grouped['life_events'].append(url)
            elif raw_group == 'details':
                grouped['details'].append(url)
            # Bỏ qua các nhóm khác không cần thiết

    # Đảm bảo luôn có tất cả các nhóm (dù là rỗng)
    for group in expected_groups:
        grouped.setdefault(group, [])

    return dict(grouped)

async def verify_client_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY_VLM_SERVICE"):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

async def format_api_output_vlm(
    dict_post_meta,
    dict_person_meta,
    dict_work2edu_meta,
    dict_lifeEvent_meta,
    dict_detail_meta,
    dict_contact_meta  # <-- thêm contact
):
    # Lấy status và error
    post_status = dict_post_meta.get("status", "fail")
    profile_status = dict_person_meta.get("status", "fail")
    workedu_status = dict_work2edu_meta.get("status", "fail")
    event_status = dict_lifeEvent_meta.get("status", "fail")
    detail_status = dict_detail_meta.get("status", "fail")
    contact_status = dict_contact_meta.get("status", "fail")  # <-- thêm contact

    # Xác định status tổng thể
    status = (
        "completed"
        if all(s == "success" for s in [post_status, profile_status, workedu_status, event_status, detail_status, contact_status])
        else "failed"
    )

    # error message chi tiết
    error_message = {
        "Profile": profile_status,
        "Post": post_status,
        "WorkEdu": workedu_status,
        "Events": event_status,
        "Detail": detail_status,
        "Contact": contact_status,
        "Profile error": dict_person_meta.get("error") if profile_status != "success" else None,
        "Post error": dict_post_meta.get("error") if post_status != "success" else None,
        "WorkEdu error": dict_work2edu_meta.get("error") if workedu_status != "success" else None,
        "Events error": dict_lifeEvent_meta.get("error") if event_status != "success" else None,
        "Detail error": dict_detail_meta.get("error") if detail_status != "success" else None,
        "Contact error": dict_contact_meta.get("error") if contact_status != "success" else None,
    }

    # Lấy thông tin từng phần
    posts = dict_post_meta.get("post_meta", [])
    profile_data = dict_person_meta.get("profile_meta", {})
    work_edu_data = dict_work2edu_meta.get("work_edu_meta", {})
    events_data = dict_lifeEvent_meta.get("life_event_meta", {}).get("events", {})
    detail_data = dict_detail_meta.get("detail_meta", {}).get("details", {})
    contact_data = dict_contact_meta.get("contact_meta", {})  # <-- trích xuất contact

    # Ghép tất cả thành dict phẳng
    data = {
        **{k: v for k, v in profile_data.items() if k not in ("status", "error")},
        "posts": posts,
        "work_education": work_edu_data,
        "life_events": events_data,
        "details": detail_data,
        "contact": contact_data,  # <-- thêm contact vào output
    }

    return {
        "data": data,
        "status": status,
        "errorMessage": str(error_message),
        "metadata": {}
    }

async def update_facebook_thirdparty(user_id: str, data: dict):
    API_BASE = os.getenv("API_BACKEND_BASE_URL")
    API_KEY = os.getenv("API_BACKEND_KEY_API")

    url = f"{API_BASE}/api/facebook/third-party/update/{user_id}"
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(url, json=data, headers=headers)
            if response.status_code == 200:
                logger.info("✅ Đã cập nhật thông tin thành công lên API Facebook third-party")
            else:
                logger.warning(f"⚠️ Không cập nhật được API Facebook third-party: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi gọi API Facebook third-party: {e}")


async def process_with_overlap(req, attribute_service, POST_PROMPT, Posts):
    urls = req.postUrls
    step = 9
    batch_size = 10
    all_dict_meta = []

    batches = []
    i = 0

    while i < len(urls):
        batch = urls[i:i + batch_size]
        if len(batch) < batch_size:
            if len(batch) < 5 and batches:
                batches[-1].extend(batch)
            else:
                batches.append(batch)
            break
        batches.append(batch)
        i += step

    # Xử lý tuần tự từng batch
    for batch in batches:
        sorted_batch = await sort_urls_by_post_index(list(set(batch)))
        try:
            result = await attribute_service.generate_info_extraction(
                image_urls=sorted_batch,
                prompt_input=POST_PROMPT,
                text_format=Posts,
                label_key='post_meta'
            )
            all_dict_meta+= result['post_meta']
        except Exception as e:
            print(f'❌ Lỗi xử lý batch: {batch}\n{e}')
    return {'post_meta': all_dict_meta, 'error': result['error'] , 'status': result['status']}
