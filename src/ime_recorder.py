import httpx
from httpx_socks import SyncProxyTransport, AsyncProxyTransport
from collections import defaultdict
# from os import environ, getenv

import time
from typing import Optional

openai_chat_api = "https://api.openai.com/v1/chat/completions"
openai_emb_api = "https://api.openai.com/v1/embeddings"
openai_key = "sk-"
proxy = 'socks5://100.64.0.15:11081'
# proxy = 'socks5://100.64.0.42:7890'


headers = {
    "Authorization": f"Bearer {openai_key}"
}

histories = defaultdict(list)


def get_reply(messages):
    transport = SyncProxyTransport.from_url(proxy)
    try:
        with httpx.Client(transport=transport) as client:
            resp = client.post(
                openai_chat_api, json={
                    "model": "gpt-3.5-turbo", "messages": messages}, 
                    headers=headers,
                timeout=5 * 60)
            data = resp.json()
            if data.get('choices'):
                reply = data['choices'][0]['message']
                return reply['content']
    except Exception as e:
        import logging
        logging.exception(e)
        return "请求失败，请重试"