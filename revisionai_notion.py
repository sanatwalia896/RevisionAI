# revisionai_notion.py

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class NotionPageLoader:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def search_all_pages(self) -> List[str]:
        url = "https://api.notion.com/v1/search"
        payload = {
            "filter": {"value": "page", "property": "object"},
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
        }
        response = requests.post(url, headers=self.headers, json=payload)
        data = response.json()
        return [
            result["id"]
            for result in data.get("results", [])
            if result["object"] == "page"
        ]

    def get_page_title(self, page_id: str) -> str:
        url = f"https://api.notion.com/v1/pages/{page_id}"
        response = requests.get(url, headers=self.headers)
        data = response.json()
        props = data.get("properties", {})
        title = "Untitled"
        for prop in props.values():
            if prop.get("type") == "title":
                title_data = prop["title"]
                if title_data:
                    title = "".join([t["plain_text"] for t in title_data])
                break
        return title

    def get_block_children(self, block_id: str) -> List[Dict[str, Any]]:
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        results = []
        while url:
            response = requests.get(url, headers=self.headers)
            data = response.json()
            results.extend(data.get("results", []))
            url = (
                f"{url}?start_cursor={data['next_cursor']}"
                if data.get("has_more")
                else None
            )
        return results

    def get_block_content(self, block: Dict[str, Any]) -> str:
        block_type = block.get("type", "unknown")
        block_data = block.get(block_type, {})
        if "rich_text" in block_data:
            return "".join([t.get("plain_text", "") for t in block_data["rich_text"]])
        elif block_type == "code":
            return f"[Code]\n" + "".join(
                [t.get("plain_text", "") for t in block_data.get("rich_text", [])]
            )
        elif block_type == "image":
            return "[Image]"
        else:
            return f"[{block_type} block]"

    def get_page_blocks(
        self, page_id: str, filter_last_edited_days: int = 0
    ) -> List[Dict[str, str]]:
        blocks = self.get_block_children(page_id)
        result = []
        cutoff = datetime.utcnow() - timedelta(days=filter_last_edited_days)
        for block in blocks:
            last_edited = block.get("last_edited_time")
            if last_edited:
                edited_dt = datetime.fromisoformat(last_edited.replace("Z", "+00:00"))
                if filter_last_edited_days > 0 and edited_dt < cutoff:
                    continue
            text = self.get_block_content(block)
            result.append({"text": text, "timestamp": last_edited})
        return result

    def get_all_page_contents(self) -> List[Dict[str, str]]:
        page_ids = self.search_all_pages()
        pages = []
        for pid in page_ids:
            title = self.get_page_title(pid)
            content_blocks = self.get_page_blocks(pid)
            content = "\n".join([b["text"] for b in content_blocks])
            pages.append({"id": pid, "title": title, "content": content})
        return pages
