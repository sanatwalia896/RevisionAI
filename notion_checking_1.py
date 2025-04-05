import requests
from typing import List, Dict, Any


class NotionAllPagesReader:
    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def search_all_pages(self) -> List[str]:
        """Search all pages shared with this integration."""
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
                title_text = prop["title"]
                if title_text:
                    title = "".join([t["plain_text"] for t in title_text])
                break
        return title

    def get_block_children(self, block_id: str) -> List[Dict[str, Any]]:
        """Recursively get children blocks."""
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        results = []
        while url:
            response = requests.get(url, headers=self.headers)
            data = response.json()
            results.extend(data.get("results", []))
            if data.get("has_more"):
                url = f"{url}?start_cursor={data['next_cursor']}"
            else:
                url = None
        return results

    def get_block_content(self, block: Dict[str, Any]) -> str:
        """Extracts readable content from a block."""
        block_type = block["type"]
        block_data = block.get(block_type, {})
        if "rich_text" in block_data:
            return "".join([t.get("plain_text", "") for t in block_data["rich_text"]])
        elif block_type == "image":
            return "[Image Block]"
        elif block_type == "code":
            code_text = "".join(
                [t.get("plain_text", "") for t in block_data.get("rich_text", [])]
            )
            return f"[Code Block]\n{code_text}"
        else:
            return f"[{block_type} block]"

    def read_page_content(self, page_id: str) -> str:
        """Reads and flattens the content of a page."""
        blocks = self.get_block_children(page_id)
        content = []
        for block in blocks:
            text = self.get_block_content(block)
            if text.strip():
                content.append(text)
        return "\n".join(content)

    def get_all_page_contents(self) -> List[Dict[str, str]]:
        """Reads all accessible pages and returns their title + content."""
        page_ids = self.search_all_pages()
        results = []
        for pid in page_ids:
            title = self.get_page_title(pid)
            content = self.read_page_content(pid)
            results.append({"id": pid, "title": title, "content": content})
        return results


if __name__ == "__main__":
    # 🔑 Replace this with your Notion integration token
    NOTION_TOKEN = "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"

    reader = NotionAllPagesReader(NOTION_TOKEN)

    print("📥 Reading all Notion pages shared with the integration...\n")

    all_pages = reader.get_all_page_contents()

    for i, page in enumerate(all_pages, 1):
        print(f"\n--- PAGE {i}: {page['title']} ---\n")
        print(page["content"])

        # ✅ Optional: Save to text files
        safe_title = page["title"].replace("/", "_").replace("\\", "_")
        with open(f"notion_page_{i}_{safe_title}.txt", "w", encoding="utf-8") as f:
            f.write(f"# {page['title']}\n\n{page['content']}")

    print(f"\n✅ Done. Read {len(all_pages)} pages.")
