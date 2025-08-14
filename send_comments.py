import requests
import os
from dotenv import load_dotenv
from app.utils.logger import logger

load_dotenv()
github_token = os.getenv("GITHUB_TOKEN_KEY")

class GitHubCommenter:
    def __init__(self, owner: str, repo: str, pr_number: int):
        self.headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json"
        }
        self.owner = owner
        self.repo = repo
        self.pr_number = pr_number
        self.issue_number = pr_number  # PR cũng là Issue

    def send_general_comment(self, message: str):
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues/{self.issue_number}/comments"
        data = {
            "body": message
        }
        response = requests.post(url, headers=self.headers, json=data)
        print("✅ General comment:", response.status_code, response.json())
        return response

    def send_file_comments(self, data: dict):
        """
        data = {
            "body": "Tổng thể PR ok",
            "event": "COMMENT",  # hoặc APPROVE, REQUEST_CHANGES
            "comments": [
                {
                    "path": "path/to/file.py",
                    "position": 5,
                    "body": "Nội dung cần chỉnh sửa"
                },
                ...
            ]
        }
        """
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/pulls/{self.pr_number}/reviews"
        response = requests.post(url, headers=self.headers, json=data)
        print("✅ Inline comments:", response.status_code, response.json())
        print(f"✅ Inline comments: {data}")
        return response


# ----------------------------
# 👇 Kiểm thử tại đây
# ----------------------------
if __name__ == "__main__":
    OWNER = "your-github-username-or-org"
    REPO = "your-repo-name"
    PR_NUMBER = 1  # thay bằng pull request number thực tế

    commenter = GitHubCommenter(owner=OWNER, repo=REPO, pr_number=PR_NUMBER)

    # Gửi comment tổng thể
    commenter.send_general_comment("💡 Review tổng thể: Code rất tốt, chỉ cần sửa vài dòng.")

    # Gửi comment theo dòng cụ thể trong file
    file_comments_data = {
        "body": "💬 Đây là nhận xét tổng quát kèm theo từng dòng",
        "event": "COMMENT",
        "comments": [
            {
                "path": "app_test.py",  # đường dẫn file tính từ root repo
                "position": 2,  # vị trí dòng theo diff, KHÔNG phải dòng gốc
                "body": "⚠️ Đề xuất đổi tên biến cho dễ hiểu hơn."
            },
            {
                "path": "app_test.py",
                "position": 5,
                "body": "✅ Cách sử dụng decorator này là hợp lý."
            }
        ]
    }

    commenter.send_file_comments(data=file_comments_data)
