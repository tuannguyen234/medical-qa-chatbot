import requests
import os
from dotenv import load_dotenv
from app.models.payloads import *

load_dotenv()  # Load environment variables from .env file

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env file")

        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
    
        self.emoji_scale = [
                                "😞",  # 0–2
                                "😐",  # 2–4
                                "😌",  # 4–6
                                "🙂",  # 6–8
                                "😃",  # 8–10
                            ]
    
    async def score_to_emoji(self, score: float) -> str:
        if score < 2:
            return self.emoji_scale[0]
        elif score < 4:
            return self.emoji_scale[1]
        elif score < 6:
            return self.emoji_scale[2]
        elif score < 8:
            return self.emoji_scale[3]
        else:
            return self.emoji_scale[4]

    async def format_message(self, data: dict) -> str:
        pr = TelegramOutputSending(**data)

        message = (
            f"<b>📦 Pull Request Score Report</b>\n"
            f"👤 <b>Committer:</b> <code>{pr.committer}</code>\n"
            f"📁 <b>Repo:</b> <code>{pr.repo}</code>\n"
            f"🔢 <b>PR Number:</b> <code>#{pr.pr_number}</code>\n\n"
            f"🔗 <b>Commit:</b> <code>{pr.commit}</code> 🛠️\n\n"
            f"<b>📊 Detail of each criterion:</b>\n"
        )

        for c in pr.criteria:
            emoji = await self.score_to_emoji(c.score)
            message += f"• <b>{c.name}:</b> <code>{c.score:.1f}</code> {emoji}\n"

        total_emoji = await self.score_to_emoji(pr.total_score)

        message += (
            f"\n<b>🌟 Total Score:</b> <code>{pr.total_score:.1f}/10</code> {total_emoji}\n\n"
            f"<b>🧾 Overall Comment:</b>\n<i>{pr.final_comment}</i>"  # No escaping needed for _
        )

        return message

    # async def format_message(self, data: dict) -> str:
    #     pr = TelegramOutputSending(**data)

    #     message = (
    #         f"*📦 Pull Request Score Report*\n"
    #         f"👤 *Committer:* `{pr.committer}`\n"
    #         f"📁 *Repo:* `{pr.repo}`\n"
    #         f"🔢 *PR Number:* `#{pr.pr_number}`\n\n"
    #         f"🔗 *Commit:* `{pr.commit}` 🛠️\n\n"
    #         f"*📊 Chi tiết điểm theo từng tiêu chí:*\n"
    #     )

    #     for c in pr.criteria:
    #         emoji = await self.score_to_emoji(c.score)
    #         message += f"• *{c.name}*: `{c.score:.1f}` {emoji}\n"

    #     total_emoji = await self.score_to_emoji(pr.total_score)

    #     message += (
    #         f"\n*🌟 Tổng điểm:* `{pr.total_score:.1f}/10` {total_emoji}\n\n"
    #         f"*🧾 Nhận xét tổng quát:*\n_{pr.final_comment}_"
    #     )


    #     return message

    async def send_message(self, message: str) -> None:
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }

        response = requests.post(self.api_url, data=payload)

        if response.status_code != 200:
            raise Exception(f"Failed to send message: {response.text}")
        else:
            print("Message sent successfully!")

# Example usage
if __name__ == "__main__":
    sample_data = {'committer': 'tuannguyen234', 
    'repo': 'medical-qa-chatbot', 
    'pr_number': 7, 
    'commit': 'Update test.py', 
    'total_score': 6.3, 
    'criteria': [{'name': 'Readability', 'score': 6.5, 'explanation': 'The llm_creator.py file is generally well-structured and documented, aiding readability, while test.py suffers from unclear function/variable names and duplicated code sections, reducing overall clarity.'}, 
    {'name': 'Test Coverage', 'score': 5.5, 'explanation': 'Test coverage is weak, especially in test.py where missing handling of control flow and incomplete functionality are noted, indicating insufficient tests or incomplete logic.'}, 
    {'name': 'Documentation', 'score': 5.5, 'explanation': 'llm_creator.py is well-documented with clear class and method descriptions, but test.py lacks adequate comments and has vague update descriptions, lowering overall documentation quality.'}, 
    {'name': 'Security', 'score': 6.5, 'explanation': 'Security is moderate; llm_creator.py handles API keys but lacks validation and relies on environment variables without checks, while test.py has basic input handling but no validation against harmful inputs.'}, 
    {'name': 'Performance', 'score': 6.5, 'explanation': "Performance is generally acceptable; llm_creator.py efficiently processes configurations, though it could reduce overhead with default parameters, and test.py's async usage is appropriate but may cause bottlenecks if input queries are not optimized."}, 
    {'name': 'Modularity', 'score': 7.0, 'explanation': 'llm_creator.py demonstrates good modular design with factory methods and separation of concerns, whereas test.py has duplicated logic and imports, which detracts from modularity and maintainability.'}], 
    'final_comment': 'This PR shows strengths in the llm_creator.py file with good design principles, adherence to coding standards, and clear documentation. However, there are notable areas for improvement, especially in test.py, which has critical syntax issues, poor documentation, and incomplete logic. Security could be enhanced by adding validation for critical configuration values and environment variables. Performance and modularity can be improved by consolidating duplicated code and refining async usage. Addressing these points will significantly increase the overall quality and robustness of the PR.'}

    import asyncio

    async def main():
        messenger = TelegramNotifier()
        message = await messenger.format_message(sample_data)
        await messenger.send_message(message)

    asyncio.run(main())
