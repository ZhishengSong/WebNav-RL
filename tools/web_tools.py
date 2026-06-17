from __future__ import annotations

from env.browser_env import BrowserEnv


class WebTools:
    def __init__(self, env: BrowserEnv) -> None:
        self.env = env

    def open_page(self, page_id: str) -> dict:
        return self.env.open_page(page_id)

    def click(self, element_id: str) -> dict:
        return self.env.click(element_id)

    def get_visible_text(self) -> dict:
        return self.env.get_visible_text()

    def submit_answer(self, answer: str) -> dict:
        return self.env.submit_answer(answer)
