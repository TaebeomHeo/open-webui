"""
title: Example Action
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
"""

from typing import Optional
from pydantic import BaseModel
import asyncio


class Action:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")
        response = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "메시지 작성",
                    "message": "추가할 메시지를 작성하세요",
                    "placeholder": "메시지 입력",
                },
            }
        )
        print(response)

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "메시지 추가 중", "done": False},
                }
            )
            await asyncio.sleep(1)
            await __event_emitter__({"type": "message", "data": {"content": response}})
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "메시지 추가 완료", "done": True},
                }
            )

        # return body

        pass
