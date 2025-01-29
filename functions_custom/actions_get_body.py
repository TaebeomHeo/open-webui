"""
title: Get Body contents Action
author: bombbie
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator

import os
import requests
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
        # print(self.valves, body)  # Prints the configuration options and the input body
        # IMPORTANT:
        # messages 리스트의 마지막 요소의 content에 접근
        last_message_content = body["messages"][-1]["content"]
        print(f"last message : {last_message_content}")

        # 팝업 창으로 마지막 메시지 내용 표시

        response = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "마지막 메시지",
                    "message": "확인용",
                    "placeholder": last_message_content,
                },
            }
        )
        print("사용자 응답:", response)

        # 상태 업데이트 (선택사항)
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
                    "data": {"description": "마지막 메시지 표시 완료", "done": True},
                }
            )
        pass
