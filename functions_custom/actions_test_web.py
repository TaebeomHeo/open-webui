"""
title: Test last messages
author: bombbie
version: 0.1.0
required_open_webui_version: 0.3.9
"""
# 웹 브라우저 테스트 프로그램
# FIXME: 아래 경고
# TODO: 아래와 같은 경고 발생 in open-webui
# WARNI [browser_use.browser.context] Failed to force close browser context: asyncio.run() cannot be called from a running event loop
# /Users/bombbie/pythonenv3.11/lib/python3.11/site-packages/browser_use/browser/context.py:187: RuntimeWarning: coroutine 'BrowserContext.close' was never awaited
#   logger.warning(f'Failed to force close browser context: {e}')
# RuntimeWarning: Enable tracemalloc to get the object allocation traceback
# WARNI [browser_use.agent.service] No history or first screenshot to create GIF from
# FIXME: open-webui 에서 console로 로깅안됨... 
# TODO: open-webui 로깅방법을 준수해서 로깅하자구.  

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from dotenv import load_dotenv

load_dotenv()

import os
import requests
import asyncio

import pdb  # 디버깅을 위한 모듈 추가
import logging
import json
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_web")

config = BrowserContextConfig(
    cookies_file="/Users/bombbie/CodeLearning/ai-browser/logs/cookies.json",
    wait_for_network_idle_page_load_time=3.0,
    browser_window_size={"width": 1280, "height": 1100},
    locale="en-US",
    # user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    viewport_expansion=500,
    # allowed_domains=['google.com', 'wikipedia.org'],
    save_recording_path="/Users/bombbie/CodeLearning/ai-browser/logs",
)

browser = Browser()
context = BrowserContext(config=config, browser=browser)


llm = ChatOpenAI(model="gpt-4o")
# llm = ChatOpenAI( model="gpt-4o-mini")


async def run_agent(
    task,
    context,
    llm,
    save_conversation_path="/Users/bombbie/CodeLearning/ai-browser/logs/conversation.json",
    use_vision=False,
):
    logger.debug("Creating agent...")
    agent = Agent(
        browser_context=context,
        task=task,  # 함수 인자로 받은 task 사용
        llm=llm,
        save_conversation_path=save_conversation_path,  # save chat logs
        use_vision=use_vision,
    )
    logger.info("Agent created, starting run...")
    result = await agent.run()

    # AgentHistoryList 객체의 유용한 정보 추출 및 출력
    try:
        # 실행 기록에서 유용한 정보 추출
        urls = result.urls()
        screenshots = result.screenshots()
        action_names = result.action_names()
        extracted_content = result.extracted_content()
        errors = result.errors()
        model_actions = result.model_actions()
        final_result = result.final_result()
        is_done = result.is_done()
        has_errors = result.has_errors()
        model_thoughts = result.model_thoughts()
        action_results = result.action_results()

        # 포맷하여 출력
        logger.info("Execution Summary:")
        logger.info(f"Visited URLs: {urls}")
        logger.info(f"Screenshots: {screenshots}")
        logger.info(f"Executed Actions: {action_names}")
        logger.info(f"Extracted Content: {extracted_content}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Model Actions: {model_actions}")
        logger.info(f"Final Result: {final_result}")
        logger.info(f"Completed Successfully: {is_done}")
        logger.info(f"Errors Occurred: {has_errors}")
        logger.info(f"Model Thoughts: {model_thoughts}")
        logger.info(f"Action Results: {action_results}")

    except Exception as e:
        logger.error(f"Error processing result: {e}")
    finally:
        await browser.close()

    return final_result


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

        test_result = await run_agent(last_message_content, context=context, llm=llm)

        # 상태 업데이트 (선택사항)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "메시지 추가 중", "done": False},
                }
            )
            await asyncio.sleep(1)
            await __event_emitter__(
                {"type": "message", "data": {"content": f"\n------\n{test_result}"}}
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "마지막 메시지 표시 완료", "done": True},
                }
            )

        # return {"result": "마지막 메시지를 성공적으로 표시했습니다."}
        pass
