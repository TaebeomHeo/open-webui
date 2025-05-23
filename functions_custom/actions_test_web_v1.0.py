"""
title: Browser Agent with Logging - Fixed Version
author: bombbie
version: 0.2.1
required_open_webui_version: 0.3.9
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from dotenv import load_dotenv

load_dotenv()

import os
import requests
import asyncio
import subprocess  # 누락된 import 추가
import logging
import json
import argparse
import time
import platform
import signal


# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_web")


def is_chrome_debugging_available(port=9222, retries=3, retry_interval=2):
    """Chrome 디버깅 포트가 활성화되어 있는지 확인"""
    url = f"http://127.0.0.1:{port}/json/version"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"기존 Chrome 디버깅 세션 발견 (포트: {port})")
                return True
        except requests.exceptions.RequestException as e:
            print(f"연결 시도 {attempt + 1}/{retries} 실패: {e}")

        if attempt < retries - 1:
            time.sleep(retry_interval)

    return False


def kill_existing_chrome():
    """기존 Chrome 프로세스 종료"""
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["pkill", "-f", "Google Chrome"], check=False)
        elif system == "Windows":
            subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], check=False)
        else:  # Linux
            subprocess.run(["pkill", "-f", "chrome"], check=False)
        
        print("기존 Chrome 프로세스를 종료했습니다.")
        time.sleep(2)  # 프로세스 종료 대기
    except Exception as e:
        print(f"Chrome 프로세스 종료 중 오류: {e}")


def start_chrome_debugging(port=9222, kill_existing=True):
    """Chrome을 디버깅 모드로 시작"""
    system = platform.system()

    # 기존 Chrome 프로세스 종료 (선택사항)
    if kill_existing:
        kill_existing_chrome()

    # 사용자 데이터 디렉토리 설정
    temp_dir = os.path.join(os.path.expanduser("~"), ".chrome-debug-data")
    os.makedirs(temp_dir, exist_ok=True)

    if system == "Darwin":  # macOS
        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif system == "Windows":
        chrome_paths = [
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
        ]
        chrome_path = None
        for path in chrome_paths:
            if os.path.exists(path):
                chrome_path = path
                break
        if not chrome_path:
            raise Exception("Chrome 실행 파일을 찾을 수 없습니다.")
    else:  # Linux
        chrome_path = "google-chrome"

    # Chrome 실행 파일 존재 확인
    if system == "Darwin" and not os.path.exists(chrome_path):
        raise Exception(f"Chrome 실행 파일을 찾을 수 없습니다: {chrome_path}")

    cmd = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={temp_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-plugins",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "--remote-allow-origins=*",  # CORS 문제 해결
        "--disable-web-security",
        "--disable-features=VizDisplayCompositor"
    ]

    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if system != "Windows" else None
        )
        print(f"Chrome 디버깅 모드로 시작됨 (PID: {process.pid}, 포트: {port})")

        # 시작 대기 (더 긴 시간)
        for i in range(10):  # 최대 10초 대기
            if is_chrome_debugging_available(port):
                print(f"Chrome 디버깅 세션 준비 완료 (시도: {i+1})")
                return process
            print(f"Chrome 시작 대기 중... ({i+1}/10)")
            time.sleep(1)

        print("Chrome 디버깅 세션이 시작되었지만 연결할 수 없습니다.")
        return process
    except Exception as e:
        print(f"Chrome 시작 중 오류 발생: {e}")
        return None


async def run_agent(
    task,
    save_conversation_path="/Users/bombbie/CodeLearning/ai-browser/logs/conversation.json",
    use_vision=False,
):
    # 전역 변수로 선언
    browser = None
    context = None
    chrome_process = None

    print("initializing config...")
    load_dotenv()
    
    # 로그 디렉토리 설정
    log_dir = "/Users/bombbie/CodeLearning/browser-use-fork/my_examples/logs"
    os.makedirs(log_dir, exist_ok=True)
    os.environ["PLAYWRIGHT_LOG_DIR"] = log_dir

    # PlaywrightLogger 싱글톤 인스턴스 초기화
    try:
        import browser_use.playwright_logger
        from browser_use.playwright_logger import get_playwright_logger
        browser_use.playwright_logger._instance = None
        logger = get_playwright_logger(log_dir=log_dir)
    except Exception as e:
        print(f"PlaywrightLogger 초기화 오류: {e}")

    config = BrowserContextConfig(
        cookies_file="/Users/bombbie/CodeLearning/ai-browser/logs/cookies.json",
        wait_for_network_idle_page_load_time=3.0,
        browser_window_size={"width": 1280, "height": 1100},
        locale="en-US",
        highlight_elements=True,
        viewport_expansion=500,
        save_recording_path="/Users/bombbie/CodeLearning/ai-browser/logs",
    )

    print(f"BrowserContextConfig 생성 완료 => {config}")

    # 브라우저 초기화 시도
    debugging_port = 9222
    browser_created = False
    
    try:
        # 방법 1: 기존 Chrome 디버깅 세션 확인 및 연결
        if is_chrome_debugging_available(port=debugging_port):
            print("기존 Chrome 디버깅 세션에 연결 시도...")
            try:
                browser = Browser(
                    config=BrowserConfig(
                        headless=False,
                        cdp_url=f"http://127.0.0.1:{debugging_port}",
                    )
                )
                browser_created = True
                print("기존 Chrome 세션에 연결 성공")
            except Exception as e:
                print(f"기존 Chrome 세션 연결 실패: {e}")
        
        # 방법 2: 새 Chrome 디버깅 세션 시작
        if not browser_created:
            print("새 Chrome 디버깅 세션 시작...")
            chrome_process = start_chrome_debugging(port=debugging_port)
            
            if chrome_process and is_chrome_debugging_available(port=debugging_port):
                try:
                    browser = Browser(
                        config=BrowserConfig(
                            headless=False,
                            cdp_url=f"http://127.0.0.1:{debugging_port}",
                        )
                    )
                    browser_created = True
                    print("새 Chrome 디버깅 세션에 연결 성공")
                except Exception as e:
                    print(f"새 Chrome 디버깅 세션 연결 실패: {e}")
        
        # 방법 3: 직접 Chrome 인스턴스 생성 (fallback)
        if not browser_created:
            print("직접 Chrome 인스턴스 생성...")
            browser = Browser(
                config=BrowserConfig(
                    headless=False,
                    chrome_instance_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                )
            )
            browser_created = True
            print("직접 Chrome 인스턴스 생성 성공")

    except Exception as e:
        error_msg = f"Browser 초기화 최종 실패: {str(e)}"
        print(error_msg)
        return error_msg

    if not browser_created:
        error_msg = "모든 브라우저 연결 방법이 실패했습니다."
        print(error_msg)
        return error_msg

    print(f"Browser 생성 완료 => {browser}")

    # BrowserContext 생성
    try:
        context = BrowserContext(config=config, browser=browser)
        print("BrowserContext 생성 완료")
    except Exception as e:
        error_msg = f"BrowserContext 초기화 오류: {str(e)}"
        print(error_msg)
        return error_msg

    # LLM 및 Agent 생성
    try:
        llm = ChatOpenAI(model="gpt-4o")
        print("LLM 생성 완료")

        print("Agent 생성 중...")
        agent = Agent(
            browser_context=context,
            task=task,
            llm=llm,
            save_conversation_path=save_conversation_path,
            use_vision=use_vision,
        )
        print("Agent 생성 완료")
        print(f"실행할 작업: {task}")
        
        # Agent 실행
        result = await agent.run()
        
        # 결과 처리
        try:
            final_result = result.final_result() if hasattr(result, 'final_result') else str(result)
            is_done = result.is_done() if hasattr(result, 'is_done') else True
            errors = result.errors() if hasattr(result, 'errors') else []
            
            print("=== 실행 결과 ===")
            print(f"완료 여부: {is_done}")
            print(f"최종 결과: {final_result}")
            print(f"오류: {errors}")
            
        except Exception as e:
            print(f"결과 처리 중 오류: {e}")
            final_result = "결과 처리 중 오류가 발생했습니다."

    except Exception as e:
        error_msg = f"Agent 실행 오류: {str(e)}"
        print(error_msg)
        return error_msg
    
    # 로그 파일 읽기
    log_file_path = os.path.join(log_dir, "automation_log.log")
    log_content = ""

    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            print(f"로그 파일 읽기 성공: {log_file_path}")
        else:
            log_content = "로그 파일을 찾을 수 없습니다."
            print(f"로그 파일 없음: {log_file_path}")
    except Exception as e:
        log_content = f"로그 파일 읽기 오류: {str(e)}"
        print(f"로그 파일 읽기 오류: {e}")

    # final_result 문자열 변환
    if not isinstance(final_result, str):
        try:
            final_result = json.dumps(final_result, ensure_ascii=False, indent=2)
        except:
            final_result = str(final_result)

    # 결과와 로그 결합
    combined_result = f"{final_result}\n\n--- AUTOMATION LOG ---\n\n{log_content}"
    
    print("Agent 실행 완료")
    return combined_result


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
        last_message_content = body["messages"][-1]["content"]
        print(f"last message : {last_message_content}")

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

        # Agent 실행
        try:
            test_result = await run_agent(last_message_content)
        except Exception as e:
            test_result = f"Agent 실행 중 오류 발생: {str(e)}"
            print(test_result)

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
                    "data": {"description": "테스트 결과 표시 완료", "done": True},
                }
            )

        pass