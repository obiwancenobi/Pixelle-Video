"""
Seedance video generation API client (ByteDance ARK)

"""

import os
import time
import logging
import requests
import base64
from typing import Optional

logger = logging.getLogger(__name__)

class SeedanceVideoClient:
    """
    Seedance video generation client (ByteDance ARK)
    Supports image-to-video generation using an async flow: submit task -> poll -> download
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local_proxy: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.base_url = (base_url or os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        self.local_proxy = local_proxy
        self.timeout = timeout

        if not self.api_key:
            logger.warning("SeedanceVideoClient: ARK_API_KEY is not set")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _proxies(self) -> Optional[dict]:
        if not self.local_proxy:
            return None
        return {"http": self.local_proxy, "https": self.local_proxy}

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str = "doubao-seedance-2-0-260128",
        duration: int = 5,
        **kwargs
    ) -> str:
        """
        Full image-to-video flow

        Args:
            prompt: Prompt
            image_path: Local path of the input image; falls back to text-to-video when empty
            save_path: Output video save path
            model: Model name
            duration: Video duration
        """
        if not self.api_key:
            raise RuntimeError("ARK_API_KEY not set.")

        # 1. Submit task
        task_id = self._submit_task(prompt, image_path, model, duration, **kwargs)

        # 2. Poll and wait
        video_url = self._poll_until_done(task_id)

        # 3. Download video
        self._download_video(video_url, save_path)

        return video_url

    def _submit_task(self, prompt: str, image_path: Optional[str], model: str, duration: int, **kwargs) -> str:
        # Endpoint path updated per the Seedance 2.0 docs
        url = f"{self.base_url}/contents/generations/tasks"

        # Build the content array
        content = []
        if prompt:
            content.append({
                "type": "text",
                "text": prompt
            })

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image does not exist: {image_path}")

            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            image_base64 = f"data:{mime};base64,{img_data}"

            # Image-to-video - first frame
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                },
                "role": "first_frame"
            })

        payload = {
            "model": model,
            "content": content,
            "duration": duration,
            "ratio": kwargs.get("ratio", "adaptive"),
            "resolution": kwargs.get("resolution", "720p")
        }

        # Merge other optional parameters (e.g. seed, watermark)
        for key in ["seed", "watermark", "generate_audio"]:
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        logger.info(f"SeedanceVideoClient: submitting task model={model}, duration={duration}s")
        resp = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            proxies=self._proxies(),
        )
        
        if not resp.ok:
            logger.error(f"Seedance submission failed: {resp.text}")
            resp.raise_for_status()

        data = resp.json()
        task_id = data.get("id")
        if not task_id:
            raise RuntimeError(f"Seedance API did not return a task ID: {data}")
            
        return task_id

    def _poll_until_done(self, task_id: str, max_polls: int = 120, interval: int = 5) -> str:
        # Query endpoint path updated to match
        url = f"{self.base_url}/contents/generations/tasks/{task_id}"
        
        for i in range(max_polls):
            resp = requests.get(url, headers=self._headers(), timeout=30, proxies=self._proxies())
            resp.raise_for_status()
            data = resp.json()
            
            status = data.get("status")
            if status == "succeeded":
                # Per the actual response body, the URL is at content.video_url or video_url
                video_url = data.get("content", {}).get("video_url") or data.get("video_url")
                if not video_url:
                    raise RuntimeError(f"Seedance task succeeded but did not return a video URL: {data}")
                return video_url
            elif status in ("failed", "expired"):
                error_msg = data.get("error", {}).get("message") or data.get("status_msg") or "Unknown error"
                raise RuntimeError(f"Seedance video generation {status}: {error_msg}")

            logger.debug(f"SeedanceVideoClient: task in progress {task_id}, status={status}, poll={i+1}")
            time.sleep(interval)

        raise TimeoutError(f"Seedance video generation timed out (task_id={task_id})")

    def _download_video(self, url: str, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        resp = requests.get(url, stream=True, timeout=120, proxies=self._proxies())
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"SeedanceVideoClient: video saved: {save_path}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ── Test parameters (modify as needed) ──
    # IMAGE_PATH = "code/result/image/test_avail/test_input.png"
    IMAGE_PATH = "code/result/image/test_avail/test_input_human.jpg"
    OUTPUT_PATH = "code/result/video/test_avail/seedance_test_output.mp4"
    PROMPT = "女生把财务报表交给男生，男生看到后喜极而泣"
    # MODELS = ["doubao-seedance-2-0-fast-260128", "doubao-seedance-2-0-260128"]
    MODELS = ["doubao-seedance-2-0-fast-260128"]
    DURATION = 5

    print("=== Seedance (ARK) image-to-video test ===")
    api_key = Config.ARK_API_KEY
    base_url = Config.ARK_BASE_URL

    if not api_key:
        print("✗ ARK_API_KEY is not set, please check the .env configuration")
        sys.exit(1)

    if not os.path.exists(IMAGE_PATH):
        print(f"✗ Input image does not exist: {IMAGE_PATH}")
        sys.exit(1)

    print(f"  API Key    : {api_key[:6]}***{api_key[-4:]}")
    print(f"  Base URL   : {base_url}")

    for model in MODELS:
        print("\n" + "="*40)
        print(f"  Input image: {IMAGE_PATH}")
        print(f"  Output path: {OUTPUT_PATH}")
        print(f"  Model      : {model}")
        print(f"  Duration   : {DURATION}s")
        if PROMPT:
            print(f"  Prompt     : {PROMPT[:80]}")

        try:
            client = SeedanceVideoClient(api_key=api_key, base_url=base_url)
            print("✓ Client initialized successfully")

            start = time.time()
            video_url = client.generate_video(
                prompt=PROMPT,
                image_path=IMAGE_PATH,
                save_path=OUTPUT_PATH,
                model=model,
                duration=DURATION,
            )
            elapsed = time.time() - start

            print(f"✓ Video generation complete! Took {elapsed:.1f}s")
            print(f"  Remote URL : {video_url}")
            print(f"  Local file : {os.path.abspath(OUTPUT_PATH)}")
            print(f"  File size  : {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"✗ Failed: {e}")
            sys.exit(1)
        break  # Only test the first model
