"""
Kling AI video generation client
Text-to-video (text2video) / image-to-video (image2video) functionality based on the Kling API
Supported models: kling-v3, kling-v2-6, kling-v2-5-turbo
"""

import os
import io
import ssl
import time
import base64
import logging
from typing import Optional

import jwt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

logger = logging.getLogger(__name__)

# Kling API base address
KLING_BASE_URL = "https://api-beijing.klingai.com"


class _TLSAdapter(HTTPAdapter):
    """HTTPS adapter that enforces TLS 1.2, compatible with older LibreSSL versions"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_default_certs()
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _build_session(max_retries: int = 3) -> requests.Session:
    """Create a requests Session with a TLS adapter and automatic retries"""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = _TLSAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _proxy_dict(local_proxy: Optional[str]) -> dict:
    if not local_proxy:
        return {}
    return {"http": local_proxy, "https": local_proxy}


class KlingVideoClient:
    """
    Kling AI video generation client
    Uses JWT (HMAC-SHA256) authentication, calling the /v1/videos/text2video or /v1/videos/image2video endpoints
    """

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local_proxy: Optional[str] = None,
        token_ttl: int = 1800,
        poll_interval: int = 5,
        max_polls: int = 120,
    ) -> None:
        """
        Args:
            access_key: Kling API Access Key
            secret_key: Kling API Secret Key
            base_url:   Kling API base URL (defaults to the Beijing node)
            token_ttl:  JWT validity period (seconds), defaults to 30 minutes
            poll_interval: Polling interval (seconds)
            max_polls:  Maximum number of polls
        """
        self.access_key = access_key or os.getenv("KLING_ACCESS_KEY", "")
        self.secret_key = secret_key or os.getenv("KLING_SECRET_KEY", "")
        self.base_url = (base_url or os.getenv("KLING_BASE_URL", "")).rstrip("/") or KLING_BASE_URL
        self.local_proxy = local_proxy
        self.token_ttl = token_ttl
        self.poll_interval = poll_interval
        self.max_polls = max_polls

        if not self.access_key or not self.secret_key:
            logger.warning(
                "KlingVideoClient: KLING_ACCESS_KEY / KLING_SECRET_KEY not set, please check the configuration"
            )

        # Use a Session with enforced TLS 1.2 + automatic retries
        self._session = _build_session()

    # ─── JWT authentication ───

    def _generate_token(self) -> str:
        """
        Generate a JWT Token using the Access Key / Secret Key
        Algorithm: HS256
        Payload:
          - iss: Access Key
          - iat: issued-at time
          - exp: expiration time
          - nbf: not-before time
        """
        now = int(time.time())
        payload = {
            "iss": self.access_key,
            "iat": now,
            "exp": now + self.token_ttl,
            "nbf": now - 5,  # allow 5 seconds of clock skew
        }
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token

    def _auth_headers(self) -> dict:
        """Build request headers with JWT authentication"""
        token = self._generate_token()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    # ─── Image processing ───

    @staticmethod
    def _encode_image(image_path: str, quality: int = 85) -> str:
        """
        Encode a local image into a Base64 string
        Kling requires: do not add the data:image/xxx;base64, prefix, pass the Base64 string directly
        Image size ≤ 10MB, width/height ≥ 300px, aspect ratio 1:2.5 ~ 2.5:1
        """
        try:
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Image compression failed ({image_path}), using the original file: {e}")
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

    # ─── Create task ───

    def _submit_task(
        self,
        image_path: Optional[str],
        prompt: str = "",
        negative_prompt: str = "",
        model_name: str = "kling-v3",
        mode: str = "pro",
        duration: str = "5",
        cfg_scale: float = 0.5,
        sound: str = "",
        aspect_ratio: str = "16:9",
    ) -> str:
        """Submit a text-to-video or image-to-video task.

        Args:
            image_path: local image path; when empty, the text-to-video endpoint is called
            prompt: positive prompt (≤2500 characters)
            negative_prompt: negative prompt (≤2500 characters)
            model_name: Kling model name (kling-v3 / kling-v2-6 / kling-v2-5-turbo)
            mode: generation mode std (standard) / pro (high quality)
            duration: video duration, v3: "3"~"15", v2: "5" or "10"
            cfg_scale: degree of freedom [0,1], higher values follow the prompt more closely
            sound: whether to generate sound "on"/"off"
            aspect_ratio: aspect ratio for text-to-video

        Returns:
            task_id: task ID
        """
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

        # Determine the duration range based on the model series
        model_lower = model_name.lower()
        is_v3 = "v3" in model_lower or "video-o1" in model_lower
        is_v26 = any(tag in model_lower for tag in ("v2-6", "v2.6"))

        if is_v3:
            # The v3 series supports 3~15s
            clamped = str(min(max(int(duration), 3), 15))
        else:
            # The v2 series only supports 5 or 10
            clamped = "10" if int(duration) >= 8 else "5"

        body = {
            "model_name": model_name,
            "mode": mode,
            "duration": clamped,
        }
        endpoint = "image2video"
        if image_path:
            body["image"] = self._encode_image(image_path)
        else:
            endpoint = "text2video"
            body["aspect_ratio"] = aspect_ratio

        # sound parameter handling
        # v3 / v2-6: sound is on by default unless sound="off" is explicitly set
        # For v2-6, sound=on must be paired with pro mode
        # kling-v2-5-turbo does not support sound
        if is_v3 or is_v26:
            if sound == "off":
                body["sound"] = "off"
            else:
                body["sound"] = "on"
                # For v2-6, sound=on must be paired with pro mode; v3 has no such restriction
                if is_v26 and mode != "pro":
                    mode = "pro"
                    body["mode"] = mode
                    logger.info("KlingVideoClient: v2-6 sound=on requires pro mode, switched automatically")
        elif sound == "on":
            logger.warning(f"KlingVideoClient: model {model_name} does not support the sound parameter, ignored")

        if prompt:
            body["prompt"] = prompt
        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        url = f"{self.base_url}/v1/videos/{endpoint}"
        headers = self._auth_headers()

        logger.info(
            f"KlingVideoClient: submitting {endpoint} task model={model_name}, "
            f"mode={mode}, duration={clamped}s, sound={body.get('sound', 'off')}"
        )

        resp = self._session.post(
            url,
            json=body,
            headers=headers,
            timeout=300,
            proxies=_proxy_dict(self.local_proxy),
        )
        if not resp.ok:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            logger.error(f"KlingVideoClient: HTTP {resp.status_code}, response: {err_body}")
            resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(
                f"Kling API error: code={data.get('code')}, message={data.get('message')}"
            )

        task_id = data["data"]["task_id"]
        logger.info(f"KlingVideoClient: task submitted task_id={task_id}")
        return task_id

    # ─── Query task ───

    def _query_task(self, task_id: str, endpoint: str = "image2video") -> dict:
        """
        Query the status of a single task

        Returns:
            The data field from the API response
        """
        url = f"{self.base_url}/v1/videos/{endpoint}/{task_id}"
        headers = self._auth_headers()

        resp = self._session.get(
            url,
            headers=headers,
            timeout=30,
            proxies=_proxy_dict(self.local_proxy),
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise RuntimeError(
                f"Kling query API error: code={data.get('code')}, message={data.get('message')}"
            )

        return data["data"]

    # ─── Poll and wait ───

    def _poll_until_done(self, task_id: str, endpoint: str = "image2video") -> dict:
        """
        Poll the task until it completes or fails

        Returns:
            Task result data

        Raises:
            RuntimeError: task failed
            TimeoutError: exceeded the maximum number of polls
        """
        for attempt in range(self.max_polls):
            result = self._query_task(task_id, endpoint=endpoint)
            status = result.get("task_status", "")

            if status == "succeed":
                logger.info(f"KlingVideoClient: task completed task_id={task_id}")
                return result
            elif status == "failed":
                msg = result.get("task_status_msg", "Unknown error")
                raise RuntimeError(f"Kling video generation failed: {msg} (task_id={task_id})")
            else:
                # submitted / processing
                logger.debug(
                    f"KlingVideoClient: task in progress task_id={task_id}, "
                    f"status={status}, attempt={attempt + 1}/{self.max_polls}"
                )
                time.sleep(self.poll_interval)

        raise TimeoutError(f"Kling video generation timed out (task_id={task_id}, waited {self.max_polls * self.poll_interval}s)")

    # ─── Download video ───

    @staticmethod
    def _download_video(video_url: str, save_path: str) -> None:
        """Download the video from a URL to local disk"""
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        # Use a TLS-secure Session for the download as well
        dl_session = _build_session(max_retries=2)
        resp = dl_session.get(video_url, stream=True, timeout=600)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"KlingVideoClient: video saved: {save_path}")

    # ─── Main entry point ───

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str],
        save_path: str,
        model: str = "kling-v3",
        duration: int = 5,
        mode: str = "pro",
        cfg_scale: float = 0.5,
        negative_prompt: str = "",
        sound: str = "",
        aspect_ratio: str = "16:9",
    ) -> str:
        """
        Full text-to-video / image-to-video flow: submit task → poll and wait → download video

        Args:
            prompt: video description prompt
            image_path: local path of the input image; when empty, text-to-video is called
            save_path: output video save path
            model: Kling model name (kling-v3 / kling-v2-6 / kling-v2-5-turbo)
            duration: video duration (seconds), v3: 3~15, v2: 5 or 10
            mode: generation mode "std" (standard) or "pro" (high quality)
            cfg_scale: degree of freedom [0,1]
            negative_prompt: negative prompt
            sound: whether to generate sound "on"/"off"
            aspect_ratio: aspect ratio for text-to-video

        Returns:
            video_url: remote video URL
        """
        # 1. Submit the task
        endpoint = "image2video" if image_path else "text2video"
        task_id = self._submit_task(
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_name=model,
            mode=mode,
            duration=str(duration),
            cfg_scale=cfg_scale,
            sound=sound,
            aspect_ratio=aspect_ratio,
        )

        # 2. Poll and wait
        result = self._poll_until_done(task_id, endpoint=endpoint)

        # 3. Extract the video URL
        videos = result.get("task_result", {}).get("videos", [])
        if not videos:
            raise RuntimeError(f"Kling task succeeded but returned no video data (task_id={task_id})")

        video_url = videos[0].get("url", "")
        if not video_url:
            raise RuntimeError(f"Kling task succeeded but the video URL is empty (task_id={task_id})")

        # 4. Download to local disk
        self._download_video(video_url, save_path)

        return video_url


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ── Test parameters (modify as needed) ──
    IMAGE_PATH = "code/result/image/test_avail/test_input.png"
    OUTPUT_PATH = "code/result/video/test_avail/kling_test_output.mp4"
    PROMPT = ""
    MODEL = "kling-v3"         # kling-v3 / kling-v2-6 / kling-v2-5-turbo
    DURATION = 5               # v3: 3~15, v2: 5 or 10
    MODE = "pro"               # std or pro
    SOUND = ""                 # "" = auto-enable, "on", "off"

    print("=== Kling image-to-video test ===")
    ak = Config.KLING_ACCESS_KEY
    sk = Config.KLING_SECRET_KEY
    base_url = Config.KLING_BASE_URL
    if not ak or not sk:
        print("✗ KLING_ACCESS_KEY / KLING_SECRET_KEY not set, please check the .env configuration")
        sys.exit(1)

    if not os.path.exists(IMAGE_PATH):
        print(f"✗ Input image does not exist: {IMAGE_PATH}")
        sys.exit(1)

    print(f"  Access Key : {ak[:6]}***{ak[-4:]}")
    print(f"  Base URL   : {base_url}")
    print(f"  Input image : {IMAGE_PATH}")
    print(f"  Output path : {OUTPUT_PATH}")
    print(f"  Model       : {MODEL}")
    print(f"  Duration    : {DURATION}s")
    print(f"  Mode        : {MODE}")
    print(f"  Sound       : {SOUND or 'auto'}")
    if PROMPT:
        print(f"  Prompt      : {PROMPT[:80]}")
    print("-" * 40)

    try:
        client = KlingVideoClient(access_key=ak, secret_key=sk, base_url=base_url)
        print("✓ Client initialized successfully")

        start = time.time()
        video_url = client.generate_video(
            prompt=PROMPT,
            image_path=IMAGE_PATH,
            save_path=OUTPUT_PATH,
            model=MODEL,
            duration=DURATION,
            mode=MODE,
            sound=SOUND,
        )
        elapsed = time.time() - start

        print(f"✓ Video generation complete! Took {elapsed:.1f}s")
        print(f"  Remote URL : {video_url}")
        print(f"  Local file : {os.path.abspath(OUTPUT_PATH)}")
        print(f"  File size  : {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)
