"""
Seedream image generation API client
ByteDance ARK - doubao-seedream-5-0-260128 model
"""

import os
import time
import logging
from typing import Optional, List, Dict
import httpx
from openai import OpenAI

# Model name mapping table (old name -> new name)
MODEL_NAME_MAP: Dict[str, str] = {
    # doubao-seedream-5-0 family
    "doubao-seedream-5-0": "doubao-seedream-5-0-260128",
    # doubao-seedream-4-5 family
    "doubao-seedream-4-5": "doubao-seedream-4-5-251128",
    # doubao-seedream-4-0 family
    "doubao-seedream-4-0": "doubao-seedream-4-0-250828",
}


def normalize_model_name(model: str) -> str:
    """
    Normalize the model name

    Args:
        model: the model name passed in

    Returns:
        the normalized model name
    """
    return MODEL_NAME_MAP.get(model, model)


class SeedreamClient:
    """
    Seedream image generation client (ByteDance ARK)
    Supports text-to-image generation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local_proxy: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        """
        Initialize the Seedream client

        Args:
            api_key: ARK API Key
            base_url: ARK API base URL
            timeout: HTTP request timeout in seconds
        """
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.base_url = base_url or "https://ark.cn-beijing.volces.com/api/v3"
        self.local_proxy = local_proxy
        self.timeout = timeout

        if not self.api_key:
            logging.warning(
                "SeedreamClient missing api_key. Set ARK_API_KEY."
            )

        client_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": timeout,
        }
        if self.local_proxy:
            client_kwargs["http_client"] = httpx.Client(proxy=self.local_proxy, timeout=timeout)

        self.client = OpenAI(**client_kwargs)

    def generate_image(
        self,
        prompt: str,
        session_id: str,
        model: str = "doubao-seedream-4-5-251128",
        size: str = "1920*1080",
        image_paths: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate images

        Args:
            prompt: the prompt text
            session_id: task or session ID, used to build the storage path
            model: model name
            size: resolution of the generated image, e.g. "1920*1080", "1024*1024"
            image_paths: list of reference image paths or URLs (image-to-image)
            **kwargs: other generation parameters

        Returns:
            list of generated image paths
        """
        if not self.api_key:
            raise RuntimeError("ARK_API_KEY not set.")

        # Normalize the model name (old name -> new name)
        model = normalize_model_name(model)

        # Handle resolution (Seedream requires at least 3686400 pixels)
        # Common 2K/4K resolutions
        size_map = {
            # 16:9
            "1920*1080": (1920, 1080),
            "2048*1080": (2048, 1080),  # 2K cinema
            "2560*1440": (2560, 1440),  # 2K QHD
            "3840*2160": (3840, 2160),  # 4K UHD
            "4096*2160": (4096, 2160),  # 4K cinema
            # 9:16
            "1080*1920": (1080, 1920),
            "1080*2048": (1080, 2048),
            "1440*2560": (1440, 2560),
            "2160*3840": (2160, 3840),
            "2160*4096": (2160, 4096),
            # 1:1
            "1024*1024": (1024, 1024),
            "2048*2048": (2048, 2048),  # 2K square
            # 4:3
            "1920*1440": (1920, 1440),
            "2560*1920": (2560, 1920),
            # 3:4
            "1440*1920": (1440, 1920),
            "1920*2560": (1920, 2560),
        }

        width, height = 1920, 1080  # default
        min_pixels = 3686400

        if size:
            parts = size.split("*")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                width, height = w, h

        # Ensure the minimum pixel requirement is met
        if width * height < min_pixels:
            # Look for a common resolution with the same aspect ratio
            aspect_ratio = width / height
            for (w, h) in size_map.values():
                if abs(w / h - aspect_ratio) < 0.01 and w * h >= min_pixels:
                    width, height = w, h
                    break
            else:
                # No suitable match found, scale up proportionally
                scale = (min_pixels / (width * height)) ** 0.5
                width = int(width * scale)
                height = int(height * scale)
                width = width if width % 2 == 0 else width + 1
                height = height if height % 2 == 0 else height + 1

        # Build extra_body
        extra_body = {
            "watermark": False,
            "sequential_image_generation": "disabled",
        }

        # Add other parameters
        if "seed" in kwargs:
            extra_body["seed"] = kwargs["seed"]
        if "quality" in kwargs:
            extra_body["quality"] = kwargs["quality"]
        if "style" in kwargs:
            extra_body["style"] = kwargs["style"]

        # Handle reference images (image-to-image)
        image_urls = []
        if image_paths and len(image_paths) > 0:
            # Process reference images: supports URLs and local files
            ref_images = []
            for p in image_paths:
                if p.startswith("http"):
                    ref_images.append(p)
                elif os.path.exists(p):
                    # Convert to a base64 URL
                    import base64
                    with open(p, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(p)[1].lower()
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    ref_images.append(f"data:{mime};base64,{img_data}")
            extra_body["image"] = ref_images

        # Call the API
        if image_paths and len(image_paths) > 0:
            # Image-to-image - image is passed in extra_body
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=f"{width}x{height}",
                response_format="url",
                extra_body=extra_body,
            )
        else:
            # Text-to-image
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=f"{width}x{height}",
                response_format="url",
                extra_body=extra_body,
            )

        # Download images to local storage
        generated_paths = []
        if response.data:
            for idx, img_data in enumerate(response.data):
                if img_data.url:
                    local_path = self._download_image(
                        img_data.url, session_id, idx
                    )
                    if local_path:
                        generated_paths.append(local_path)

        return generated_paths

    def _download_image(self, url: str, session_id: str, idx: int) -> Optional[str]:
        """Download an image from a URL to local storage"""
        import requests

        # Build the storage path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_dir = os.path.join(base_dir, "code", "result", "image", str(session_id))
        os.makedirs(result_dir, exist_ok=True)

        file_name = f"seedream_{int(time.time())}_{idx}.png"
        file_path = os.path.join(result_dir, file_name)

        try:
            proxies = {"http": self.local_proxy, "https": self.local_proxy} if self.local_proxy else None
            response = requests.get(url, timeout=self.timeout, proxies=proxies)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path
        except Exception as e:
            logging.error(f"Failed to download image from {url}: {e}")
            return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config  # load .env

    print("=== Seedream availability test ===")
    api_key = os.getenv("ARK_API_KEY", "")
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    if not api_key:
        print("✗ ARK_API_KEY not set, skipping")
        sys.exit(1)
    print(f"  API Key: {api_key[:6]}***{api_key[-4:]}")
    print(f"  Base URL: {base_url}")

    client = SeedreamClient(api_key=api_key, base_url=base_url)

    # === Test 1: Text-to-image ===
    prompt = "星际穿越，黑洞，黑洞里冲出一辆支离破碎的复古列车，视觉冲击力，电影大片，末日既视感"
    print(f"\n[Test 1: Text-to-image] Prompt: {prompt}")
    t0 = time.time()
    try:
        paths = client.generate_image(
            prompt=prompt,
            session_id="test_avail",
            model="doubao-seedream-5-0-260128",
            size="1920*1080",
        )
        elapsed = time.time() - t0
        if paths:
            print(f"✓ Generated {len(paths)} image(s) ({elapsed:.1f}s): {paths}")
        else:
            print(f"✗ Returned empty list ({elapsed:.1f}s)")
    except Exception as e:
        print(f"✗ Image generation failed: {e}")

    # === Test 2: Image-to-image ===
    # Requires an existing reference image path
    ref_image_path = "code/result/image/test_avail/test_input.png"
    if os.path.exists(ref_image_path):
        prompt_i2i = "将这只猫变成赛博朋克风格"
        print(f"\n[Test 2: Image-to-image] Prompt: {prompt_i2i}")
        print(f"  Reference image: {ref_image_path}")
        t0 = time.time()
        try:
            paths = client.generate_image(
                prompt=prompt_i2i,
                session_id="test_avail",
                model="doubao-seedream-5-0-260128",
                size="1920*1080",
                image_paths=[ref_image_path],
            )
            elapsed = time.time() - t0
            if paths:
                print(f"✓ Generated {len(paths)} image(s) ({elapsed:.1f}s): {paths}")
            else:
                print(f"✗ Returned empty list ({elapsed:.1f}s)")
        except Exception as e:
            print(f"✗ Image-to-image failed: {e}")
    else:
        print(f"\n[Test 2: Image-to-image] ✗ Reference image not found: {ref_image_path}")
        print("  Skipping image-to-image test, run the text-to-image test first to generate a reference image")
