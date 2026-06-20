import os
import time
import uuid
import logging

try:
    from .image_processor import ImageProcessor
except ImportError:
    from image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ReplicateImageClient:
    """Replicate image generation client.

    The model name is the Replicate ref with a ``replicate:`` prefix, e.g.
    ``replicate:black-forest-labs/flux-1.1-pro``. The prefix is stripped before
    the call; everything after it is passed to ``replicate.run`` verbatim.

    ponytail: input is the common {prompt, aspect_ratio, image} shape that
    Flux/SDXL-family models accept. Models with exotic param names need a
    per-model adapter — add an elif here if/when that happens.
    """

    def __init__(self, api_token=None, local_proxy=None, timeout=300.0):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        self.local_proxy = local_proxy
        self.timeout = timeout
        self.image_processor = ImageProcessor(local_proxy=local_proxy)

    def _client(self):
        if not self.api_token:
            raise RuntimeError(
                "REPLICATE_API_TOKEN not set. Configure api_providers.replicate.api_token "
                "to use replicate: models."
            )
        import replicate  # lazy: only import when a replicate model is used
        return replicate.Client(api_token=self.api_token)

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model.split("replicate:", 1)[-1].strip()

    def generate_image(self, prompt, model, save_dir, size="1024*1024",
                       video_ratio="16:9", image_paths=None, session_id=None):
        ref = self._strip_prefix(model)
        os.makedirs(save_dir, exist_ok=True)

        inputs = {"prompt": prompt, "aspect_ratio": video_ratio}
        opened = None
        if image_paths:
            first = image_paths[0]
            if first.startswith("http"):
                inputs["image"] = first
            elif os.path.exists(first):
                opened = open(first, "rb")
                inputs["image"] = opened

        try:
            logger.info(f"Replicate image: {ref}")
            output = self._client().run(ref, input=inputs)
        finally:
            if opened:
                opened.close()

        items = output if isinstance(output, list) else [output]
        paths = []
        for item in items:
            path = os.path.join(save_dir, f"replicate_{int(time.time())}_{uuid.uuid4().hex[:6]}.png")
            if self._save_item(item, path):
                paths.append(path)
        return paths

    def _save_item(self, item, path) -> bool:
        # FileOutput (replicate>=0.25) exposes .read(); otherwise it's a URL string.
        if hasattr(item, "read"):
            with open(path, "wb") as f:
                f.write(item.read())
            return True
        url = str(item)
        if url.startswith("http"):
            return self.image_processor.download_image(url, path)
        return False
