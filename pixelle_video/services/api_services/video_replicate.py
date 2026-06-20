import os
import logging

try:
    from .image_processor import ImageProcessor
except ImportError:
    from image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ReplicateVideoClient:
    """Replicate video generation client.

    Model name is the Replicate ref prefixed with ``replicate:``, e.g.
    ``replicate:minimax/video-01``. Prefix is stripped before the call.

    ponytail: input is the common {prompt, image} shape (text/image-to-video).
    Models with other param names need a per-model adapter — add an elif here.
    """

    def __init__(self, api_token=None, local_proxy=None, timeout=600.0):
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

    def generate_video(self, prompt, image_path, save_path, model,
                       duration=5, video_ratio="16:9", **_ignored):
        ref = self._strip_prefix(model)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        inputs = {"prompt": prompt}
        opened = None
        if image_path:
            if str(image_path).startswith("http"):
                inputs["image"] = image_path
            elif os.path.exists(image_path):
                opened = open(image_path, "rb")
                inputs["image"] = opened

        try:
            logger.info(f"Replicate video: {ref}")
            output = self._client().run(ref, input=inputs)
        finally:
            if opened:
                opened.close()

        item = output[0] if isinstance(output, list) else output
        if hasattr(item, "read"):  # FileOutput
            with open(save_path, "wb") as f:
                f.write(item.read())
            return save_path
        url = str(item)
        if url.startswith("http") and self.image_processor.download_image(url, save_path):
            return save_path
        raise RuntimeError(f"Replicate video returned no usable output: {item!r}")
