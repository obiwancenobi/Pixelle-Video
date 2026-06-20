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

    Input shape is model-specific, so generic Pixelle inputs (image, audio,
    driving video) are mapped per model family in ``_build_inputs``. Anything
    unrecognized falls back to the common {prompt, image} text/image-to-video
    shape. A new model whose param names differ needs its own branch there.
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
        # Pass our timeout to httpx; the default read timeout (~60s) trips on
        # long-running models like mimic-motion before the prediction finishes.
        return replicate.Client(api_token=self.api_token, timeout=self.timeout)

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model.split("replicate:", 1)[-1].strip()

    @staticmethod
    def _resolve_ref(client, ref):
        """Pin bare ``owner/name`` refs to their latest version.

        Official models run by bare name, but community models (e.g. cjwbw/sadtalker)
        return 404 unless called as ``owner/name:version``. Already-pinned refs pass through.
        """
        if ":" in ref:
            return ref
        try:
            version = getattr(client.models.get(ref), "latest_version", None)
            if version is not None and getattr(version, "id", None):
                return f"{ref}:{version.id}"
        except Exception as exc:
            logger.warning(f"Could not resolve latest version for {ref}: {exc}")
        return ref

    @staticmethod
    def _file_arg(path, opened):
        """Coerce a path into a Replicate file input: pass URLs as-is, open local
        files (appending the handle to ``opened`` so the caller can close it)."""
        if not path:
            return None
        if str(path).startswith("http"):
            return path
        if os.path.exists(path):
            fh = open(path, "rb")
            opened.append(fh)
            return fh
        return None

    def _build_inputs(self, ref_lower, prompt, image, audio, motion_video, opened):
        """Map generic Pixelle inputs to a specific model's input schema.

        Add an elif for each new model whose param names differ from these.
        """
        if "sadtalker" in ref_lower:
            # talking head: portrait image + driving audio
            source = self._file_arg(image, opened)
            driven = self._file_arg(audio, opened)
            if not source or not driven:
                raise RuntimeError(
                    "SadTalker needs a source image and driving audio "
                    f"(got image={bool(source)}, audio={bool(driven)})."
                )
            return {"source_image": source, "driven_audio": driven}

        if "mimic-motion" in ref_lower or "mimicmotion" in ref_lower:
            # motion transfer: appearance image + driving pose video
            appearance = self._file_arg(image, opened)
            motion = self._file_arg(motion_video, opened)
            if not appearance or not motion:
                raise RuntimeError(
                    "MimicMotion needs an appearance image and a driving motion video "
                    f"(got image={bool(appearance)}, motion_video={bool(motion)})."
                )
            return {"appearance_image": appearance, "motion_video": motion}

        # generic text/image-to-video
        inputs = {"prompt": prompt}
        img = self._file_arg(image, opened)
        if img is not None:
            inputs["image"] = img
        return inputs

    def generate_video(self, prompt, image_path, save_path, model,
                       duration=5, video_ratio="16:9",
                       reference_image_path=None, reference_image_paths=None,
                       reference_audio_path=None, audio_path=None,
                       first_clip_path=None, reference_video_paths=None,
                       **_ignored):
        ref = self._strip_prefix(model)
        ref_lower = ref.lower()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Collapse the various generic input slots into image / audio / driving-video.
        image = image_path or reference_image_path or (reference_image_paths or [None])[0]
        audio = reference_audio_path or audio_path
        motion_video = first_clip_path or (reference_video_paths or [None])[0]

        opened = []
        inputs = self._build_inputs(ref_lower, prompt, image, audio, motion_video, opened)
        inputs = {k: v for k, v in inputs.items() if v is not None}

        try:
            client = self._client()
            run_ref = self._resolve_ref(client, ref)
            logger.info(f"Replicate video: {run_ref} inputs={list(inputs)}")
            output = client.run(run_ref, input=inputs)
        finally:
            for fh in opened:
                fh.close()

        item = output[0] if isinstance(output, list) else output
        if hasattr(item, "read"):  # FileOutput
            with open(save_path, "wb") as f:
                f.write(item.read())
            return save_path
        url = str(item)
        if url.startswith("http") and self.image_processor.download_image(url, save_path):
            return save_path
        raise RuntimeError(f"Replicate video returned no usable output: {item!r}")
