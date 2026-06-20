import os
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
import logging


class ImageProcessor:
    """
    Image processing and upload utility class.
    Supports: image processing, splitting, stitching, and uploading to Alibaba Cloud OSS.
    """

    # Alibaba Cloud DashScope upload configuration
    UPLOAD_API_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
    
    def __init__(self,
                 image_path='',
                 api_key: str = "sk-bcab316d69a7414faa9dc29737019333",
                 model_name: str = "wan2.6-i2v-flash",
                 local_proxy: str | None = None):
        """
        Initialize the image processor.

        Args:
            image_path: Image file path (optional, used to process an existing image)
            api_key: DashScope API Key (used for uploads; can be read from the DASHSCOPE_API_KEY environment variable)
            model_name: Model name, defaults to wan2.6-i2v-flash
        """
        # Image processing section
        if image_path != '':
            self.image_path = image_path
            self.image = Image.open(image_path)
            self.image_np = np.array(self.image)
            self.width, self.height = self.image_np.shape[1], self.image_np.shape[0]
        else:
            self.image_path = None
            self.image = None
            self.image_np = None
            self.width = None
            self.height = None
        
        # Upload functionality section
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model_name = model_name
        self.local_proxy = local_proxy

    def _proxies(self):
        if not self.local_proxy:
            return None
        return {"http": self.local_proxy, "https": self.local_proxy}

    @staticmethod
    def check_column_white(column_pixels):
        """Check whether a column is almost entirely white."""
        is_almost_white = np.logical_or(column_pixels == 254, column_pixels == 255)
        white_pixels_ratio = np.mean(np.all(is_almost_white, axis=-1))
        return white_pixels_ratio >= 0.98  # At least 98% of pixels are white

    def find_white_section(self, start, end):
        """Find white sections within the specified range."""
        white_sections = []
        in_white_section = False
        start_index = 0

        for col in range(start, end):
            column_pixels = self.image_np[:, col, :]
            if self.check_column_white(column_pixels):
                if not in_white_section:
                    start_index = col
                    in_white_section = True
            else:
                if in_white_section:
                    white_sections.append((start_index, col))
                    in_white_section = False

        if in_white_section:
            white_sections.append((start_index, end))

        return white_sections

    def split_image(self):
        """Split the image down the middle into left and right halves."""
        start_col = self.width * 2 // 5
        end_col = self.width * 3 // 5
        white_sections = self.find_white_section(start_col, end_col)

        if white_sections:
            middle_section = white_sections[len(white_sections) // 2]
            mid_col = (middle_section[0] + middle_section[1]) // 2
        else:
            raise ValueError("No suitable white column found within the specified range")

        left_box = (0, 0, mid_col, self.height)
        right_box = (mid_col, 0, self.width, self.height)
        left_image = self.image.crop(left_box)
        right_image = self.image.crop(right_box)

        save_dir, filename = os.path.split(self.image_path)
        base, extension = os.path.splitext(filename)

        left_image_path = os.path.join(save_dir, base + '_front' + extension)
        right_image_path = os.path.join(save_dir, base + '_back' + extension)
        left_image.save(left_image_path)
        right_image.save(right_image_path)

        return left_image_path, right_image_path
    
    def stitch_images(self, image_paths, output_path):
        """Stitch multiple images together."""
        if not image_paths:
            raise ValueError("No image paths provided")
        sample_image = Image.open(image_paths[0])
        single_width, single_height = sample_image.size
        num_images = len(image_paths)
        total_desired_width = single_width
        total_current_width = single_width * num_images
        total_width_to_cut = max(0, total_current_width - total_desired_width)
        width_to_cut_per_image = total_width_to_cut // num_images
        stitched_image = Image.new('RGB', (total_desired_width, single_height), "white")
        current_x = 0
        
        for path in image_paths:
            image = Image.open(path)
            if width_to_cut_per_image > 0:
                left_margin = width_to_cut_per_image // 2
                right_margin = image.width - width_to_cut_per_image + left_margin
                image = image.crop((left_margin, 0, right_margin, image.height))
            stitched_image.paste(image, (current_x, 0))
            current_x += image.width
        
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        stitched_image.save(output_path)
        return output_path
    
    def download_image(self, image_url, save_path, max_retries=3):
        """
        Download an image with a retry mechanism and SSL error handling.

        Args:
            image_url: Image URL
            save_path: Local save path
            max_retries: Maximum number of retries
        """
        import time
        import urllib3
        
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    image_url, 
                    timeout=(10, 30),
                    stream=True,
                    verify=True,
                    proxies=self._proxies(),
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                )
                
                if response.status_code == 200:
                    with open(save_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                    print(f"✓ Image downloaded successfully: {save_path}")
                    return True
                else:
                    print(f"Download failed, status code: {response.status_code}")

            except requests.exceptions.SSLError as e:
                print(f"SSL error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print("Retrying download with SSL verification disabled...")
                    try:
                        response = requests.get(
                            image_url, 
                            timeout=(10, 30),
                            stream=True,
                            verify=False,
                            proxies=self._proxies(),
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                        )
                        if response.status_code == 200:
                            with open(save_path, 'wb') as file:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        file.write(chunk)
                            print(f"✓ Image downloaded successfully (SSL verification disabled): {save_path}")
                            return True
                    except Exception as fallback_error:
                        print(f"Still failed after disabling SSL verification: {fallback_error}")
                        raise

            except requests.exceptions.Timeout as e:
                print(f"Timeout error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    raise
                    
            except Exception as e:
                print(f"Download error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    raise
        
        return False

    def resize_image(self, image_path):
        """Resize the image (add blank space at the top)."""
        original_image = Image.open(image_path)
        width, height = original_image.size
        top_blank_height = height // 2
        final_height = height + top_blank_height
        final_width = int(final_height * 5 / 3)
        new_image = Image.new("RGB", (final_width, final_height), color="white")
        left = (final_width - width) // 2
        top = top_blank_height
        new_image.paste(original_image, (left, top))
        new_image.save(image_path)
        return image_path

    def has_black_borders(self, image_path, threshold=10, black_limit=20):
        """Check whether the image has black borders."""
        img = Image.open(image_path)
        pixels = img.load()
        width, height = img.size
        
        def is_black_pixel(pixel):
            return all(x <= black_limit for x in pixel)
        
        # Check the top and bottom borders
        for y in range(threshold):
            if all(is_black_pixel(pixels[x, y]) for x in range(width)):
                return True
            if all(is_black_pixel(pixels[x, height - 1 - y]) for x in range(width)):
                return True
        
        # Check the left and right borders
        for x in range(threshold):
            if all(is_black_pixel(pixels[x, y]) for y in range(height)):
                return True
            if all(is_black_pixel(pixels[width - 1 - x, y]) for y in range(height)):
                return True
        
        return False

    # ===== Image upload functionality =====

    def get_upload_policy(self):
        """
        Get the file upload credentials.

        Returns:
            policy_data: A dictionary containing the credentials required for uploading

        Raises:
            Exception: When obtaining the upload credentials fails
        """
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set; cannot use the image upload service")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "action": "getPolicy",
            "model": self.model_name
        }
        
        response = requests.get(
            self.UPLOAD_API_URL,
            headers=headers,
            params=params,
            proxies=self._proxies(),
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get upload policy: {response.text}")
        
        return response.json()['data']
    
    def upload_file_to_oss(self, policy_data: dict, file_path: str) -> str:
        """
        Upload the file to temporary OSS storage.

        Args:
            policy_data: Upload credential data
            file_path: Local file path

        Returns:
            oss_url: OSS URL (format: oss://...)

        Raises:
            Exception: When the upload fails
        """
        file_name = Path(file_path).name
        # Sanitize filename for upload to avoid issues with spaces/characters
        safe_file_name = "".join([c if c.isalnum() or c in ('-','_','.') else '_' for c in file_name])
        
        key = f"{policy_data['upload_dir']}/{safe_file_name}"
        
        with open(file_path, 'rb') as file:
            files = {
                'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
                'Signature': (None, policy_data['signature']),
                'policy': (None, policy_data['policy']),
                'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
                'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
                'key': (None, key),
                'success_action_status': (None, '200'),
                'file': (safe_file_name, file)
            }
            
            response = requests.post(
                policy_data['upload_host'],
                files=files,
                proxies=self._proxies(),
            )
            if response.status_code != 200:
                raise Exception(f"Failed to upload file: {response.text}")
        
        # Construct OSS URL correctly: oss://<bucket>/<key>
        # Extract bucket from upload_host (e.g., https://dashscope-instant.oss-cn-beijing.aliyuncs.com)
        upload_host = policy_data['upload_host']
        bucket_name = ""
        if '://' in upload_host:
            domain = upload_host.split('://')[1]
            bucket_name = domain.split('.')[0]
        
        if bucket_name:
            return f"oss://{bucket_name}/{key}"
        else:
            # Fallback if parsing fails (though unlikely for standard OSS hosts)
            # If the original code's assumption that key was self-sufficient was somehow valid, logic is here.
            # But normally, oss://<key> is wrong if key doesn't have bucket.
            return f"oss://{key}"
    
    def upload(self, file_path: str) -> str:
        """
        Upload a file to Alibaba Cloud OSS and get its URL (unified interface method).

        Args:
            file_path: Local file path

        Returns:
            oss_url: OSS URL, usable for 48 hours

        Raises:
            FileNotFoundError: When the file does not exist
            RuntimeError: When the API Key is not set
            Exception: When the upload fails
        """
        # Check whether the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set; cannot use the image upload service")

        # 1. Get the upload credentials (note: the upload credential endpoint is rate-limited)
        policy_data = self.get_upload_policy()

        # 2. Upload the file to OSS
        oss_url = self.upload_file_to_oss(policy_data, file_path)

        # 3. Compute the expiration time
        expire_time = datetime.now() + timedelta(hours=48)

        logging.info(f"File uploaded successfully: {file_path}")
        logging.info(f"  OSS URL: {oss_url}")
        logging.info(f"  Expiration time: {expire_time.strftime('%Y-%m-%d %H:%M:%S')} (48 hours)")
        
        return oss_url

    def collage_images(self, image_paths, output_path):
        """
        Collage functionality: stitch multiple images horizontally.
        Args:
            image_paths: List of image paths
            output_path: Output file path
        """
        if not image_paths:
            return None
        
        images = []
        for p in image_paths:
            try:
                img = Image.open(p)
                images.append(img)
            except Exception as e:
                logging.error(f"Cannot open image {p}: {e}")
        
        if not images:
            return None

        # Normalize height: resize the other images to match the first image's height
        base_height = images[0].height
        resized_images = []
        for img in images:
            if img.height != base_height:
                ratio = base_height / img.height
                new_width = int(img.width * ratio)
                resized_images.append(img.resize((new_width, base_height)))
            else:
                resized_images.append(img)
        
        total_width = sum(img.width for img in resized_images)
        new_im = Image.new('RGB', (total_width, base_height))
        
        x_offset = 0
        for img in resized_images:
            new_im.paste(img, (x_offset, 0))
            x_offset += img.width
            
        new_im.save(output_path)
        return output_path
