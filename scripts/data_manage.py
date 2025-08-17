
import os
import hashlib
from PIL import Image




# 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 PICTURES COUNT
# Set the path to your folder
folder_path = r'A:\\New folder\\project 1\\gender\\female'  # Update this to your folder path

# Correct variable name: image_extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Count images
image_count = sum(1 for file in os.listdir(folder_path) if file.lower().endswith(image_extensions))

print(f"Total number of images in the folder: {image_count}")













# 👨‍🔧👨‍🔧👨‍🔧 scraping images from the internet
# """
# Quick Pexels image-grabber.
# Downloads the first N photos that match the search term (default: "girl").
# """

# import pathlib, requests, time
# from io import BytesIO
# from urllib.parse import urlparse
# from PIL import Image
# from tqdm import tqdm

# # ──------------------------  CONFIG  ---------------------------- #
# API_KEY  = "63kLpPfGANYl5tWioEO98qb5tjHNk7vAzb7NLzxYu5C9bdEnXOToo9jh"  # <-- your key
# QUERY    = "men with makeup"          # search keyword
# N_PHOTOS =   4000          # how many images to fetch
# SIZE     = "large"         # Pexels sizes: original | large | medium | small | tiny
# OUT_DIR  = pathlib.Path("pexels_" + QUERY.replace(" ", "_"))
# OUT_DIR.mkdir(exist_ok=True)
# # ───────────────────────────────────────────────────────────────── #

# HEADERS = {"Authorization": API_KEY}
# BASE    = "https://api.pexels.com/v1/search"

# def filename_from_url(url: str) -> str:
#     """Extract clean filename from a Pexels src URL."""
#     return pathlib.Path(urlparse(url).path).name

# def download_and_save(url: str, dst: pathlib.Path) -> None:
#     """Download an image and save it as JPEG."""
#     r = requests.get(url, timeout=30)
#     r.raise_for_status()
#     img = Image.open(BytesIO(r.content)).convert("RGB")
#     img.save(dst, "JPEG", quality=94)

# def pexels_search(query: str, per_page: int = 80):
#     """Yield photo dicts until N_PHOTOS reached or no more results."""
#     grabbed, page = 0, 1
#     while grabbed < N_PHOTOS:
#         resp = requests.get(
#             f"{BASE}?query={query}&per_page={per_page}&page={page}",
#             headers=HEADERS,
#             timeout=15,
#         )
#         resp.raise_for_status()
#         photos = resp.json().get("photos", [])
#         if not photos:
#             break
#         for photo in photos:
#             if grabbed >= N_PHOTOS:
#                 return
#             yield photo
#             grabbed += 1
#         page += 1
#         time.sleep(0.2)  # gentle on the API

# if __name__ == "__main__":
#     print(f"Fetching up to {N_PHOTOS} “{QUERY}” photos …")
#     for p in tqdm(pexels_search(QUERY), total=N_PHOTOS):
#         src    = p["src"][SIZE]
#         dst_fn = filename_from_url(src)
#         dst    = OUT_DIR / dst_fn
#         try:
#             download_and_save(src, dst)
#         except Exception as e:
#             print(f"⚠️  Skipped {dst_fn}: {e}")
#     print(f"✅ Done! Images saved in “{OUT_DIR}”.")














# 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 delete duplicate images in sinlge folder 
# import os
# import hashlib
# from pathlib import Path

# # Set your folder path\
# folder_path = r'A:\\CODING FILES\\pexels_Buddhist_monk'  # <-- change this

# # Allowed image formats
# image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

# # Dictionary to store file hashes
# hash_dict = {}
# deleted_count = 0

# # Go through all files in the folder
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#     ext = Path(filename).suffix.lower()
    
#     if ext in image_extensions and os.path.isfile(file_path):
#         # Compute MD5 hash of file contents
#         with open(file_path, 'rb') as f:
#             file_hash = hashlib.md5(f.read()).hexdigest()

#         if file_hash in hash_dict:
#             # Duplicate found, delete it
#             os.remove(file_path)
#             print(f"🗑️ Deleted duplicate: {filename}")
#             deleted_count += 1
#         else:
#             hash_dict[file_hash] = filename

# print(f"\n✅ Total duplicates deleted: {deleted_count}")




# # 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 pictures count
# # set the path to your folder 
# import os
# import hashlib
# from PIL import Image
# folder_path = r"A:\\New folder\\project 1\\gender\\male"

# # correct varibale name: image_extensions
# image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# # Count images
# image_count = sum(1 for file in os.listdir(folder_path) if file.lower().endswith(image_extensions))
# print(f"toral number of images in the folder are: {image_count}")
















# 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 copy and paste any number of images from one folder to another
# import os
# import os
# import shutil
# import random

# # === SETTINGS (EDIT THESE) ===
# SOURCE_FOLDER = r"A:\\New folder\\project 1\\gender\\Male"        # Change this
# DESTINATION_FOLDER = r"A:\\New folder\\project 1\\gender\\male 1"     # Change this
# NUM_IMAGES_TO_COPY = 57000                                  # Change this

# def copy_images(source_folder, destination_folder, num_images):
#     os.makedirs(destination_folder, exist_ok=True)

#     valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
#     image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]

#     if not image_files:
#         print("No images found in the source folder.")
#         return

#     if num_images > len(image_files):
#         print(f"Only {len(image_files)} images found. Copying all.")
#         num_images = len(image_files)

#     selected_images = random.sample(image_files, num_images)

#     for image in selected_images:
#         src_path = os.path.join(source_folder, image)
#         dest_path = os.path.join(destination_folder, image)
#         shutil.copy2(src_path, dest_path)

#     print(f"Copied {num_images} image(s) from '{source_folder}' to '{destination_folder}'.")

# # === RUN ===
# copy_images(SOURCE_FOLDER, DESTINATION_FOLDER, NUM_IMAGES_TO_COPY)



















# 👨‍🔧👨‍🔧👨‍🔧 renaming the images 
# import os
# import sys
# from tqdm import tqdm # For a nice progress bar

# def rename_images_in_folder():
#     """
#     Renames all image files in a specified folder with a new prefix
#     and sequential numbering, preserving their original file extensions.
#     """
#     print("Welcome to the Image Renamer!")
#     print("This script will rename all image files in a chosen folder.")

#     # --- USER INPUT SECTION: MODIFY THESE VARIABLES DIRECTLY IN THE CODE ---
#     # 1. Enter the FULL path to the folder containing your images.
#     #    Example for Windows: r"C:\Users\YourName\Documents\MyImages"
#     #    Example for macOS/Linux: "/home/youruser/pictures/humans"
#     FOLDER_PATH = r"A:\\CODING FILES\\pexels_Buddhist_monk" # <--- SET YOUR FOLDER PATH HERE

#     # 2. Enter the desired prefix for the new filenames (e.g., 'human_pic').
#     #    If left empty (e.g., NEW_PREFIX = ""), files will be named '1.jpg', '2.png', etc.
#     NEW_PREFIX = "lll_monks" # <--- SET YOUR DESIRED PREFIX HERE

#     # 3. Confirm if you want to proceed with renaming. Set to True to proceed, False to cancel.
#     CONFIRM_RENAME = True # <--- SET TO True TO PROCEED WITH RENAMING
#     # -----------------------------------------------------------------------

#     folder_path = FOLDER_PATH
#     prefix = NEW_PREFIX
#     confirm = "yes" if CONFIRM_RENAME else "no"

#     # Validate folder path (no longer asks in loop, just checks once)
#     if not os.path.isdir(folder_path):
#         print(f"\n❌ Error: The provided FOLDER_PATH '{folder_path}' is not a valid directory.")
#         print("Please edit the 'FOLDER_PATH' variable in the code to a correct path.")
#         return

#     if not prefix:
#         print("\n❗ Warning: No prefix entered in 'NEW_PREFIX'. Files will be named '1.jpg', '2.png', etc.")
#         if not CONFIRM_RENAME: # If user didn't explicitly confirm, treat as cancellation
#             print("🚫 Renaming cancelled because 'CONFIRM_RENAME' is False and no prefix was set.")
#             return

#     # Define common image extensions (case-insensitive)
#     image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

#     # --- 3. Get list of image files ---
#     image_files = []
#     for filename in os.listdir(folder_path):
#         # Check if the file has an image extension
#         if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(image_extensions):
#             image_files.append(filename)

#     if not image_files:
#         print(f"\n😔 No image files found in '{folder_path}' with common extensions ({', '.join(image_extensions)}).")
#         print("Please ensure your images are in the folder and have standard extensions.")
#         return

#     # Sort files to ensure consistent numbering (e.g., alphabetical order)
#     image_files.sort()

#     print(f"\nFound {len(image_files)} image files in '{folder_path}'.")
#     print(f"They will be renamed with the prefix '{prefix}' (or sequentially if no prefix given).")
#     print("Example: Your first image will become '{}{}{}'".format(prefix or '', 1, os.path.splitext(image_files[0])[1]))

#     # --- Confirmation before proceeding ---
#     if confirm != 'yes': # This now relies on CONFIRM_RENAME variable
#         print("🚫 Renaming cancelled by user (CONFIRM_RENAME set to False).")
#         return

#     # --- 4. Rename the files ---
#     renamed_count = 0
#     # Use tqdm for a progress bar
#     for i, old_filename in enumerate(tqdm(image_files, desc="Renaming Images", unit="file")):
#         try:
#             # Get the original file extension
#             # os.path.splitext returns a tuple: (root, ext)
#             # e.g., ('my_image', '.jpg')
#             _, original_extension = os.path.splitext(old_filename)

#             # Construct the new filename
#             # The number 'i + 1' ensures numbering starts from 1
#             new_filename = f"{prefix}{i + 1}{original_extension}"

#             old_filepath = os.path.join(folder_path, old_filename)
#             new_filepath = os.path.join(folder_path, new_filename)

#             # Handle cases where the new filename might already exist (unlikely with sequential naming)
#             # Or if original file names contain the prefix and number.
#             if os.path.exists(new_filepath):
#                 print(f"❗ Warning: New filename '{new_filename}' already exists. Skipping '{old_filename}'.")
#                 continue

#             os.rename(old_filepath, new_filepath)
#             renamed_count += 1
#         except Exception as e:
#             print(f"\n❌ Error renaming '{old_filename}': {e}")
#             # Decide if you want to continue or stop on error
#             continue # Continue to the next file even if one fails

#     print(f"\n🎉 Renaming complete! Successfully renamed {renamed_count} out of {len(image_files)} image files.")
#     print(f"All renamed images are now in: {folder_path}")

# # Run the function when the script is executed
# if __name__ == "__main__":
#     rename_images_in_folder()















# 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 doubling the images
# import os
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm

# # # Paths
# source_folder = r"A:\\New folder\\project 1\\gender\\female"
# output_folder = r"A:\\New folder\\project 1\\gender\\female"
# os.makedirs(output_folder, exist_ok=True)

# # Augmentations
# augment = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=1.0),  # Flip
#     transforms.RandomRotation(25),           # Rotate
#     transforms.ColorJitter(brightness=0.3),  # Change brightness
# ])

# # Apply augmentations
# for file in tqdm(os.listdir(source_folder)):
#     if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#         path = os.path.join(source_folder, file)
#         img = Image.open(path).convert("RGB")

#         # Save original
#         img.save(os.path.join(output_folder, file))

#         # Save augmented
#         augmented = augment(img)
#         new_name = f"aug_{file}"
#         augmented.save(os.path.join(output_folder, new_name))

# print("✅ Done: Images doubled with augmentation.")












# 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 FOR DELETING DUPLICATE IMAGES
# import os
# import hashlib
# import warnings
# from PIL import Image

# # Suppress palette image warning
# warnings.filterwarnings("ignore", category=UserWarning)

# def get_image_hash(image_path):
#     try:
#         with Image.open(image_path) as img:
#             img.load()  # Force-load image data into memory
#             if img.mode in ("P", "RGBA"):
#                 img = img.convert("RGB")
#             else:
#                 img = img.convert("RGB")
#             return hashlib.md5(img.tobytes()).hexdigest()
#     except Exception as e:
#         print(f"⚠️ Failed to process {image_path}: {e}")
#         try:
#             os.remove(image_path)
#             print(f"🗑️ Deleted corrupted image: {os.path.basename(image_path)}")
#         except Exception as del_error:
#             print(f"❌ Failed to delete corrupted image: {del_error}")
#         return None

# def collect_hashes(folder_path):
#     hashes = {}
#     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     print(f"📁 Total number of images in the folder: {len(image_files)}")

#     for file in image_files:
#         path = os.path.join(folder_path, file)
#         img_hash = get_image_hash(path)
#         if img_hash:
#             hashes[img_hash] = path
#     return hashes

# # 🔁 Update with your actual folder paths
# train_folder = r"A:\\CODING FILES\\asdasdf"
# test_folder = r"A:\\CODING FILES\\pexels_Buddhist_monk"

# # Collect hashes
# train_hashes = collect_hashes(train_folder)
# test_hashes = collect_hashes(test_folder)

# # Find duplicates
# duplicate_hashes = set(train_hashes) & set(test_hashes)

# # Delete duplicates from test
# deleted_count = 0
# for h in duplicate_hashes:
#     duplicate_path = test_hashes[h]
#     try:
#         os.remove(duplicate_path)
#         print(f"🗑️ Deleted duplicate from test: {os.path.basename(duplicate_path)}")
#         deleted_count += 1
#     except Exception as e:
#         print(f"❌ Failed to delete {duplicate_path}: {e}")

# print(f"\n✅ Total duplicates deleted from test folder: {deleted_count}")



# Optimized Unsplash image-grabber (faster)
# Requires: pip install requests pillow tqdm

import pathlib, requests, time
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ── CONFIG ───────────────────────────────────────── #
ACCESS_KEY = "-h5JctTVx0QwjS3UlRZ3qTJADjtWMdNVfka3cvcKW6o"  # <-- your Unsplash Access Key
QUERY      = "male model with eyeliner lipstick dramatic lighting"     # search keyword
N_PHOTOS   = 500                  # how many images to fetch
THREADS    = 5                     # number of parallel downloads
SIZE_KEY   = "regular"             # use "regular" for faster downloads
OUT_DIR    = pathlib.Path("unsplash_" + QUERY.replace(" ", "_"))
OUT_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────── #

BASE_URL = "https://api.unsplash.com/search/photos"

def download_and_save(photo):
    """Download an image and save as JPEG."""
    try:
        url = photo["urls"][SIZE_KEY]
        dst = OUT_DIR / (photo["id"] + ".jpg")
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(dst, "JPEG", quality=90)
    except Exception as e:
        print(f"⚠️ Skipped {photo['id']}: {e}")

def unsplash_search(query: str, per_page: int = 30):
    """Yield photo dicts until N_PHOTOS reached or no more results."""
    grabbed, page = 0, 1
    while grabbed < N_PHOTOS:
        resp = requests.get(
            BASE_URL,
            params={
                "query": query,
                "page": page,
                "per_page": per_page,
                "client_id": ACCESS_KEY
            },
            timeout=10
        )
        resp.raise_for_status()
        photos = resp.json().get("results", [])
        if not photos:
            break
        for photo in photos:
            if grabbed >= N_PHOTOS:
                return
            yield photo
            grabbed += 1
        page += 1
        time.sleep(0.1)  # gentle on the API

if __name__ == "__main__":
    print(f"Fetching up to {N_PHOTOS} “{QUERY}” photos …")
    photos = list(unsplash_search(QUERY))
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        list(tqdm(executor.map(download_and_save, photos), total=len(photos)))
    print(f"✅ Done! Images saved in “{OUT_DIR}”.")

