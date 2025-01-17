import os
import csv
import time
from openai import OpenAI

client = OpenAI(api_key="")
import requests

NUM_IMAGES_PER_CLASS = 100
BATCH_SIZE = 5
IMAGE_SIZE = "256x256"
OUTPUT_DIR = "flowers"
LABELS_CSV = "labels.csv"

# Prompts for each emotion
EMOTION_PROMPTS = {
    "sad": (
        "a sad flower, drawn by hand, black and white, "
        "with a visibly sad expression"
    ),
    "happy": (
    " a happy flower, drawn by hand, black and white, "
        "with a visibly sad expression"
    ),
    "angry": (
        " a angry flower, drawn by hand, black and white, "
        "with a visibly sad expression"
    ),
}


def generate_flowers(emotion, prompt, num_images, batch_size, image_size, csv_writer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_batches = num_images // batch_size

    print(f"Generating {num_images} '{emotion}' flowers in {num_batches} batches...")
    for batch_idx in range(num_batches):
        print(f"  Requesting batch {batch_idx + 1}/{num_batches}...")


        while True:
            try:
                response = client.images.generate(prompt=prompt,
                n=batch_size,
                size=image_size)
                break
            except Exception as e:

                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    print("    Rate limit encountered. Waiting 60 seconds before retry...")
                    time.sleep(60)
                else:
                    raise e


        for i, data_item in enumerate(response.data):
            image_url = data_item.url
            image_data = requests.get(image_url).content

            image_number = batch_idx * batch_size + i + 1
            filename = f"{emotion}_flower_{image_number}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(image_data)


            csv_writer.writerow([filename, emotion])
            print(f"    Saved: {filepath}")

        time.sleep(60)

    print(f"Done generating {num_images} '{emotion}' flowers!\n")


def main():
    file_exists = os.path.isfile(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)


        if not file_exists or os.path.getsize(LABELS_CSV) == 0:
            writer.writerow(["filename", "label"])

        for emotion, prompt in EMOTION_PROMPTS.items():
            generate_flowers(
                emotion=emotion,
                prompt=prompt,
                num_images=NUM_IMAGES_PER_CLASS,
                batch_size=BATCH_SIZE,
                image_size=IMAGE_SIZE,
                csv_writer=writer
            )

    print(f"images have been generated.\nCheck '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()