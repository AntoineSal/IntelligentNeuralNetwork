import os
import requests
import zipfile
import io

def download_and_extract():
    # URL directe alternative (souvent plus stable)
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    valid_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    test_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    
    output_dir = "INNv2/data/wikitext-2"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading to {output_dir}...")
    
    files = {
        "train.txt": url,
        "valid.txt": valid_url,
        "test.txt": test_url
    }
    
    for filename, file_url in files.items():
        print(f"Fetching {filename}...")
        try:
            r = requests.get(file_url)
            if r.status_code == 200:
                with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(r.text)
                print(f"✓ {filename} saved ({len(r.text)} chars)")
            else:
                print(f"❌ Error downloading {filename}: {r.status_code}")
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    download_and_extract()

