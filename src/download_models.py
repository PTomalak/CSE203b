import os
import urllib.request
import zipfile
import sys
from rich.console import Console

console = Console()

def download_and_extract():
    url = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    zip_path = os.path.join(script_dir, "models.zip")
    
    # 1. Check if models already exist inside the directory
    if os.path.exists(models_dir) and os.listdir(models_dir):
        console.print(f"[bold yellow]Models already present[/bold yellow] at {models_dir}")
        return

    # 2. Check if zip already exists to skip download
    if os.path.exists(zip_path):
        console.print(f"[bold cyan]Zip file already exists[/bold cyan] at {zip_path}. Skipping download.")
    else:
        console.print(f"[bold cyan]Downloading pre-trained models[/bold cyan]...")
        try:
            def hook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(100, int(count * block_size * 100 / total_size))
                    sys.stdout.write(f"\rDownloading: {percent}%")
                    sys.stdout.flush()
                    
            urllib.request.urlretrieve(url, zip_path, reporthook=hook)
            print() 
            console.print("[bold green]Download complete.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Download failed:[/bold red] {e}")
            return

    # 3. Extract contents into the ./models directory
    try:
        os.makedirs(models_dir, exist_ok=True)
        console.print(f"Extracting into {models_dir}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
            
        console.print(f"[bold green]Extraction complete.[/bold green]")
        os.remove(zip_path)
        console.print("Cleaned up zip file.")
        
    except Exception as e:
        console.print(f"[bold red]Extraction failed:[/bold red] {e}")

if __name__ == "__main__":
    download_and_extract()
