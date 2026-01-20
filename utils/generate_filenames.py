import os
from sklearn.model_selection import train_test_split

def generate_filenames():
    base_path = r"d:\Pratik\Landslide\Dataset\archive"
    train_img_path = os.path.join(base_path, "TrainData", "img")
    
    # Get all .h5 files
    files = sorted([f for f in os.listdir(train_img_path) if f.endswith(".h5")])
    
    # Split into train and validation (using 0.2 ratio as in original code)
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
    
    # Save to utils/filenames/
    output_dir = r"d:\Pratik\Landslide\utils\filenames"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for item in sorted(train_files):
            f.write(f"{item}\n")
            
    with open(os.path.join(output_dir, "val.txt"), "w") as f:
        for item in sorted(val_files):
            f.write(f"{item}\n")
            
    print(f"Generated train.txt ({len(train_files)} files) and val.txt ({len(val_files)} files) in {output_dir}")

if __name__ == "__main__":
    generate_filenames()
