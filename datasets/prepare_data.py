import os

def generate_file_list(data_path, split):
    img_dir = os.path.join(data_path, split, "img")
    gt_dir = os.path.join(data_path, split, "gt")
    output_file = os.path.join(data_path, f"{split}.txt")

    with open(output_file, "w") as f:
        for img_file in sorted(os.listdir(img_dir)):
            if img_file.startswith("img_") and img_file.endswith((".jpg", ".png", ".jpeg")):
                img_path = f"./datasets/{split}/img/{img_file}"
                
                # Modify ground truth filename to match gt_img_x.txt format
                img_number = os.path.splitext(img_file)[0].split("_")[1]
                gt_filename = f"gt_img_{img_number}.txt"
                gt_path = f"./datasets/{split}/gt/{gt_filename}"

                if os.path.exists(os.path.join(gt_dir, gt_filename)):
                    f.write(f"{img_path}\t{gt_path}\n")
                else:
                    print(f"Warning: No ground truth for {img_file}")

    print(f"{split}.txt generated successfully.")

# Define dataset path
dataset_path = "/home/rajratan_unix/DBNet.pytorch/datasets"

# Generate train.txt and test.txt
generate_file_list(dataset_path, "train")
generate_file_list(dataset_path, "test")