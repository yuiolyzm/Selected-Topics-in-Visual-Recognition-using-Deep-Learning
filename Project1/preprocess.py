""" count images in each subfolder """
import os
import matplotlib.pyplot as plt
import numpy as np

# Set dataset directory
train_dir = "data/train"

# Count images in each subfolder (class)
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)}

# Sort by class index
sorted_classes = sorted(class_counts.keys(), key=lambda x: int(x))  
sorted_counts = [class_counts[cls] for cls in sorted_classes]

# Display each class count
for cls, count in zip(sorted_classes, sorted_counts):
    print(f"Class {cls}: {count} images")

# Create bar plot (grouped by 25 classes per chart)
num_classes = len(sorted_classes)
num_groups = (num_classes // 25) + (1 if num_classes % 25 != 0 else 0)

for i in range(num_groups):
    start, end = i * 25, min((i + 1) * 25, num_classes)
    plt.figure(figsize=(12, 5))
    plt.bar(sorted_classes[start:end], sorted_counts[start:end], color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title(f"Image Count per Class (Part {i+1})")
    plt.xticks(rotation=45)
    plt.show()

""" mapping and change the prediction """
# import pandas as pd

# def remap_predictions(csv_path, output_path):
#     # Read CSV file
#     df = pd.read_csv(csv_path)
    
#     # Create mapping dictionary
#     mapping = {}
    
#     mapping[0] = 0
#     mapping[1] = 1
#     for original in range(2, 100):
#         base = (original - 1) // 11  # Find base group (0-9)
#         offset = original - (base * 11 + 1)  # Determine new range start
#         if offset == 0:
#             mapping[original] = base + 1
#         else:
#             mapping[original] = (base + 1) * 10 + offset - 1
    
#     # Apply mapping
#     df['pred_label'] = df['pred_label'].map(mapping)
#     df['image_name'] = df['image_name'].str.replace('.jpg', '', regex=False)
    
#     # Save the modified CSV
#     df.to_csv(output_path, index=False)
#     print(f"Mapping complete. Output saved to {output_path}")

# # Example usage
# remap_predictions("prediction8.csv", "prediction8-1.csv")


""" mapping """
# import pandas as pd
# mapping = {}

# mapping[0] = 0
# mapping[1] = 1
# for original in range(2, 100):
#     base = (original - 1) // 11  # Find base group (0-9)
#     offset = original - (base * 11 + 1)  # Determine new range start
#     if offset == 0:
#         mapping[original] = base + 1
#     else:
#         mapping[original] = (base + 1) * 10 + offset - 1

# for k, v in mapping.items():
#     print(f"{k} -> {v}")


""" sort """
# import pandas as pd

# # read reference and target CSV
# df_reference = pd.read_csv(r"C:\code_sem\senior\VRDL\Project1_csv\prediction_try.csv")  
# df_target = pd.read_csv("prediction.csv")  # need to sort this

# df_target_sorted = df_target.set_index("image_name").loc[df_reference["image_name"]].reset_index()
# df_target_sorted.to_csv("prediction1.csv", index=False)

# print("Sorting completed! Saved as target_sorted.csv")
