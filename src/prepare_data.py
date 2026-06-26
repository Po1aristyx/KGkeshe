import os
import csv
import re
import shutil

archive_dir = r"E:\code\keshecode\archive"
ready_data_dir = r"E:\code\keshecode\ready_data"

def process_csv(filepath, dest_dir, filename):
    valid_rows = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Determine delimiter by reading first few lines
        first_line = f.readline()
        delimiter = '|' if '|' in first_line else ','
        f.seek(0)
        
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
            # Find indices for node_1, node_2, edge if possible, otherwise assume first 3
            # But task says: "提取前三列命名为 node_1, node_2, edge"
        except StopIteration:
            return False

        for row in reader:
            if len(row) >= 3:
                # Some rows might have empty nodes or edges, we keep them if they have 3 cols,
                # but better to strip spaces.
                n1, n2, e = row[0].strip(), row[1].strip(), row[2].strip()
                if n1 and n2 and e:
                    valid_rows.append([n1, n2, e])
    
    if len(valid_rows) > 0:
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        with open(dest_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['node_1', 'node_2', 'edge'])
            writer.writerows(valid_rows)
        return True
    return False

def main():
    if not os.path.exists(archive_dir):
        print(f"Archive dir not found: {archive_dir}")
        return

    # Mapping from archive subdirectory to (year, conference)
    dir_map = {
        "AAAI2023": ("2023", "AAAI"),
        "CVPR2023": ("2023", "CVPR"),
        "NeurIPS2022": ("2022", "NeurIPS"),
        "NeurIPS2023": ("2023", "NeurIPS"),
        "AAAI2024": ("2024", "AAAI"),
        "CVPR2024": ("2024", "CVPR"),
        "NeurIPS2024": ("2024", "NeurIPS"),
    }

    count = 0
    for subdir in os.listdir(archive_dir):
        if subdir not in dir_map:
            continue
        year, conf = dir_map[subdir]
        dest_dir = os.path.join(ready_data_dir, year, conf)
        
        # Traverse the subdirectory to find all files ending with _graph.csv or just .csv inside graph/csv folders
        source_subdir = os.path.join(archive_dir, subdir)
        for root, dirs, files in os.walk(source_subdir):
            for file in files:
                if file.endswith("graph.csv") or (file.endswith(".csv") and "chunk" not in file):
                    # Exclude chunks
                    if "chunks" in file.lower() or "nodes_data" in file.lower() or "edges_data" in file.lower():
                        continue
                    # It's a graph csv
                    filepath = os.path.join(root, file)
                    if process_csv(filepath, dest_dir, file):
                        count += 1

    print(f"Phase 1.1 Complete: Processed and standardized {count} graph CSV files into {ready_data_dir}")

if __name__ == "__main__":
    main()
