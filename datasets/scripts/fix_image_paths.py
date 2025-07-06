# file: fix_image_paths.py

import json
import argparse
import os
from tqdm import tqdm

def process_jsonl_file(input_file_path, output_file_path):
    """
    Reads a JSONL file, removes a specific prefix from the 'image' field,
    and writes the corrected data to a new JSONL file.
    """
    # 定义需要被移除的路径前缀
    prefix_to_remove = "allava_laion/images/"
    
    lines_processed = 0
    lines_modified = 0

    print(f"Starting to process file: {input_file_path}")
    print(f"Corrected data will be saved to: {output_file_path}")

    try:
        # 使用 with 语句安全地打开输入和输出文件
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            # 为了处理大文件，我们逐行读取
            for line in tqdm(infile, desc="Processing lines"):
                lines_processed += 1
                try:
                    # 将每一行的JSON字符串解析为Python字典
                    data = json.loads(line)
                    
                    # 检查 'image' 键是否存在且其值是字符串
                    if 'image' in data and isinstance(data['image'], str):
                        original_path = data['image']
                        
                        # 如果路径以指定前缀开头，则进行修正
                        if original_path.startswith(prefix_to_remove):
                            # 移除前缀
                            data['image'] = original_path[len(prefix_to_remove):]
                            lines_modified += 1
                    
                    # 将处理后（或未处理）的字典转换回JSON字符串，并写入新文件
                    outfile.write(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    print(f"\nWarning: Skipping malformed JSON line #{lines_processed}: {line.strip()}")
                    continue

        print("\nProcessing complete!")
        print(f"Total lines processed: {lines_processed}")
        print(f"Total lines modified:  {lines_modified}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="Fix image paths in a JSONL file by removing the 'allava_laion/images/' prefix.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the source JSONL file with incorrect paths."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to the new JSONL file where corrected data will be saved."
    )
    
    args = parser.parse_args()

    # --- 安全检查，防止覆盖原始文件 ---
    if os.path.abspath(args.input_file) == os.path.abspath(args.output_file):
        print("Error: Input and output file paths cannot be the same. Please choose a different name for the output file.")
    else:
        process_jsonl_file(args.input_file, args.output_file)