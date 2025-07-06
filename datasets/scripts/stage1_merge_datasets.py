# file: stage1_merge_datasets.py

import json
import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def load_json_data(json_path):
    """
    安全地读取JSON/JSONL文件，支持两种格式
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"文件不存在: {json_path}")
    
    data = []
    try:
        # 首先尝试作为标准JSON文件读取
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ 成功读取JSON文件: {json_path.name}")
    except json.JSONDecodeError:
        # 如果失败，尝试作为JSONL文件读取
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            print(f"✅ 成功读取JSONL文件: {json_path.name}")
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析JSON/JSONL文件 {json_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"读取文件时发生错误 {json_path}: {e}")
    
    return data

def process_and_normalize_data(json_path, image_base_dir, desc="Processing", keep_full_path=True):
    """
    读取JSON文件，提取image和caption，并处理图片路径。
    
    Args:
        json_path: JSON文件路径
        image_base_dir: 图片基础目录
        desc: 进度条描述
        keep_full_path: 是否保留完整路径（推荐为True）
    """
    data = load_json_data(json_path)
    processed_data = []
    skipped_count = 0
    filename_counts = defaultdict(int)
    
    for item in tqdm(data, desc=desc):
        try:
            caption = ""
            image_path = ""

            # 提取图片路径
            if 'image' in item:
                original_path = item['image']
                if keep_full_path:
                    # 保留相对路径结构，便于训练时加载
                    image_path = str(Path(original_path).as_posix())
                else:
                    # 只保留文件名（原始逻辑）
                    image_path = Path(original_path).name
                
                # 统计文件名重复情况
                filename_counts[Path(original_path).name] += 1
            
            # 提取caption
            if 'conversations' in item and len(item['conversations']) >= 2:
                # LLaVA格式的对话数据
                for conv in item['conversations']:
                    if conv.get('from') in ['gpt', 'assistant']:
                        caption = conv.get('value', '').strip()
                        break
            elif 'caption' in item:
                # 直接的caption字段
                caption = str(item['caption']).strip()
            
            # 数据验证
            if not image_path or not caption:
                skipped_count += 1
                continue
            
            # 检查caption质量
            if len(caption) < 5:  # 过短的caption可能质量不高
                skipped_count += 1
                continue
            
            # 🔧 关键修复：为caption添加<image>标签（如果还没有的话）
            if not caption.startswith('<image>'):
                caption = '<image>\n' + caption
            
            processed_data.append({
                "id": item.get('id', f"{Path(json_path).stem}_{len(processed_data)}"),
                "image": image_path,
                "caption": caption,
                "source": Path(json_path).stem  # 添加数据源标识
            })
            
        except Exception as e:
            print(f"⚠️  处理数据项时发生错误: {e}")
            skipped_count += 1
            continue
    
    # 输出统计信息
    print(f"📊 {desc} 统计:")
    print(f"   - 原始数据: {len(data)} 条")
    print(f"   - 有效数据: {len(processed_data)} 条")
    print(f"   - 跳过数据: {skipped_count} 条")
    print(f"   - 已为所有caption添加<image>标签")
    
    # 检查文件名重复情况
    duplicates = {name: count for name, count in filename_counts.items() if count > 1}
    if duplicates and not keep_full_path:
        print(f"⚠️  发现重复文件名: {len(duplicates)} 个")
        print("   建议使用 keep_full_path=True 避免路径冲突")
    
    return processed_data

def validate_and_merge_datasets(datasets_info, output_path, image_base_dirs=None, keep_full_path=True):
    """
    验证并合并多个数据集
    
    Args:
        datasets_info: [(json_path, description), ...] 数据集信息列表
        output_path: 输出文件路径
        image_base_dirs: 对应的图片基础目录列表
        keep_full_path: 是否保留完整路径
    """
    print("🚀 开始处理和合并数据集...")
    all_data = []
    source_stats = {}
    
    if image_base_dirs is None:
        image_base_dirs = [None] * len(datasets_info)
    
    # 处理每个数据集
    for i, ((json_path, desc), image_base_dir) in enumerate(zip(datasets_info, image_base_dirs)):
        try:
            processed_data = process_and_normalize_data(
                json_path, image_base_dir, desc, keep_full_path
            )
            all_data.extend(processed_data)
            source_stats[desc] = len(processed_data)
            print(f"✅ {desc}: {len(processed_data)} 条数据")
        except Exception as e:
            print(f"❌ 处理 {desc} 时发生错误: {e}")
            continue
    
    if not all_data:
        raise ValueError("没有有效的数据可以合并！")
    
    # 数据打乱
    print(f"\n📊 总计样本数: {len(all_data)} 条，正在打乱数据...")
    random.seed(42)  # 设置随机种子以便复现
    random.shuffle(all_data)
    
    # 写入输出文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 正在将数据写入到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(all_data, desc="Writing final data"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 输出最终统计
    print("\n🎉 数据处理完成！")
    print(f"📁 合并后的数据集已保存到: {output_path}")
    print(f"📊 各数据源统计:")
    for source, count in source_stats.items():
        percentage = (count / len(all_data)) * 100
        print(f"   - {source}: {count} 条 ({percentage:.1f}%)")
    
    return output_path

def fix_existing_alignment_data(input_file, output_file):
    """
    修复已经存在的对齐数据文件，为其添加<image>标签
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的修复后JSONL文件路径
    """
    print("🔧 修复现有数据：为caption添加<image>标签...")
    
    fixed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="修复数据"):
            if not line.strip():
                continue
            
            total_count += 1
            item = json.loads(line.strip())
            
            # 检查caption是否已经包含<image>标签
            if 'caption' in item and not item['caption'].startswith('<image>'):
                # 添加<image>标签到caption开头
                item['caption'] = '<image>\n' + item['caption']
                fixed_count += 1
            
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 修复完成！")
    print(f"   - 总样本数: {total_count}")
    print(f"   - 修复样本数: {fixed_count}")
    print(f"   - 输出文件: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="合并Stage1对齐训练数据集")
    parser.add_argument("--allava_json", type=str, 
                       default=r"D:\SoftData\VSCODE\顶会论文\MedPLIB-main\20250607\datasets\ALLaVA-Caption-LAION-4V.json",
                       help="ALLaVA数据集JSON文件路径")
    parser.add_argument("--llava_json", type=str,
                       default=r"D:\SoftData\VSCODE\顶会论文\MedPLIB-main\20250607\datasets\chat.json", 
                       help="LLaVA预训练数据集JSON文件路径")
    parser.add_argument("--output", type=str,
                       default=r"D:\SoftData\VSCODE\顶会论文\MedPLIB-main\20250607\datasets\final_alignment_data_fixed.jsonl",
                       help="输出JSONL文件路径")
    parser.add_argument("--keep_full_path", action="store_true", default=True,
                       help="是否保留完整的图片路径（推荐）")
    parser.add_argument("--fix_existing", type=str, default="",
                       help="修复现有文件：指定已存在的JSONL文件路径来添加<image>标签")
    
    args = parser.parse_args()
    
    # 如果指定了修复现有文件的选项
    if args.fix_existing:
        if not Path(args.fix_existing).exists():
            print(f"❌ 文件不存在: {args.fix_existing}")
            return 1
        
        # 生成修复后的文件名
        input_path = Path(args.fix_existing)
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
        
        fix_existing_alignment_data(args.fix_existing, str(output_path))
        return 0
    
    # 正常的数据合并流程
    datasets_info = [
        (args.llava_json, "LLaVA-Pretrain"),
        (args.allava_json, "ALLaVA-4V")
    ]
    
    # 执行合并
    try:
        output_file = validate_and_merge_datasets(
            datasets_info=datasets_info,
            output_path=args.output,
            keep_full_path=args.keep_full_path
        )
        print(f"\n✨ 任务完成！输出文件: {output_file}")
        print("🎯 修复说明: 所有caption都已自动添加<image>标签，可直接用于FILA训练")
        
    except Exception as e:
        print(f"❌ 执行过程中发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())