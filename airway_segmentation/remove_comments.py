import re
import os
from pathlib import Path

def remove_comments_from_code(code):
    lines = code.split('\n')
    result = []
    
    for line in lines:
        stripped = line.lstrip()
        
        if not stripped:
            result.append('')
            continue
        
        if stripped.startswith('#'):
            continue
        
        if '#' in line:
            in_string = False
            quote_char = None
            cleaned_line = []
            i = 0
            while i < len(line):
                char = line[i]
                
                if char in ['"', "'"]:
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char and (i == 0 or line[i-1] != '\\'):
                        in_string = False
                        quote_char = None
                    cleaned_line.append(char)
                elif char == '#' and not in_string:
                    break
                else:
                    cleaned_line.append(char)
                i += 1
            
            line = ''.join(cleaned_line).rstrip()
        
        result.append(line)
    
    return '\n'.join(result)

def process_file(file_path):
    print(f"Processing: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        cleaned = remove_comments_from_code(source)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        print(f"  ✓ Cleaned successfully")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    base_dir = Path(__file__).parent
    
    python_files = [
        "airwais_seg.py",
        "airway_gap_filler.py",
        "airway_graph.py",
        "airway_refinement.py",
        "combine_reports.py",
        "compare_results.py",
        "fibrosis_scoring.py",
        "main_pipeline.py",
        "parenchymal_metrics.py",
        "preprocessin_cleaning.py",
        "recalculate_both_scores.py",
        "recalculate_fibrosis_scores.py",
        "skeleton_cleaner.py"
    ]
    
    print("="*80)
    print("REMOVING COMMENTS FROM ALL SCRIPTS")
    print("="*80)
    print()
    
    success_count = 0
    for file_name in python_files:
        file_path = base_dir / file_name
        if file_path.exists():
            if process_file(file_path):
                success_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print()
    print("="*80)
    print(f"COMPLETED: {success_count}/{len(python_files)} files processed")
    print("="*80)

if __name__ == "__main__":
    main()
