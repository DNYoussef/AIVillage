import os
import shutil

def move_techniques():
    # Source and destination directories
    src_dir = 'agents/magi/techniques'
    dst_dir = 'agents/base/techniques'
    
    # List of technique files to move
    technique_files = [
        'tree_of_thoughts.py',
        'program_of_thoughts.py',
        'self_ask.py',
        'least_to_most.py',
        'contrastive_chain.py',
        'memory_of_thought.py',
        'choice_annealing.py',
        'prompt_chaining.py',
        'self_consistency.py',
        'evolutionary_tournament.py'
    ]
    
    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)
    
    # Move each technique file
    for file in technique_files:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)
        
        if os.path.exists(src_path):
            print(f"Moving {file}...")
            shutil.copy2(src_path, dst_path)
            print(f"Successfully moved {file}")
        else:
            print(f"Warning: {file} not found in source directory")

if __name__ == "__main__":
    move_techniques()
