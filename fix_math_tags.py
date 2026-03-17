import re
import os
import glob

def fix_content(content):
    # Order matters: more specific to less specific
    
    # 1. $^{&}$lt;sup>...</sup> -> $ ^{...} $
    # We saw it is literally $^{&}$lt;sup>
    content = re.sub(r'\$\^\{&\}\$lt;sup>(.*?)</sup>', r'$ ^{\1} $', content)
    
    # 2. $^{&}$lt;sub>...</sub> -> $ _{...} $
    content = re.sub(r'\$\^\{&\}\$lt;sub>(.*?)</sub>', r'$ _{\1} $', content)
    
    # 3. $^{...}$lt;sup>... -> $ ^{...} $
    # Pattern: $^{A}$lt;sup>B</sup>
    content = re.sub(r'\$\^\{(.*?)\}\$lt;sup>(.*?)</sup>', r'$ ^{\2} $', content)
    # Pattern: $^{A}$lt;sup>B
    content = re.sub(r'\$\^\{(.*?)\}\$lt;sup>(.*?)', r'$ ^{\2} $', content)
    
    # Also handle without the $ between } and lt
    content = re.sub(r'\$\^\{(.*?)\}lt;sup>(.*?)</sup>', r'$ ^{\2} $', content)
    content = re.sub(r'\$\^\{(.*?)\}lt;sup>(.*?)', r'$ ^{\2} $', content)

    # 4. lt;sup>...</sup> -> $ ^{...} $
    content = re.sub(r'lt;sup>(.*?)</sup>', r'$ ^{\1} $', content)
    
    # 5. lt;sub>...</sub> -> $ _{...} $
    content = re.sub(r'lt;sub>(.*?)</sub>', r'$ _{\1} $', content)

    # 6. Any other instances of "lt;sup" or "lt;sub"
    # To be safe, we only replace if it's "lt;sup" or "lt;sub"
    content = re.sub(r'lt;sup', r' $ ^{ ', content)
    content = re.sub(r'lt;sub', r' $ _{ ', content)
    
    return content

def main():
    files = glob.glob('**/*.md', recursive=True) + glob.glob('**/*.qmd', recursive=True)
    for file_path in files:
        if os.path.basename(file_path) == 'fix_math_tags.py':
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = fix_content(content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed {file_path}")

if __name__ == "__main__":
    main()
