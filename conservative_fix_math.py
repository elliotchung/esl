
import re
import os
import sys

def fix_math(content):
    lines = content.split('\n')
    new_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if not stripped:
            new_lines.append(line)
            i += 1
            continue
            
        if stripped.startswith("$$") or stripped.startswith("$"):
            new_lines.append(line)
            i += 1
            continue
            
        if stripped.startswith("\\begin{aligned}") or stripped.startswith("\\begin{cases}"):
            env_name = "aligned" if "aligned" in stripped else "cases"
            block = [stripped]
            i += 1
            while i < len(lines) and f"\\end{{{env_name}}}" not in lines[i]:
                block.append(lines[i].strip())
                i += 1
            if i < len(lines):
                block.append(lines[i].strip())
                i += 1
                # Check for tag
                tag = ""
                if i < len(lines) and (r"\tag{" in lines[i] or re.match(r"^\s*\(\d+\.\d+\)\s*$", lines[i])):
                    tag = lines[i].strip()
                    if tag.startswith("("): tag = f"\\tag{{{tag[1:-1]}}}"
                    i += 1
                
                if tag:
                    new_lines.append("$$\n" + "\n".join(block) + "\n" + tag + "\n$$")
                else:
                    new_lines.append("$$\n" + "\n".join(block) + "\n$$")
            continue

        math_indicators = [
            r"\\", r" = ", r" \sum", r" \hat", r" \beta", r" \mathbf", r" \frac", r" \operatorname", r"\tag{"
        ]
        
        # KEY CHANGE: Added '$' not in line
        if (line.startswith(' ') or line.startswith('\t')) and any(ind in line for ind in math_indicators) and '$' not in line and not stripped.startswith('!') and not stripped.startswith('-') and not stripped.startswith('*'):
            # Check for tag on next line
            if i + 1 < len(lines) and (r"\tag{" in lines[i+1] or re.match(r"^\s*\(\d+\.\d+\)\s*$", lines[i+1])):
                tag = lines[i+1].strip()
                if tag.startswith("("): tag = f"\\tag{{{tag[1:-1]}}}"
                new_lines.append(f"$$\n{stripped} {tag}\n$$")
                i += 2
            else:
                new_lines.append(f"$$\n{stripped}\n$$")
                i += 1
        else:
            new_lines.append(line)
            i += 1
            
    return '\n'.join(new_lines)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            with open(arg, 'r') as f:
                content = f.read()
            new_content = fix_math(content)
            with open(arg, 'w') as f:
                f.write(new_content)
