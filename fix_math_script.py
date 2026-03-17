
import re
import os
import sys

def fix_math(content):
    # Task 1: Protect existing math mode blocks
    tokens = re.split(r"(\$\$.*?\$\$|\$.*?\$)", content, flags=re.DOTALL)
    
    def process_math(text):
        # Replace HTML tags inside a math block
        text = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", text)
        text = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", text)
        return text

    # Unicode math characters to wrap when outside math mode
    unicode_math = "βσλθεα≈ℓΓΣ∈ρμΦπΔεǫˆˇ"
    
    for i in range(len(tokens)):
        if tokens[i].startswith("$"):
            # Inside math mode
            tokens[i] = process_math(tokens[i])
        else:
            # Outside math mode
            # 1. Replace <sup> and <sub> with $...$
            tokens[i] = re.sub(r"<(sup|sub)>(.*?)</\1>", lambda m: f"${'^' if m.group(1)=='sup' else '_'}{{{m.group(2)}}}$", tokens[i])
            # 2. Wrap unicode characters
            for char in unicode_math:
                tokens[i] = tokens[i].replace(char, f"${char}$")
                
    content = "".join(tokens)

    # Task 2: Wrap blocks starting with \begin{...} in $$ if not already
    # This regex tries to find the whole environment including possible leading = or following \tag
    # We look for lines that contain \begin but aren't in $$
    lines = content.split('\n')
    new_lines = []
    in_math_block = False
    
    # We use a state machine to track if we are in a block that needs wrapping
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if stripped.startswith("$$"):
            # Already in display math
            if stripped.endswith("$$") and stripped != "$$":
                new_lines.append(line)
            else:
                # Multi-line $$
                new_lines.append(line)
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    new_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    new_lines.append(lines[i])
            i += 1
            continue

        if "\\begin{aligned}" in line or "\\begin{cases}" in line:
            # Found a block
            block_lines = [line.strip()]
            i += 1
            env_name = "aligned" if "aligned" in line else "cases"
            while i < len(lines) and f"\\end{{{env_name}}}" not in lines[i]:
                block_lines.append(lines[i].strip())
                i += 1
            if i < len(lines):
                last_line = lines[i].strip()
                block_lines.append(last_line)
                i += 1
                # Check if next line is a \tag
                if i < len(lines) and (r"\tag{" in lines[i] or re.match(r"^\s*\(\d+\.\d+\)\s*$", lines[i])):
                    block_lines.append(lines[i].strip())
                    i += 1
            
            # Wrap the block
            new_lines.append("\n$$" + " ".join(block_lines) + "$$\n")
            continue

        # Regular line, check for standalone equations
        math_indicators = [
            r"\\", r" = ", r" \sum", r" \hat", r" \beta", r" \mathbf", r" \frac", r" \operatorname", r" \tag{"
        ]
        
        if (line.startswith(' ') or line.startswith('\t')) and any(ind in line for ind in math_indicators) and not stripped.startswith('!') and not stripped.startswith('-'):
            # Math candidate
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

    content = '\n'.join(new_lines)
    
    # Cleanup: remove triple $ and excessive newlines
    content = re.sub(r"\${3,}", "$$", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    
    return content

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            with open(arg, 'r') as f:
                content = f.read()
            new_content = fix_math(content)
            with open(arg, 'w') as f:
                f.write(new_content)
