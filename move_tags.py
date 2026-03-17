
import re
import os
import sys

def move_tags(content):
    # Move \tag{...} from inside aligned/cases to outside
    content = re.sub(r"(\\tag\{.*?\})\s*\\end\{(aligned|cases)\}", r"\\end{\2} \1", content)
    # Also handle tags like (2.47)
    content = re.sub(r"(\(\d+\.\d+\))\s*\\end\{(aligned|cases)\}", r"\\end{\2} \\tag{\1}", content)
    return content

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            with open(arg, 'r') as f:
                content = f.read()
            new_content = move_tags(content)
            if new_content != content:
                with open(arg, 'w') as f:
                    f.write(new_content)
