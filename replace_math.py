import os
import re
import sys

unicode_map = {
    'β': r'\beta',
    'σ': r'\sigma',
    'λ': r'\lambda',
    'θ': r'\theta',
    'ε': r'\epsilon',
    'α': r'\alpha',
    '≈': r'\approx',
    'ℓ': r'\ell',
    'Γ': r'\Gamma',
    'Σ': r'\Sigma',
    '∈': r'\in',
    'ρ': r'\rho',
    'µ': r'\mu',
    'η': r'\eta',
    'τ': r'\tau',
    'ω': r'\omega',
    'δ': r'\delta',
    '∆': r'\Delta',
    '∇': r'\nabla',
    '∂': r'\partial',
    '∞': r'\infty',
    '≤': r'\le',
    '≥': r'\ge',
    '±': r'\pm',
    '×': r'\times',
    '·': r'\cdot',
    '√': r'\sqrt{}',
    '‖': r'\|',
    'ǫ': r'\epsilon',
}

def process_content(content):
    # 1. Protect existing math blocks
    math_blocks = []
    def save_math(match):
        block = match.group(0)
        # Inside math, just replace unicode characters
        for u_char, l_cmd in unicode_map.items():
            block = block.replace(u_char, l_cmd)
        
        # Also handle hat cases inside math: e.g. ˆf or Σˆ
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            block = block.replace(f'ˆ{c}', f'\\hat{{{c}}}')
            block = block.replace(f'{c}ˆ', f'\\hat{{{c}}}')
            
        # And unicode with hats inside math
        for u_char, l_cmd in unicode_map.items():
            l_cmd = unicode_map[u_char]
            block = block.replace(f'ˆ{u_char}', f'\\hat{{{l_cmd}}}')
            block = block.replace(f'{u_char}ˆ', f'\\hat{{{l_cmd}}}')
        
        placeholder = f"__MATH_BLOCK_{len(math_blocks)}__"
        math_blocks.append(block)
        return placeholder

    # Replace $$...$$ first, then $...$
    content = re.sub(r'\$\$.*?\$\$', save_math, content, flags=re.DOTALL)
    content = re.sub(r'\$.*?\$', save_math, content)

    # 2. Handle special combined cases outside math
    
    # || · ||<sup>2</sup> -> $\|\cdot\|^2$
    content = re.sub(r'\|\|\s*·\s*\|\|\s*<sup>(.*?)</sup>', lambda m: r'$\|\cdot\|' + f'^{{{m.group(1)}}}$', content)
    # Generic || · ||
    content = re.sub(r'\|\|\s*·\s*\|\|', lambda m: r'$\|\cdot\|$', content)

    # Support for <sup> and <sub> with multiple letters
    def sub_sup(match):
        prefix_hat = match.group(1) or ""
        word = match.group(2)
        suffix_hat = match.group(3) or ""
        sup = match.group(4)
        
        l_word = word
        for u_char, l_cmd in unicode_map.items():
            l_word = l_word.replace(u_char, l_cmd)
        
        if prefix_hat or suffix_hat:
            l_word = f"\\hat{{{l_word}}}"
        
        return f"${l_word}^{{{sup}}}$"

    content = re.sub(r'(ˆ)?([a-zA-ZβσλθεαℓΓΣρµητωδ∆∇∂]+)(ˆ)?\s*<sup>(.*?)</sup>', sub_sup, content)

    def sub_sub(match):
        prefix_hat = match.group(1) or ""
        word = match.group(2)
        suffix_hat = match.group(3) or ""
        sub = match.group(4)
        
        l_word = word
        for u_char, l_cmd in unicode_map.items():
            l_word = l_word.replace(u_char, l_cmd)
        
        if prefix_hat or suffix_hat:
            l_word = f"\\hat{{{l_word}}}"
        
        return f"${l_word}_{{{sub}}}$"

    content = re.sub(r'(ˆ)?([a-zA-ZβσλθεαℓΓΣρµητωδ∆∇∂]+)(ˆ)?\s*<sub>(.*?)</sub>', sub_sub, content)

    # 3. Handle hat cases remaining outside math
    def sub_hat(match):
        char = match.group(1)
        l_char = unicode_map.get(char, char)
        return f"$\\hat{{{l_char}}}$"

    content = re.sub(r'ˆ([a-zA-ZβσλθεαℓΓΣρµητωδ∆∇∂])', sub_hat, content)
    content = re.sub(r'([a-zA-ZβσλθεαℓΓΣρµητωδ∆∇∂])ˆ', sub_hat, content)

    # 4. Handle remaining individual Unicode characters
    for u_char in sorted(unicode_map.keys(), key=len, reverse=True):
        l_cmd = unicode_map[u_char]
        content = content.replace(u_char, f"${l_cmd}$")

    # 5. Restore math blocks
    for i, block in enumerate(math_blocks):
        content = content.replace(f"__MATH_BLOCK_{i}__", block)

    # 6. Post-processing: merge adjacent math blocks
    # Preserving the exact space captured
    content = re.sub(r'\$([^\$]+)\$(\s*)\$([^\$]+)\$', lambda m: f'${m.group(1)}{m.group(2)}{m.group(3)}$', content)
    content = re.sub(r'\$([^\$]+)\$(\s*)\$([^\$]+)\$', lambda m: f'${m.group(1)}{m.group(2)}{m.group(3)}$', content)

    return content

if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = process_content(content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Processed {file_path}")
        else:
            print(f"No changes in {file_path}")
