
import re
import os
import sys

def replace_unicode_math(content):
    unicode_map = {
        "β": r"$\beta$", "σ": r"$\sigma$", "λ": r"$\lambda$", "θ": r"$\theta$",
        "ε": r"$\epsilon$", "α": r"$\alpha$", "≈": r"$\approx$", "ℓ": r"$\ell$",
        "Γ": r"$\Gamma$", "Σ": r"$\Sigma$", "∈": r"$\in$", "ρ": r"$\rho$",
        "µ": r"$\mu$", "η": r"$\eta$", "τ": r"$\tau$", "ω": r"$\omega$",
        "δ": r"$\delta$", "∆": r"$\Delta$", "∇": r"$\nabla$", "∂": r"$\partial$",
        "∞": r"$\infty$", "≤": r"$\le$", "≥": r"$\ge$", "±": r"$\pm$",
        "×": r"$\times$", "·": r"$\cdot$", "√": r"$\sqrt{}$", "‖": r"$\|$",
        "ǫ": r"$\epsilon$", "ϕ": r"$\phi$", "Φ": r"$\Phi$", "π": r"$\pi$",
        "→": r"$\to$", "⇒": r"$\Rightarrow$", "⇔": r"$\Leftrightarrow$"
    }
    
    # 1. Handle superscripts and subscripts FIRST
    content = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", content)
    content = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", content)

    # 2. Replace Unicode characters
    def replace_outside_math(text):
        for char, repl in unicode_map.items():
            text = text.replace(char, repl)
        return text

    def replace_inside_math(text):
        for char, repl in unicode_map.items():
            text = text.replace(char, repl.strip("$"))
        return text

    tokens = re.split(r"(\$\$.*?\$\$|\$.*?\$)", content, flags=re.DOTALL)
    for i in range(len(tokens)):
        if tokens[i].startswith("$"):
            tokens[i] = replace_inside_math(tokens[i])
        else:
            tokens[i] = replace_outside_math(tokens[i])
            
    content = "".join(tokens)
    
    # Final cleanup of double $ created by the above
    content = content.replace("$$", "MATH_DISPLAY_MARKER")
    content = content.replace("$$", "$")
    content = content.replace("MATH_DISPLAY_MARKER", "$$")
    
    return content

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            with open(arg, 'r') as f:
                content = f.read()
            new_content = replace_unicode_math(content)
            if new_content != content:
                with open(arg, 'w') as f:
                    f.write(new_content)
