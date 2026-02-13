import sys
import copy
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
import traceback

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def load_css(path):
    if not os.path.exists(path):
        print(f"DEBUG: CSS file not found at {path}")
        raise FileNotFoundError(f"CSS file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def inject_style(root, css_text):
    style = ET.Element("{%s}style" % SVG_NS)
    style.text = css_text
    root.insert(0, style)


def strip_inline_style(elem):
    for attr in ["fill", "stroke"]:
        if attr in elem.attrib:
            del elem.attrib[attr]

    if "style" in elem.attrib:
        parts = elem.attrib["style"].split(";")
        parts = [
            p for p in parts
            if not p.strip().lower().startswith("fill")
            and not p.strip().lower().startswith("stroke")
        ]
        if parts:
            elem.attrib["style"] = ";".join(parts)
        else:
            del elem.attrib["style"]


def hide_future_steps(elem, max_step):
    cls = elem.attrib.get("class", "")
    for token in cls.split():
        if token.startswith("step"):
            try:
                n = int(token.replace("step", ""))
                if n > max_step:
                    style = elem.attrib.get("style", "")
                    if "display:none" not in style:
                        style = (style + ";display:none").strip(";")
                    elem.attrib["style"] = style
            except ValueError:
                pass


def process_step(tree, css_text, step):
    root = tree.getroot()
    inject_style(root, css_text)
    for elem in root.iter():
        strip_inline_style(elem)
        hide_future_steps(elem, step)
    return tree


def try_render_with_cairosvg(svg_in, png_out, width):
    if os.environ.get("SKIP_CAIROSVG") == "1":
        return False
    
    print("DEBUG: Attempting to import CairoSVG...")
    try:
        import cairosvg
        print("DEBUG: CairoSVG imported. Converting...")
        cairosvg.svg2png(url=svg_in, write_to=png_out, output_width=width)
        return os.path.exists(png_out)
    except Exception as e:
        print(f"DEBUG: CairoSVG failed: {e}")
        return False


def render_with_inkscape(inkscape_cmd, svg_in, png_out, width, timeout_sec):
    svg_abs = os.path.abspath(svg_in)
    png_abs = os.path.abspath(png_out)
    
    if not os.path.exists(inkscape_cmd):
        print(f"DEBUG: Inkscape not found at: {inkscape_cmd}")
        return False

    cmd_v1 = [
        inkscape_cmd,
        svg_abs,
        "--export-type=png",
        f"--export-filename={png_abs}",
        f"--export-width={width}",
    ]

    cmd_v092 = [
        inkscape_cmd,
        f"--export-png={png_abs}",
        f"--export-width={width}",
        svg_abs,
    ]

    for cmd in [cmd_v1, cmd_v092]:
        print(f"DEBUG: Executing: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, 
                check=False, 
                timeout=timeout_sec, 
                capture_output=True, 
                text=True
            )
            if os.path.exists(png_abs):
                return True
            print(f"DEBUG: Command failed. Code: {result.returncode}, Stderr: {result.stderr}")
        except Exception as e:
            print(f"DEBUG: Subprocess error: {e}")
    return False


def render_png_4k(svg_in, png_out, inkscape_path=None):
    width = 3840
    timeout_sec = 60

    # Priority 1: Use Inkscape if path is explicitly provided
    if inkscape_path:
        print(f"DEBUG: Using provided Inkscape: {inkscape_path}")
        return render_with_inkscape(inkscape_path, svg_in, png_out, width, timeout_sec)

    # Priority 2: Try CairoSVG if no Inkscape path was given
    if try_render_with_cairosvg(svg_in, png_out, width):
        return True

    # Priority 3: Try to find Inkscape in PATH or ENV
    inkscape_cmd = os.environ.get("INKSCAPE", "").strip() or shutil.which("inkscape")
    if inkscape_cmd:
        print(f"DEBUG: Found Inkscape in environment: {inkscape_cmd}")
        return render_with_inkscape(inkscape_cmd, svg_in, png_out, width, timeout_sec)
    
    return False


def main():
    if len(sys.argv) not in (3, 5):
        print("Usage: python build_svg.py <input.svg> <palette.css> [--inkscape <path>]")
        return

    svg_path = sys.argv[1]
    css_path = sys.argv[2]
    inkscape_path = sys.argv[4] if len(sys.argv) == 5 else None

    if not os.path.exists(svg_path):
        print(f"ERROR: SVG file not found: {svg_path}")
        return

    try:
        css_text = load_css(css_path)
        base_tree = ET.parse(svg_path)
        base_name = os.path.splitext(os.path.basename(svg_path))[0]

        detected_max_step = 1
        for elem in base_tree.getroot().iter():
            cls = elem.attrib.get("class", "")
            for token in cls.split():
                if token.startswith("step"):
                    try:
                        n = int(token.replace("step", ""))
                        if n > detected_max_step:
                            detected_max_step = n
                    except ValueError:
                        pass

        for step in range(1, detected_max_step + 1):
            print(f"\n--- Step {step} ---")
            tree_copy = copy.deepcopy(base_tree)
            result = process_step(tree_copy, css_text, step)

            out_svg = f"{base_name}_step{step}.svg"
            result.write(out_svg, encoding="utf-8", xml_declaration=True)
            print(f"Saved: {out_svg}")

            out_png = f"{base_name}_step{step}.png"
            if render_png_4k(out_svg, out_png, inkscape_path=inkscape_path):
                print(f"Success: {out_png}")
            else:
                print(f"Failed: {out_png}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
