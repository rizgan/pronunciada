"""Regenerate every graphic, fill the generated parts of the site page, and copy the link preview into site/assets/.

    python scripts/graphics/build.py        # needs Google Chrome or Chromium, and ffmpeg
"""
from __future__ import annotations

import runpy
import shutil
from pathlib import Path

from common import ROOT

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    for script in ("make_logo.py", "make_banner.py", "make_charts.py", "make_site.py", "make_benchmarks.py"):      # make_video.py and make_cli_gif.py are run by hand
        runpy.run_path(str(HERE / script), run_name="__main__")
    shutil.copy2(ROOT / "docs/assets/social.jpg", ROOT / "site/assets/social.jpg")
    print("copied site assets")
