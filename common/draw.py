import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "Songti SC"]
matplotlib.rcParams["axes.unicode_minus"] = False


def configure_chinese_font() -> None:
    candidates = ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False