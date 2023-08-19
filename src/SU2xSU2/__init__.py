# install scientific.mplstyle following https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/__init__.py
import matplotlib.pyplot as plt
import SU2xSU2

# register the included stylesheet in the matplotlib style library
style_path = SU2xSU2.__path__[0]
stylesheet = plt.style.core.read_style_directory(style_path)
# Update dictionary of styles
plt.style.core.update_nested_dict(plt.style.library, stylesheet)
#plt.style.core.available[:] = sorted(plt.style.library.keys())
plt.style.core.reload_library()