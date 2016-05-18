import pstats, cProfile
import pyximport, numpy
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

import train_weights as tw
 

cProfile.runctx("tw.train()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
