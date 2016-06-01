# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# df1 = pd.DataFrame({
#     'name': ['Danny', 'Jess', 'Joey', 'D.J.', 'Steph', 'Michelle'],
#     'age': [29, 24, 29, 10, 5, 0],
#     'room': random_integer(10, 10)
# })
# print(df1)


# 出力結果
#    age      name sex
# 0   29     Danny   m
# 1   24      Jess   m
# 2   29      Joey   m
# 3   10      D.J.   f
# 4    5     Steph   f
# 5    0  Michelle   f

# room
print np.random.random_integers(10, size=10)

# 専有面積(m2、8〜904m2)
print np.random.random(10) * 896 + 8

# 築年数(0〜70年)
print np.random.random_integers(70, size=10)
