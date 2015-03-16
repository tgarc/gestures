trunc_coords = lambda shape,xy: tuple(x if x >= 0 and x <= dimsz
                                      else (0 if x < 0 else dimsz)
                                      for dimsz,x in zip(shape[::-1],xy))
inttuple = lambda xy: tuple(map(int,xy))
