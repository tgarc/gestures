from matplotlib.animation import Animation
import matplotlib.pyplot as plt
import itertools
import warnings
from collections import deque

class AsyncAnimation(Animation):
    def __init__(self,fig,init_artists=[],event_source=None,interval=1):
        self.__data = deque()
        self.__lag = 0

        self._fig = fig
        self._interval = interval
        self._init_artists = init_artists

        if event_source is None:
            event_source = fig.canvas.new_timer(interval=self._interval)

        if callable(self._init_artists):
            artists = self._init_artists()
        else:
            artists = self._init_artists
        self.__data.append((zip(artists,[None]*len(artists)),False))

        Animation.__init__(self,fig,event_source=event_source,blit=True)

    def new_frame_seq(self):
        return itertools.count()

    def _init_draw(self):
        # Initialize the drawing either using the given init_func or by
        # calling the draw function with the first item of the frame sequence.
        # For blitting, the init_func should return a sequence of modified
        # artists.
        # self._drawn_artists = self._init_artists()
        # self.__data.append((zip(self._drawn_artists,[None]*len(self._drawn_artists)),False))
        self._draw_frame(next(self.new_frame_seq()))

    def update_artists(self,artists,redraw=False):
        assert(self.event_source is not None)

        if len(self.__data) == self.__lag > 1: 
            warnings.warn("Artists queue is behind by %d" % len(self.__data))

        self.__data.append((artists,redraw))
        self.__lag = len(self.__data)
        self._fig.canvas.flush_events()

    def _draw_next_frame(self, *args):
        # carry on if there's nothing to draw right now
        if self.__data:
            Animation._draw_next_frame(self,*args)

    def _draw_frame(self,framedata):
        artdata, redraw = self.__data.popleft()

        artists = []
        for (a,d) in artdata:
            artists.append(a)
            if d is not None:
                a.set_data(d)

        self._drawn_artists = artists
