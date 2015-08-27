from matplotlib.animation import Animation
import matplotlib.pyplot as plt
import itertools
import warnings


class GUI(Animation):
    def __init__(self,fig,init_func=None,interval=1):
        self.__data = []
        self.__lag = 0

        self._fig = fig
        self._framedata = itertools.count()
        self._init_func = init_func
        self._interval = interval
        self._drawn_artists = []

        timer = fig.canvas.new_timer(interval=self._interval)
        self.timer = timer

        Animation.__init__(self,fig,event_source=timer,blit=True)
        # self._fig.canvas.draw()
        # self._fig.canvas.flush_events()
        # self._fig.canvas.start_event_loop(1e-6)
        # self._fig.canvas.start_event_loop(1e-6)
        plt.pause(1e-6)

    def update_artists(self,artists,redraw=False):
        if self.event_source is None:
            raise Exception

        if len(self.__data) == self.__lag > 1: 
            warnings.warn("Artists queue is behind by %d" % len(self.__data))

        self.__data.append((artists,redraw))
        self.__lag = len(self.__data)
        self._fig.canvas.flush_events()

    def _draw_next_frame(self, *args):
        if self.__data:
            Animation._draw_next_frame(self,*args)

    def _draw_frame(self,framedata):
        artdata, redraw = self.__data.pop(0)

        artists = []
        for (a,d) in artdata:
            artists.append(a)
            a.set_data(d)

        self._drawn_artists = artists
