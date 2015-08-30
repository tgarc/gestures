from matplotlib.animation import Animation
import matplotlib.pyplot as plt
import itertools
import warnings
from collections import deque


class AsyncAnimation(Animation):
    """
    Makes an animation by reading from a user filled queue every *interval*
    milliseconds. User adds to the queue by calling *update_artists*.

    Note that data is not copied before being put into the queue; if a reference
    is passed (such as a numpy ndarray) whatever data the reference is pointing
    to at draw time will be plotted.

    *init_func* is a function used to draw a clear frame. If not given, the
    results of drawing from the first item in the frames sequence will be
    used. This function will be called once before the first frame.

    *event_source* Default event source to trigger polling. By default this is a
    timer with an *interval* millisecond timeout. This option is given to permit
    sharing timers between animation objects for syncing animations.
    """
    def __init__(self,fig,init_func=None,event_source=None,interval=10):
        self._data = deque()
        self._lag = 0
        self._drawn = False

        self._fig = fig
        self._interval = interval
        self._init_func = init_func

        if event_source is None:
            event_source = fig.canvas.new_timer(interval=self._interval)

        Animation.__init__(self,fig,event_source=event_source,blit=True)

    def new_frame_seq(self):
        return itertools.count()

    def _init_draw(self):
        if self._init_func is not None:
            self._drawn_artists = self._init_func()

    def update_artists(self,*artists):
        assert(self.event_source is not None)

        if len(self._data) == self._lag > 1: 
            warnings.warn("Artists queue is behind by %d" % len(self._data))

        self._data.append(artists)
        self._lag = len(self._data)
        self._fig.canvas.flush_events()

    def _draw_next_frame(self, *args):
        # carry on if there's nothing to draw right now
        if self._data:
            Animation._draw_next_frame(self,*args)

    def _draw_frame(self,framedata):
        artdata = self._data.popleft()

        artists = []
        for (a,d) in artdata:
            artists.append(a)
            if d is not None:
                a.set_data(d)
        self._drawn_artists = artists
    

import signal
class App(AsyncAnimation):
    def __init__(self,*args,**kwargs):
        AsyncAnimation.__init__(self,*args,**kwargs)
        signal.signal(signal.SIGINT,self._stop)

    def _stop(self,*args):
        self._data.clear()
        AsyncAnimation._stop(self,*args)
        
    def __nonzero__(self):
        return self.event_source is not None

from gestures.utils.framebuffer import FrameBuffer
class VideoApp(App):
    def __init__(self,fig,cap=None,**kwargs):
        self._cap = cap
        if cap is None:
            self._cap = FrameBuffer.from_argv()
        App.__init__(self,fig,**kwargs)

    @property
    def cap(self):
        return self._cap

    def _stop(self,*args):
        self._cap.close()
        App._stop(self,*args)

