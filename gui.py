import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import numpy as np

class DemoGUI(object):
    def __init__(self,scale,imshape,interval=1):
        self.fig = plt.figure(dpi=100)
        self.axes = {}
        self.artists = []

        gridspec = GridSpec(2,2)
        self.axes['raw'] = gridspec.new_subplotspec((0, 0),rowspan=2)
        self.axes['draw'] = gridspec.new_subplotspec((0, 1))
        self.axes['match'] = gridspec.new_subplotspec((1, 1))
        for k,sp in self.axes.items(): self.axes[k] = self.fig.add_subplot(sp)

        self.axes['raw'].set_title('raw')
        self.axes['raw'].set_xticklabels([])
        self.axes['raw'].set_yticklabels([])
        self.axes['match'].grid(which='both')

        self.lines = {}
        self.lines['template'] = self.axes['match'].plot((),(),marker='x',color='g')[0]
        self.lines['query'] = self.axes['match'].plot((),(),marker='o',color='b')[0]
        self.lines['draw'] = self.axes['draw'].plot((),(),color='b',linewidth=5)[0]

        self.axes['match'].set_ylim(-scale//2-10,scale//2+10)
        self.axes['match'].set_xlim(-scale//2-10,scale//2+10)
        self.axes['draw'].set_ylim(imshape[0],0)
        self.axes['draw'].set_xlim(imshape[1],0)
        self.axes['draw'].xaxis.tick_top()

        self.imdisp = self.axes['raw'].imshow(np.zeros(imshape,dtype=np.uint8))

        self.fig.tight_layout()
        self.bg_cache = {ax:self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.fig.axes}

        self.draw_state = 0
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.timer = self.fig.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._update)
        self.queuelen = 0

    def show(self):
        self.fig.show()
        self.timer.start()

    def onclick(self,event):
        self.draw_state = (self.draw_state+1)%5
        if self.draw_state == 0:
            self.axes['raw'].title.set_text('raw')
        elif self.draw_state == 1:
            self.axes['raw'].title.set_text('skin')
        elif self.draw_state == 2:
            self.axes['raw'].title.set_text('motion')            
        elif self.draw_state == 3:
            self.axes['raw'].title.set_text('backproject')
        elif self.draw_state == 4:
            self.axes['raw'].title.set_text('background')
        self.timer.stop()
        self.fig.canvas.draw()
        self.timer.start()        

    def update_artists(self,artists,redraw=False):
        self.artists.append((artists,redraw))
        if (len(self.artists) - self.queuelen) > 0:
            print "Warning: Artist queue fell behind by %d" % (len(self.artists)-1)
        self.queuelen = len(self.artists)
        self.fig.canvas.get_tk_widget().update()

    def _update(self):
        if not self.artists: return
        artists, redraw = self.artists.pop(0)

        update_axes = set(a.axes for a,d in artists)
        for ax in update_axes:
            self.fig.canvas.restore_region(self.bg_cache[ax])
        for (a,d) in artists:
            a.set_data(d)
            a.axes.draw_artist(a)

        if redraw:
            self.fig.canvas.draw()
        else:
            for ax in update_axes: self.fig.canvas.blit(ax.bbox)

    def close(self):
        plt.close(self.fig)
