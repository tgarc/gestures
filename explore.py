import h5py
import matplotlib.pyplot as plt

fn = '/home/tdos/gestures/libras.hdf5'
libras_fh = h5py.File(fn,'r')
libras = libras_fh['libras'].value

try:
    fig = plt.figure()
    txt = fig.suptitle('',fontsize=12)
    ax = fig.add_subplot(111)
    line = plt.Line2D((),(),marker='o',markerfacecolor='k')
    ax.add_artist(line)
    ax.grid(which='both')
    ax.minorticks_on()

    databuff = iter(libras[['x','y','class']])
    timer = fig.canvas.new_timer(interval=1000)

    def update():
        x,y,c = databuff.next()
        line.set_data(x,y)
        txt.set_text("Class '%s'" % c)

        # ax.draw_artist(line)
        # fig.draw_artist(txt)
        # fig.canvas.blit(ax.bbox)
        # fig.canvas.blit(txt)
        
        fig.canvas.draw()

    def onclick(event):
        update()
        timer.start()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    timer.add_callback(update)

    update()        #draw first frame
    timer.start()

    plt.show()
except KeyboardInterrupt:
    pass
finally:
    plt.close(fig)
    fig.canvas.mpl_disconnect(cid)
