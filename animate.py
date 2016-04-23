#!/usr/bin/env python

import os
import sys
import re
import numpy as np
try:
    import pydicom
except:
    import dicom as pydicom
import matplotlib.pyplot as plot
import matplotlib.animation as animation

FRAME_NAME_PATTERN = 'IM-(\d{4,})-(\d{4})\.dcm'

def get_frames(folder_name):
    files = []
    for file_name in os.listdir(folder_name):
        match = re.match(FRAME_NAME_PATTERN, file_name)
        if match:
            files.append(file_name)
        else:
            print >>sys.stderr, 'file %s is ignored' % file_name
    # sort by time
    files = sorted(files, key=lambda f: int(re.match(FRAME_NAME_PATTERN, f).group(2)))
    frames = []
    for f in files:
        full_path = os.path.join(folder_name, f)
        ds = pydicom.read_file(full_path)
        frames.append(np.array(ds.pixel_array))
    print 'total number of frames %s' % len(frames)
    return frames

def animate(frames):
    if isinstance(frames, (np.ndarray, np.generic)):
        frames = [frames[i] for i in xrange(frames.shape[0])]
    if not frames:
        return
    if isinstance(frames[0], dict):
        frames = [f['pixel'] for f in frames]

    figure = plot.figure()
    image = plot.imshow(frames[0], animated=True)
    def update_func(idx):
        image.set_array(frames[idx % len(frames)])
        return image
    anime = animation.FuncAnimation(figure, update_func, interval=100, blit=False)
    plot.show()

def main(folder_name):
    frames = get_frames(folder_name)
    animate(frames)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s path_to_slice (i.e. raw_data/train/1/study/2ch_21/)' % sys.argv[0]
        sys.exit(1)
    main(sys.argv[1])
