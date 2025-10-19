#!/home/jpcostel/Projects/pumpkin_lights/neopixel_venv/bin/python3
import time
import board
import neopixel

# Define pins and number of LEDs per ring
PIXELS_PER_RING = 16
PIXELS=64
RINGS=4
RING1_PIN = board.D18
#RING2_PIN = board.D21

COLORS=((128, 0, 255), (0, 255, 0), (255, 64, 0), (255,0,0))

# Create NeoPixel objects
ring1 = neopixel.NeoPixel(RING1_PIN, PIXELS, brightness=0.75, auto_write=False)
#ring2 = neopixel.NeoPixel(RING2_PIN, PIXELS_PER_RING, brightness=0.5, auto_write=False)

# Function to fill a ring with a color
def fill_ring(ring, color):
    ring.fill(color)
    ring.show()

def offset_color(ring, color, pixel, colors=COLORS, count=PIXELS, ppr=PIXELS_PER_RING):
    ring_offset = pixel // ppr
    color_index = (color + ring_offset) % len(colors)
    ring[pixel] = colors[color_index]
    # print("ring_offset:{} pixel:{} ppr:{} color_index:{} color:{}".format(ring_offset, pixel, ppr, color_index, color, ))

def fifo(ring, on_color, colors=COLORS, off_color=(0,0,0), count=PIXELS, width=0, sleep=0.025):
    for i in range(count):
        ring.fill(off_color)
        ring[i] = colors[on_color]
        if width:
            for j in range(width):
                if i+j < count:
                    offset_color(ring, on_color, i+j)
        ring.show()
        time.sleep(sleep)
    ring.fill((0,0,0))
    ring.show()

def fifo4(ring, on_color, colors=COLORS, off_color=(0,0,0), count=PIXELS_PER_RING, width=0, sleep=0.025):
    for i in range(count):
        ring.fill(off_color)
        for j in range(width+1):
            set4(ring, i+j, on_color)
        ring.show()
        time.sleep(sleep)
    ring.fill((0,0,0))
    ring.show()

def set4(ring, position, color, colors=COLORS, offset=PIXELS_PER_RING, count=PIXELS):
    indices = (position, position+offset, position+(2*offset), position+(3*offset))
    for index in indices:
        if index < count and index >= 0:
            offset_color(ring, color, index)

def bounce(ring, on_color, colors=COLORS, off_color=(0,0,0), count=PIXELS, width=1, sleep=0.02):
    for i in range(count):
        ring.fill(off_color)
        ring[i] = colors[on_color]
        for j in range(width):
            if i+j < count:
                offset_color(ring, on_color, i+j)
            if i-j >= 0:
                offset_color(ring, on_color, i-j)
        ring.show()
        time.sleep(sleep)
    for i in range(count-1, -1, -1):
        ring.fill(off_color)
        ring[i] = colors[on_color]
        for j in range(width):
            if i+j < count:
                offset_color(ring, on_color, i+j)
            if i-j >= 0:
                offset_color(ring, on_color, i-j)
        ring.show()
        time.sleep(sleep)
    ring.fill((0,0,0))
    ring.show()

def bounce4(ring, on_color, colors=COLORS, off_color=(0,0,0), count=PIXELS_PER_RING, width=1, sleep=0.05):
    for i in range(count):
        ring.fill(off_color)
        set4(ring, i, on_color)
        for j in range(width):
            set4(ring, i+j, on_color)
        ring.show()
        time.sleep(sleep)

    for i in range(count-1, -1, -1):
        ring.fill(off_color)
        set4(ring, i, on_color)
        for j in range(width):
            set4(ring, i+j, on_color)
        ring.show()
        time.sleep(sleep)
    ring.fill(off_color)
    ring.show()

def spindown(ring, color, start=0.01, end=0.05, steps=20, width=1):
    step = (end - start)/steps
    for n in range(1, steps + 1):
        nsleep = start + n*step
        #print("nsleep:{} n:{}".format(nsleep, n))
        fifo4(ring, color, width=width, sleep=nsleep)


def spinup(ring, color, start=0.05, end=0.01, steps=20, width=1):
    for n in range(1, steps + 1):
        step = (start - end)/steps
        nsleep = start - n*((start-end)/steps)
        #print("nsleep:{} n:{}".format(nsleep, n))
        fifo4(ring, color, width=width, sleep=nsleep)

def alternate4(ring, on_color, colors=COLORS, off_color=(0,0,0), count=10, sleep=-.25):
    rings = 4
    ppr = 16
    rings = [[True, 0], [False, 16], [True, 32], [False, 48]]
    for i in range(count):
        for r in rings:
            if r[0]:
                #Turn this ring on and set it false
                r[0] = False
                for pix in range(r[1], (r[1]+ppr)):
                    offset_color(ring, on_color, pix)
            else:
                r[0] = True
                for pix in range(r[1], (r[1]+ppr)):
                    offset_color(ring, 0, pix, colors=[(0,0,0)])
        ring.show()
        time.sleep(sleep)
    ring.fill((0,0,0))
    ring.show()

def alternate(ring, on_color, colors=COLORS, width=3, off_color=(0,0,0), sleep=0.1, pixels=PIXELS):
    count = 0
    for i in range(width):
        ring.fill(off_color)
        for pix in range(pixels):
            mod = pix % width
            if mod == count:
                offset_color(ring, on_color, pix)
        count += 1
        ring.show()
        time.sleep(sleep)

def alternate_left(ring, on_color, colors=COLORS, width=3, off_color=(0,0,0), sleep=0.1, pixels=PIXELS):
    count=width - 1
    for i in range(width):
        ring.fill(off_color)
        for pix in range(pixels):
            mod = pix % width
            if mod == count:
                offset_color(ring, on_color, pix)
        count -= 1
        ring.show()
        time.sleep(sleep)

# Example animation
def main():
    colors = [0,1,2,3]
    width = 3
    fast = 0.005
    slow = 0.035
    steps = 6
    try:
        while True:
            for color in colors:
                alternate4(ring1, color, sleep=0.5)
                for j in range(2,5):
                    for i in range(10):
                        alternate(ring1, color, width=j)
                    for i in range(10):
                        alternate_left(ring1, color, width=j)
                spinup(ring1, color, start=slow, end=fast, steps=steps, width=width)
                for n in range(20):
                    fifo4(ring1, color, sleep=fast, width=width)
                spindown(ring1, color, start=fast, end=slow, steps=steps, width=width)
                alternate4(ring1, color, sleep=0.5)
    except KeyboardInterrupt:
        ring1.fill((0, 0, 0))
        ring1.show()

if __name__ == "__main__":
    main()

