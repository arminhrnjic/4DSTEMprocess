import numpy as np
import mib
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math
from tkinter import filedialog as fd
import cv2
########################################################################################################################
Plotting=True
Writing=True
########################################################################################################################
# Open a file - now automatic
filename = fd.askopenfilename(initialdir='/',filetypes=[("Merlin binary","*.mib")])
print(filename)

with open(filename, 'rb') as f:
    buffer_mib = f.read()
########################################################################################################################
# Detector settings (Resolution can be 128 or 256 or 512)
Resolution = 512
Frame_To_See=1
########################################################################################################################
#Ru_TiON_3_128_DF111.tif
# Load parameter to buffer
try:
    data_from_buffer = mib.loadMib(buffer_mib, scan_size=(Resolution * Resolution))
    Frame = data_from_buffer[Frame_To_See - 1]
except:
    try:
        Resolution=256
        data_from_buffer = mib.loadMib(buffer_mib, scan_size=(Resolution * Resolution))
        Frame = data_from_buffer[Frame_To_See - 1]
    except:
        Resolution = 128
        data_from_buffer = mib.loadMib(buffer_mib, scan_size=(Resolution * Resolution))
        Frame = data_from_buffer[Frame_To_See - 1]

########################################################################################################################

def center_Of_Mass(datab):
    """Calculates center of mass from the binary merlin raw data and returns amplitude and angle
        Input: Binary data
        Output: x, y, angle, amplitude
    """

    def cartesian_to_polar(x, y):
        """Converts cartesian coordinates to polar
            Input: x,y
            Output: amplitude and angle
        """
        r = math.sqrt((x * x) + (y * y))
        theta = math.atan(y / x)

        return (r, theta)


    cOMs=[]

    # Finding center of mass using numpy
    for frame in datab:
        cOMs.append(nd.center_of_mass(frame))

    cOMx=[i[0] for i in cOMs]
    cOMy=[i[1] for i in cOMs]

    center_cOMx=sum(cOMx)/len(cOMx)
    center_cOMy=sum(cOMy)/len(cOMy)

    cOMx=cOMx-center_cOMx
    cOMy=cOMy-center_cOMy


    r_out=[] # Amplitude
    theta_out=[] # Angle

    for i in range(len(cOMx)):
        r_out.append(cartesian_to_polar(cOMx[i],cOMy[i])[0])
        theta_out.append(cartesian_to_polar(cOMx[i],cOMy[i])[1])

    return (r_out,theta_out)

def dpc_mask(h, w, center=None):
    """
    Creates four segments for a dpc mask
    Input: height, width, center*
    Output: Segment1, Segment2, Segment3, Segment4 [binary]
    """
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))

    # Create a matrix of ones with size 100x100
    InputImage = np.ones((h, w))

    # Get the size of the input image
    SizeX, SizeY = InputImage.shape

    # Define center of the matrix
    CenterX = 55
    CenterY = 60

    # Create the mesh-grid
    X, Y = np.meshgrid(np.linspace(1, SizeX, SizeX) - center[0], np.linspace(1, SizeY, SizeY) - center[1])

    # Calculate Z1 and Z2
    Z1 = X + Y
    Z2 = Y - X

    # Set values of Z1 and Z2 less than 0 to 0
    Z1[Z1 < 0] = 0
    Z2[Z2 < 0] = 0

    # Change Z1 and Z2 to logical values
    Z1 = Z1.astype(bool)
    Z2 = Z2.astype(bool)

    # Calculate Z3 and Z4
    Z3 = np.logical_not(Z1)
    Z4 = np.logical_not(Z2)

    # Create the segments
    Segment4 = np.logical_and(Z1, Z2)
    Segment1 = np.logical_and(Z1, Z4)
    Segment2 = np.logical_and(Z2, Z3)
    Segment3 = np.logical_and(Z3, Z4)


    return [Segment1,Segment2, Segment3, Segment4]

def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates circular binary mask
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius

    return mask

########################################################################################################################
img=cv2.cvtColor(Frame, cv2.COLOR_GRAY2BGR)
images=[cv2.cvtColor(data_from_buffer[i],cv2.COLOR_GRAY2BGR) for i in range(Resolution*Resolution)]
# Create global variables for circle parameters
x = Resolution//2
y = Resolution//2
r = 10
ro = 20
selected_image = 0
button_pressed=False

def on_button_pressed(state):
    global button_pressed
    button_pressed=True
# Create function to update circle
def update_circle(val):
    global x, y, r, ro, button_pressed
    x = cv2.getTrackbarPos("x", "image")
    y = cv2.getTrackbarPos("y", "image")
    r = cv2.getTrackbarPos("r", "image")
    ro = cv2.getTrackbarPos("r_out", "image")

# Create function to update selected image
def update_image(val):
    global selected_image
    selected_image = cv2.getTrackbarPos("Image", "image")

# Create trackbars for circle parameters
cv2.namedWindow("image")



cv2.createTrackbar("x", "image", 0, img.shape[1], update_circle)
cv2.createTrackbar("y", "image", 0, img.shape[0], update_circle)
cv2.createTrackbar("r", "image", 0, 100, update_circle)
cv2.createTrackbar("r_out", "image", 20, 200, update_circle)
# Create trackbar for selected image
cv2.createTrackbar("Image", "image", 0, len(images) - 1, update_image)
cv2.createTrackbar("Push right", "image", 0, 1, on_button_pressed)
# Display image with circle
while True:
    img = images[selected_image]
    img_circle = img.copy()
    cv2.circle(img_circle, (x, y), r, (0, 255, 0), 2)
    cv2.circle(img_circle, (x, y), ro, (0, 255, 0), 2)
    # Naming a window
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Using resizeWindow()
    cv2.resizeWindow("image", 1024, 1024)
    img_circle=cv2.resize(img_circle, (1024, 1024))
    cv2.imshow("image", img_circle)

    key = cv2.waitKey(1) & 0xFF
    #if key == ord("r"):
      #break
    if button_pressed:
        break


print(selected_image)
# Create binary mask
mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

img_masked = cv2.bitwise_and(img, img, mask=mask)

# Display masked image
cv2.imshow("Masked Image", img_masked)
cv2.waitKey(0)

Inner_mask_x=x
Inner_mask_y=y
Inner_mask_r=r
Outer_mask_x = Inner_mask_x
Outer_mask_y = Inner_mask_y
Outer_mask_r = ro

########################################################################################################################
# Creating mask

# Masks for DF
mask_inner = create_circular_mask(256, 256, (Inner_mask_x, Inner_mask_y), Inner_mask_r)
mask_out = create_circular_mask(256, 256, (Outer_mask_x, Outer_mask_y), Outer_mask_r)
mask_inner = mask_inner.astype(int)
mask_out = mask_out.astype(int)
mask_BF=mask_inner
mask_inner = np.where((mask_inner == 0) | (mask_inner == 1), mask_inner ^ 1, mask_inner)


# Masks for DPC
dpc_s1=dpc_mask(256,256,(Inner_mask_x, Inner_mask_y))[0]
dpc_s2=dpc_mask(256,256,(Inner_mask_x, Inner_mask_y))[1]
dpc_s3=dpc_mask(256,256,(Inner_mask_x, Inner_mask_y))[2]
dpc_s4=dpc_mask(256,256,(Inner_mask_x, Inner_mask_y))[3]


# Not needed for now
merlin_type = np.dtype([('header', np.string_, 768), ('data', np.dtype('>u2'), (256, 256))]) # Not needed for now

########################################################################################################################
# Plot frame
Frame = data_from_buffer[selected_image - 1]

# Plot mask
figure, axes = plt.subplots()
in_mask_plot = plt.Circle((Inner_mask_x, Inner_mask_y), Inner_mask_r, fill=False)
out_mask_plot = plt.Circle((Outer_mask_x, Outer_mask_y), Outer_mask_r, fill=False)
axes.imshow(Frame)
axes.set_aspect(1)
axes.add_artist(in_mask_plot)
axes.add_artist(out_mask_plot)
plt.title('Frame: ' + str(selected_image))
plt.show()


########################################################################################################################
# BF image formation
BF_image = np.zeros(Resolution * Resolution)
for i in range(Resolution*Resolution):
    Frame = data_from_buffer[i]
    out_frame = Frame*mask_BF


    BF_image[i] = np.sum(out_frame)

BF_image = np.reshape(BF_image, (Resolution, Resolution))

########################################################################################################################
# DF image formation
DF_image = np.zeros(Resolution * Resolution)
for i in range(Resolution*Resolution):
    Frame = data_from_buffer[i]
    out_frame = Frame
    # Virtual detector logic
    out_frame_DF_in = out_frame * mask_inner
    out_frame_DF = out_frame_DF_in * mask_out


    DF_image[i] = np.sum(out_frame_DF)

DF_image = np.reshape(DF_image, (Resolution, Resolution))

########################################################################################################################

"""
DPC - Differential phase contrast
"""


# DCP x image formation
DPCx_image = np.zeros(Resolution * Resolution)
for i in range(Resolution*Resolution):
    Frame = data_from_buffer[i]
    out_frame = Frame
    # Virtual detector logic
    out_frame_DF_in = out_frame * mask_inner
    out_frame_DF = out_frame_DF_in * mask_out


    out_frame_DPC1x= out_frame_DF * dpc_s1
    out_frame_DPC2x= out_frame_DF * dpc_s2

    DPCx_image[i] = np.sum(out_frame_DPC1x) - np.sum(out_frame_DPC2x)
DPCx_image = np.reshape(DPCx_image, (Resolution, Resolution))

# DCP y image formation
DPCy_image = np.zeros(Resolution * Resolution)
for i in range(Resolution * Resolution):
    Frame = data_from_buffer[i]
    out_frame = Frame
    # Virtual detector logic
    out_frame_DF_in = out_frame * mask_inner
    out_frame_DF = out_frame_DF_in * mask_out

    out_frame_DPC1y = out_frame_DF * dpc_s3
    out_frame_DPC2y = out_frame_DF * dpc_s4
    DPCy_image[i] = np.sum(out_frame_DPC1y) - np.sum(out_frame_DPC2y)
DPCy_image = np.reshape(DPCy_image, (Resolution, Resolution))

########################################################################################################################
"""Center of mass calculation"""

r_plot=np.array(center_Of_Mass(data_from_buffer)[0]) # Calculate amplitude
theta_plot=np.array(center_Of_Mass(data_from_buffer)[1]) # Calculate angle
r_plot = np.reshape(r_plot, (Resolution, Resolution)) # Reshape format
theta_plot = np.reshape(theta_plot, (Resolution, Resolution)) # Reshape format

########################################################################################################################

def WritingCSV(Resolution,Filename,image,name):

    with open(filename[:-4]+'_'+name+'.csv','w') as f:
        f.write('x'+','+'y'+','+name+'\n')
        for i in range(Resolution):
            for j in range(Resolution):

                f.write(str(i)+','+str(-j)+','+str(image[i][j])+'\n')

def WritingCSV_CoM(Resolution,Filename,image1,image2,name):

    with open(filename[:-4]+'_'+name+'.csv','w') as f:
        f.write('x'+','+'y'+','+name+'\n')
        for i in range(Resolution):
            for j in range(Resolution):

                f.write(str(i)+','+str(-j)+','+str(image1[i][j])+','+str(image2[i][j])+'\n')

if Writing is True:
    WritingCSV(Resolution, filename, DF_image, 'DF_image')
    WritingCSV(Resolution, filename, BF_image, 'BF_image')
    WritingCSV(Resolution, filename, DPCx_image, 'DPCx_image')
    WritingCSV(Resolution, filename, DPCy_image, 'DPCy_image')
    WritingCSV_CoM(Resolution, filename, r_plot, theta_plot, 'CoM_image')
else:
    print('Turn the variable "Writing" to True to save the images as csv')

if Plotting is True:
    # Plotting final images
    plt.figure("Bright field image")
    plt.imshow(BF_image,cmap='Greys')
    plt.figure("Dark field image")
    plt.imshow(DF_image,cmap='Greys')
    plt.figure("Differential phase contrast - X image")
    plt.imshow(DPCx_image,cmap='Greys')
    plt.figure("Differential phase contrast - Y image")
    plt.imshow(DPCy_image,cmap='Greys')
    plt.figure("Center of mass - amplitude image")
    plt.imshow(r_plot,cmap='Greys')
    plt.figure("Center of mass - angle image")
    plt.imshow(theta_plot,cmap='Greys')
    plt.show()
else:
    print('Turn the variable "Plotting" to True to show the images')