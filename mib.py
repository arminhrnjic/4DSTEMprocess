import numpy as np
import os

class mib_properties(object):
    '''Class covering Merlin MIB file properties.'''
    def __init__(self):
        '''Initialisation of default MIB properties. Single detector, 1 frame, 12 bit'''
        self.path = ''
        self.buffer = True
        self.merlin_size = (256,256)
        self.single = True
        self.quad = False
        self.raw = False
        self.dyn_range = '12-bit'
        self.packed = False
        self.pixeltype = np.uint16
        self.headsize = 384
        self.offset = 0
        self.addCross = False
        self.scan_size = (1,1)
        self.xy = 1
        self.numberOfFramesInFile = 1
        self.gap = 0
        self.quadscale = 1
        self.detectorgeometry = '1x1'
        self.frameDouble = 1
        self.roi_rows = 256
        

    def show(self):
        '''Show current properties of the Merlin file. Use get_mib_properties(path/buffer) to populate'''
        if not self.buffer:
            print("\nPath:",self.path)
        else:
            print("\nData is from a buffer")
        if self.single: 
            print("\tData is single")
        if self.quad: 
            print("\tData is quad")
            print("\tDetector geometry",self.detectorgeometry)
        print("\tData pixel size",self.merlin_size)
        if self.raw: 
            print("\tData is RAW")
        else: 
            print("\tData is processed")
        print("\tPixel type:", np.dtype(self.pixeltype))
        print("\tDynamic range:", self.dyn_range)
        print("\tHeader size:", self.headsize,"bytes")
        print("\tNumber of frames in the file/buffer:", self.numberOfFramesInFile)
        print("\tNumber of frames to be read:", self.xy)

def get_mib_properties(head):
    '''parse header of a MIB data and return object containing frame parameters'''    

    # init frame properties
    fp = mib_properties()
    # read detector size
    fp.merlin_size = (int(head[4]),int(head[5]))

    # test if RAW 
    if head[6] == 'R64': fp.raw = True

    if head[7].endswith('2x2'):            
            fp.detectorgeometry = '2x2'
    if head[7].endswith('Nx1'):
            fp.detectorgeometry = 'Nx1'    

    # test if single
    if head[2] == '00384': 
        fp.single = True
    # test if quad and read full quad header
    if head[2] == '00768': 
        # read quad data 
        with open(path,'rb') as f:
            head = f.read(768).decode().split(',')
        fp.headsize = 768
        fp.quad = True
        fp.single = False

    # set bit-depths for processed data (binary is U08 as well)
    if not fp.raw:
        if head[6] == 'U08': 
            fp.pixeltype = np.dtype('uint8')
            fp.dyn_range = '1 or 6-bit'
        if head[6] == 'U16': 
            fp.pixeltype = np.dtype('>u2')
            fp.dyn_range = '12-bit'
        if head[6] == 'U32': 
            fp.pixeltype = np.dtype('>u4')
            fp.dyn_range = '24-bit'

    return fp

def processedMib(mib_prop):
    '''load processed mib file, return is memmapped numpy file of specified geometry'''

    # define numpy type for MerlinEM frame according to file properties
    merlin_frame_dtype = np.dtype([('header', np.string_, mib_prop.headsize), ('data', mib_prop.pixeltype, mib_prop.merlin_size)])
    
    # generate offset in bytes
    offset = mib_prop.offset * merlin_frame_dtype.itemsize
    
    # map the file to memory, if a numpy or memmap array is given, work with it as with a buffer
    # buffer needs to have the exact structure of MIB file, if it is read from TCPIP interface it needs to drop first 15 bytes which describe the stream size. Also watch for the coma in front of the stream.
    if type(mib_prop.path)==str: 
        data = np.memmap(
            mib_prop.path,
            dtype=merlin_frame_dtype,
            offset=mib_prop.offset,
            shape=mib_prop.scan_size 
            )
    if type(mib_prop.path)==bytes:
        data = np.frombuffer(
            mib_prop.path,
            dtype=merlin_frame_dtype,
            count = mib_prop.xy,
            offset=mib_prop.offset
            )
        data = data.reshape(mib_prop.scan_size)

    # remove header data and return
    return data['data']

def loadMib(path_buffer,scan_size=(1,1)):
    '''Load Quantum Detectors MIB file from a path or a memory buffer.\nScan_size=(x,y) is 2D size of the STEM scan or scan_size=(n) if image stack is loaded, debug=True prints header file.'''
    
    #read header from the start of the file or buffer
    if type(path_buffer)==str: 
        try:
            with open(path_buffer,'rb') as f:
                head = f.read(384).decode().split(',')
                f.seek(0, os.SEEK_END)
                filesize = f.tell()
        except:
            print('File does not contain MIB header')
            return 1
    if type(path_buffer)==bytes:
        try:
            filesize = len(path_buffer)
            head = path_buffer[:384].decode().split(',')
        except:
            print('Buffer does not contain MIB header')
            return 1

    #parse header info 
    mib_prop = get_mib_properties(head)
    mib_prop.path = path_buffer

    #correct for buffer/file logic
    if type(path_buffer)==str: 
        mib_prop.buffer = False

    #find the size of the data
    merlin_frame_dtype = np.dtype([('header', np.string_, mib_prop.headsize), ('data', mib_prop.pixeltype, mib_prop.merlin_size)])
    mib_prop.numberOfFramesInFile = filesize // merlin_frame_dtype.itemsize

    mib_prop.scan_size = scan_size
    if type(scan_size)==int: mib_prop.xy = scan_size
    if type(scan_size)==tuple: mib_prop.xy = scan_size[0]*scan_size[1]

    #show file properties
    mib_prop.show()
    
    if mib_prop.xy>mib_prop.numberOfFramesInFile:
        print("Error:\nRequested number of files:",mib_prop.xy,"is smaller than the number of available files:",mib_prop.numberOfFramesInFile)
        return 1

    if mib_prop.raw: 
        print('RAW MIB data not supported.')
        return 1
    else: 
        return processedMib(mib_prop)

