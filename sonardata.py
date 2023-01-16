# imports
import os
import sys
import numpy as np
import math
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata
import geopandas as gpd
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from tqdm.notebook import tqdm

class SonarData:
  """
  This class can process sonar measurements from a Totalscan Transducer Lowrance, Elite 7Ti. Two hidden methods convert the .sl2 file to a usable format, 
  after which the measurements can be assessed individually and flagged according to their amplitude profile.
  Visualizing this can be done with the SonarVisualizer class.
  Processing of the sonar measurements can only be done if the channel of the sonar is 0 throughout the entire measurement campaign.

  Parameters:
    filename = the name(s) of the .sl2 file which comes directly from the sonar
    filepath = the path to the folder in which the .sl2 file is stored
  
  Methods included:
    load() = method to load (multiple) sl2 file(s) and read them. It converts the files and produces one dataset to work with
    flag() = method to classify the data for Posidonia Oceanica
    flag_set() = method to classify the data for Posidonia Oceanica with a set range on the sonar device
    saveascsv() = method to save the loaded (and flagged) dataset to a csv file for post processing and visualizing

  Created by: Sofie Schijvenaars, 24/10/2022
  """

  # this method initiates the class
  # provide the filename of the .sl2 file and the path to the folder in which the .sl2 file is stored
  def __init__(self, filename, filepath):
    self.path = filepath
    self.name = str(filename)
    files = []
    for i in filename:
      files.append(filepath+i)
    self.files = files
  
  # this method changes the representation of the class
  # when the instance is printed, the filename is shown
  def __repr__(self):
    string = 'Files: ' + self.name + ', methods: self.load(), self.flag(), self.flag_set(), self.saveascsv()'
    return string
  
  ####################################################################################

  # some initial functions to make life easier:

  @staticmethod
  def print_bold(text):
    print(f'\033[1m{text}\033[0m')
  
  @property
  def is_loaded(self):
    if hasattr(self, 'ds'):
      return True
    else:
      return False

  @property
  def is_flagged(self):
    if hasattr(self, 'flags'):
      return True
    else:
      return False

  ####################################################################################

  # a hidden method to convert .sl2 files to block_data
  # all relevant information is extracted from the binary .sl2 file in this method
  # most important are the depth, longitude, latitude and the bouncing data (amplitudes)
  def __sl2toblock(self, filename):
    """
    A function to convert .sl2 files to a dictionary called block_data.
    The .sl2 files processed here are from a Totalscan Transducer Lowrance, Elite 7Ti.
    Only files of channel 0 can be processed with this function.

    Output:
        A dictionary with the following attributes for each measurement:
            waterDepthM = the waterdepth in meters. It is positive, going down from the watersurface.
            psize = the size of the bouncing data in bytes.
            frame_index = the index of the start of each frame (one frame equals one sounding plus additional variables in a header)
            channel = the channel of the transducer in use, on the time of acquisition
            upperlM = the upper depth limit of the transducer in use
            lowerLM = the lower depth limit of the transducer in use
            longitude = longitude in degrees
            latitude = latitude in degrees
            temp = temperature of the water in Celsius
            speed = the speed of the GPS of the transducer in use, in knots
            bdata = the sounding (just a list of values for each observation)
    
    Sources: https://www.memotech.franken.de/FileFormats/Navico_SLG_Format.pdf, https://github.com/nwhoffman/sl2PyDecode
    
    Created by: Sofie Schijvenaars, 12/09/2022
    """
    
    # Constants for conversions
    rad_polar = 6356752.3142 # polar radius of the Earth in meters
    feet2m = 1/3.2808399  # feet to meter conversions

    with open(filename, "rb") as myfile:
        # set some safety constants
        VIEW_DATA_MAX_SIZE = sys.maxsize
        SONAR_DATA_MAX_SIZE = 3072  # the length of the bouncing data is always the same for channel 0

        # find the size of the file (in bytes)
        sl2file_size = os.path.getsize(filename)
        print('--> File size (bytes) :', sl2file_size)
        print('--> Decoding....(may take a few seconds)')

        # --------- Dtypes used-----------------
        # Dtype to find the size of each block
        dt_block = np.dtype({'blockSize': ('<u2', 28)})

        # Dtype for the rest of the data from each block
        sl2blocks_dtype = np.dtype(dict(
          names=['depth', 'packetSize' , 'channel','upper_limit', 
                'lower_limit', 'FrameIndex' ,'longitude', 
                'latitude', 'temperature','speed'],
          offsets=[64, 34, 32, 40, 44, 36, 108, 112, 104, 100], # counted from start of the block in bytes
          formats=["f4", np.ushort ,np.ushort,"f4","f4", 
                  np.uint, "<u4", "<u4", "f4", "f4"],
        ))

        # Dtype for the bouncing data
        bouncing_data = np.dtype({'bouncing': (np.uint8 , 144)})

        # ---------------------------------------

        # file header is 8 bytes
        block_offset = 8 

        # empty list for block locations
        block_offset_list = [] 

        # create a list with all block offsets
        while block_offset < sl2file_size:
            # start at the beginning of each block, Just locating after the header for the first block
            myfile.seek(block_offset,0)
            # read the block size
            block_size = np.fromfile(myfile, dtype=dt_block, count=1)
            # increase the block position marker
            block_offset = block_offset + block_size[0][0] 
            # list of offsets to use later
            block_offset_list.append(block_offset) 

        # position in block_offset_list
        i=0 
        # list to hold the data
        block_data = [] 
        # initiate the counter
        data_size = 0
        # iterate over each block
        for i in range(len(block_offset_list)-1):
            # find the beginning of each block
            myfile.seek(block_offset_list[i],0) 
            # get data with the dtype created before
            data_array = np.fromfile(myfile, dtype=sl2blocks_dtype, count=1)

            # save retrieved data into single variables
            depth = data_array[0][0]
            psize = data_array[0][1]
            channel = data_array[0][2]
            upperl = data_array[0][3]
            lowerl = data_array[0][4]
            findex = data_array[0][5]
            lon = data_array[0][6]
            lat = data_array[0][7]
            temp = data_array[0][8]
            speed = data_array[0][9]

            # now let's read the bouncing data
            # repositioning
            myfile.seek(block_offset_list[i] + 144 , 0)
            # get data with the bouncing data dtype 
            bdata = np.fromfile(myfile, dtype='uint8', count=psize)

            if (channel == 0):       

                # convert depth, lower and upper limit to meters (negative depth)
                depthM = (depth * feet2m)
                upperlM = (upperl * feet2m)
                lowerlM = (lowerl * feet2m)

                # convert coords to degrees
                longitude =  lon / rad_polar * (180.0 / math.pi)
                latitude = ((2*math.atan(math.exp(lat/rad_polar)))-(math.pi/2)) * (180/math.pi)

                # fill the block data list with dictionaries
                block = {'waterDepthM': depthM, 'psize': psize ,'frame_index': findex, 'channel' : channel,
                        'upperlM' : upperlM, 'lowerLM' : lowerlM, 'longitude': longitude, 'latitude': latitude,
                        'temp' : temp, 'speed' : speed, 'bdata': bdata, 'lon_m': lon, 'lat_m': lat }
                block_data.append(block)
                # increment the counter
                data_size += 1
            # check if the counter has reached the MAX value
            # stop the reading when the counter has reached MAX value
            if data_size >= VIEW_DATA_MAX_SIZE:
                # print("Max capacity reached")
                break  
        myfile.close() # close the file opened in the while loop
    print('--> Decoding complete. Number of blocks decoded: ', len(block_data))

    # add another constant
    BOUNCING_DATA_MAX_SIZE = len(block_data)
    # self.block_data = block_data

    # paste the block_data dictionaries together
    if hasattr(self, 'block_data'):
      self.block_data.extend(block_data)
    else:
      self.block_data = block_data
    
    return

  ####################################################################################

  # a hidden method to combine multiple files into one block_data attribute
  def __combineblocks(self):
    """
    This function is meant to combine multiple .sl2 files. Use only when the files are from the same transect!
    Output is equal to __sl2toblock() method. It loops over the files and pastes the block data together.
    NOTE: the frame index will not be fixed with this method!

    Created by: Sofie Schijvenaars, 16/09/2022
    """
    for i in self.files:
      self.__sl2toblock(i)
    return

  ####################################################################################

  # a hidden method to convert the block_data to an xarray Dataset
  def __blocktoxarray(self):
    """
    This function creates an xarray Dataset from the dictionary given. The dictionary has to be of the same format as the output from sl2toblock().
    Xarray Datasets are are easy to work with and give a comprehensive and easy overview of the data in the dataset.

    Output:
      An xarray Dataset which contains all variables with labels
  
    Created by: Sofie Schijvenaars, 16/09/2022
    """

    # call the data
    block_data = self.block_data

    # defining variables needed (lots of arrays)
    data_size = 0
    ds_dict = {}
    frame_index_arr = []
    depth_arr = []
    lower_arr = []
    upper_arr = []
    psize_arr = []
    lat_arr = []
    lon_arr = []

    # creating a nparray for adding my bouncing data
    BOUNCING_DATA_MAX_SIZE = len(block_data)
    SONAR_DATA_MAX_SIZE = 3072            # size of the bouncing data is always 3072 for channel = 0
    bouncing_data = np.zeros((BOUNCING_DATA_MAX_SIZE, SONAR_DATA_MAX_SIZE), dtype='uint8')

    # in this for loop i am simply restoring the block_data into arrays that i will use to fill the xarray dataset
    for d in block_data:
      
        # data from a single dataframe
        b_data = d['bdata']
        # channel
        channel = d['channel']  
        # frame_index
        frame_index_arr.append(d['frame_index'])   
        # depth
        depth_arr.append(d['waterDepthM'])   
        # upper and lower limits
        lower_arr.append(d['lowerLM'])
        upper_arr.append(d['upperlM']) 
        # size
        psize_arr.append(d['psize'])  
        #adding bouncing data into the previously created nparray (2d array)
        bouncing_data[data_size][0:len(b_data)] = b_data
        bouncing_data[data_size][len(b_data):] = np.zeros(SONAR_DATA_MAX_SIZE - len(b_data), dtype='uint8')
        # latitude
        lat_arr.append(d['latitude'])
        # longitude
        lon_arr.append(d['longitude'])

        #checking the size of the data 
        data_size +=1
          
    # now that I have all data stored in the array, as I want to work with xarray I am adding the labels to them
    # (channel and frame index that will be used for the plot and measurement units)

    def create_record(array):
      return ( ['channel', 'frame_index'], [array], {'units': 'meters'} )
    
    # water_depth
    ds_dict['water_depth'] =  create_record(depth_arr[0:data_size])
    # lower limit
    ds_dict['lowerLM'] =  create_record(lower_arr[0:data_size])
    # upper limit
    ds_dict['upperlM'] =  create_record(upper_arr[0:data_size])
    # size of data
    ds_dict['data_size'] =  create_record(psize_arr[0:data_size])
    # latitude
    ds_dict['latitude'] =  create_record(lat_arr[0:data_size])
    # longitude
    ds_dict['longitude'] =  create_record(lon_arr[0:data_size])

    # bouncing data
    ds_dict['data'] = (
                ['channel', 'frame_index', 'depth_bin'], 
                [bouncing_data[0:data_size]], 
                {'units': 'amplitude'}
                    )

    #finally I am able to create my dataset
    ds = xr.Dataset(ds_dict, coords={
        'depth_bin': (['depth_bin'], range(0, len(bouncing_data[0]))),
        'frame_index': (['frame_index'], frame_index_arr[0:data_size]),
        'channel': (['channel'], [channel])
    })

    self.ds = ds
    print('--> Conversion complete! Number of blocks converted: ', data_size)
    return
  
  ####################################################################################

  # a method to convert the format .sl2 to a usable format, i.e. xarrayDataset
  def load(self):
    """
    This function calls the functions _sl2toblock(), _combine() and _blocktoxarray() to decode and convert the binary .sl2 files given from the sonar.
    The output is an xarray Dataset, which is easy to work with and gives a comprehensive and easy overview of the data in the dataset.

    Output:
      xarray Dataset of a sonar measurement. Attributes include:
        water_depth = the waterdepth in meters. It is positive, going down from the watersurface.
        lower_limit = the lower depth limit of the transducer in use
        upper_limit = the upper depth limit of the transducer in use
        data_size = the size of the bouncing data (always 3072 in the case of a Totalscan Transducer Lowrance, Elite 7Ti)
        data = the sounding (amplitudes of the signal) (a list of values for each observation)
      a dictionary with with important information extracted from the .sl2 files. See __sl2toblock for more information
      
    Created by: Sofie Schijvenaars, 05/10/2022
    """
    if self.is_loaded:
      self.print_bold('{} already loaded. Yay!'.format(self.name))
    else:
      self.print_bold('Converting {}'.format(self.name))
      self.__combineblocks()
      self.__blocktoxarray()
      self.print_bold('{} loaded. Yay!'.format(self.name))

  ####################################################################################

  # a method to extract the seagrass from an xarray Dataset of sonar observations
  def flag(self, skip = 600, threshold = 90, running_mean_value = 200, running_mean_value2 = 300, max_height = 2, min_height = 0.1, max_depth = 45, smoothing = 0.35,  force=False):
    """
    This function takes the sonar data from a measurement dataset and extracts the seagrass from it.
    First of all, a given amount of pings of each measurement is skipped before the extracting begins.
    This is due to noise in the measurement, caused by the motor of a boat or the oars of a kayak.

    Input:
      skip = the amount of pings which should be skipped before the extracting begins. Default is 600 pings.
      threshold = the threshold at which a signal is flagged as seagrass. This corresponds to the amount of pings counted in the maximum 10% of the signal strength. Default is 90.
      running_mean_value = the width of the running mean window used before any extraction starts. This happens AFTER skipping. Default is 200 pings.
      running_mean_value2 = the width of the running mean window used after extraction. This is used to smooth any outliers/false positives/negatives. Default is 300 pings.
      max_height = the maximum height of the seagrass. Default is 2 meters.
      min_height = the minimum height of the seagrass. Default is 10 cm.
      max_depth = the maximum depth at which seagrass can occur. Default is 45 meters.
      smoothing = weight variable for the smoothing of the flags. 0 - 0.5 is in favor of seagrass. 0.5 - 1 is in favor of no seagrass. Default is 0.35.
      force = if True, the method is applied again despite self.is_flagged = True. Default is False.
    
    Output:
      An array with equal length as the amount of observations and flagged with 1 when seagrass is detected and 0 otherwise.
    
    Created by: Sofie Schijvenaars, 05/10/2022
    Updated by: Sofie Schijvenaars, 06/12/2022
    """

    # STEP 1: check whether the data has not been processed/flagged yet
    if not self.is_loaded:
      raise TypeError("{} not yet loaded".format(self.name))
    elif len(np.unique(self.ds.lowerLM.values[0])) == 1:
      raise TypeError("{} has a fixed range. Please use self.flag_fixed()".format(self.name))
    else:
      # check whether the data has not been flagged yet or if we want to force another flagging
      if self.is_flagged and force or not self.is_flagged:

        self.print_bold('Flagging...')
        # define a running mean
        def running_mean(x, N):
          cumsum = np.cumsum(np.insert(x, 0, 0)) 
          return (cumsum[N:] - cumsum[:-N]) / float(N)
        
        # STEP 2: load the data and set initial values
        ds = self.ds
        depth = ds.water_depth.values[0]
        # the frame index is not imported from ds, because they get messed up when the instance consists of multiple files
        frame_index = np.arange(1, len(ds.water_depth.values[0])+1, 1)
        x = ds.data.values[0]
        flags = np.zeros(len(x))
        max_indices = np.zeros(len(x))
        count_zeros = 0
        nan_indices = np.zeros(len(x))

        # STEP 3: fix the 0 values of the depth
        lowerLM = ds.lowerLM.values[0]
        upperLM = ds.upperlM.values[0]
        depth_new = depth[depth != 0]
        frame_index_new = frame_index[depth != 0]
        depth_fixed = np.interp(frame_index, frame_index_new, depth_new)

        # STEP 4: create the array to skip the first few pings of each measurement
        depth_bin = (lowerLM - upperLM)/3072
        distance_to_skip = 0.5*depth_fixed
        blocks_to_skip = (lowerLM <= 4)
        pings_to_skip = distance_to_skip/depth_bin
        pings_to_skip_int = pings_to_skip.round().astype(int)
        skip_new = pings_to_skip_int*blocks_to_skip
        skip_new[skip_new == 0] = skip
        skip_new[skip_new > 2000] = 2000

        # loop over all measurements
        for i in tqdm(range(len(x))):
          # STEP 5: smooth the signals
          # to account for the motor of the boat and the noise that this causes, a given amount of pings of a measurement is skipped
          smoothed = running_mean(x[i][skip_new[i]:], running_mean_value)

          # STEP 6: find the signal
          max_i_viz = np.argmax(x[i][skip_new[i]:])
          max_i = np.argmax(smoothed)
          # sometimes, the maximum is found at index 0. This is due to the noise and the measurement is therefore taken in too shallow water or the sonar is out of the water.
          if max_i == 0:
            flags[i] = 0
            count_zeros += 1
            nan_indices[i] = i
          else:
            # find the minimum before the maximum computed
            min_i = np.argmin(smoothed[:max_i])
            # find the distance from the max index to the min index
            dist = max_i - min_i
            # extract the total signal for the seafloor
            signal = smoothed[max_i-dist:max_i+dist]

            # STEP 7: compute the histogram for the seafloor signal
            nhist = np.histogram(signal)

            # STEP 8: flag signals according to different variables
            # Histogram: if the last column of the histogram is the largest and it has more than a certain amount of counts, it can be flagged as seagrass.
            # this value is arbitrary and can vary for different transects.
            if np.max(nhist[0]) == nhist[0][-1] and np.max(nhist[0]) > threshold:
              flags[i] = 1
            # Depth: if the depth exceeds a certain threshold, it cannot be seagrass
            if depth_fixed[i] > max_depth:
              flags[i] = 0
            # Seagrass height: if the signal height is over a certain threshold, it cannot be seagrass
            if dist*depth_bin[i] > max_height and dist*depth_bin[i] < min_height:
              flags[i] = 0

          max_indices[i] = max_i_viz+skip_new[i]

        print('--> The amount of measurements which were too shallow to process or the sonar is out of the water:', count_zeros)
        flags_smoothed = (running_mean(flags, running_mean_value2) > smoothing)*1
        add = int(running_mean_value2/2)
        front = np.zeros(add)
        back = np.zeros(add-1)
        flags_smoothed_full = np.concatenate((front, flags_smoothed, back))
        flags_smoothed_full[nan_indices.astype(int)] = np.nan
        self.flags = flags_smoothed_full
        self.max_indices = max_indices
        self.print_bold('Flagged! Yay!')
      else:
        self.print_bold('{} already flagged. Yay!'.format(self.name))
    return
  
  ####################################################################################

  def flag_fixed(self, signal_width = 1100, depth_limit = 8, dist_signal = 0.3, dist_nosignal = 1, diff_limit = 50, rm1 = 20, bin_limit = 22, height_limit = 3, smoothing = 0.45, force=False):
    """
    This method is meant to process sonar data acquired in narrow, steep bays (E.g. Vroulia bay in Lipsi, Greece).
    It takes the sonar data from a measurement dataset and extracts the seagrass from it.
    The range of the sonar device must be set on one value and not change with depth. If it changes with depth, please use the self.flag() method.
    The method splits the observations into 'deep' and 'shallow' measurements. The shallow measurements will be processed in a similar way as in the self.flag() method.
    The deep measurements will be processes differently. Two signals are compared to each other at a distance of dist_signal and dist_nosignal meters from the seabed.
    When the difference between the two measurements is significant, it means that there is biomass at dist_signal meters above the seabed, which is classified as seagrass.
    Filters used in this method: running mean and Kalman filter.

    Input:
      signal_width = half of the water depth will be skipped initially. The signal_width states how much pixels should be considered after the initial skip for the seabed signal.
                      Make sure that the full seabed signal is included. Default is 1100 pixels.
      depth_limit = limit in meters where to split the dataset. Everything above the depth limit will be processed
                    in a similar way as in the self.flag() method. Default is 8 meters.
      dist_signal = distance in meters above the seabed to assess the presence of seagrass. Default is 0.3 meter.
      dist_nosignal = distance in meters above the seabed which has a high confidence of not having seagrass. Default is 1 meter.
      diff_limit = the limit in amplitude which the difference between the amplitude at dist_signal and dist_nosignal has to exceed in order to
                    classify the observation as seagrass. Default is 50.
      rm1 = the width of the running mean window used to smooth the histogram method of the shallow observations. Default is 20.
      bin_limit = the threshold which the last bin of the histogram method has to exceed in order to classify the observation as seagrass.
                    This corresponds to the amount of pings counted in the maximum 10% of the signal strength. Default is 22.
      height_limit = a limit in meters of the height of the seagrass signal. When the height exceeds this limit it is likely not seagrass. Default is 3 meters.
      smoothing = weight variable for the smoothing of the flags. 0 - 0.5 is in favor of seagrass. 0.5 - 1 is in favor of no seagrass. Default is 0.45.
      force = if True, the method is applied again despite self.is_flagged = True. Default is False.
    
    Output:
      An array with equal length as the amount of observations and flagged with 1 when seagrass is detected and 0 otherwise.
      An array with the indices of the maximum amplitude found within the amplitude profile of each observation. This equals the depth of the seabed.

    NOTES:
      This method is tested only for lower limits of 50 meters (range of 25 meters on sonar device)! When other lower limits are considered, use other values for the variables.
      This method is only tested on Vroulia bay in Lipsi, Greece. Other bays can have a significantly lower accuracy and different settings for the variables.

    Created by: Sofie Schijvenaars, 06/12/2022
    """
    # STEP 1: check whether the data has not been processed/flagged yet
    if not self.is_loaded:
      raise TypeError("{} not yet loaded".format(self.name))
    elif len(np.unique(self.ds.lowerLM.values[0])) != 1:
      raise TypeError("{} has variable range. Please use self.flag()".format(self.name))
    else:
      # check whether the data has not been flagged yet or if we want to force another flagging
      if self.is_flagged and force or not self.is_flagged:
        self.print_bold('Flagging...')

        # STEP 2: define functions
        def Kalman(data):
          f = KalmanFilter (dim_x=2, dim_z=1)

          # state transition matrix
          f.F = np.array([[1.,1.],
                          [0.,1.]])
          
          # measurement function
          f.H = np.array([[1.,0.]])

          # covariance matrix
          f.P *= 1000.

          # measurement noise
          f.R = 250

          # process noise
          f.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.01)

          # initial state
          f.x = np.array([data[0], 0.])
          s = np.zeros(len(data))
          v = np.zeros(len(data))

          # loop over all pings
          for j in range(len(data)-1):
            z = data[j+1]
            f.predict()
            f.update(z)
            s[j] = f.x[0]
            v[j] = f.x[1]
          return s

        def running_mean(x, N):
          cumsum = np.cumsum(np.insert(x, 0, 0))
          return (cumsum[N:] - cumsum[:-N]) / float(N)

        # STEP 3: call variables
        data = self.ds.data.values[0]
        depth = self.ds.water_depth.values[0]
        lowerlimits = self.ds.lowerLM.values[0]

        ffull = np.arange(0,len(data))
        max_indices = np.zeros(len(data))
        bin_count = np.copy(max_indices)
        flags = np.copy(max_indices)
        dist = np.copy(max_indices)
        signaldiff = np.copy(max_indices)

        # STEP 4: fix the 0 values of the depth
        depth_new = depth[depth != 0]
        frame_index_new = ffull[depth != 0]
        depth_fixed = np.interp(ffull, frame_index_new, depth_new)

        # STEP 5: create the array to skip half the depth of each measurement
        depth_bin = lowerlimits/3072
        distance_to_skip = 0.5*depth_fixed
        skip = np.round(distance_to_skip/depth_bin).astype(int)
        skip[skip > 2000] = 2000
        skip_new = skip

        # STEP 6: identify some variables
        mtopixels = np.unique(lowerlimits)/3072
        pixels2m = np.round(2/mtopixels).astype(int)[0]
        cm10signal = np.round(dist_signal/mtopixels).astype(int)[0]
        cmnosignal = np.round(dist_nosignal/mtopixels).astype(int)[0]

        # STEP 7: loop through all the measurements
        for i in tqdm(range(len(data))):
          s = data[i][skip_new[i]:skip_new[i]+signal_width]
          k = Kalman(s)

          max_i = np.argmax(k)
          max_indices[i] = max_i+skip_new[i]
          min_i = np.argmin(k[:max_i.astype(int)])

          if depth_fixed[i] < depth_limit:
            start = (max_i-min_i).astype(int)
            end = (2*max_i+min_i).astype(int)
            signal = k[start:end]
            dist[i] = len(signal)*mtopixels
            hist = np.histogram(signal)
            bin_count[i] = hist[0][-1]
          else:
            start = max((max_i-pixels2m).astype(int), 0)
            end = (2*max_i+min_i).astype(int)
            signal = k[start:end]
            bin_count[i] = np.histogram(signal)[0][-1]
            signal10 = k[max_i.astype(int)-cm10signal]
            signalno = k[max_i.astype(int)-cmnosignal]
            signaldiff[i] = signal10 - signalno
            if signaldiff[i] > diff_limit:
              flags[i] = 1

        # STEP 8: smooth the shallow part
        bin_smoothed = running_mean(bin_count, rm1)
        add = int(rm1/2)
        front = np.zeros(add)
        back = np.zeros(add-1)
        bin_new = np.concatenate((front, bin_smoothed, back))
        for i in range(len(bin_count)):
          if bin_count[i] != 0:
            if bin_new[i] > bin_limit and dist[i] < height_limit:
              flags[i] = 1
        
        flags_smoothed = (running_mean(flags, rm1) > smoothing)*1
        flags_smoothed_full = np.concatenate((front, flags_smoothed, back))

        self.flags = flags_smoothed_full
        self.max_indices = max_indices
      else:
        self.print_bold('{} already flagged. Yay!'.format(self.name))
    return
  
  ####################################################################################

  def saveascsv(self, output_filename):
    """
    This function saves the data as a csv file, but the data needs to be loaded with the load() method and classified with the flag() method first.
    """
    if not self.is_loaded:
      raise TypeError('{} not loaded yet'.format(self.name))
    else:
      depth = self.ds.water_depth.values[0]
      lat = self.ds.latitude.values[0]
      lon = self.ds.longitude.values[0]
      if self.is_flagged:
        flags = self.flags
        d = {'lat': lat, 'lon': lon, 'depth': depth, 'flag': flags}
        print('--> Data is flagged')
      else:
        d = {'lat': lat, 'lon': lon, 'depth': depth}
        print('--> Data is not flagged')
      data = pd.DataFrame(data=d)
      data.to_csv(r'/content/drive/MyDrive/Archipelagos/Output/{}'.format(output_filename))
    return