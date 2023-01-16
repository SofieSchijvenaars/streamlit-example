import matplotlib.pyplot as plt
import numpy as np
from sonardata import SonarData

class SonarVisualizer():
  """
  When the sonar campaigns need visualizing, this class can be used.

  Methods included:
    plot_raw() = plot the raw sonar data
    plot_flagged() = plot the classified sonar data
    plot_interpolated_depth() = plot the interpolated depth data in a map
    plot_raw_depth() = plot the raw depth data in a map
  """

  def __init__(self, data, landmask = None):
    if not isinstance(data, SonarData):
      raise TypeError('Visualizer must be initialized with a SonarData dataset')
    if not data.is_flagged:
      raise ValueError('Visualizer needs a flagged SonarData dataset')
    
    self.data = data
    self.landmask = landmask
  
  ####################################################################################

  # some initial functions to make life easier
  
  @property
  def is_interpolated(self):
    if hasattr(self, 'interpolated_depth'):
      return True
    else:
      return False


  ####################################################################################

  def plot_raw(self):
    """
    This function visualizes the raw data extracted from the sonar data.
    """
    plt.figure(figsize=(20,4))
    plt.imshow(self.data.ds.data.values[0].T)
    plt.show()
    return

  ####################################################################################

  def plot_flagged(self):
    """
    This function visualizes the flagged data extracted from the sonar data.
    """
    plt.figure(figsize=(20,4))
    plt.imshow(self.data.ds.data.values[0].T)
    plt.plot(self.data.max_indices*self.data.flags,'r.',markersize=0.5, label = 'Seagrass flag')
    plt.legend()
    plt.show()
    return
  
  ####################################################################################

  # function to interpolate depth data
  def __interp_depth(self, method = 'linear'):
    """""
    A function to interpolate the depth of a sonar measurement campaign.
    See scipy.interpolate.griddata for more information on the interpolation method used.

    Input:
      df = the dataset which contains the depth of the sonar campaign. It can also be a list of dictionaries.
      method = the method used for interpolating: 'nearest', 'linear' or 'cubic'
    
    Output:
      z = the interpolated depth values with a spatial resolution of 0.000001 degrees (CRS:4326).
      lon = the original longitude measurements of the sonar
      lat = the original latitude measurements of the sonar
      depth = the original depth measurements of the sonar
      x_min = the minimum longitude of the interpolated points
      x_max = the maximum longitude of the interpolated points
      y_min = the minimum latitude of the interpolated points
      y_max = the maximum latitude of the interpolated points
    
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    Created by: Sofie Schijvenaars, 19/09/2022
    """""

    depth = self.data.ds.water_depth[0]
    lat = self.data.ds.latitude[0]
    lon = self.data.ds.longitude[0]

    x_min = min(lon) - 0.0005
    x_max = max(lon) + 0.0005
    y_min = min(lat) - 0.0005
    y_max = max(lat) + 0.0005

    grid_x = np.arange(x_min, x_max, 0.000001)
    grid_y = np.arange(y_min, y_max,0.000001)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    z = griddata((lon,lat), depth, (grid_xx, grid_yy), method)
    self.interpolated_depth = z
    return

  ####################################################################################

  # function to interpolate flag data
  def __interp_flag(self, method = 'linear'):
    """""
    A function to interpolate the flag of a sonar measurement campaign.
    See scipy.interpolate.griddata for more information on the interpolation method used.

    Input:
      df = the dataset which contains the depth of the sonar campaign. It can also be a list of dictionaries.
      method = the method used for interpolating: 'nearest', 'linear' or 'cubic'
    
    Output:
      z = the interpolated depth values with a spatial resolution of 0.000001 degrees (CRS:4326).
      lon = the original longitude measurements of the sonar
      lat = the original latitude measurements of the sonar
      depth = the original depth measurements of the sonar
      x_min = the minimum longitude of the interpolated points
      x_max = the maximum longitude of the interpolated points
      y_min = the minimum latitude of the interpolated points
      y_max = the maximum latitude of the interpolated points
    
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    Created by: Sofie Schijvenaars, 19/09/2022
    """""

    flags = self.data.flags
    lat = self.data.ds.latitude[0]
    lon = self.data.ds.longitude[0]

    x_min = min(lon) - 0.0005
    x_max = max(lon) + 0.0005
    y_min = min(lat) - 0.0005
    y_max = max(lat) + 0.0005

    grid_x = np.arange(x_min, x_max, 0.000001)
    grid_y = np.arange(y_min, y_max,0.000001)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    z = griddata((lon,lat), flags, (grid_xx, grid_yy), method)
    z = (z > 0.5)*1
    self.interpolated_flags = z
    return

  ####################################################################################

  # method to visualize the interpolated depth data
  def plot_interpolated_depth(self, method = 'linear'):
    """
    This function plots the interpolated depth.
    """
    print('Interpolating...')
    self.__interp_depth(method)
    print('Visualizing...')
    depth = self.data.ds.water_depth[0]
    lat = self.data.ds.latitude[0]
    lon = self.data.ds.longitude[0]

    plt.rcParams['figure.figsize'] = [15, 10]
    fig, ax = plt.subplots()
    landMask = gpd.read_file(self.landmask)
    # plot landmask first
    landMask.plot(ax=ax,color='grey', edgecolor='black')

    ax.set_xlim(min(lon) - 0.0015, max(lon) + 0.0015)
    ax.set_ylim(min(lat) - 0.0015, max(lat) + 0.0015)

    i = ax.imshow(self.interpolated_depth,
              extent=(min(lon) - 0.0005, max(lon) + 0.0005, min(lat) - 0.0005, max(lat) + 0.0005),
              origin='lower',alpha = 0.5)
    ax.scatter(lon, lat, c=depth, s=0.5,label='Original measurements')
    fig.colorbar(i,ax=ax,label='Depth [m]')
    ax.legend()
    plt.show()
    return

  ####################################################################################

  # method to visualize the raw depth data
  def plot_raw_depth(self):
    """
    This function plots the raw depth data.
    """
    depth = self.data.ds.water_depth[0]
    lat = self.data.ds.latitude[0]
    lon = self.data.ds.longitude[0]

    plt.rcParams['figure.figsize'] = [15, 10]
    fig, ax = plt.subplots()
    landMask = gpd.read_file(self.landmask)
    # plot landmask first
    landMask.plot(ax=ax,color='grey', edgecolor='black')

    ax.set_xlim(min(lon) - 0.0015, max(lon) + 0.0015)
    ax.set_ylim(min(lat) - 0.0015, max(lat) + 0.0015)

    i = ax.scatter(lon, lat, c=depth, s=0.5,label='Original measurements')
    fig.colorbar(i,ax=ax,label='Depth [m]')
    ax.legend()
    plt.show()
    return