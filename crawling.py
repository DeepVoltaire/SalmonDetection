from utils import *

fish_coordinates = [
    (52.591901, -128.521975),
    (50.882396, -127.496930),
    (50.878538, -126.902387),
    (50.854201, -126.758715),
    (50.850617, -126.716857),
    (50.847869, -126.319231),
    (50.837714, -126.663131),
    (50.834701, -126.496432),
    (50.832302, -127.520585),
    (50.831666, -126.597147),
    (50.796923, -126.495746),
    (50.749069, -127.683667), # found with NN
    (50.746123, -127.647787), # found with NN
    (50.743125, -127.448952),
    (50.733599, -127.444264),
    (50.707619, -126.664039),
    (50.697870, -126.256680),
    (50.670949, -126.476713),
    (50.656792, -126.665785), #new
    (50.653974, -126.289050),
    (50.649603, -126.618208), #new
    (50.618472, -126.704306),
    (50.607944, -126.363262),
    (50.601005, -126.348642),
    (50.566306, -126.274258),
    (50.594348, -127.571985),
    (50.488321, -125.356540),
    (50.485700, -127.876225),
    (50.474602, -127.787872),
    (50.473781, -125.808549),
    (50.468923, -127.835091),
    (50.458708, -127.890031),
    (50.425624, -125.305282),
    (50.414876, -125.768263),
    (50.414630, -125.659615),
    (50.409818, -125.700438),
    (50.408549, -125.340233),
    (50.392891, -125.362241),
    (50.349968, -125.342870),
    (50.324312, -125.260480),
    (50.309412, -125.316563),
    (50.302343, -125.338198),
    (50.286035, -125.349444),
    (50.178898, -125.327057),
    (50.177378, -125.344332),
    (50.046627, -127.296477),
    (50.037817, -127.176002),
    (50.020139, -127.193640),
    (50.028678, -124.717693),# new
    (49.891797, -126.814856), #new
    (49.886826, -126.791493),
    (49.879720, -126.760589),
    (49.867223, -126.759452),
    (49.854614, -124.227246),
    (49.854365, -124.218791),
    (49.854393, -124.231923), #new
    (49.835393, -124.055283), # new
    (49.796591, -124.097778), # new
    (49.779334, -124.153914), # new
    (49.672485, -124.086384),
    (49.670991, -123.861535),
    (49.659395, -126.479839),
    (49.658978, -126.452150), #new
    (49.656206, -126.431379), #new
    (49.647667, -123.724143),
    (49.640080, -123.658476),
    (49.638588, -126.329635),
    (49.635475, -123.782979),
    (49.628771, -123.845134),
    (49.615316, -123.834076), #new
    (49.615065, -126.057087), #new
    (49.598437, -123.787204),
    (49.565561, -123.781296),
    (49.512665, -123.491802),
    (49.459149, -126.235764), # new
    (49.404472, -126.152111),
    (49.375583, -126.091917),
    (49.341726, -125.952694), # new
    (49.327123, -125.798584),
    (49.324218, -126.048529),
    (49.307867, -126.007018),
    (49.305597, -125.816070),
    (49.294125, -126.070052),
    (49.279077, -125.830551),
    (49.265400, -125.812512),
    (49.258769, -125.870399),
    (49.256889, -125.841672),
    (49.247028, -125.905806),
    (49.235057, -125.751812),
#   (49.225361, -125.757639), # in between two
    (49.212221, -125.887158),
    (49.214303, -125.767151),
    (49.165746, -125.769318),
    (49.133107, -125.783026),
    (49.129157, -125.753921),
    (49.119247, -125.725661),
    (49.014239, -125.030291),
    (48.943252, -124.987273),
    (48.938815, -124.991092),
    (48.814243, -124.667558),
    (48.582611, -124.348857),
    
    # OTHER FISHERIES
    (52.798845, -128.401589),
    (52.795894, -128.311540),
    (52.785498, -128.331592),
    (52.785389, -128.414116),
    (50.965119, -127.453544),
    (50.906201, -127.341426),
    (50.901341, -127.938465),
    (50.821342, -127.555000),
    (50.721725, -126.568060),
    (50.521590, -128.008791),
    (50.533085, -126.226017),
    (49.862455, -124.224756),
    (49.858030, -124.215966),
]

fish_coordinates = [
    (49.512665, -123.491802),
]

what_is_this = [
    (50.029086, -124.747982),
    (50.095887, -125.017094),
    (50.100317, -125.005351)
]

delete_from_no_nets = [
    "2019-11-17_49.259,-125.909_zoom16.png",
    "2019-11-17_50.011,-127.176_zoom16.png",
    "2019-11-17_49.208,-125.752_zoom16.png",
    "2019-11-17_49.279,-125.816_zoom16.png",
    "2019-11-17_49.657,-126.412_zoom16.png",
    "2019-11-17_49.657,-126.412_zoom16.png",
    "2019-11-17_50.824,-126.496_zoom16.png",
    "2019-11-17_49.634,-126.312_zoom16.png",
    "2019-11-17_50.469,-127.873_zoom16.png",
    "2019-11-17_49.887,-126.83_zoom16.png",
    "2019-11-17_49.306,-125.831_zoom16.png",
    "2019-11-17_50.459,-127.876_zoom16.png",
    "2019-11-17_49.247,-125.867_zoom16.png",
    "2019-11-17_50.459,-127.852_zoom16.png",
    "2019-11-17_50.047,-127.194_zoom16.png",
    "2019-11-17_50.459,-127.838_zoom16.png",
    "2019-11-17_49.3,-125.799_zoom16.png",
    "2019-11-17_49.241,-125.767_zoom16.png",
    "2019-11-17_49.333,-125.816_zoom16.png",
    "2019-11-17_50.539,-126.236_zoom16.png",
    "2019-11-17_50.486,-127.89_zoom16.png",
    "2019-11-17_50.496,-127.873_zoom16.png",
    "2019-11-17_49.241,-125.767_zoom16.png",
    "2019-11-17_50.824,-126.496_zoom16.png",
    "2019-11-17_49.241,-125.767_zoom16.png",
    "2019-11-17_49.624,-126.074_zoom16.png",
         ]

LONG_PER_KM = 0.0118  # 0.0118 long at this latitude is roughly 1 km
LAT_PER_KM = 0.009  # 0.009 lat is roughly 1 km

# Adapted from Charles He
class SatImageryRequestHandler:
    def __init__(self, _save_dir="sat_downloads", driver=None, filename_prefix='CF13_Jun_13_screenshot', zoom_level=17):
        """ Handles imagery request """
        self.zoom_level = zoom_level
        self._save_dir = _save_dir
        self.driver = driver
        self.filename_prefix = filename_prefix

    def request_screenshot(self, lat, long):
        """ Wednesday, November 14, 2018 (11-14-18)
        Searches or creates a new screenshot
        """
        # round lat/long to make coarser coordinates, reduces excessive replication
        _lat = round(lat, 3)
        _long = round(long, 3)

        self.savepath =  "{}/{}_{},{}_zoom{}.png".format(self._save_dir, self.filename_prefix, _lat, _long, self.zoom_level)
        os.makedirs(self._save_dir, exist_ok=True)

        # find file in directory
        if not os.path.exists(self.savepath):
            print(f"Obtaining screenshot at coordinates {_lat}, {_long}...", end='')
            self._get_screenshot_from_bing(long, lat)
            return True
        else:
            print(f'Screenshot appears to exist: {self.savepath}')
            return False

    def _get_screenshot_from_bing(self, _long, _lat):
        """ Tuesday, November 13, 2018 (11-13-18)
        Get screenshot from an address"""

        # Using style=a, because it shows the aerial view of the map
        # style=h would be aerial view with labels
        # more here: https://docs.microsoft.com/en-us/bingmaps/articles/create-a-custom-map-url
        _other_params = '&lvl={}&style=a'.format(self.zoom_level)

        # note that bing uses latitude first, longitude second
        url = 'https://www.bing.com/maps?cp=' + str(_lat) + '~' + str(_long) + _other_params
        self.driver.set_window_size(1800, 1800)

        self.driver.get(url)
        time.sleep(.9)

        # remove classes
        remove_class_list = ['b_footer', 'b_hPanel', 'taskBar']

        for class_to_remove in remove_class_list:
            try:
                element = self.driver.find_element_by_class_name(class_to_remove)

                self.driver.execute_script("""
                var element = arguments[0];
                element.parentNode.removeChild(element);
                """, element)
            except Exception as e:
                print(e)
            #(NoSuchElementException, StaleElementReferenceException):
                pass

        # remove ids
        remove_id_list = ['MicrosoftNav', 'id_h', 'b_notificationContainer']
        for id_to_remove in remove_id_list:
            try:
                element = self.driver.find_element_by_id(id_to_remove)
                self.driver.execute_script("""
                        var element = arguments[0];
                        element.parentNode.removeChild(element);
                        """, element)
            except Exception as e:
                print(e)
            #(NoSuchElementException, StaleElementReferenceException):
                pass

        time.sleep(1.4)

        self.driver.save_screenshot(str(self.savepath))
        print('Screenshot obtained at {}.'.format(str(self.savepath)))


def get_total_memory_percent(app_name='firefox.exe'):
    """
    Friday, June 14, 2019

    Returns total memory usage

    :param app_name: application name to get all processes of
    :return:
    """
    total_memory = 0
    for proc in psutil.process_iter():
        if proc.name() == 'firefox.exe':
            total_memory += proc.memory_percent()

    return total_memory

def load_webdriver():
    # Installation:
    # !pip install selenium
    # !conda install -c conda-forge geckodriver --yes

    # im Terminal:
    # apt-get update
    # apt install firefox
    
    profile = webdriver.FirefoxProfile()
    options = Options()
    options.headless = True # because we do not have any monitor
    driver = webdriver.Firefox(firefox_profile=profile, options=options)
    # executable_path="/usr/local/bin/geckodriver", 
#     executable_path='/media/austausch/popien/MachineLearning/CharlesProject/FishProject/geckodriver', 
    return driver


# +
def get_random_crops_around_nets(zoom_levels=[16], _save_dir="data/no_nets", 
                                 filename_prefix='2019-11-17', download_how_many=100):
    max_lat = max([k for k,v in fish_coordinates])
    min_lat = min([k for k,v in fish_coordinates])
    diff_lat = max_lat-min_lat
    max_long = max([v for k,v in fish_coordinates])
    min_long = min([v for k,v in fish_coordinates])
    diff_long = max_long-min_long
    
    coords = [j.split("/")[-1].split("_")[1] for j in glob.glob("data/no_nets/*")] + [
        j.split("/")[-1].split("_")[1] for j in glob.glob("data/inference_random/*")]
    no_fish_coordinates = []
    for coord in coords:
        lat, long = map(float, coord.split(","))
        no_fish_coordinates.append((lat, long))
    
    download_count = 0
    while True:
        try:
            driver = load_webdriver()
            for zoom_level in zoom_levels:
                rr = SatImageryRequestHandler(driver=driver, _save_dir=_save_dir, 
                                              filename_prefix=filename_prefix, zoom_level=zoom_level)
                while True:
                    skip = False
                    new_lat = min_lat+(diff_lat*np.random.random())
                    new_long = min_long+(diff_long*np.random.random())
                    for lat, long in fish_coordinates:
                        if abs(new_lat-lat)<LAT_PER_KM*3 and abs(new_long-long)<LONG_PER_KM*3:
                            print("Discard Lat {:.3f} and Long {:.3f} as its too close to {:.3f} and {:.3f}".format(new_lat, new_long, lat, long))
                            skip= True
                            break
                    for lat, long in no_fish_coordinates:
                        if abs(new_lat-lat)<LAT_PER_KM*3 and abs(new_long-long)<LONG_PER_KM*3:
                            print("Discard Lat {:.3f} and Long {:.3f} as its too close to {:.3f} and {:.3f}".format(new_lat, new_long, lat, long))
                            skip= True
                            break
                    if skip==True:
                        continue
                    img_downloaded = rr.request_screenshot(lat=new_lat, long=new_long)
                    if img_downloaded: 
                        download_count += 1
                        no_fish_coordinates.append((new_lat, new_long))
                    if get_total_memory_percent() > 25:
                        # driver can load too much memory, then we need to reload it
                        print(f"Total memory used by webdriver reported to be {get_total_memory_percent():.2f}% of system,"
                              f" loading new webdriver")
                        driver.quit()
                        driver = load_webdriver()
                    if download_count==download_how_many:
                        return
        except Exception as e:
            print(e)
            print("Start again.")
            
def get_crops_around_nets(zoom_levels=[16], _save_dir="data/no_nets", filename_prefix='2019-11-17'):
    while True:
        try:
            driver = load_webdriver()
            for zoom_level in zoom_levels:
                    rr = SatImageryRequestHandler(driver=driver, _save_dir=_save_dir, filename_prefix=filename_prefix, zoom_level=zoom_level)
                    for coords in fish_coordinates:
                #         rr.request_screenshot(lat=coords[0], long=coords[1])
                        for j in [-1, 0, 1]:
                            for k in [-1, 0, 1]:
                                skip = False
                                if j==0 and k==0: continue
                                elif j==-1 and k==-1: continue
                                elif j==1 and k==1: continue
                                new_lat = coords[0]+j*LAT_PER_KM*3
                                new_long = coords[1]+k*LONG_PER_KM*3.25
                                for lat, long in fish_coordinates:
                                    if abs(new_lat-lat)<LAT_PER_KM and abs(new_long-long)<LONG_PER_KM:
                                        print("Discard Lat {:.3f} and Long {:.3f} as its too close to {:.3f} and {:.3f}".format(new_lat, new_long, lat, long))
                                        skip= True
                                        break
                                if skip==True:
                                    continue
                                rr.request_screenshot(lat=new_lat, long=new_long)
                                if get_total_memory_percent() > 25:
                                    # driver can load too much memory, then we need to reload it
                                    print(f"Total memory used by webdriver reported to be {get_total_memory_percent():.2f}% of system,"
                                          f" loading new webdriver")
                                    driver.quit()
                                    driver = load_webdriver()
            break
        except Exception as e:
            print(e)
            print("Start again.")
