"""
Proyecto Vision Artificial 201910

Author: Andres Howard, Carlos Diaz
"""
# import threading
# import time
import argparse
import cv2
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision


# set this to true if you want to fly for the demo
TEST_FLIGHT = True


class UserVision:
    """ Vision class for Parrot Vision
    """
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self):
        """ Save pictures from Vision
        """
        print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()

        if img is not None:
            filename = "test_image_%06d.png" % self.index
            cv2.imwrite(filename, img)
            self.index += 1


def arguments():
    """ Parse command line arguments
    Return:
        array
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-a', '--address',
                                 default='d0:3a:4d:78:e6:36',
                                 help='mac address of drone')
    args = vars(argument_parser.parse_args())
    return args


def fly(address):
    """ Fly the drone located in address
    """
    mambo = Mambo(address, use_wifi=True)
    print("trying to connect")
    success = mambo.connect(num_retries=3)
    print("connected:", success)

    if success:
        # get the state information
        print("sleeping")
        mambo.smart_sleep(1)
        mambo.ask_for_state_update()
        mambo.smart_sleep(1)

        print("Preparing to open vision")
        mambo_vision = DroneVision(mambo, is_bebop=False, buffer_size=30)
        user_vision = UserVision(mambo_vision)
        mambo_vision.set_user_callback_function(user_vision.save_pictures,
                                                user_callback_args=None)
        success = mambo_vision.open_video()
        print("Success in opening vision is,", success)

        if success:
            print("Vision successfully started!")

            if TEST_FLIGHT:
                print("taking off")
                mambo.safe_takeoff(5)

                if mambo.sensors.flying_state != "emergency":
                    print("flying state is ", mambo.sensors.flying_state)
                    print('altitude: ', mambo.sensors.altitude)

                print("landing")
                mambo.safe_land(5)
            else:
                print("Sleeeping for 15 seconds - move the mambo around")
                mambo.smart_sleep(15)

            print("Ending the sleep and vision")
            mambo_vision.close_video()

            mambo.smart_sleep(5)

        print("disconnecting")
        mambo.disconnect()


if __name__ == '__main__':
    ARGS = arguments()
    fly(ARGS['address'])
