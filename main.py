'''
Proyecto Vision Artificial 201910

Author: Andres Howard, Carlos Diaz
'''
# import threading
# import time
import argparse
import os
import cv2
import inspect
import time

from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision

from instructionscanner import settings
from instructionscanner.process import get_board_commands


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
        print('in save pictures on image %d ' % self.index)

        img = self.vision.get_latest_valid_picture()

        if img is not None:
            filename = 'mambo_image_%06d.png' % self.index
            cv2.imwrite(filename, img)
            self.index += 1

    def get_frame(self):
        """ Calls save_pictures and XXX function from instruction scanner
        """
        self.save_pictures()
        #self.close_video()
        self.instructions = test.test()
        print(self.instructions)


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


def run_instructions(mambo, instructions):
    """ Run instructions from instruction scanner
    """
    assert isinstance(mambo, Mambo), 'Mambo must be a Mambo instance'
    assert isinstance(instructions, list), 'Instructions must be a string'
    print('running instructions')
    for instruction in instructions:
        if instruction == settings.GO_UP:
            print('going up')
            mambo.fly_direct(roll=0, pitch=0, yaw=0,
                             vertical_movement=40, duration=2)
        elif instruction == settings.TURN_LEFT:
            print('turning left')
            mambo.turn_degrees(-90)
        elif instruction == settings.TURN_RIGHT:
            print('turning right')
            mambo.turn_degrees(90)
        elif instruction == settings.BACKFLIP:
            mambo.flip(direction='back')
        elif instruction == settings.GO_FORWARD:
            print('going forward')
            mambo.fly_direct(roll=0, pitch=50, yaw=0,
                             vertical_movement=0, duration=2)
        elif instruction == settings.GO_BACKWARD:
            print('going backward')
            mambo.fly_direct(roll=0, pitch=-50, yaw=0,
                             vertical_movement=0, duration=2)


def fly(address):
    """ Fly the drone located in address
    """
    mambo = Mambo(address, use_wifi=True)
    print('trying to connect')
    success = mambo.connect(num_retries=3)
    print('connected:', success)

    if success:
        # get the state information
        print('sleeping')
        mambo.smart_sleep(1)
        mambo.ask_for_state_update()
        mambo.smart_sleep(1)

        if success:
            print('Vision successfully started!')

            if TEST_FLIGHT:
                print('taking off')
                mambo.safe_takeoff(5)

                print('Preparing to open vision')
                mambo_vision = DroneVision(
                    mambo, is_bebop=False, buffer_size=30)
                user_vision = UserVision(mambo_vision)
                mambo_vision.set_user_callback_function(user_vision.get_frame,
                                                        user_callback_args=None)
                directory = os.path.join(os.path.dirname(inspect.getfile(DroneVision)), 'images')
                print('Las fotos están en:', directory)
                success = mambo_vision.open_video()
                print('Success in opening vision is,', success)

                time.sleep(1)
                mambo_vision.close_video()
                instructions = get_board_commands(directory)
                print(instructions)
                #run_instructions(mambo, instructions)

                #if user_vision.instructions and user_vision.instructions != 'ERROR':
                #    try:
                #        print('Ending the sleep and vision')
                #        mambo_vision.close_video()
                #        run_instructions(mambo, user_vision.instructions)
                #    except Exception as ex:
                #        print(ex)
                #        print('landing')
                #        mambo.safe_land(5)

                if mambo.sensors.flying_state != 'emergency':
                    print('flying state is ', mambo.sensors.flying_state)
                    print('altitude: ', mambo.sensors.altitude)

                print('landing')
                mambo.safe_land(5)
            else:
                print('Sleeeping for 15 seconds - move the mambo around')
                mambo.smart_sleep(15)

            mambo.smart_sleep(5)

        # mambo_vision.close_video()
        print('disconnecting')
        
        mambo.disconnect()


if __name__ == '__main__':
    ARGS = arguments()
    fly(ARGS['address'])
