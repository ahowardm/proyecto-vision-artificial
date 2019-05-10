""" Settings file for instruction scanner
"""
import settings


def test():
    """ Return a list of numbers to test main
    Return:
      list of strings
    """
    return [settings.GO_UP, settings.TURN_LEFT, settings.GO_FORWARD,
            settings.TURN_RIGHT, settings.GO_FORWARD]

if __name__ == '__main__':
    print(test())
