import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(myPath,'..'))
from param_formatter import ParamFormatter

def test_param_formatter():
    p = ParamFormatter()
    assert(p.to_number("Distance","1.15 mi") == 1.15)
    assert(p.to_number("Duration","21m:24s") == 1284)
    assert(p.to_number("Duration","24s") == 24)
    assert(p.to_number("Max. Speed", "4.3 mph") == 13.953488)
    assert(p.to_number("Max. Speed", "4,300 mph") == 0.013953)
    assert(p.to_number("Max. Speed", "4.3 min/mi") == 4.3)
    assert(p.to_number("Hydration", "3.17L") == 3.17)
    assert(p.to_number("Avg. Heart Rate", "124") == 124.0)

if __name__ == "__main__":
    test_param_formatter()
