from unit import Unit
from exception import InvalidValueException

"""
converts parameters to and from their raw form (such as "11m:15s") to "675s" or just 675
"""
class ParamFormatter(object):

    def __init__(self):
        self.param_formatters = {"Duration" : self.duration_to_number,
                            "Distance" : self.default_to_number,
                            "Max. Speed" : self.minmi_to_number,
                            "Avg. Speed" : self.minmi_to_number,
                            "user_id" : self.string_to_int,
                            "workout_id" : self.string_to_int,
                            "Weather" : self.null_converter,
                            "sport" : self.null_converter,
                            "Hydration" : self.hydration_to_number,
                            "Calories" : self.default_to_number,
                            "Min. Altitude" : self.default_to_number,
                            "Max. Altitude" : self.default_to_number,
                            "Total Ascent" : self.default_to_number,
                            "Total Descent" : self.default_to_number,
                            "date-time" : self.null_converter,
                            "Cadence" : self.default_to_number}

    def to_number(self, param, value, with_unit = False):
        # convert "11m:15s" to "675s" (if with_unit = True) or 675 if with_unit = False)
        if (not Unit.is_defined(param)):
            raise Exception("Unit for the given parameter %s has not been defined" % (param))

        if (self.param_formatters.has_key(param)):
            f = self.param_formatters[param]
            v = f(param, value)
        else:
            raise Exception("Unknown parameter %s in ParamFormatter" % (param))

        if (with_unit):
            return str(v) + Unit.get(param)
        else:
            return v
    
    def duration_to_number(self, param, value):
        parts = value.split(":")
        v = 0
        assert(Unit.get("Duration") == "s")  # following code assumes the unit is seconds
        for p in parts:
            p_value = int(p[0:len(p) - 1])
            p_unit = p[len(p) - 1]
            if (p_unit == "s"):
                v += p_value
            elif(p_unit == "m"):
                v += p_value * 60
            elif(p_unit == "h"):
                v += p_value * 3600
            else:
                raise Exception("Invalid time unit %s" % (p_unit))
        return v

    def minmi_to_number(self, param, value):
        # string is of the form "12:03 min/mi" or "12.3 mph"
        if (value == "-"):
            raise InvalidValueException(param, value)
        parts = value.split(" ")
        unit = parts[1]
        if (unit == "min/mi"):
            parts = parts[0].split(":")
            v = float(parts[0]) + float(parts[1])/60.0  # convert to float minutes value
        else:
            v = float(parts[0])
        if (Unit.get(param) != unit):
            v = Unit.convert(unit, Unit.get(param), v)  # convert units
        return v

    def hydration_to_number(self, param, value):
        # string is of the form "32.3L"
        unit = value[-1]
        assert(unit == "L")
        assert(Unit.get("Hydration") == unit)
        return float(value[:-1])

    def default_to_number(self, param, value):
        # default format is "<value> <unit>"
        if (value == "-"):
            raise InvalidValueException(param, value)
        parts = value.split()
        return float(parts[0])

    def null_converter(self, param, value):
        return value

    def string_to_int(self, param, value):
        return int(value)

if __name__ == "__main__":
    p = ParamFormatter()
    print p.to_number("Distance","1.15 mi")
    print p.to_number("Duration","21m:24s")
    print p.to_number("Duration","24s")
