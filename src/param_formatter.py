from unit import Unit
from exception import InvalidValueException, InvalidParamException

"""
converts parameters to and from their raw form (such as "11m:15s") to "675s" or just 675
"""
class ParamFormatter(object):

    def __init__(self, precision = 6):
        self.precision = precision  # number of digits after decimal point
        self.param_formatters = {"Duration" : self.duration_to_number,
                            "Distance" : self.value_space_unit_to_number,
                            "Max. Speed" : self.value_space_unit_to_number,
                            "Avg. Speed" : self.value_space_unit_to_number,
                            "user_id" : self.null_converter,
                            "workout_id" : self.null_converter,
                            "Weather" : self.null_converter,
                            "sport" : self.null_converter,
                            "Hydration" : self.value_unit_to_number,
                            "Calories" : self.value_space_unit_to_number,
                            "Min. Altitude" : self.value_space_unit_to_number,
                            "Max. Altitude" : self.value_space_unit_to_number,
                            "Total Ascent" : self.value_space_unit_to_number,
                            "Total Descent" : self.value_space_unit_to_number,
                            "Avg. Heart Rate" : self.value_to_number,
                            "Max. Heart Rate" : self.value_to_number,
                            "HR After Test" : self.value_to_number,
                            "Temperature" : self.value_to_number,
                            "date-time" : self.null_converter,
                            "Humidity" : self.value_unit_to_number,
                            "Cadence" : self.value_space_unit_to_number,
                            "Wind" : self.value_space_unit_to_number,
                            "Steps" : self.value_space_unit_to_number,
                            "Fitness Score" : self.value_to_number,
                            "Fitness Level" : self.null_converter,
                            "Avg. Steps/Min" : self.value_space_unit_to_number}

    def to_number(self, param, value):
        # convert "11m:15s" to 675
        if (not Unit.is_defined(param)):
            #raise Exception("Unit for the given parameter %s has not been defined" % (param))
            raise InvalidParamException(param)

        if (self.param_formatters.has_key(param)):
            f = self.param_formatters[param]
            v = f(param, value)
        else:
            raise Exception("Unknown parameter %s in ParamFormatter" % (param))

        return v
    
    def duration_to_number(self, param, value):
        # the value is of the form 16m:31s
        parts = value.split(":")
        v = 0
        assert(param == "Duration")
        assert(Unit.get("Duration") == "s")  # following code assumes the unit is seconds
        for p in parts:
            #p_value = int(p[0:len(p) - 1])
            p_value = self.str_to_float(param, p[0:len(p) - 1])
            p_unit = p[len(p) - 1]
            if (p_unit == "s"):
                v += p_value
            elif(p_unit == "m"):
                v += p_value * 60
            elif(p_unit == "h"):
                v += p_value * 3600
            elif(p_unit == "d"):
                v += p_value * 86400
            else:
                raise Exception("Invalid time unit.. param = %s, value = %s" % (param, value))
        return v

    """
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
        v = round(v, self.precision)
        return v
    """

    def value_space_unit_to_number(self, param, value):
        # default format is "<value> <unit>"
        if (value == "-"):
            raise InvalidValueException(param, value)
        parts = value.split()
        unit = parts[1]
        if (unit == "min/mi" and (":" in parts[0])):
            v_parts = parts[0].split(":")
            #v = float(v_parts[0]) + float(v_parts[1])/60.0  # convert to float minutes value
            v = self.str_to_float(param, v_parts[0]) + self.str_to_float(param, v_parts[1])/60.0  # convert to float minutes value
        else:
            #v = float(parts[0])
            v = self.str_to_float(param, parts[0])
        if (Unit.get(param) != unit):
            v = Unit.convert(unit, Unit.get(param), v)
        return round(v, self.precision)

    def value_unit_to_number(self, param, value):
        # string is of the form "32.3L"
        unit = value[-1]
        #v = float(value[:-1])
        v = self.str_to_float(param, value[:-1])
        if (Unit.get(param) != unit):
            v = Unit.convert(unit, Unit.get(param), v)
        return round(v, self.precision)
    
    def null_converter(self, param, value):
        return value

    def value_to_number(self, param, value):
        #return round(float(value), self.precision)
        return round(self.str_to_float(param, value), self.precision)

    def str_to_float(self, param, value):
        try:
            return float(value.replace(",",""))
        except ValueError:
            #print "param = %s, value = %s" % (param, value)
            raise InvalidValueException(param, value)

if __name__ == "__main__":
    p = ParamFormatter()
    print p.to_number("Distance","1.15 mi")
    print p.to_number("Duration","21m:24s")
    print p.to_number("Duration","24s")
