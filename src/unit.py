
"""
defines the unit for each parameter in the data
"""
class Unit(object):
    params = {"alt":"ft",
                "distance" : "mi",
                "hr" : "bpm",
                "Avg. Heart Rate" : "bpm",
                "Max. Heart Rate" : "bpm",
                "HR After Test" : "bpm",
                "speed" : "mph",
                "pace" : "min/mi",
                "Distance" : "mi",
                "Duration" : "s",
                "Max. Speed" : "min/mi",
                "Avg. Speed" : "min/mi",
                "Hydration" : "L",
                "Total Ascent" : "ft",
                "Total Descent" : "ft",
                "Max. Altitude" : "ft",
                "Min. Altitude" : "ft",
                "Calories" : "kcal",
                "Cadence" : "rpm",
                "Temperature" : "F",
                "Humidity" : "%",
                "Wind" : "mph",
                "Avg. Steps/Min" : "steps",
                "Steps" : "steps",
                "Fitness Score" : None,
                "Fitness Level" : None,
                "date-time" : None,
                "user_id" : None,
                "workout_id" : None,
                "Weather" : None,
                "sport" : None}
    @staticmethod
    def get(param):
        if ("(avg)" in param):
            param = param.replace("(avg)","")
        if (Unit.params.has_key(param)):
            return Unit.params[param]
        else:
            return "UNKNOWN"

    @staticmethod
    def is_defined(param):
        return (Unit.params.has_key(param))

    @staticmethod
    def minmi_to_mph(minmi):
        return 1.0 / (float(minmi) / 60.0)

    @staticmethod
    def mph_to_minmi(mph):
        return (1.0 / (float(mph)) * 60.0)

    @staticmethod
    def convert(src_unit, dest_unit, src_value):
        func_name = src_unit.replace("/","") + "_to_" + dest_unit.replace("/","")
        return getattr(Unit, func_name)(src_value)

if __name__ == "__main__":
    print Unit.convert("min/mi","mph",1.0)

